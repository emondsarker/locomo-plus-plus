"""
Evaluation script for the persuasion memory benchmark.

Runs model responses through an LLM judge and computes aggregate scores.
Also generates a human evaluation template.

Usage:
    python eval/evaluate.py \
        --responses data/responses/model_responses.json \
        --triggers data/filtered/triggers_filtered.json \
        --cues data/filtered/cues_filtered.json \
        --output eval/results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.utils import call_llm, parse_json_from_response, load_json, save_json
from judge_prompt import format_judge_prompt


LABEL_SCORES = {"correct": 2, "partial": 1, "wrong": 0, "stale": -1}


def run_llm_judge(triggers, cues, responses, judge_model="gpt-4o"):
    """Run LLM judge on all trigger responses.

    The judge outputs labels: correct, partial, or wrong.
    Stale detection is handled in post-processing (apply_stale_labels)
    to avoid anchoring the judge. See methodology/05_bias_and_validity.md §T2.
    """
    # Index cues by (user_id, topic) so we can provide full evidence
    # for Phase 2 triggers (all phases, not just same-phase cues).
    # See methodology/05_bias_and_validity.md §T3.
    cues_by_user_topic = {}
    for c in cues:
        key = (c["user_id"], c["topic"])
        cues_by_user_topic.setdefault(key, []).append(c)

    results = []
    for trigger in triggers:
        tid = trigger["trigger_id"]
        model_response = responses.get(tid, "")

        if not model_response:
            results.append({
                "trigger_id": tid,
                "label": "wrong",
                "principle_detected": "none",
                "memory_awareness": False,
                "phase_awareness": False,
                "reason": "No response provided",
            })
            continue

        # Provide all cues for this user-topic pair so the judge sees
        # the full preference timeline (Phase 1 + drift + Phase 2).
        related_cues = cues_by_user_topic.get(
            (trigger["user_id"], trigger["topic"]), []
        )

        prompt = format_judge_prompt(trigger, related_cues, model_response)

        try:
            response = call_llm(prompt, model=judge_model, temperature=0.0, max_tokens=512)
            judgment = parse_json_from_response(response)
            judgment["trigger_id"] = tid
            results.append(judgment)
        except Exception as e:
            print(f"  Judge error for {tid}: {e}")
            results.append({
                "trigger_id": tid,
                "label": "wrong",
                "principle_detected": "unknown",
                "memory_awareness": False,
                "phase_awareness": False,
                "reason": f"Judge error: {e}",
            })

    return results


def apply_stale_labels(judgments, triggers):
    """Post-processing: reclassify Phase 2 judgments as 'stale' when the
    judge-detected principle matches the outdated Phase 1 effective principle.

    This is separated from the judge to prevent anchoring bias.
    The judge identifies the principle; this function determines staleness.
    See methodology/05_bias_and_validity.md §T2.
    """
    trigger_by_id = {t["trigger_id"]: t for t in triggers}

    for j in judgments:
        trigger = trigger_by_id.get(j["trigger_id"])
        if not trigger:
            continue

        # Only Phase 2 triggers can be stale
        if trigger.get("phase") != 2:
            continue

        stale_principle = trigger.get("stale_principle")
        if not stale_principle:
            continue

        detected = j.get("principle_detected", "").lower().replace(" ", "_")
        if detected == stale_principle:
            j["label"] = "stale"
            j["reason"] = (
                f"Reclassified as stale: model used {stale_principle} "
                f"(Phase 1 effective) on a Phase 2 trigger. "
                f"Original judge reason: {j.get('reason', '')}"
            )

    return judgments


def compute_metrics(judgments, triggers):
    """Compute aggregate metrics from judge results, including phase-aware metrics."""
    trigger_by_id = {t["trigger_id"]: t for t in triggers}

    # Overall
    labels = [j["label"] for j in judgments]
    total = len(labels)
    correct = labels.count("correct")
    partial = labels.count("partial")
    wrong = labels.count("wrong")
    stale = labels.count("stale")
    avg_score = sum(LABEL_SCORES.get(l, 0) for l in labels) / max(total, 1)

    overall = {
        "total": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "stale": stale,
        "correct_pct": round(100 * correct / max(total, 1), 2),
        "partial_pct": round(100 * partial / max(total, 1), 2),
        "wrong_pct": round(100 * wrong / max(total, 1), 2),
        "stale_pct": round(100 * stale / max(total, 1), 2),
        "avg_score": round(avg_score, 4),
        "memory_awareness_pct": round(
            100 * sum(1 for j in judgments if j.get("memory_awareness")) / max(total, 1), 2
        ),
        "phase_awareness_pct": round(
            100 * sum(1 for j in judgments if j.get("phase_awareness")) / max(total, 1), 2
        ),
    }

    # Per topic
    per_topic = defaultdict(lambda: {"correct": 0, "partial": 0, "wrong": 0, "stale": 0, "total": 0})
    for j in judgments:
        t = trigger_by_id.get(j["trigger_id"], {})
        topic = t.get("topic", "unknown")
        per_topic[topic][j["label"]] += 1
        per_topic[topic]["total"] += 1

    for topic in per_topic:
        n = per_topic[topic]["total"]
        per_topic[topic]["correct_pct"] = round(100 * per_topic[topic]["correct"] / n, 2)
        per_topic[topic]["avg_score"] = round(
            sum(LABEL_SCORES.get(l, 0) * per_topic[topic].get(l, 0) for l in LABEL_SCORES)
            / (n * 2) * 100, 2  # normalize to 0-100
        )

    # Per phase (Phase 1 vs Phase 2)
    per_phase = defaultdict(lambda: {"correct": 0, "partial": 0, "wrong": 0, "stale": 0, "total": 0})
    for j in judgments:
        t = trigger_by_id.get(j["trigger_id"], {})
        phase = t.get("phase", 1)
        per_phase[phase][j["label"]] += 1
        per_phase[phase]["total"] += 1

    for phase in per_phase:
        n = per_phase[phase]["total"]
        per_phase[phase]["correct_pct"] = round(100 * per_phase[phase]["correct"] / n, 2)
        per_phase[phase]["avg_score"] = round(
            sum(LABEL_SCORES.get(l, 0) * per_phase[phase].get(l, 0) for l in LABEL_SCORES)
            / (n * 2) * 100, 2
        )

    # Per principle (which effective principle was being tested)
    per_principle = defaultdict(lambda: {"correct": 0, "partial": 0, "wrong": 0, "stale": 0, "total": 0})
    for j in judgments:
        t = trigger_by_id.get(j["trigger_id"], {})
        principle = t.get("effective_principle", "unknown")
        per_principle[principle][j["label"]] += 1
        per_principle[principle]["total"] += 1

    for p in per_principle:
        n = per_principle[p]["total"]
        per_principle[p]["correct_pct"] = round(100 * per_principle[p]["correct"] / n, 2)

    # Drift detection metrics
    phase2_triggers = [t for t in triggers if t.get("phase") == 2]
    phase2_judgments = [
        j for j in judgments
        if trigger_by_id.get(j["trigger_id"], {}).get("phase") == 2
    ]

    drift_metrics = {}
    if phase2_triggers:
        # Drift detection rate: % of Phase 2 triggers answered correctly
        drift_correct = sum(1 for j in phase2_judgments if j["label"] == "correct")
        drift_detection_rate = round(100 * drift_correct / len(phase2_judgments), 2) if phase2_judgments else 0

        # Stale rate: % of Phase 2 triggers where model uses phase 1 principle
        stale_count = sum(1 for j in phase2_judgments if j["label"] == "stale")
        stale_rate = round(100 * stale_count / len(phase2_judgments), 2) if phase2_judgments else 0

        drift_metrics = {
            "phase_2_triggers": len(phase2_triggers),
            "drift_detection_rate_pct": drift_detection_rate,
            "stale_rate_pct": stale_rate,
        }

    return {
        "overall": overall,
        "per_topic": dict(per_topic),
        "per_phase": dict(per_phase),
        "per_principle": dict(per_principle),
        "drift_metrics": drift_metrics,
    }


def generate_human_eval_template(triggers, cues, responses, output_path):
    """Generate a JSON template for human annotators."""
    cue_by_id = {c["cue_id"]: c for c in cues}

    items = []
    for trigger in triggers:
        tid = trigger["trigger_id"]
        model_response = responses.get(tid, "[NO RESPONSE]")
        phase = trigger.get("phase", 1)
        stale_principle = trigger.get("stale_principle")

        related_cues = [
            {
                "cue_id": cid,
                "phase": cue_by_id[cid].get("phase", 1),
                "topic": cue_by_id[cid]["topic"],
                "principle_used": cue_by_id[cid].get("principle_used"),
                "outcome": cue_by_id[cid].get("outcome"),
                "dialogue": cue_by_id[cid]["dialogue"],
            }
            for cid in trigger.get("related_cue_ids", [])
            if cid in cue_by_id
        ]

        # No stale hint for annotators — they should independently identify
        # the principle used and whether it matches outdated preferences.
        # See methodology/05_bias_and_validity.md §T2.
        items.append({
            "trigger_id": tid,
            "user_name": trigger["user_name"],
            "topic": trigger["topic"],
            "phase": phase,
            "effective_principle": trigger["effective_principle"],
            "ineffective_principle": trigger["ineffective_principle"],
            "trigger_text": trigger["trigger_text"],
            "model_response": model_response,
            "related_cues": related_cues,
            "annotation": {
                "label": "__FILL: correct / partial / wrong__",
                "principle_detected": "__FILL: one of: reciprocity, commitment_consistency, social_proof, authority, liking, scarcity, unity__",
                "memory_awareness": "__FILL: true / false__",
                "phase_awareness": "__FILL: true / false (understands current phase)__",
                "notes": "__OPTIONAL: any additional observations__",
            },
        })

    save_json(items, output_path)
    print(f"Human eval template saved to {output_path} ({len(items)} items)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate persuasion memory")
    parser.add_argument("--responses", type=str, required=True,
                        help="JSON file mapping trigger_id → model response text")
    parser.add_argument("--triggers", type=str, default="data/filtered/triggers_filtered.json")
    parser.add_argument("--cues", type=str, default="data/filtered/cues_filtered.json")
    parser.add_argument("--output", type=str, default="eval/results/")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--human-eval-only", action="store_true",
                        help="Only generate human eval template, skip LLM judge")
    args = parser.parse_args()

    triggers = load_json(args.triggers)
    cues = load_json(args.cues)
    responses = load_json(args.responses)  # {trigger_id: response_text}

    os.makedirs(args.output, exist_ok=True)

    # Generate human eval template
    generate_human_eval_template(
        triggers, cues, responses,
        os.path.join(args.output, "human_eval_template.json"),
    )

    if args.human_eval_only:
        return

    # Run LLM judge
    print(f"Running LLM judge ({args.judge_model}) on {len(triggers)} triggers...")
    judgments = run_llm_judge(triggers, cues, responses, judge_model=args.judge_model)

    # Post-processing: reclassify stale labels based on principle detection
    # (judge is not told which principle is stale; we determine it here)
    judgments = apply_stale_labels(judgments, triggers)

    save_json(judgments, os.path.join(args.output, "judgments.json"))

    # Compute metrics
    metrics = compute_metrics(judgments, triggers)
    save_json(metrics, os.path.join(args.output, "metrics.json"))

    # Print summary
    o = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"PERSUASION MEMORY EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total triggers: {o['total']}")
    print(f"Correct: {o['correct']:3d} ({o['correct_pct']:5.1f}%)")
    print(f"Partial: {o['partial']:3d} ({o['partial_pct']:5.1f}%)")
    print(f"Wrong:   {o['wrong']:3d} ({o['wrong_pct']:5.1f}%)")
    print(f"Stale:   {o['stale']:3d} ({o['stale_pct']:5.1f}%)  [uses outdated phase]")
    print(f"Avg score: {o['avg_score']:6.4f} / 2.0")
    print(f"Memory awareness: {o['memory_awareness_pct']:5.1f}%")
    print(f"Phase awareness:  {o['phase_awareness_pct']:5.1f}%")

    print(f"\nPer Phase:")
    for phase, stats in sorted(metrics.get("per_phase", {}).items()):
        print(f"  Phase {phase}: correct={stats['correct_pct']:5.1f}%  avg_score={stats['avg_score']:6.2f}  (n={stats['total']})")

    print(f"\nPer Topic:")
    for topic, stats in sorted(metrics["per_topic"].items()):
        print(f"  {topic:25s}  correct={stats['correct_pct']:5.1f}%  (n={stats['total']})")

    print(f"\nPer Effective Principle:")
    for p, stats in sorted(metrics["per_principle"].items()):
        print(f"  {p:30s}  correct={stats['correct_pct']:5.1f}%  (n={stats['total']})")

    if metrics.get("drift_metrics"):
        print(f"\nDRIFT DETECTION METRICS (Phase 2 only):")
        dm = metrics["drift_metrics"]
        print(f"  Phase 2 triggers: {dm['phase_2_triggers']}")
        print(f"  Drift detection rate: {dm['drift_detection_rate_pct']:5.1f}%")
        print(f"  Stale rate (using outdated phase): {dm['stale_rate_pct']:5.1f}%")


if __name__ == "__main__":
    main()
