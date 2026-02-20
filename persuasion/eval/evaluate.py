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


LABEL_SCORES = {"correct": 2, "partial": 1, "wrong": 0}


def run_llm_judge(triggers, cues, responses, judge_model="gpt-4o"):
    """Run LLM judge on all trigger responses."""
    cue_by_id = {c["cue_id"]: c for c in cues}

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
                "reason": "No response provided",
            })
            continue

        # Get related cues
        related_cues = [
            cue_by_id[cid] for cid in trigger.get("related_cue_ids", [])
            if cid in cue_by_id
        ]

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
                "reason": f"Judge error: {e}",
            })

    return results


def compute_metrics(judgments, triggers):
    """Compute aggregate metrics from judge results."""
    trigger_by_id = {t["trigger_id"]: t for t in triggers}

    # Overall
    labels = [j["label"] for j in judgments]
    total = len(labels)
    correct = labels.count("correct")
    partial = labels.count("partial")
    wrong = labels.count("wrong")
    avg_score = sum(LABEL_SCORES.get(l, 0) for l in labels) / max(total, 1)

    overall = {
        "total": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "correct_pct": round(100 * correct / max(total, 1), 2),
        "partial_pct": round(100 * partial / max(total, 1), 2),
        "wrong_pct": round(100 * wrong / max(total, 1), 2),
        "avg_score": round(avg_score, 4),
        "memory_awareness_pct": round(
            100 * sum(1 for j in judgments if j.get("memory_awareness")) / max(total, 1), 2
        ),
    }

    # Per topic
    per_topic = defaultdict(lambda: {"correct": 0, "partial": 0, "wrong": 0, "total": 0})
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

    # Per principle (which effective principle was being tested)
    per_principle = defaultdict(lambda: {"correct": 0, "partial": 0, "wrong": 0, "total": 0})
    for j in judgments:
        t = trigger_by_id.get(j["trigger_id"], {})
        principle = t.get("effective_principle", "unknown")
        per_principle[principle][j["label"]] += 1
        per_principle[principle]["total"] += 1

    for p in per_principle:
        n = per_principle[p]["total"]
        per_principle[p]["correct_pct"] = round(100 * per_principle[p]["correct"] / n, 2)

    return {
        "overall": overall,
        "per_topic": dict(per_topic),
        "per_principle": dict(per_principle),
    }


def generate_human_eval_template(triggers, cues, responses, output_path):
    """Generate a JSON template for human annotators."""
    cue_by_id = {c["cue_id"]: c for c in cues}

    items = []
    for trigger in triggers:
        tid = trigger["trigger_id"]
        model_response = responses.get(tid, "[NO RESPONSE]")

        related_cues = [
            {
                "cue_id": cid,
                "topic": cue_by_id[cid]["topic"],
                "principle_used": cue_by_id[cid]["principle_used"],
                "outcome": cue_by_id[cid]["outcome"],
                "dialogue": cue_by_id[cid]["dialogue"],
            }
            for cid in trigger.get("related_cue_ids", [])
            if cid in cue_by_id
        ]

        items.append({
            "trigger_id": tid,
            "user_name": trigger["user_name"],
            "topic": trigger["topic"],
            "effective_principle": trigger["effective_principle"],
            "ineffective_principle": trigger["ineffective_principle"],
            "trigger_text": trigger["trigger_text"],
            "model_response": model_response,
            "related_cues": related_cues,
            "annotation": {
                "label": "__FILL: correct / partial / wrong__",
                "principle_detected": "__FILL: which principle the model used__",
                "memory_awareness": "__FILL: true / false__",
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

    save_json(judgments, os.path.join(args.output, "judgments.json"))

    # Compute metrics
    metrics = compute_metrics(judgments, triggers)
    save_json(metrics, os.path.join(args.output, "metrics.json"))

    # Print summary
    o = metrics["overall"]
    print(f"\n{'='*50}")
    print(f"PERSUASION MEMORY EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total triggers: {o['total']}")
    print(f"Correct: {o['correct']} ({o['correct_pct']}%)")
    print(f"Partial: {o['partial']} ({o['partial_pct']}%)")
    print(f"Wrong:   {o['wrong']} ({o['wrong_pct']}%)")
    print(f"Avg score: {o['avg_score']} / 2.0")
    print(f"Memory awareness: {o['memory_awareness_pct']}%")

    print(f"\nPer Topic:")
    for topic, stats in sorted(metrics["per_topic"].items()):
        print(f"  {topic:25s}  correct={stats['correct_pct']:5.1f}%  (n={stats['total']})")

    print(f"\nPer Effective Principle:")
    for p, stats in sorted(metrics["per_principle"].items()):
        print(f"  {p:30s}  correct={stats['correct_pct']:5.1f}%  (n={stats['total']})")


if __name__ == "__main__":
    main()
