"""
Step 3: Generate trigger queries — new persuasion scenarios that test whether
the model learned from past cues which principle works for this user on this topic.

Triggers are semantically distant from the cues but in the same topic domain.
The model must recall past successes/failures and choose the right approach.

Usage:
    python scripts/03_generate_triggers.py \
        --profiles data/profiles/profiles.json \
        --cues data/cues/cues.json \
        --output data/triggers/
"""

import argparse
import random
from utils import (
    PRINCIPLES,
    PRINCIPLE_DESCRIPTIONS,
    TOPICS,
    TOPIC_DESCRIPTIONS,
    call_llm,
    parse_json_from_response,
    load_json,
    save_json,
)

TRIGGER_PROMPT = """\
You are generating trigger scenarios for a persuasion memory benchmark.

## Context:
- Persuadee name: {name}
- Persuadee backstory: {backstory}
- Topic domain: {topic} ({topic_desc})
- Phase: {phase} (the persuadee's current preference phase)

## Prior cue dialogues the model will have seen (for reference — the trigger must be \
DIFFERENT from these):
{cue_summaries}

## Task:
Generate {count} trigger scenarios. Each trigger is a NEW situation where {name} \
brings up a problem, decision, or question in the **{topic}** domain. The Persuader \
(model being tested) must figure out the right persuasion approach based on what \
worked/failed in the prior cues above.

## Constraints:
- Each trigger must be a SINGLE utterance from {name} (the persuadee)
- It must be a natural conversational statement, question, or reflection
- It must be in the same topic domain but involve a **different specific situation** \
  than any of the prior cues
- It must be **semantically distant** from the cue dialogues:
  * Do NOT reuse nouns, verbs, or phrases from the cues
  * Do NOT reference the prior conversations
  * The trigger should stand alone as a natural conversational turn
- The trigger should be **underspecified** — multiple persuasion approaches could \
  seem reasonable without knowing the user's preferences
- Include a time_gap field (how long after the cues this might occur)

Output ONLY a valid JSON array:
[
  {{
    "trigger_text": "U: ...",
    "scenario_brief": "one-line description of the new scenario",
    "time_gap": "one week / several weeks / a few months / several months"
  }},
  ...
]
"""


def summarize_cues(cues_for_topic, phase=None):
    """Create a summary of prior cues for the prompt.

    Summaries use neutral labels (Approach A/B) instead of principle names
    to prevent leaking the correct answer into trigger generation.
    See methodology/05_bias_and_validity.md §T1.
    """
    summaries = []
    for i, c in enumerate(cues_for_topic):
        # Filter by phase if specified
        if phase is not None and c.get("phase") != phase:
            continue

        # Skip drift events and erosion cues in summary
        if c.get("phase") in ["drift_event", "erosion"]:
            continue

        outcome_word = "receptive" if c.get("outcome") == "positive" else "resistant"
        approach_label = f"Approach {chr(65 + i)}"  # A, B, C, ...
        summaries.append(
            f"- Scenario: {c.get('scenario_brief', 'N/A')}. "
            f"{approach_label} was used → {c['user_name']} was {outcome_word}."
        )
    return "\n".join(summaries) if summaries else "[No cues for this phase]"


def generate_triggers_for_profile(profile, cues_by_topic, model, triggers_per_topic=3,
                                  triggers_per_phase=None):
    """Generate trigger queries for one user profile, handling phases for drifting users."""
    if triggers_per_phase is None:
        triggers_per_phase = (2, 3)  # (phase_1_count, phase_2_count) for drifting topics

    triggers = []
    trigger_counter = 0
    topic_prefs = profile["preference_map"]
    name = profile["name"]
    backstory = profile["backstory"]

    for topic in TOPICS:
        topic_cues = cues_by_topic.get(topic, [])
        if not topic_cues:
            print(f"  Warning: no cues for {profile['user_id']}/{topic}, skipping")
            continue

        prefs = topic_prefs[topic]
        drifts = prefs.get("drifts", False)

        if drifts:
            # Generate Phase 1 triggers (2 triggers)
            phase1_cue_ids = [c["cue_id"] for c in topic_cues if c.get("phase") == 1]
            phase1_cues_summary = summarize_cues(topic_cues, phase=1)

            phase1 = prefs["phase_1"]

            prompt = TRIGGER_PROMPT.format(
                name=name,
                backstory=backstory,
                topic=topic,
                topic_desc=TOPIC_DESCRIPTIONS[topic],
                phase="1 (initial preferences)",
                cue_summaries=phase1_cues_summary,
                count=triggers_per_phase[0],
            )

            try:
                response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
                trigger_items = parse_json_from_response(response)
            except Exception as e:
                print(f"  Error generating phase_1 triggers for {profile['user_id']}/{topic}: {e}")
                trigger_items = []

            for t in trigger_items:
                trigger_counter += 1
                triggers.append({
                    "trigger_id": f"{profile['user_id']}_trig_{trigger_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": name,
                    "topic": topic,
                    "phase": 1,
                    "effective_principle": phase1["effective"],
                    "ineffective_principle": phase1["ineffective"],
                    "stale_principle": None,  # No stale for phase 1
                    "trigger_text": t["trigger_text"],
                    "scenario_brief": t.get("scenario_brief", ""),
                    "time_gap": t.get("time_gap", "several weeks"),
                    "related_cue_ids": phase1_cue_ids,
                })

            # Generate Phase 2 triggers (3 triggers)
            # Phase 2 related_cue_ids include ALL cues (Phase 1 + drift + Phase 2)
            # because the model sees the full conversation timeline before Phase 2
            # triggers. See methodology/05_bias_and_validity.md §T3.
            phase2_cue_ids = [c["cue_id"] for c in topic_cues]
            phase2_cues_summary = summarize_cues(topic_cues, phase=2)

            phase2 = prefs["phase_2"]

            prompt = TRIGGER_PROMPT.format(
                name=name,
                backstory=backstory,
                topic=topic,
                topic_desc=TOPIC_DESCRIPTIONS[topic],
                phase="2 (updated preferences after drift)",
                cue_summaries=phase2_cues_summary,
                count=triggers_per_phase[1],
            )

            try:
                response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
                trigger_items = parse_json_from_response(response)
            except Exception as e:
                print(f"  Error generating phase_2 triggers for {profile['user_id']}/{topic}: {e}")
                trigger_items = []

            for t in trigger_items:
                trigger_counter += 1
                triggers.append({
                    "trigger_id": f"{profile['user_id']}_trig_{trigger_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": name,
                    "topic": topic,
                    "phase": 2,
                    "effective_principle": phase2["effective"],
                    "ineffective_principle": phase2["ineffective"],
                    "stale_principle": phase1["effective"],  # Phase 1 effective = stale for phase 2
                    "trigger_text": t["trigger_text"],
                    "scenario_brief": t.get("scenario_brief", ""),
                    "time_gap": t.get("time_gap", "several weeks"),
                    "related_cue_ids": phase2_cue_ids,
                })
        else:
            # Non-drifting topic: generate standard triggers (3 triggers, phase 1)
            related_cue_ids = [c["cue_id"] for c in topic_cues if c.get("phase") == 1]
            cue_summary = summarize_cues(topic_cues, phase=1)

            phase1 = prefs["phase_1"]

            prompt = TRIGGER_PROMPT.format(
                name=name,
                backstory=backstory,
                topic=topic,
                topic_desc=TOPIC_DESCRIPTIONS[topic],
                phase="1 (stable preferences)",
                cue_summaries=cue_summary,
                count=triggers_per_topic,
            )

            try:
                response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
                trigger_items = parse_json_from_response(response)
            except Exception as e:
                print(f"  Error generating triggers for {profile['user_id']}/{topic}: {e}")
                trigger_items = []

            for t in trigger_items:
                trigger_counter += 1
                triggers.append({
                    "trigger_id": f"{profile['user_id']}_trig_{trigger_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": name,
                    "topic": topic,
                    "phase": 1,
                    "effective_principle": phase1["effective"],
                    "ineffective_principle": phase1["ineffective"],
                    "stale_principle": None,  # No phase 2 for non-drifting
                    "trigger_text": t["trigger_text"],
                    "scenario_brief": t.get("scenario_brief", ""),
                    "time_gap": t.get("time_gap", "several weeks"),
                    "related_cue_ids": related_cue_ids,
                })

    return triggers


def main():
    parser = argparse.ArgumentParser(description="Generate trigger queries")
    parser.add_argument("--profiles", type=str, default="data/profiles/profiles.json")
    parser.add_argument("--cues", type=str, default="data/cues/cues.json")
    parser.add_argument("--output", type=str, default="data/triggers/")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--triggers-per-topic", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    profiles = load_json(args.profiles)
    all_cues = load_json(args.cues)
    print(f"Loaded {len(profiles)} profiles, {len(all_cues)} cues")

    # Index cues by user_id and topic
    cues_index = {}
    for c in all_cues:
        key = (c["user_id"], c["topic"])
        cues_index.setdefault(key, []).append(c)

    all_triggers = []
    for profile in profiles:
        print(f"Generating triggers for {profile['user_id']} ({profile['name']})...")
        cues_by_topic = {
            topic: cues_index.get((profile["user_id"], topic), [])
            for topic in TOPICS
        }
        triggers = generate_triggers_for_profile(
            profile, cues_by_topic, args.model,
            triggers_per_topic=args.triggers_per_topic,
        )
        all_triggers.extend(triggers)
        print(f"  Generated {len(triggers)} triggers")

    out_path = f"{args.output}/triggers.json"
    save_json(all_triggers, out_path)
    print(f"\nTotal: {len(all_triggers)} triggers saved to {out_path}")


if __name__ == "__main__":
    main()
