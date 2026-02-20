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


def summarize_cues(cues_for_topic):
    """Create a summary of prior cues for the prompt (without revealing principles)."""
    summaries = []
    for c in cues_for_topic:
        outcome_word = "receptive" if c["outcome"] == "positive" else "resistant"
        summaries.append(
            f"- Scenario: {c['scenario_brief']}. "
            f"Persuader's approach: {PRINCIPLE_DESCRIPTIONS[c['principle_used']]} "
            f"→ {c['user_name']} was {outcome_word}."
        )
    return "\n".join(summaries)


def generate_triggers_for_profile(profile, cues_by_topic, model, triggers_per_topic=3):
    """Generate trigger queries for one user profile."""
    triggers = []
    trigger_counter = 0

    for topic in TOPICS:
        topic_cues = cues_by_topic.get(topic, [])
        if not topic_cues:
            print(f"  Warning: no cues for {profile['user_id']}/{topic}, skipping")
            continue

        prefs = profile["preference_map"][topic]
        related_cue_ids = [c["cue_id"] for c in topic_cues]

        prompt = TRIGGER_PROMPT.format(
            name=profile["name"],
            backstory=profile["backstory"],
            topic=topic,
            topic_desc=TOPIC_DESCRIPTIONS[topic],
            cue_summaries=summarize_cues(topic_cues),
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
                "user_name": profile["name"],
                "topic": topic,
                "effective_principle": prefs["effective"],
                "ineffective_principle": prefs["ineffective"],
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
    args = parser.parse_args()

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
