"""
Step 2: Generate cue dialogues — persuasion attempts with implicit outcomes.

Each cue is a short Persuader (P) ↔ Persuadee (U) exchange that shows a persuasion
attempt using a specific principle on a specific topic. The outcome (success/failure)
is conveyed implicitly through the persuadee's reaction.

We generate MULTIPLE cues per user per topic (2-3) so the model has repeated
observations to learn from.

Usage:
    python scripts/02_generate_cues.py --profiles data/profiles/profiles.json --output data/cues/
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

POSITIVE_CUE_PROMPT = """\
You are generating short persuasion dialogue snippets for a memory benchmark.

## Context:
- Persuadee name: {name}
- Persuadee backstory: {backstory}
- Topic: {topic} ({topic_desc})
- Persuasion principle used: **{principle}** — {principle_desc}
- Outcome: The persuasion **succeeds**. The persuadee is receptive and convinced.

## Task:
Generate {count} distinct short dialogue exchanges (3-5 turns each) between:
- **P** (Persuader): uses the specified principle naturally in conversation
- **U** ({name}, Persuadee): responds in a way that **implicitly shows** they are convinced

## Constraints:
- The principle must be woven naturally into P's approach — do NOT name the principle
- U's receptivity must be shown through behavior/words, NOT stated explicitly \
  (e.g., "That makes sense, I'll try it" not "I am persuaded by social proof")
- Each dialogue should involve a **different specific scenario** within the topic domain
- Make the dialogues feel like natural, casual conversation
- Vary the specific situations (e.g., for personal_finance: one about investing, \
  one about saving for a purchase, one about debt)
- Keep each dialogue to 3-5 turns total (alternating P and U)

Output ONLY a valid JSON array:
[
  {{
    "scenario_brief": "one-line description of the specific scenario",
    "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."
  }},
  ...
]
"""

NEGATIVE_CUE_PROMPT = """\
You are generating short persuasion dialogue snippets for a memory benchmark.

## Context:
- Persuadee name: {name}
- Persuadee backstory: {backstory}
- Topic: {topic} ({topic_desc})
- Persuasion principle used: **{principle}** — {principle_desc}
- Outcome: The persuasion **fails**. The persuadee resists, deflects, or is unmoved.

## Task:
Generate {count} distinct short dialogue exchanges (3-5 turns each) between:
- **P** (Persuader): uses the specified principle naturally in conversation
- **U** ({name}, Persuadee): responds in a way that **implicitly shows** they are NOT convinced

## Constraints:
- The principle must be woven naturally into P's approach — do NOT name the principle
- U's resistance must be shown through behavior/words, NOT stated explicitly \
  (e.g., "I don't know, I'd rather figure this out myself" not "I reject your authority argument")
- U should not be rude — they simply aren't moved by this approach
- Each dialogue should involve a **different specific scenario** within the topic domain
- Make the dialogues feel like natural, casual conversation
- Keep each dialogue to 3-5 turns total (alternating P and U)

Output ONLY a valid JSON array:
[
  {{
    "scenario_brief": "one-line description of the specific scenario",
    "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."
  }},
  ...
]
"""


def generate_cues_for_profile(profile, model, positive_per_topic=2, negative_per_topic=1):
    """Generate cue dialogues for one user profile."""
    cues = []
    cue_counter = 0

    for topic in TOPICS:
        prefs = profile["preference_map"][topic]
        eff_principle = prefs["effective"]
        ineff_principle = prefs["ineffective"]

        # Generate positive cues (effective principle → success)
        prompt = POSITIVE_CUE_PROMPT.format(
            name=profile["name"],
            backstory=profile["backstory"],
            topic=topic,
            topic_desc=TOPIC_DESCRIPTIONS[topic],
            principle=eff_principle,
            principle_desc=PRINCIPLE_DESCRIPTIONS[eff_principle],
            count=positive_per_topic,
        )

        try:
            response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
            pos_dialogues = parse_json_from_response(response)
        except Exception as e:
            print(f"  Error generating positive cues for {profile['user_id']}/{topic}: {e}")
            pos_dialogues = []

        for d in pos_dialogues:
            cue_counter += 1
            cues.append({
                "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                "user_id": profile["user_id"],
                "user_name": profile["name"],
                "topic": topic,
                "principle_used": eff_principle,
                "outcome": "positive",
                "scenario_brief": d.get("scenario_brief", ""),
                "dialogue": d["dialogue"],
            })

        # Generate negative cues (ineffective principle → failure)
        prompt = NEGATIVE_CUE_PROMPT.format(
            name=profile["name"],
            backstory=profile["backstory"],
            topic=topic,
            topic_desc=TOPIC_DESCRIPTIONS[topic],
            principle=ineff_principle,
            principle_desc=PRINCIPLE_DESCRIPTIONS[ineff_principle],
            count=negative_per_topic,
        )

        try:
            response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
            neg_dialogues = parse_json_from_response(response)
        except Exception as e:
            print(f"  Error generating negative cues for {profile['user_id']}/{topic}: {e}")
            neg_dialogues = []

        for d in neg_dialogues:
            cue_counter += 1
            cues.append({
                "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                "user_id": profile["user_id"],
                "user_name": profile["name"],
                "topic": topic,
                "principle_used": ineff_principle,
                "outcome": "negative",
                "scenario_brief": d.get("scenario_brief", ""),
                "dialogue": d["dialogue"],
            })

    return cues


def main():
    parser = argparse.ArgumentParser(description="Generate cue dialogues")
    parser.add_argument("--profiles", type=str, default="data/profiles/profiles.json")
    parser.add_argument("--output", type=str, default="data/cues/")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--positive-per-topic", type=int, default=2,
                        help="Number of positive cues per topic per user")
    parser.add_argument("--negative-per-topic", type=int, default=1,
                        help="Number of negative cues per topic per user")
    args = parser.parse_args()

    profiles = load_json(args.profiles)
    print(f"Loaded {len(profiles)} profiles")

    all_cues = []
    for profile in profiles:
        print(f"Generating cues for {profile['user_id']} ({profile['name']})...")
        cues = generate_cues_for_profile(
            profile, args.model,
            positive_per_topic=args.positive_per_topic,
            negative_per_topic=args.negative_per_topic,
        )
        all_cues.extend(cues)
        print(f"  Generated {len(cues)} cues")

    out_path = f"{args.output}/cues.json"
    save_json(all_cues, out_path)
    print(f"\nTotal: {len(all_cues)} cues saved to {out_path}")

    # Print summary
    pos = sum(1 for c in all_cues if c["outcome"] == "positive")
    neg = sum(1 for c in all_cues if c["outcome"] == "negative")
    print(f"  Positive: {pos}, Negative: {neg}")


if __name__ == "__main__":
    main()
