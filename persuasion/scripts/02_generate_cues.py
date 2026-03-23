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


DRIFT_EVENT_PROMPT = """\
You are generating a natural utterance that implicitly signals a life event causing \
a shift in persuasion preferences.

## Context:
- Persuadee name: {name}
- Topic: {topic}
- The persuadee previously responded well to: **{old_principle}**
- But they are now shifting away from this principle

## Task:
Generate a SINGLE natural utterance from {name} that reveals (implicitly, not explicitly):
- They had a negative experience with {old_principle}
- Their preference is now shifting toward a different approach

The utterance should:
- Be conversational and natural (like talking to a friend)
- NOT explicitly say "I've changed my mind" or "social proof no longer works"
- Instead, reveal through their experience or reflection
- Be 1-2 sentences only
- Stand alone as a natural turn

## Example:
If {old_principle}="social_proof" for personal_finance:
  "Ugh, I lost so much following what everyone was doing with those meme stocks last year."

Output ONLY the dialogue line:
U: ...
"""

EROSION_PROMPT = """\
You are generating negative cues showing an ineffective principle gradually failing.

## Context:
- Persuadee name: {name}
- Persuadee backstory: {backstory}
- Topic: {topic} ({topic_desc})
- Principle that is FAILING: **{old_principle}** — {principle_desc}
- Outcome: The persuasion **fails**. The persuadee's resistance is growing.

## Task:
Generate 3 distinct short dialogue exchanges (3-5 turns each) showing {name} \
becoming increasingly frustrated with {old_principle}. Each should show:
- A different specific scenario within the topic domain
- {old_principle} is being used but isn't working
- The persuadee's frustration or resistance is increasing progressively

Make the dialogues feel like natural, casual conversation.

Output ONLY a valid JSON array:
[
  {{
    "scenario_brief": "one-line description",
    "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."
  }},
  ...
]
"""


def _generate_drift_event(name, topic, old_principle, new_principle, model):
    """Generate a single drift event utterance."""
    prompt = DRIFT_EVENT_PROMPT.format(
        name=name,
        topic=topic,
        old_principle=old_principle,
    )

    try:
        response = call_llm(prompt, model=model, temperature=0.7, max_tokens=512)
        # Extract just the U: line
        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("U:"):
                return line.strip()
        return response.strip()  # Fallback to full response
    except Exception as e:
        print(f"  Error generating drift event: {e}")
        return None  # Caller must handle None (no placeholder strings)


def _generate_erosion_cues(name, topic, old_principle, new_principle, model):
    """Generate 3 erosion cues showing principle failure."""
    prompt = EROSION_PROMPT.format(
        name=name,
        backstory="",  # Use empty for now to fit in token budget
        topic=topic,
        topic_desc=TOPIC_DESCRIPTIONS.get(topic, topic),
        old_principle=old_principle,
        principle_desc=PRINCIPLE_DESCRIPTIONS[old_principle],
    )

    try:
        response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
        erosion_items = parse_json_from_response(response)
        return [e["dialogue"] for e in erosion_items]
    except Exception as e:
        print(f"  Error generating erosion cues: {e}")
        return []  # Empty list; caller must handle (no placeholder strings)


def generate_cues_for_profile(profile, model, positive_per_topic=2, negative_per_topic=1):
    """Generate cue dialogues for one user profile, handling phases and drift."""
    cues = []
    cue_counter = 0
    name = profile["name"]
    backstory = profile["backstory"]
    is_stable = profile.get("is_stable", False)
    drift_type = profile.get("drift_type")

    for topic in TOPICS:
        topic_pref = profile["preference_map"][topic]
        drifts = topic_pref.get("drifts", False)

        # Phase 1 cues (always generate)
        phase1 = topic_pref["phase_1"]
        eff1 = phase1["effective"]
        ineff1 = phase1["ineffective"]

        # Positive cues for phase 1
        prompt = POSITIVE_CUE_PROMPT.format(
            name=name,
            backstory=backstory,
            topic=topic,
            topic_desc=TOPIC_DESCRIPTIONS[topic],
            principle=eff1,
            principle_desc=PRINCIPLE_DESCRIPTIONS[eff1],
            count=positive_per_topic,
        )

        try:
            response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
            pos_dialogues = parse_json_from_response(response)
        except Exception as e:
            print(f"  Error generating positive phase_1 cues for {profile['user_id']}/{topic}: {e}")
            pos_dialogues = []

        for d in pos_dialogues:
            cue_counter += 1
            cues.append({
                "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                "user_id": profile["user_id"],
                "user_name": name,
                "topic": topic,
                "phase": 1,
                "principle_used": eff1,
                "outcome": "positive",
                "scenario_brief": d.get("scenario_brief", ""),
                "dialogue": d["dialogue"],
            })

        # Negative cues for phase 1
        prompt = NEGATIVE_CUE_PROMPT.format(
            name=name,
            backstory=backstory,
            topic=topic,
            topic_desc=TOPIC_DESCRIPTIONS[topic],
            principle=ineff1,
            principle_desc=PRINCIPLE_DESCRIPTIONS[ineff1],
            count=negative_per_topic,
        )

        try:
            response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
            neg_dialogues = parse_json_from_response(response)
        except Exception as e:
            print(f"  Error generating negative phase_1 cues for {profile['user_id']}/{topic}: {e}")
            neg_dialogues = []

        for d in neg_dialogues:
            cue_counter += 1
            cues.append({
                "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                "user_id": profile["user_id"],
                "user_name": name,
                "topic": topic,
                "phase": 1,
                "principle_used": ineff1,
                "outcome": "negative",
                "scenario_brief": d.get("scenario_brief", ""),
                "dialogue": d["dialogue"],
            })

        # If topic drifts, generate drift event/erosion and phase 2 cues
        if drifts:
            # Generate drift event (event-type) or erosion cues (accumulation-type)
            if drift_type == "event":
                drift_event = _generate_drift_event(name, topic, eff1, ineff1, model)
                if drift_event is not None:
                    cue_counter += 1
                    cues.append({
                        "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                        "user_id": profile["user_id"],
                        "user_name": name,
                        "topic": topic,
                        "phase": "drift_event",
                        "dialogue": drift_event,
                    })
                else:
                    print(f"  WARNING: drift event generation failed for {profile['user_id']}/{topic}, skipping")
            else:  # accumulation
                erosion_cues = _generate_erosion_cues(name, topic, eff1, ineff1, model)
                for erosion_dialogue in erosion_cues:
                    cue_counter += 1
                    cues.append({
                        "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                        "user_id": profile["user_id"],
                        "user_name": name,
                        "topic": topic,
                        "phase": "erosion",
                        "dialogue": erosion_dialogue,
                    })

            # Phase 2 cues (for drifting topics)
            phase2 = topic_pref["phase_2"]
            eff2 = phase2["effective"]
            ineff2 = phase2["ineffective"]

            # Positive cues for phase 2
            prompt = POSITIVE_CUE_PROMPT.format(
                name=name,
                backstory=backstory,
                topic=topic,
                topic_desc=TOPIC_DESCRIPTIONS[topic],
                principle=eff2,
                principle_desc=PRINCIPLE_DESCRIPTIONS[eff2],
                count=positive_per_topic,
            )

            try:
                response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
                pos_dialogues = parse_json_from_response(response)
            except Exception as e:
                print(f"  Error generating positive phase_2 cues for {profile['user_id']}/{topic}: {e}")
                pos_dialogues = []

            for d in pos_dialogues:
                cue_counter += 1
                cues.append({
                    "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": name,
                    "topic": topic,
                    "phase": 2,
                    "principle_used": eff2,
                    "outcome": "positive",
                    "scenario_brief": d.get("scenario_brief", ""),
                    "dialogue": d["dialogue"],
                })

            # Negative cues for phase 2
            prompt = NEGATIVE_CUE_PROMPT.format(
                name=name,
                backstory=backstory,
                topic=topic,
                topic_desc=TOPIC_DESCRIPTIONS[topic],
                principle=ineff2,
                principle_desc=PRINCIPLE_DESCRIPTIONS[ineff2],
                count=negative_per_topic,
            )

            try:
                response = call_llm(prompt, model=model, temperature=0.7, max_tokens=2048)
                neg_dialogues = parse_json_from_response(response)
            except Exception as e:
                print(f"  Error generating negative phase_2 cues for {profile['user_id']}/{topic}: {e}")
                neg_dialogues = []

            for d in neg_dialogues:
                cue_counter += 1
                cues.append({
                    "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": name,
                    "topic": topic,
                    "phase": 2,
                    "principle_used": ineff2,
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

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

    # Print summary — count by outcome, skipping drift/erosion cues which lack outcome
    pos = sum(1 for c in all_cues if c.get("outcome") == "positive")
    neg = sum(1 for c in all_cues if c.get("outcome") == "negative")
    print(f"  Positive: {pos}, Negative: {neg}")


if __name__ == "__main__":
    main()
