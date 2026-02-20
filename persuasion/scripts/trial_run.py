"""
Trial run: generate 2 profiles + a few cues to verify quality before full run.

Usage (run from persuasion/ directory):
    python scripts/trial_run.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils import call_llm, parse_json_from_response, save_json, PRINCIPLE_DESCRIPTIONS, TOPIC_DESCRIPTIONS

TRIAL_PROFILE_PROMPT = """\
Generate 2 unique persuadee profiles for a persuasion memory benchmark.

Each profile represents a person with personality traits that determine which \
persuasion strategies work on them for different topics.

Cialdini's 7 Principles: reciprocity, commitment_consistency, social_proof, \
authority, liking, scarcity, unity.

Topics: personal_finance, health_fitness, career, taxes_legal, technology, \
social_relationships, education, lifestyle.

For each profile create:
1. A first name
2. A short backstory (2-3 sentences)
3. A preference_map: for EACH of the 8 topics, assign "effective" (one principle \
that works best) and "ineffective" (one that works worst). They must be different.

Preferences should feel psychologically plausible given the backstory.

Output ONLY a valid JSON array:
[
  {
    "name": "...",
    "backstory": "...",
    "preference_map": {
      "personal_finance": {"effective": "...", "ineffective": "..."},
      "health_fitness": {"effective": "...", "ineffective": "..."},
      "career": {"effective": "...", "ineffective": "..."},
      "taxes_legal": {"effective": "...", "ineffective": "..."},
      "technology": {"effective": "...", "ineffective": "..."},
      "social_relationships": {"effective": "...", "ineffective": "..."},
      "education": {"effective": "...", "ineffective": "..."},
      "lifestyle": {"effective": "...", "ineffective": "..."}
    }
  }
]
"""


def make_cue_prompt(name, backstory, topic, principle, outcome):
    outcome_word = "succeeds" if outcome == "positive" else "fails"
    reaction = "receptive and convinced" if outcome == "positive" else "resistant and unmoved"

    return f"""\
Generate 1 short persuasion dialogue (3-5 turns) between P (Persuader) and U ({name}).

Context:
- {name}: {backstory}
- Topic: {topic} ({TOPIC_DESCRIPTIONS[topic]})
- Principle used: {principle} — {PRINCIPLE_DESCRIPTIONS[principle]}
- Outcome: Persuasion {outcome_word}. {name} is {reaction}.

Constraints:
- Do NOT name the principle — weave it naturally into P's approach
- Show the outcome through U's reaction, not explicitly
- Natural, casual conversation

Output ONLY valid JSON:
[{{"scenario_brief": "...", "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."}}]
"""


def make_trigger_prompt(name, backstory, topic, cue_summary):
    return f"""\
Generate 1 trigger scenario for a persuasion memory benchmark.

Context:
- Persuadee: {name} — {backstory}
- Topic: {topic} ({TOPIC_DESCRIPTIONS[topic]})

Prior interactions the model observed:
{cue_summary}

Generate a NEW situation where {name} brings up a problem/decision in the {topic} \
domain. It must be semantically different from the prior cues (different words, \
different specific situation). It should be a single utterance from {name}.

Output ONLY valid JSON:
[{{"trigger_text": "U: ...", "scenario_brief": "...", "time_gap": "several weeks"}}]
"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="haiku")
    args = parser.parse_args()

    out_dir = "data/trial/"
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Generate 2 profiles
    print("=" * 50)
    print(f"STEP 1: Generating 2 profiles (model={args.model})...")
    print("=" * 50)
    response = call_llm(TRIAL_PROFILE_PROMPT, model=args.model)
    print("\nRaw response:\n")
    print(response[:2000])
    print("\n")

    try:
        profiles = parse_json_from_response(response)
        for i, p in enumerate(profiles):
            p["user_id"] = f"trial_user_{i+1:02d}"
        save_json(profiles, f"{out_dir}/trial_profiles.json")
        print(f"Parsed {len(profiles)} profiles successfully\n")
    except Exception as e:
        print(f"FAILED to parse profiles: {e}")
        print("Stopping trial. Check the raw response above.")
        return

    # Step 2: Generate cues for first profile, first 2 topics only
    profile = profiles[0]
    name = profile["name"]
    backstory = profile["backstory"]
    pmap = profile["preference_map"]

    print("=" * 50)
    print(f"STEP 2: Generating cues for {name}...")
    print("=" * 50)

    cues = []
    topics_to_test = ["personal_finance", "health_fitness"]

    for topic in topics_to_test:
        eff = pmap[topic]["effective"]
        ineff = pmap[topic]["ineffective"]

        # One positive cue
        print(f"\n  Generating positive cue: {topic} / {eff}...")
        resp = call_llm(make_cue_prompt(name, backstory, topic, eff, "positive"), model=args.model)
        print(f"  Response preview: {resp[:300]}\n")
        try:
            parsed = parse_json_from_response(resp)
            cue = parsed[0]
            cue.update({
                "cue_id": f"trial_cue_{len(cues)+1}",
                "user_id": profile["user_id"],
                "topic": topic,
                "principle_used": eff,
                "outcome": "positive",
            })
            cues.append(cue)
            print(f"  OK: {cue['scenario_brief']}")
        except Exception as e:
            print(f"  FAILED: {e}")

        # One negative cue
        print(f"\n  Generating negative cue: {topic} / {ineff}...")
        resp = call_llm(make_cue_prompt(name, backstory, topic, ineff, "negative"), model=args.model)
        print(f"  Response preview: {resp[:300]}\n")
        try:
            parsed = parse_json_from_response(resp)
            cue = parsed[0]
            cue.update({
                "cue_id": f"trial_cue_{len(cues)+1}",
                "user_id": profile["user_id"],
                "topic": topic,
                "principle_used": ineff,
                "outcome": "negative",
            })
            cues.append(cue)
            print(f"  OK: {cue['scenario_brief']}")
        except Exception as e:
            print(f"  FAILED: {e}")

    save_json(cues, f"{out_dir}/trial_cues.json")
    print(f"\nSaved {len(cues)} cues")

    # Step 3: Generate 1 trigger per topic
    print("\n" + "=" * 50)
    print(f"STEP 3: Generating triggers for {name}...")
    print("=" * 50)

    triggers = []
    for topic in topics_to_test:
        topic_cues = [c for c in cues if c["topic"] == topic]
        cue_summary = "\n".join(
            f"- {c['scenario_brief']} (principle: {c['principle_used']}, "
            f"outcome: {c['outcome']})"
            for c in topic_cues
        )

        print(f"\n  Generating trigger: {topic}...")
        resp = call_llm(make_trigger_prompt(name, backstory, topic, cue_summary), model=args.model)
        print(f"  Response preview: {resp[:300]}\n")
        try:
            parsed = parse_json_from_response(resp)
            trig = parsed[0]
            trig.update({
                "trigger_id": f"trial_trig_{len(triggers)+1}",
                "user_id": profile["user_id"],
                "topic": topic,
                "effective_principle": pmap[topic]["effective"],
                "ineffective_principle": pmap[topic]["ineffective"],
                "related_cue_ids": [c["cue_id"] for c in topic_cues],
            })
            triggers.append(trig)
            print(f"  OK: {trig['scenario_brief']}")
        except Exception as e:
            print(f"  FAILED: {e}")

    save_json(triggers, f"{out_dir}/trial_triggers.json")
    print(f"\nSaved {len(triggers)} triggers")

    # Summary
    print("\n" + "=" * 50)
    print("TRIAL RUN COMPLETE")
    print("=" * 50)
    print(f"Output: {out_dir}")
    print(f"  trial_profiles.json  — {len(profiles)} profiles")
    print(f"  trial_cues.json      — {len(cues)} cues")
    print(f"  trial_triggers.json  — {len(triggers)} triggers")
    print(f"\nTotal claude -p calls: {2 + len(cues) + len(triggers)}")
    print("\nReview the output files. If quality looks good, run the full pipeline.")


if __name__ == "__main__":
    main()
