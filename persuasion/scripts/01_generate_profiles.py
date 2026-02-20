"""
Step 1: Generate persuadee profiles with topic→principle preference maps.

Each profile has:
- A name and short backstory
- A preference_map: for each topic, which principle is most/least effective

Usage:
    python scripts/01_generate_profiles.py --num-profiles 20 --output data/profiles/
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
    save_json,
)

PROFILE_PROMPT = """\
You are generating realistic persuadee profiles for a persuasion memory benchmark.

Each profile represents a person with consistent personality traits that determine \
which persuasion strategies work on them for different topics.

## Cialdini's 7 Principles of Persuasion:
{principle_list}

## Topic Domains:
{topic_list}

## Task:
Generate {count} unique persuadee profiles. For each profile, create:
1. A first name
2. A short backstory (2-3 sentences: age, occupation, personality traits) that makes \
their preferences feel coherent and realistic
3. A preference_map: for EACH of the 8 topics, assign:
   - "effective": the ONE principle that works best on this person for this topic
   - "ineffective": the ONE principle that works worst on this person for this topic
   (effective and ineffective must be different principles)

## Constraints:
- Make profiles diverse in age, occupation, personality, and cultural background
- Preferences should feel psychologically plausible given the backstory
- The SAME person may respond to different principles on different topics \
  (e.g., authority for taxes but social proof for fitness)
- Across all profiles, ensure variety — don't make everyone prefer authority for taxes
- Do NOT reuse the same effective/ineffective principle for every topic within one profile
- Each profile's preferences should reflect their personality

{diversity_hint}

Output ONLY a valid JSON array:
[
  {{
    "name": "...",
    "backstory": "...",
    "preference_map": {{
      "personal_finance": {{"effective": "...", "ineffective": "..."}},
      "health_fitness": {{"effective": "...", "ineffective": "..."}},
      "career": {{"effective": "...", "ineffective": "..."}},
      "taxes_legal": {{"effective": "...", "ineffective": "..."}},
      "technology": {{"effective": "...", "ineffective": "..."}},
      "social_relationships": {{"effective": "...", "ineffective": "..."}},
      "education": {{"effective": "...", "ineffective": "..."}},
      "lifestyle": {{"effective": "...", "ineffective": "..."}}
    }}
  }},
  ...
]
"""


def build_principle_list():
    return "\n".join(
        f"- **{p}**: {PRINCIPLE_DESCRIPTIONS[p]}" for p in PRINCIPLES
    )


def build_topic_list():
    return "\n".join(
        f"- **{t}**: {TOPIC_DESCRIPTIONS[t]}" for t in TOPICS
    )


def generate_profiles(num_profiles, model, batch_size=5):
    """Generate profiles in batches to improve diversity."""
    all_profiles = []
    principle_list = build_principle_list()
    topic_list = build_topic_list()

    generated_names = set()

    for batch_idx in range(0, num_profiles, batch_size):
        count = min(batch_size, num_profiles - batch_idx)

        diversity_hint = ""
        if generated_names:
            diversity_hint = (
                f"Names already used (do NOT reuse): {', '.join(sorted(generated_names))}.\n"
                f"Make these profiles distinctly different from prior batches."
            )

        prompt = PROFILE_PROMPT.format(
            principle_list=principle_list,
            topic_list=topic_list,
            count=count,
            diversity_hint=diversity_hint,
        )

        print(f"Generating batch {batch_idx // batch_size + 1} ({count} profiles)...")
        response = call_llm(prompt, model=model, temperature=0.8, max_tokens=4096)
        profiles = parse_json_from_response(response)

        for i, profile in enumerate(profiles):
            profile["user_id"] = f"user_{len(all_profiles) + 1:02d}"
            generated_names.add(profile["name"])
            all_profiles.append(profile)

        print(f"  Got {len(profiles)} profiles (total: {len(all_profiles)})")

    return all_profiles[:num_profiles]


def validate_profiles(profiles):
    """Basic validation of generated profiles."""
    errors = []
    for p in profiles:
        for topic in TOPICS:
            if topic not in p["preference_map"]:
                errors.append(f"{p['user_id']}: missing topic {topic}")
                continue
            eff = p["preference_map"][topic]["effective"]
            ineff = p["preference_map"][topic]["ineffective"]
            if eff not in PRINCIPLES:
                errors.append(f"{p['user_id']}/{topic}: unknown effective principle '{eff}'")
            if ineff not in PRINCIPLES:
                errors.append(f"{p['user_id']}/{topic}: unknown ineffective principle '{ineff}'")
            if eff == ineff:
                errors.append(f"{p['user_id']}/{topic}: effective == ineffective ({eff})")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate persuadee profiles")
    parser.add_argument("--num-profiles", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/profiles/")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    profiles = generate_profiles(args.num_profiles, args.model, args.batch_size)

    errors = validate_profiles(profiles)
    if errors:
        print(f"\nValidation warnings ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    out_path = f"{args.output}/profiles.json"
    save_json(profiles, out_path)
    print(f"\nSaved {len(profiles)} profiles to {out_path}")


if __name__ == "__main__":
    main()
