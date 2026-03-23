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
    DRIFT_TYPES,
    MIN_DRIFTING_TOPICS,
    MAX_DRIFTING_TOPICS,
    call_llm,
    parse_json_from_response,
    save_json,
)

PROFILE_PROMPT = """\
You are generating realistic persuadee profiles for a persuasion memory benchmark \
with temporal drift (users' preferences can change over the course of a conversation).

Each profile represents a person with personality traits and evolving persuasion preferences.

## Cialdini's 7 Principles of Persuasion:
{principle_list}

## Topic Domains:
{topic_list}

## Profile Type: {profile_type}

{type_specific_instructions}

## Task:
Generate {count} unique persuadee profiles. For each profile, create:
1. A first name
2. A short backstory (2-3 sentences: age, occupation, personality traits) that justifies preferences
3. An is_stable flag: {is_stable}
4. A drift_type (if is_stable is false): "event" or "accumulation"
5. A preference_map: for EACH of the 8 topics, assign phases and drift flags:
   - Each topic has: phase_1 (required), phase_2 (if drifts: true), drifts (true/false)
   - Each phase has: effective (1 principle), ineffective (1 principle)
   - For drifting topics: phase_2 effective MUST differ from phase_1 effective
   - For non-drifting topics: omit phase_2, set drifts: false

## Constraints:
- Make profiles diverse in age, occupation, personality, and cultural background
- Preferences should feel psychologically plausible given the backstory
- The SAME person may respond to different principles on different topics
- Across all profiles, ensure variety — don't make everyone prefer authority for taxes
- Do NOT reuse the same effective/ineffective principle for every topic within one profile
- For drifting profiles: realistic life changes that would cause preference shifts \
  (e.g., getting burned by social proof in finance → now trust authority)

{diversity_hint}

Output ONLY a valid JSON array:
[
  {{
    "name": "...",
    "backstory": "...",
    "is_stable": {is_stable},
    "drift_type": {drift_type_value},
    "preference_map": {{
      "personal_finance": {{
        "phase_1": {{"effective": "...", "ineffective": "..."}},
        "phase_2": {{"effective": "...", "ineffective": "..."}},
        "drifts": true|false
      }},
      ...
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


def generate_profiles(num_profiles, model, batch_size=5, stable_count=8):
    """Generate profiles in batches, with mix of stable and drifting users."""
    all_profiles = []
    principle_list = build_principle_list()
    topic_list = build_topic_list()

    generated_names = set()
    drifting_count = num_profiles - stable_count

    # Generate stable profiles first
    print(f"\n=== Generating {stable_count} STABLE profiles ===")
    stable_profiles = _generate_profile_batch(
        stable_count, model, batch_size, principle_list, topic_list,
        profile_type="stable", generated_names=generated_names
    )
    all_profiles.extend(stable_profiles)

    # Generate drifting profiles
    print(f"\n=== Generating {drifting_count} DRIFTING profiles ===")
    drifting_profiles = _generate_profile_batch(
        drifting_count, model, batch_size, principle_list, topic_list,
        profile_type="drifting", generated_names=generated_names
    )
    all_profiles.extend(drifting_profiles)

    # Enforce MAX_DRIFTING_TOPICS: if the LLM produced more drifting topics
    # than allowed, randomly select which topics to keep as drifting and
    # revert the rest to stable. See methodology/05_bias_and_validity.md.
    for profile in all_profiles:
        if profile.get("is_stable"):
            continue
        drifting_topics = [
            t for t in TOPICS
            if profile["preference_map"].get(t, {}).get("drifts", False)
        ]
        if len(drifting_topics) > MAX_DRIFTING_TOPICS:
            keep = set(random.sample(drifting_topics, MAX_DRIFTING_TOPICS))
            reverted = []
            for t in drifting_topics:
                if t not in keep:
                    profile["preference_map"][t]["drifts"] = False
                    profile["preference_map"][t].pop("phase_2", None)
                    reverted.append(t)
            print(f"  Clamped {profile['name']}: {len(drifting_topics)} → {MAX_DRIFTING_TOPICS} drifting topics "
                  f"(reverted: {', '.join(reverted)})")

    # Shuffle to break correlation between generation order and user_id.
    # Without this, early-batch profiles (which had max name diversity) would
    # always be low user_ids. See methodology/05_bias_and_validity.md §T4.
    random.shuffle(all_profiles)

    # Assign user IDs after shuffle
    for i, profile in enumerate(all_profiles):
        profile["user_id"] = f"user_{i + 1:02d}"

    return all_profiles[:num_profiles]


def _generate_profile_batch(count, model, batch_size, principle_list, topic_list,
                            profile_type="stable", generated_names=None):
    """Generate a batch of profiles of a specific type."""
    if generated_names is None:
        generated_names = set()

    all_profiles = []

    for batch_idx in range(0, count, batch_size):
        batch_count = min(batch_size, count - batch_idx)

        # Profile type instructions
        if profile_type == "stable":
            is_stable = "true"
            drift_type_value = "null"
            type_specific_instructions = (
                "## Stable Profiles\n"
                "Generate profiles where preferences do NOT change. "
                "Their persuasion strategy effectiveness remains constant throughout."
            )
        else:  # drifting
            is_stable = "false"
            drift_type_value = '"event" or "accumulation"'
            type_specific_instructions = (
                "## Drifting Profiles\n"
                "Generate profiles where BETWEEN 1 AND 4 topics (inclusive) drift to new preferences.\n"
                "IMPORTANT: At least 4 of the 8 topics MUST remain stable (drifts: false, no phase_2).\n"
                "Most topics should NOT drift — drift is the exception, not the rule.\n"
                "- CHOOSE drift_type for the user: 'event' (1 life-changing moment) or 'accumulation' (gradual erosion)\n"
                "- For EACH drifting topic: phase_1 effective ≠ phase_2 effective\n"
                "- For NON-drifting topics: set drifts: false, omit phase_2"
            )

        diversity_hint = ""
        if generated_names:
            diversity_hint = (
                f"Names already used (do NOT reuse): {', '.join(sorted(generated_names))}.\n"
                f"Make these profiles distinctly different from prior batches."
            )

        prompt = PROFILE_PROMPT.format(
            principle_list=principle_list,
            topic_list=topic_list,
            profile_type=profile_type,
            type_specific_instructions=type_specific_instructions,
            is_stable=is_stable,
            drift_type_value=drift_type_value,
            count=batch_count,
            diversity_hint=diversity_hint,
        )

        print(f"Generating batch {batch_idx // batch_size + 1} of {profile_type} ({batch_count} profiles)...")
        response = call_llm(prompt, model=model, temperature=0.8, max_tokens=4096)
        profiles = parse_json_from_response(response)

        for profile in profiles:
            generated_names.add(profile["name"])
            all_profiles.append(profile)

        print(f"  Got {len(profiles)} profiles (total: {len(all_profiles)})")

    return all_profiles


def validate_profiles(profiles):
    """Validate generated profiles with phase/drift schema."""
    errors = []
    for p in profiles:
        # Check is_stable and drift_type consistency
        is_stable = p.get("is_stable", False)
        drift_type = p.get("drift_type")

        if is_stable and drift_type is not None:
            errors.append(f"{p['user_id']}: is_stable=true but drift_type={drift_type}")
        if not is_stable and drift_type not in DRIFT_TYPES:
            errors.append(f"{p['user_id']}: is_stable=false but invalid drift_type '{drift_type}'")

        for topic in TOPICS:
            if topic not in p["preference_map"]:
                errors.append(f"{p['user_id']}: missing topic {topic}")
                continue

            topic_pref = p["preference_map"][topic]
            drifts = topic_pref.get("drifts", False)

            # Check phase_1 (always required)
            if "phase_1" not in topic_pref:
                errors.append(f"{p['user_id']}/{topic}: missing phase_1")
                continue

            phase1 = topic_pref["phase_1"]
            eff1 = phase1.get("effective")
            ineff1 = phase1.get("ineffective")

            if eff1 not in PRINCIPLES:
                errors.append(f"{p['user_id']}/{topic}/phase_1: unknown effective '{eff1}'")
            if ineff1 not in PRINCIPLES:
                errors.append(f"{p['user_id']}/{topic}/phase_1: unknown ineffective '{ineff1}'")
            if eff1 and ineff1 and eff1 == ineff1:
                errors.append(f"{p['user_id']}/{topic}/phase_1: effective == ineffective ({eff1})")

            # Check phase_2 (required if drifts=true, should be absent if drifts=false)
            if drifts:
                if "phase_2" not in topic_pref:
                    errors.append(f"{p['user_id']}/{topic}: drifts=true but missing phase_2")
                    continue

                phase2 = topic_pref["phase_2"]
                eff2 = phase2.get("effective")
                ineff2 = phase2.get("ineffective")

                if eff2 not in PRINCIPLES:
                    errors.append(f"{p['user_id']}/{topic}/phase_2: unknown effective '{eff2}'")
                if ineff2 not in PRINCIPLES:
                    errors.append(f"{p['user_id']}/{topic}/phase_2: unknown ineffective '{ineff2}'")
                if eff2 and ineff2 and eff2 == ineff2:
                    errors.append(f"{p['user_id']}/{topic}/phase_2: effective == ineffective ({eff2})")

                # For drifting topics: phase_2 effective must differ from phase_1
                if eff1 and eff2 and eff1 == eff2:
                    errors.append(
                        f"{p['user_id']}/{topic}: drifts=true but phase_1.effective == phase_2.effective ({eff1})"
                    )
            else:
                if "phase_2" in topic_pref:
                    errors.append(f"{p['user_id']}/{topic}: drifts=false but phase_2 present")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate persuadee profiles")
    parser.add_argument("--num-profiles", type=int, default=20)
    parser.add_argument("--stable-count", type=int, default=8,
                        help="Number of stable (non-drifting) users")
    parser.add_argument("--output", type=str, default="data/profiles/")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    profiles = generate_profiles(
        args.num_profiles, args.model, args.batch_size,
        stable_count=args.stable_count
    )

    errors = validate_profiles(profiles)
    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print("\nWARNING: Profiles have validation errors. Review before using downstream.")

    out_path = f"{args.output}/profiles.json"
    save_json(profiles, out_path)
    print(f"\nSaved {len(profiles)} profiles to {out_path}")

    # Report drift distribution for transparency
    stable = sum(1 for p in profiles if p.get("is_stable"))
    drifting = len(profiles) - stable
    event_count = sum(1 for p in profiles if p.get("drift_type") == "event")
    accum_count = sum(1 for p in profiles if p.get("drift_type") == "accumulation")
    drift_topic_counts = []
    for p in profiles:
        if not p.get("is_stable"):
            dt = sum(1 for t in TOPICS if p["preference_map"].get(t, {}).get("drifts"))
            drift_topic_counts.append(dt)
    print(f"\nDistribution:")
    print(f"  Stable: {stable}, Drifting: {drifting} (event={event_count}, accumulation={accum_count})")
    if drift_topic_counts:
        print(f"  Drifting topics per user: min={min(drift_topic_counts)}, max={max(drift_topic_counts)}, "
              f"mean={sum(drift_topic_counts)/len(drift_topic_counts):.1f}")


if __name__ == "__main__":
    main()
