"""
Trial run: generate 3 profiles (1 stable + 2 drifting) + cues + triggers
to verify quality before full run.

Usage (run from persuasion/scripts directory):
    python trial_run.py [--model haiku]
"""

import sys
import os
import random
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import importlib.util

from utils import (
    TOPICS, PRINCIPLES, PRINCIPLE_DESCRIPTIONS, TOPIC_DESCRIPTIONS,
    call_llm, parse_json_from_response, save_json, load_json
)

# Import modules that start with numbers using importlib
def _load_module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('.py', ''), filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

profiles_module = _load_module('01_generate_profiles.py')
cues_module = _load_module('02_generate_cues.py')
triggers_module = _load_module('03_generate_triggers.py')

generate_profiles = profiles_module.generate_profiles
validate_profiles = profiles_module.validate_profiles
generate_cues_for_profile = cues_module.generate_cues_for_profile
generate_triggers_for_profile = triggers_module.generate_triggers_for_profile


def main():
    parser = argparse.ArgumentParser(description="Trial run with drift-aware profiles")
    parser.add_argument("--model", type=str, default="haiku")
    args = parser.parse_args()

    out_dir = "data/trial"
    os.makedirs(f"{out_dir}/profiles", exist_ok=True)
    os.makedirs(f"{out_dir}/cues", exist_ok=True)
    os.makedirs(f"{out_dir}/triggers", exist_ok=True)

    print("=" * 60)
    print("TRIAL RUN: Dynamic Persuasion Profiles")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Profiles: 3 (1 stable, 2 drifting)")
    print()

    # Step 1: Generate profiles
    print("STEP 1: Generating 3 profiles (1 stable + 2 drifting)...")
    print("-" * 60)

    random.seed(42)
    profiles = generate_profiles(
        num_profiles=3,
        model=args.model,
        batch_size=5,
        stable_count=1
    )

    errors = validate_profiles(profiles)
    if errors:
        print(f"\n⚠️  Validation warnings ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ All profiles valid!")

    # Assign user IDs
    for i, profile in enumerate(profiles):
        profile["user_id"] = f"trial_user_{i + 1:02d}"

    save_json(profiles, f"{out_dir}/profiles/profiles.json")
    print(f"\n✅ Generated {len(profiles)} profiles")
    for p in profiles:
        is_stable = p.get("is_stable", False)
        drift_type = p.get("drift_type")
        status = "STABLE" if is_stable else f"DRIFTING ({drift_type})"
        print(f"   - {p['user_id']}: {p['name']} [{status}]")

    # Step 2: Generate cues for each profile
    print("\n" + "=" * 60)
    print("STEP 2: Generating cues for each profile...")
    print("-" * 60)

    all_cues = []
    for profile in profiles:
        uid = profile["user_id"]
        name = profile["name"]
        is_stable = profile.get("is_stable", False)

        print(f"\n  {uid} ({name}) - generating cues...")
        try:
            cues = generate_cues_for_profile(profile, args.model)
            all_cues.extend(cues)

            # Count by phase
            phase_counts = {}
            for c in cues:
                phase = c.get("phase", 1)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            print(f"    ✅ Generated {len(cues)} cues")
            # Sort with integers first, then strings
            sorted_phases = sorted(phase_counts.items(), key=lambda x: (isinstance(x[0], str), x[0]))
            for phase, count in sorted_phases:
                print(f"       Phase {phase}: {count} cues")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    save_json(all_cues, f"{out_dir}/cues/cues.json")
    print(f"\n✅ Total: {len(all_cues)} cues saved")

    # Step 3: Generate triggers for each profile
    print("\n" + "=" * 60)
    print("STEP 3: Generating triggers for each profile...")
    print("-" * 60)

    all_triggers = []
    for profile in profiles:
        uid = profile["user_id"]
        name = profile["name"]

        # Index cues by user and topic
        cues_by_topic = {}
        for c in all_cues:
            if c["user_id"] == uid:
                topic = c["topic"]
                cues_by_topic.setdefault(topic, []).append(c)

        print(f"\n  {uid} ({name}) - generating triggers...")
        try:
            triggers = generate_triggers_for_profile(profile, cues_by_topic, args.model)
            all_triggers.extend(triggers)

            # Count by phase
            phase_counts = {}
            for t in triggers:
                phase = t.get("phase", 1)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

            print(f"    ✅ Generated {len(triggers)} triggers")
            # Sort with integers first, then strings
            sorted_phases = sorted(phase_counts.items(), key=lambda x: (isinstance(x[0], str), x[0]))
            for phase, count in sorted_phases:
                print(f"       Phase {phase}: {count} triggers")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    save_json(all_triggers, f"{out_dir}/triggers/triggers.json")
    print(f"\n✅ Total: {len(all_triggers)} triggers saved")

    # Summary
    print("\n" + "=" * 60)
    print("TRIAL RUN COMPLETE ✅")
    print("=" * 60)
    print(f"\nOutput location: {out_dir}/")
    print(f"  profiles/")
    print(f"    └─ profiles.json ({len(profiles)} profiles)")
    print(f"  cues/")
    print(f"    └─ cues.json ({len(all_cues)} cues)")
    print(f"  triggers/")
    print(f"    └─ triggers.json ({len(all_triggers)} triggers)")

    print(f"\nProfile breakdown:")
    stable = sum(1 for p in profiles if p.get("is_stable"))
    drifting = len(profiles) - stable
    print(f"  Stable: {stable}")
    print(f"  Drifting: {drifting}")

    print(f"\nNext steps:")
    print(f"  1. Inspect outputs: cat {out_dir}/profiles/profiles.json | head -50")
    print(f"  2. Check drift schema: cat {out_dir}/profiles/profiles.json | grep 'phase_2'")
    print(f"  3. Verify cue phases: cat {out_dir}/cues/cues.json | grep '\"phase\"'")
    print(f"  4. Check trigger stale: cat {out_dir}/triggers/triggers.json | grep 'stale_principle'")
    print(f"\n  If quality looks good, run full pipeline:")
    print(f"    python 01_generate_profiles.py --num-profiles 20 --stable-count 8")
    print(f"    python 02_generate_cues.py --profiles ../data/profiles/profiles.json")
    print(f"    python 03_generate_triggers.py --profiles ../data/profiles/profiles.json --cues ../data/cues/cues.json")
    print(f"    python 05_assemble_conversations.py --profiles ../data/profiles/profiles.json --cues ../data/cues/cues.json --triggers ../data/triggers/triggers.json")


if __name__ == "__main__":
    main()
