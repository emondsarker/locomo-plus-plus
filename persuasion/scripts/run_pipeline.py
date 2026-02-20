"""
Full pipeline runner — generates the complete persuasion benchmark in batches.

Runs steps 1-3 in 10 batches (2 profiles per batch), then steps 4-5 once at the end.
Progress is saved after each batch so you can resume if interrupted.

Usage:
    cd persuasion/
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --model sonnet          # use sonnet instead
    python scripts/run_pipeline.py --resume-from-batch 4   # resume from batch 4
    python scripts/run_pipeline.py --skip-generation       # skip to filter+assemble
"""

import argparse
import json
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    PRINCIPLES,
    PRINCIPLE_DESCRIPTIONS,
    TOPICS,
    TOPIC_DESCRIPTIONS,
    call_llm,
    parse_json_from_response,
    save_json,
    load_json,
)

def progress_bar(current, total, label="", width=30, extra=""):
    """Print a progress bar that updates in place."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    term_width = shutil.get_terminal_size((80, 20)).columns
    line = f"\r    [{bar}] {current}/{total} {label}"
    if extra:
        line += f" ({extra})"
    # Pad to clear previous line, but don't exceed terminal width
    line = line[:term_width - 1].ljust(term_width - 1)
    print(line, end="", flush=True)


def progress_done(msg=""):
    """Finish a progress bar line."""
    print(f"\r    {msg}" + " " * 40)


NUM_BATCHES = 10
PROFILES_PER_BATCH = 2
TOTAL_PROFILES = NUM_BATCHES * PROFILES_PER_BATCH  # 20

POSITIVE_CUES_PER_TOPIC = 2
NEGATIVE_CUES_PER_TOPIC = 1
TRIGGERS_PER_TOPIC = 3

# ─── Prompts ──────────────────────────────────────────────────────────────

PROFILE_PROMPT = """\
Generate {count} unique persuadee profiles for a persuasion memory benchmark.

Each profile represents a person with personality traits that determine which \
persuasion strategies work on them for different topics.

Cialdini's 7 Principles: reciprocity, commitment_consistency, social_proof, \
authority, liking, scarcity, unity.

Topics: personal_finance, health_fitness, career, taxes_legal, technology, \
social_relationships, education, lifestyle.

For each profile create:
1. A first name
2. A short backstory (2-3 sentences: age, occupation, personality)
3. A preference_map: for EACH of the 8 topics, assign:
   - "effective": the ONE principle that works best
   - "ineffective": the ONE principle that works worst
   (must be different)

Constraints:
- Diverse in age, occupation, personality, cultural background
- Preferences should feel psychologically plausible given the backstory
- Same person may respond to different principles on different topics
- Don't reuse same effective/ineffective for every topic within one profile

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
  }}
]
"""


def make_cue_prompt(name, backstory, topic, principle, outcome, count=1):
    outcome_word = "succeeds" if outcome == "positive" else "fails"
    reaction = "receptive and convinced" if outcome == "positive" else "resistant and unmoved"

    return f"""\
Generate {count} short persuasion dialogue(s) (3-5 turns each) between P (Persuader) and U ({name}).

Context:
- {name}: {backstory}
- Topic: {topic} ({TOPIC_DESCRIPTIONS[topic]})
- Principle used: {principle} — {PRINCIPLE_DESCRIPTIONS[principle]}
- Outcome: Persuasion {outcome_word}. {name} is {reaction}.

Constraints:
- Do NOT name the principle — weave it naturally into P's approach
- Show the outcome through U's reaction, not explicitly
- Natural, casual conversation
- Each dialogue should involve a different specific scenario within the topic

Output ONLY valid JSON:
[{{"scenario_brief": "...", "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."}}]
"""


def make_trigger_prompt(name, backstory, topic, cue_summary, count=3):
    return f"""\
Generate {count} trigger scenarios for a persuasion memory benchmark.

Context:
- Persuadee: {name} — {backstory}
- Topic: {topic} ({TOPIC_DESCRIPTIONS[topic]})

Prior interactions the model observed:
{cue_summary}

Generate {count} NEW situations where {name} brings up a problem/decision in the \
{topic} domain. Each must be:
- Semantically different from the prior cues (different words, different situation)
- A single natural utterance from {name}
- Underspecified — multiple persuasion approaches could seem reasonable

Output ONLY valid JSON:
[{{"trigger_text": "U: ...", "scenario_brief": "...", "time_gap": "one week / several weeks / a few months / several months"}}]
"""


# ─── Generation Functions ─────────────────────────────────────────────────

def generate_profiles(batch_idx, existing_names, model):
    """Generate PROFILES_PER_BATCH profiles for one batch."""
    diversity_hint = ""
    if existing_names:
        diversity_hint = (
            f"Names already used (do NOT reuse): {', '.join(sorted(existing_names))}.\n"
            f"Make these profiles distinctly different from those names."
        )

    prompt = PROFILE_PROMPT.format(
        count=PROFILES_PER_BATCH,
        diversity_hint=diversity_hint,
    )

    response = call_llm(prompt, model=model)
    profiles = parse_json_from_response(response)

    # Assign user_ids
    start_idx = batch_idx * PROFILES_PER_BATCH
    for i, p in enumerate(profiles):
        p["user_id"] = f"user_{start_idx + i + 1:02d}"

    return profiles


def generate_cues_for_profile(profile, model):
    """Generate all cues for one profile."""
    cues = []
    cue_counter = 0
    total_steps = len(TOPICS) * 2  # positive + negative per topic
    step = 0

    for topic in TOPICS:
        prefs = profile["preference_map"][topic]

        # Positive cues
        step += 1
        progress_bar(step, total_steps, f"{topic} +", extra=prefs["effective"])
        prompt = make_cue_prompt(
            profile["name"], profile["backstory"], topic,
            prefs["effective"], "positive", count=POSITIVE_CUES_PER_TOPIC,
        )
        try:
            resp = call_llm(prompt, model=model)
            parsed = parse_json_from_response(resp)
            for d in parsed:
                cue_counter += 1
                cues.append({
                    "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": profile["name"],
                    "topic": topic,
                    "principle_used": prefs["effective"],
                    "outcome": "positive",
                    "scenario_brief": d.get("scenario_brief", ""),
                    "dialogue": d["dialogue"],
                })
        except Exception as e:
            progress_done(f"ERROR positive cue {topic}: {e}")

        # Negative cues
        step += 1
        progress_bar(step, total_steps, f"{topic} -", extra=prefs["ineffective"])
        prompt = make_cue_prompt(
            profile["name"], profile["backstory"], topic,
            prefs["ineffective"], "negative", count=NEGATIVE_CUES_PER_TOPIC,
        )
        try:
            resp = call_llm(prompt, model=model)
            parsed = parse_json_from_response(resp)
            for d in parsed:
                cue_counter += 1
                cues.append({
                    "cue_id": f"{profile['user_id']}_cue_{cue_counter:03d}",
                    "user_id": profile["user_id"],
                    "user_name": profile["name"],
                    "topic": topic,
                    "principle_used": prefs["ineffective"],
                    "outcome": "negative",
                    "scenario_brief": d.get("scenario_brief", ""),
                    "dialogue": d["dialogue"],
                })
        except Exception as e:
            progress_done(f"ERROR negative cue {topic}: {e}")

    progress_done(f"{len(cues)} cues generated")
    return cues


def generate_triggers_for_profile(profile, cues, model):
    """Generate triggers for one profile."""
    triggers = []
    trigger_counter = 0

    cues_by_topic = {}
    for c in cues:
        cues_by_topic.setdefault(c["topic"], []).append(c)

    active_topics = [t for t in TOPICS if t in cues_by_topic]
    total_steps = len(active_topics)
    step = 0

    for topic in active_topics:
        topic_cues = cues_by_topic[topic]
        step += 1
        progress_bar(step, total_steps, topic)

        prefs = profile["preference_map"][topic]
        cue_summary = "\n".join(
            f"- {c['scenario_brief']} (approach: {PRINCIPLE_DESCRIPTIONS[c['principle_used']]}, "
            f"outcome: {'receptive' if c['outcome'] == 'positive' else 'resistant'})"
            for c in topic_cues
        )

        prompt = make_trigger_prompt(
            profile["name"], profile["backstory"], topic,
            cue_summary, count=TRIGGERS_PER_TOPIC,
        )

        try:
            resp = call_llm(prompt, model=model)
            parsed = parse_json_from_response(resp)
            for t in parsed:
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
                    "related_cue_ids": [c["cue_id"] for c in topic_cues],
                })
        except Exception as e:
            progress_done(f"ERROR trigger {topic}: {e}")

    progress_done(f"{len(triggers)} triggers generated")
    return triggers


# ─── Main Pipeline ────────────────────────────────────────────────────────

def run_batch(batch_idx, existing_names, model, data_dir):
    """Run generation for one batch of 2 profiles."""
    batch_dir = f"{data_dir}/batches/batch_{batch_idx:02d}"
    os.makedirs(batch_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BATCH {batch_idx + 1}/{NUM_BATCHES}")
    print(f"{'='*60}")

    # Step 1: Profiles
    print(f"\n  [1/3] Generating {PROFILES_PER_BATCH} profiles...")
    t0 = time.time()
    profiles = generate_profiles(batch_idx, existing_names, model)
    save_json(profiles, f"{batch_dir}/profiles.json")
    print(f"    Generated: {', '.join(p['name'] for p in profiles)} ({time.time()-t0:.1f}s)")

    all_cues = []
    all_triggers = []

    for profile in profiles:
        uid = profile["user_id"]
        name = profile["name"]

        # Step 2: Cues
        print(f"\n  [2/3] Generating cues for {name} ({uid})...")
        t0 = time.time()
        cues = generate_cues_for_profile(profile, model)
        all_cues.extend(cues)
        pos = sum(1 for c in cues if c["outcome"] == "positive")
        neg = sum(1 for c in cues if c["outcome"] == "negative")
        print(f"    {len(cues)} cues ({pos} positive, {neg} negative) ({time.time()-t0:.1f}s)")

        # Step 3: Triggers
        print(f"\n  [3/3] Generating triggers for {name} ({uid})...")
        t0 = time.time()
        triggers = generate_triggers_for_profile(profile, cues, model)
        all_triggers.extend(triggers)
        print(f"    {len(triggers)} triggers ({time.time()-t0:.1f}s)")

    save_json(all_cues, f"{batch_dir}/cues.json")
    save_json(all_triggers, f"{batch_dir}/triggers.json")

    new_names = {p["name"] for p in profiles}
    return profiles, all_cues, all_triggers, new_names


def merge_batches(data_dir):
    """Merge all batch outputs into single files."""
    all_profiles = []
    all_cues = []
    all_triggers = []

    for batch_dir in sorted((Path(f"{data_dir}/batches")).iterdir()):
        if not batch_dir.is_dir():
            continue
        try:
            all_profiles.extend(load_json(f"{batch_dir}/profiles.json"))
            all_cues.extend(load_json(f"{batch_dir}/cues.json"))
            all_triggers.extend(load_json(f"{batch_dir}/triggers.json"))
        except FileNotFoundError:
            print(f"  Warning: incomplete batch {batch_dir.name}, skipping")

    save_json(all_profiles, f"{data_dir}/profiles/profiles.json")
    save_json(all_cues, f"{data_dir}/cues/cues.json")
    save_json(all_triggers, f"{data_dir}/triggers/triggers.json")

    return all_profiles, all_cues, all_triggers


from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run full persuasion benchmark pipeline")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--resume-from-batch", type=int, default=0,
                        help="Resume from this batch index (0-indexed)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, just merge + filter + assemble")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(f"{data_dir}/batches", exist_ok=True)
    os.makedirs(f"{data_dir}/profiles", exist_ok=True)
    os.makedirs(f"{data_dir}/cues", exist_ok=True)
    os.makedirs(f"{data_dir}/triggers", exist_ok=True)

    if not args.skip_generation:
        # Collect existing names from completed batches
        existing_names = set()
        for i in range(args.resume_from_batch):
            batch_file = f"{data_dir}/batches/batch_{i:02d}/profiles.json"
            if os.path.exists(batch_file):
                for p in load_json(batch_file):
                    existing_names.add(p["name"])

        if args.resume_from_batch > 0:
            print(f"Resuming from batch {args.resume_from_batch}")
            print(f"Known names from prior batches: {sorted(existing_names)}")

        total_calls = 0
        pipeline_start = time.time()

        for batch_idx in range(args.resume_from_batch, NUM_BATCHES):
            batch_start = time.time()
            profiles, cues, triggers, new_names = run_batch(
                batch_idx, existing_names, args.model, data_dir
            )
            existing_names.update(new_names)

            batch_time = time.time() - batch_start
            # Estimate calls: 1 (profiles) + 2*8*2 (cues per profile) + 8*2 (triggers per profile)
            batch_calls = 1 + (PROFILES_PER_BATCH * len(TOPICS) * 2) + (PROFILES_PER_BATCH * len(TOPICS))
            total_calls += batch_calls

            print(f"\n  Batch {batch_idx + 1} done in {batch_time:.0f}s "
                  f"(~{batch_calls} calls, {len(cues)} cues, {len(triggers)} triggers)")

            remaining = NUM_BATCHES - batch_idx - 1
            if remaining > 0:
                est_remaining = batch_time * remaining
                print(f"  Estimated time remaining: {est_remaining/60:.0f} min ({remaining} batches)")

        total_time = time.time() - pipeline_start
        print(f"\n{'='*60}")
        print(f"  GENERATION COMPLETE")
        print(f"  Total time: {total_time/60:.1f} min, ~{total_calls} CLI calls")
        print(f"{'='*60}")

    # Merge all batches
    print("\nMerging batches...")
    all_profiles, all_cues, all_triggers = merge_batches(data_dir)
    print(f"  {len(all_profiles)} profiles, {len(all_cues)} cues, {len(all_triggers)} triggers")

    # Summary
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Profiles:  {data_dir}/profiles/profiles.json ({len(all_profiles)})")
    print(f"  Cues:      {data_dir}/cues/cues.json ({len(all_cues)})")
    print(f"  Triggers:  {data_dir}/triggers/triggers.json ({len(all_triggers)})")
    print(f"\n  Next steps:")
    print(f"    1. python scripts/04_semantic_filter.py")
    print(f"    2. python scripts/05_assemble_conversations.py")
    print(f"    3. Run your target model against the conversations")
    print(f"    4. python eval/evaluate.py --responses <response_file>")


if __name__ == "__main__":
    main()
