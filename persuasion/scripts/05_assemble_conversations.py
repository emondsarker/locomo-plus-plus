"""
Step 5: Assemble long conversations by embedding cues and triggers into
extended dialogue trajectories with distractor turns in between.

Structure per conversation (one per user profile):
- Cues are spread across the conversation at varying positions
- Repeated cues for the same topic appear at different points (reinforcement)
- Triggers appear after all related cues, separated by distractor turns
- Cross-topic cues create interference between same-topic cue and trigger

Usage:
    python scripts/05_assemble_conversations.py \
        --profiles data/profiles/profiles.json \
        --cues data/filtered/cues_filtered.json \
        --triggers data/filtered/triggers_filtered.json \
        --output data/conversations/
"""

import argparse
import random
from utils import (
    TOPICS,
    TOPIC_DESCRIPTIONS,
    call_llm,
    parse_json_from_response,
    load_json,
    save_json,
)

DISTRACTOR_PROMPT = """\
Generate {count} short, natural chitchat exchanges (2-4 turns each) between two people: \
P (a friend/advisor) and U ({name}).

These are casual filler conversations about everyday topics like weather, weekend plans, \
food, sports, movies, news, or small talk. They should NOT be about: {avoid_topics}.

Make them feel natural and varied. Each exchange should be self-contained.

Output ONLY a valid JSON array:
[
  {{
    "dialogue": "P: ...\\nU: ...\\nP: ...\\nU: ..."
  }},
  ...
]
"""

TIME_GAP_TO_DISTRACTOR_COUNT = {
    "one week": (3, 6),
    "several weeks": (6, 12),
    "a few months": (12, 20),
    "several months": (20, 30),
}


def generate_distractors(name, count, avoid_topics, model):
    """Generate distractor chitchat turns."""
    prompt = DISTRACTOR_PROMPT.format(
        count=count,
        name=name,
        avoid_topics=", ".join(avoid_topics),
    )
    try:
        response = call_llm(prompt, model=model, temperature=0.9, max_tokens=3000)
        distractors = parse_json_from_response(response)
        return [d["dialogue"] for d in distractors]
    except Exception as e:
        print(f"  Error generating distractors: {e}")
        return []


def dialogue_to_turns(dialogue_text, start_turn_id, meta=None):
    """Convert a dialogue string into a list of turn dicts."""
    turns = []
    for line in dialogue_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("P:"):
            speaker = "P"
            text = line[2:].strip()
        elif line.startswith("U:"):
            speaker = "U"
            text = line[2:].strip()
        else:
            # Handle lines without speaker prefix — attach to previous speaker
            if turns:
                turns[-1]["text"] += " " + line
            continue

        turn = {
            "turn_id": start_turn_id + len(turns),
            "speaker": speaker,
            "text": text,
        }
        if meta:
            turn["meta"] = meta
        turns.append(turn)
    return turns


def assemble_conversation(profile, cues, triggers, model):
    """Assemble a single long conversation with per-topic independent drift zones."""
    name = profile["name"]

    # Group cues and triggers by topic
    cues_by_topic = {}
    for c in cues:
        cues_by_topic.setdefault(c["topic"], []).append(c)

    triggers_by_topic = {}
    for t in triggers:
        triggers_by_topic.setdefault(t["topic"], []).append(t)

    # Determine which topics drift and assign independent drift positions
    drift_events = []  # List of {topic, drift_type, drift_turn_id}
    topic_drift_position = {}  # topic -> drift position (0-1 in middle third)

    topic_prefs = profile.get("preference_map", {})
    drifting_topics = [
        t for t in TOPICS
        if topic_prefs.get(t, {}).get("drifts", False)
    ]

    # Assign independent drift positions for each drifting topic
    # Spread them across the middle third to avoid bunching
    if drifting_topics:
        positions = sorted(random.uniform(0.3, 0.7) for _ in drifting_topics)
        for topic, pos in zip(drifting_topics, positions):
            topic_drift_position[topic] = pos

    # Plan the conversation layout with per-topic phase awareness
    segments = []  # Each segment is ("cue"|"trigger"|"distractor", data)

    # Shuffle topic order for interleaving
    active_topics = [t for t in TOPICS if t in cues_by_topic]
    random.shuffle(active_topics)

    # Round-robin cues: place one per topic in round-robin fashion,
    # but track whether we've passed the drift zone for each topic
    max_cues_per_topic = max(
        (len(v) for v in cues_by_topic.values()), default=0
    )

    for round_idx in range(max_cues_per_topic):
        for topic in active_topics:
            topic_cues = cues_by_topic.get(topic, [])
            if round_idx < len(topic_cues):
                segments.append(("cue", topic_cues[round_idx]))
                # Add distractor after each cue
                segments.append(("distractor", {"topic": topic, "count": random.randint(2, 5)}))

    # Place drift events in their assigned drift zones
    for topic in drifting_topics:
        # Find drift event and erosion cues for this topic
        topic_cues = cues_by_topic.get(topic, [])
        drift_event_cues = [c for c in topic_cues if c.get("phase") == "drift_event"]
        erosion_cues = [c for c in topic_cues if c.get("phase") == "erosion"]

        if drift_event_cues:
            # Event-type user: single drift event
            for drift_cue in drift_event_cues:
                segments.append(("drift_event", drift_cue))
                drift_events.append({
                    "topic": topic,
                    "drift_type": "event",
                    "drift_cue_id": drift_cue["cue_id"]
                })
        elif erosion_cues:
            # Accumulation-type user: erosion cues showing gradual failure
            for erosion_cue in erosion_cues:
                segments.append(("erosion", erosion_cue))
            drift_events.append({
                "topic": topic,
                "drift_type": "accumulation",
                "drift_cue_ids": [c["cue_id"] for c in erosion_cues]
            })

    # Now place triggers — each after a distractor gap
    random.shuffle(active_topics)
    for topic in active_topics:
        topic_triggers = triggers_by_topic.get(topic, [])

        # Separate phase 1 and phase 2 triggers
        phase1_triggers = [t for t in topic_triggers if t.get("phase") == 1]
        phase2_triggers = [t for t in topic_triggers if t.get("phase") == 2]

        # Phase 1 triggers come before drift
        for trigger in phase1_triggers:
            gap = trigger.get("time_gap", "several weeks")
            min_d, max_d = TIME_GAP_TO_DISTRACTOR_COUNT.get(gap, (5, 10))
            distractor_count = random.randint(min_d, max_d)

            segments.append(("distractor", {"topic": topic, "count": distractor_count}))
            segments.append(("trigger", trigger))

        # Phase 2 triggers come after drift
        for trigger in phase2_triggers:
            gap = trigger.get("time_gap", "several weeks")
            min_d, max_d = TIME_GAP_TO_DISTRACTOR_COUNT.get(gap, (5, 10))
            distractor_count = random.randint(min_d, max_d)

            segments.append(("distractor", {"topic": topic, "count": distractor_count}))
            segments.append(("trigger", trigger))

    # Now generate actual distractor dialogues in bulk
    total_distractors_needed = sum(
        s[1]["count"] for s in segments if s[0] == "distractor"
    )
    avoid_topics_desc = [TOPIC_DESCRIPTIONS[t] for t in active_topics[:3]]
    print(f"  Generating ~{total_distractors_needed} distractor exchanges...")

    distractor_pool = generate_distractors(
        name,
        min(total_distractors_needed + 5, 40),  # cap per batch
        avoid_topics_desc,
        model,
    )

    # If we need more distractors, generate additional batches
    while len(distractor_pool) < total_distractors_needed:
        more = generate_distractors(
            name, 20, avoid_topics_desc, model,
        )
        distractor_pool.extend(more)

    # Assemble turns
    all_turns = []
    turn_id = 1
    distractor_idx = 0
    trigger_metadata = []

    for seg_type, seg_data in segments:
        if seg_type == "cue":
            phase = seg_data.get("phase", 1)
            meta = {
                "type": "cue",
                "cue_id": seg_data["cue_id"],
                "topic": seg_data["topic"],
                "phase": phase,
                "principle": seg_data.get("principle_used"),
                "outcome": seg_data.get("outcome")
            }
            new_turns = dialogue_to_turns(seg_data["dialogue"], turn_id, meta=meta)
            all_turns.extend(new_turns)
            turn_id += len(new_turns)

        elif seg_type in ["drift_event", "erosion"]:
            meta = {
                "type": seg_type,
                "cue_id": seg_data["cue_id"],
                "topic": seg_data["topic"],
                "phase": seg_data.get("phase"),
            }
            new_turns = dialogue_to_turns(seg_data["dialogue"], turn_id, meta=meta)
            all_turns.extend(new_turns)
            turn_id += len(new_turns)

        elif seg_type == "trigger":
            phase = seg_data.get("phase", 1)
            meta = {
                "type": "trigger",
                "trigger_id": seg_data["trigger_id"],
                "topic": seg_data["topic"],
                "phase": phase,
            }
            new_turns = dialogue_to_turns(seg_data["trigger_text"], turn_id, meta=meta)
            all_turns.extend(new_turns)

            trigger_metadata.append({
                "trigger_id": seg_data["trigger_id"],
                "turn_id": turn_id,
                "topic": seg_data["topic"],
                "phase": phase,
                "effective_principle": seg_data["effective_principle"],
                "ineffective_principle": seg_data["ineffective_principle"],
                "stale_principle": seg_data.get("stale_principle"),
                "related_cue_ids": seg_data["related_cue_ids"],
            })
            turn_id += len(new_turns)

        elif seg_type == "distractor":
            count = seg_data["count"]
            for _ in range(count):
                if distractor_idx < len(distractor_pool):
                    d_text = distractor_pool[distractor_idx]
                    distractor_idx += 1
                    new_turns = dialogue_to_turns(d_text, turn_id,
                                                  meta={"type": "distractor"})
                    all_turns.extend(new_turns)
                    turn_id += len(new_turns)

    return {
        "conversation_id": f"conv_{profile['user_id']}",
        "user_id": profile["user_id"],
        "user_name": profile["name"],
        "backstory": profile["backstory"],
        "is_stable": profile.get("is_stable", False),
        "drift_type": profile.get("drift_type"),
        "drift_events": drift_events,
        "num_turns": len(all_turns),
        "turns": all_turns,
        "triggers": trigger_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Assemble long conversations")
    parser.add_argument("--profiles", type=str, default="data/profiles/profiles.json")
    parser.add_argument("--cues", type=str, default="data/filtered/cues_filtered.json")
    parser.add_argument("--triggers", type=str, default="data/filtered/triggers_filtered.json")
    parser.add_argument("--output", type=str, default="data/conversations/")
    parser.add_argument("--model", type=str, default="haiku")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    profiles = load_json(args.profiles)
    all_cues = load_json(args.cues)
    all_triggers = load_json(args.triggers)
    print(f"Loaded {len(profiles)} profiles, {len(all_cues)} cues, {len(all_triggers)} triggers")

    # Index by user_id
    cues_by_user = {}
    for c in all_cues:
        cues_by_user.setdefault(c["user_id"], []).append(c)

    triggers_by_user = {}
    for t in all_triggers:
        triggers_by_user.setdefault(t["user_id"], []).append(t)

    conversations = []
    for profile in profiles:
        uid = profile["user_id"]
        user_cues = cues_by_user.get(uid, [])
        user_triggers = triggers_by_user.get(uid, [])

        if not user_cues or not user_triggers:
            print(f"Skipping {uid}: no cues or triggers")
            continue

        print(f"Assembling conversation for {uid} ({profile['name']})...")
        print(f"  {len(user_cues)} cues, {len(user_triggers)} triggers")

        conv = assemble_conversation(profile, user_cues, user_triggers, args.model)
        conversations.append(conv)
        print(f"  Assembled {conv['num_turns']} turns")

    out_path = f"{args.output}/conversations.json"
    save_json(conversations, out_path)
    print(f"\nSaved {len(conversations)} conversations to {out_path}")

    # Summary stats
    total_turns = sum(c["num_turns"] for c in conversations)
    total_triggers = sum(len(c["triggers"]) for c in conversations)
    print(f"Total turns: {total_turns}, Total trigger points: {total_triggers}")


if __name__ == "__main__":
    main()
