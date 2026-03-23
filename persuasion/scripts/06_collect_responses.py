"""
Step 6: Collect target model responses at trigger points.

Feeds each assembled conversation to a target LLM and records its response
at each trigger point. The output is a flat JSON dict {trigger_id: response_text}
that evaluate.py consumes directly.

Usage:
    python scripts/06_collect_responses.py \
        --conversations data/conversations/conversations.json \
        --output data/responses/sonnet_responses.json \
        --model sonnet

    # Resume after crash:
    python scripts/06_collect_responses.py \
        --conversations data/conversations/conversations.json \
        --output data/responses/sonnet_responses.json \
        --model sonnet --resume

    # Run on a subset:
    python scripts/06_collect_responses.py \
        --conversations data/conversations/conversations.json \
        --output data/responses/sonnet_responses.json \
        --model sonnet --filter user_01,user_02
"""

import argparse
import sys
import time

from utils import call_llm, load_json, save_json


SYSTEM_PROMPT_TEMPLATE = """\
You are a trusted personal advisor having an ongoing conversation with {user_name}.
You've known them for a while and have had many conversations about various topics
in their life.

Your role is to help them think through decisions, offer perspective, and give
actionable advice. Draw on what you know about them from your past conversations —
their preferences, what has worked for them before, and what hasn't.

When they bring up a new situation, respond naturally as their advisor. Be concise
and conversational (2-4 sentences). Tailor your approach based on what you've
learned about how they respond to different kinds of guidance."""


def build_trigger_index(conversation):
    """Map trigger_ids to turn indices by scanning turn meta fields.

    Returns sorted list of dicts: {trigger_id, turn_idx, trigger_meta}.
    Logs warnings for orphaned triggers (in conversation metadata but no
    matching turn).
    """
    # Scan turns for trigger markers
    turn_triggers = {}
    for idx, turn in enumerate(conversation["turns"]):
        meta = turn.get("meta", {})
        if meta.get("type") == "trigger" and "trigger_id" in meta:
            turn_triggers[meta["trigger_id"]] = {
                "trigger_id": meta["trigger_id"],
                "turn_idx": idx,
                "trigger_meta": meta,
            }

    # Check conversation-level trigger metadata for orphans
    for t in conversation.get("triggers", []):
        tid = t["trigger_id"]
        if tid not in turn_triggers:
            print(f"  WARNING: orphaned trigger {tid} in {conversation['user_id']} "
                  f"(in metadata but no matching turn) — skipping")

    # Return sorted by turn position
    return sorted(turn_triggers.values(), key=lambda x: x["turn_idx"])


def format_context(turns, up_to_idx):
    """Format turns[0:up_to_idx] as a list of 'Speaker: text' strings.

    Meta fields are stripped. Returns a list of strings (one per turn)
    for truncation granularity.
    """
    parts = []
    for turn in turns[:up_to_idx]:
        speaker = turn["speaker"]
        label = "P" if speaker == "P" else "U"
        parts.append(f"{label}: {turn['text']}")
    return parts


def truncate_context(context_parts, max_tokens, reserved=900):
    """Truncate from the beginning, keeping the most recent turns.

    Args:
        context_parts: List of turn strings.
        max_tokens: Total token budget.
        reserved: Tokens reserved for system prompt + trigger + response.

    Returns:
        A single string of the (possibly truncated) context.
    """
    budget = max_tokens - reserved
    if budget <= 0:
        return "[... earlier conversation omitted ...]"

    # Estimate tokens as len(text) / 4
    total_tokens = sum(len(p) / 4 for p in context_parts)

    if total_tokens <= budget:
        return "\n".join(context_parts)

    # Keep turns from the end until we exceed budget
    kept = []
    used = 0
    for part in reversed(context_parts):
        part_tokens = len(part) / 4
        if used + part_tokens > budget:
            break
        kept.append(part)
        used += part_tokens

    kept.reverse()
    return "[... earlier conversation omitted ...]\n" + "\n".join(kept)


def format_prompt(system_prompt, context, trigger_text):
    """Combine system prompt + conversation history + trigger into a single prompt string."""
    return (
        f"{system_prompt}\n\n"
        f"=== Conversation History ===\n"
        f"{context}\n\n"
        f"=== Current Message ===\n"
        f"U: {trigger_text}\n\n"
        f"Respond as P:"
    )


def collect_responses_for_conversation(conversation, model, max_context_tokens,
                                       existing_responses, timeout):
    """Process all triggers in one conversation.

    Returns dict {trigger_id: response_text}.
    Skips triggers already in existing_responses.
    Builds context incrementally.
    """
    user_name = conversation.get("user_name", "the user")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_name=user_name)
    turns = conversation["turns"]

    trigger_entries = build_trigger_index(conversation)
    if not trigger_entries:
        print(f"  No triggers found in turns for {conversation['user_id']}")
        return {}

    responses = {}
    # Build context incrementally: track how far we've formatted
    formatted_parts = []
    last_formatted_idx = 0

    for entry in trigger_entries:
        tid = entry["trigger_id"]
        turn_idx = entry["turn_idx"]

        if tid in existing_responses:
            print(f"    {tid}: skipped (already collected)")
            # Still advance the formatted context past this trigger
            # so subsequent triggers have correct context
            new_parts = format_context(turns[last_formatted_idx:turn_idx], turn_idx - last_formatted_idx)
            formatted_parts.extend(new_parts)
            last_formatted_idx = turn_idx
            continue

        # Extend formatted context up to (but not including) the trigger turn
        new_parts = format_context(turns[last_formatted_idx:turn_idx], turn_idx - last_formatted_idx)
        formatted_parts.extend(new_parts)
        last_formatted_idx = turn_idx

        # Get trigger text
        trigger_text = turns[turn_idx]["text"]

        # Truncate and build prompt
        context = truncate_context(formatted_parts, max_context_tokens)
        prompt = format_prompt(system_prompt, context, trigger_text)

        try:
            response = call_llm(prompt, model=model, timeout=timeout)
            # Clean up: remove "P:" prefix if model echoes it
            response = response.strip()
            if response.startswith("P:"):
                response = response[2:].strip()
            responses[tid] = response
            print(f"    {tid}: collected ({len(response)} chars)")
        except Exception as e:
            print(f"    {tid}: ERROR — {e}")
            responses[tid] = ""

    return responses


def main():
    parser = argparse.ArgumentParser(
        description="Collect target model responses at trigger points"
    )
    parser.add_argument("--conversations", type=str,
                        default="data/conversations/conversations.json",
                        help="Path to assembled conversations JSON")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for responses JSON (e.g. data/responses/sonnet_responses.json)")
    parser.add_argument("--model", type=str, default="sonnet",
                        help="Model to use: haiku, sonnet, opus (default: sonnet)")
    parser.add_argument("--max-context-tokens", type=int, default=180000,
                        help="Max context tokens for target model (default: 180000)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping completed triggers")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds per LLM call (default: 300)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated user_ids to process (e.g. user_01,user_02)")
    args = parser.parse_args()

    conversations = load_json(args.conversations)
    print(f"Loaded {len(conversations)} conversations")

    # Filter if requested
    if args.filter:
        filter_ids = set(args.filter.split(","))
        conversations = [c for c in conversations if c["user_id"] in filter_ids]
        print(f"Filtered to {len(conversations)} conversations: {sorted(c['user_id'] for c in conversations)}")

    if not conversations:
        print("No conversations to process.")
        sys.exit(0)

    # Load existing responses for resume
    all_responses = {}
    if args.resume:
        try:
            all_responses = load_json(args.output)
            print(f"Resumed: loaded {len(all_responses)} existing responses from {args.output}")
        except (FileNotFoundError, Exception) as e:
            print(f"Resume: no existing file found ({e}), starting fresh")

    # Process each conversation
    total_triggers = 0
    total_collected = 0
    start_time = time.time()

    for i, conv in enumerate(conversations):
        uid = conv["user_id"]
        n_triggers = len(conv.get("triggers", []))
        print(f"\n[{i+1}/{len(conversations)}] {uid} ({conv.get('user_name', '?')}) — "
              f"{conv['num_turns']} turns, {n_triggers} triggers")

        responses = collect_responses_for_conversation(
            conv,
            model=args.model,
            max_context_tokens=args.max_context_tokens,
            existing_responses=all_responses,
            timeout=args.timeout,
        )

        all_responses.update(responses)
        total_triggers += n_triggers
        total_collected += len(responses)

        # Checkpoint after each conversation
        save_json(all_responses, args.output)
        print(f"  Checkpoint: {len(all_responses)} total responses saved")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total responses: {len(all_responses)}")
    print(f"Newly collected: {total_collected}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
