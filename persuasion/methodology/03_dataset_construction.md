# 03 — Dataset Construction

This document specifies every step of the data generation pipeline, the prompts
used, the parameters chosen, and the justification for each decision.

## Overview

The pipeline has 5 stages. Each depends on the output of the previous.

```
01_profiles → 02_cues → 03_triggers → 04_filter → 05_assemble
```

All generation uses an LLM via CLI. The generation model and
evaluation model are deliberately chosen independently (see §Evaluation Design).

---

## Stage 1: Profile Generation

**Script**: `01_generate_profiles.py`
**Input**: None (generates from scratch)
**Output**: `data/profiles/profiles.json` (20 profiles)

### What Is Generated

Each profile contains:
- A name and 2-3 sentence backstory (age, occupation, personality)
- An `is_stable` flag and `drift_type` (if drifting)
- A `preference_map`: for each of 8 topics, Phase 1 effective/ineffective
  principles, and Phase 2 (if the topic drifts)

### Generation Strategy

Profiles are generated in two batches — stable users first, then drifting users —
rather than mixed batches.

**Justification**: Generating stable and drifting profiles separately allows the
prompt to include type-specific instructions. A mixed prompt would need conditional
logic ("if this profile is stable, do X; if drifting, do Y") which increases prompt
complexity and error rates.

### Batch Size: 5 Profiles Per LLM Call

We generate 5 profiles per call rather than 1 or 20.

- **Why not 1**: Generating individually prevents the LLM from ensuring diversity
  across profiles. A batch of 5 lets the model contrast profiles within the batch.
- **Why not 20**: A single call generating all 20 profiles would exceed reliable
  JSON output length for most models and reduce per-profile quality.
- **Why 5**: Empirically, LLMs produce well-formed JSON arrays of 5 complex
  objects reliably. Larger batches increase malformed output rates.

### Temperature: 0.8

Higher than default (typically 0.7) to encourage demographic and personality diversity
across profiles.

**Justification**: Profile generation is a creative task where diversity is more
important than precision. We validated empirically that temperature=0.8 produces
diverse backstories without generating incoherent profiles. Temperature=1.0 produced
occasional schema violations.

### Name Deduplication

Each batch receives a list of previously generated names with the instruction
"do NOT reuse." This prevents duplicate names across batches.

**Known limitation**: This creates ordering effects — later batches are constrained
by earlier batches' name choices. See [05_bias_and_validity.md](05_bias_and_validity.md)
for mitigation.

### Validation

After generation, `validate_profiles()` checks:
- `is_stable` and `drift_type` consistency
- All 8 topics present with valid `phase_1`
- `phase_2` present iff `drifts=true`
- `effective ≠ ineffective` within each phase
- `phase_1.effective ≠ phase_2.effective` for drifting topics
- All principle names are from the canonical 7

**Current behavior**: Validation warnings are printed but do not block saving.

**Required fix before generation**: Validation must reject invalid profiles and
trigger regeneration. A profile that passes generation but fails validation
contaminates downstream data.

---

## Stage 2: Cue Dialogue Generation

**Script**: `02_generate_cues.py`
**Input**: `profiles.json`
**Output**: `data/cues/cues.json`

### What Is Generated

Short dialogue snippets (3-5 turns) between P (Persuader) and U (Persuadee)
showing a persuasion attempt with an implicit outcome.

### Cue Types and Counts

| Cue Type | Phase | Count Per Topic | Purpose |
|----------|-------|-----------------|---------|
| Positive (effective principle succeeds) | 1 | 2 | Shows model what works |
| Negative (ineffective principle fails) | 1 | 1 | Shows model what doesn't work |
| Drift event (single utterance) | "drift_event" | 1 (event-type users only) | Signals abrupt preference change |
| Erosion (principle failing repeatedly) | "erosion" | 3 (accumulation-type users only) | Signals gradual preference change |
| Positive (new effective principle succeeds) | 2 | 2 | Shows model what works now |
| Negative (new ineffective principle fails) | 2 | 1 | Shows model what doesn't work now |

### Why 2 Positive + 1 Negative Per Phase

- **2 positive cues**: A single positive example could be coincidental. Two
  examples of the same principle succeeding establish a pattern. Three would
  provide diminishing returns while increasing conversation length.
- **1 negative cue**: One counterexample establishes what *doesn't* work. More
  negatives would create an imbalanced signal (more evidence of failure than
  success), biasing the model toward elimination strategies rather than
  positive identification.
- **2:1 ratio**: Matches real-world observation patterns where you see more
  successes with a good strategy than failures with a bad one in a single
  relationship.

### Positive Cue Prompt Design

The prompt instructs the LLM to:
1. Name the principle and its description (so the LLM knows what to weave in)
2. Generate natural dialogue where the principle is used *implicitly*
3. Show the outcome through the persuadee's reaction, not explicit labels

**Why we name the principle in the generation prompt**: The generating LLM needs
to know which principle to employ. This is acceptable because the generating LLM
is not the model being tested. The principle name does not appear in the generated
dialogue itself (enforced by the constraint "do NOT name the principle").

### Negative Cue Prompt Design

Identical structure to positive cues, with outcome flipped. The prompt specifies
that the persuadee "should not be rude — they simply aren't moved."

**Justification**: Overtly hostile reactions would make negative cues trivially
distinguishable from positive ones via sentiment analysis alone, rather than
requiring principle identification.

### Drift Event Prompt Design

Generates a single natural utterance from the persuadee revealing a life experience
that explains the preference shift.

**Design choices**:
- One utterance only (not a dialogue) because drift events are experienced by the
  user, not co-constructed in conversation.
- Must not explicitly name the principle being abandoned ("I no longer trust
  authority" is too direct).
- Must not name the new principle being adopted (the model must discover this
  from Phase 2 cues, not from the drift event).

### Erosion Prompt Design

Generates 3 dialogue exchanges showing the old principle increasingly failing.

**Why 3 erosion cues**: This is the minimum needed to convey *gradual* change.
Fewer than 3 looks like a single bad experience (which would be event-type drift).
Exactly 3 provides a clear escalation trajectory without bloating the conversation.

### Temperature: 0.7

Slightly lower than profile generation because cue dialogues must be coherent
and naturalistic. Temperature=0.7 balances variety (different scenarios per cue)
with quality (grammatical, natural dialogue).

### Max Tokens: 2048

Sufficient for 2 dialogue exchanges of 3-5 turns each in JSON format. Empirically,
outputs average ~800-1200 tokens.

### Error Handling

If an LLM call fails, the cue is skipped (empty list appended).

**Known issue**: This can produce users with 0 cues for a topic, which would
create triggers with no supporting evidence. **Required fix**: Retry failed
calls up to 3 times, then flag the profile for manual review.

**Known issue**: Erosion cue fallback uses placeholder strings
(`"P: [erosion cue {i} for {topic}]\nU: I'm not convinced."`). **Required fix**:
Remove placeholder fallbacks. If erosion generation fails after retries, exclude
the topic from drifting for that user.

---

## Stage 3: Trigger Generation

**Script**: `03_generate_triggers.py`
**Input**: `profiles.json`, `cues.json`
**Output**: `data/triggers/triggers.json`

### What Is Generated

New persuasion scenarios (single utterance from the persuadee) that test whether
the model learned the correct strategy from the cues.

### Trigger Counts

| Topic Type | Phase 1 Triggers | Phase 2 Triggers | Total |
|------------|-----------------|-----------------|-------|
| Non-drifting | 3 | 0 | 3 |
| Drifting | 2 | 3 | 5 |

**Why 3 triggers per non-drifting topic**: 3 gives a small sample for per-topic
accuracy. Fewer than 3 makes per-topic metrics unreliable. More than 3 would
bloat the conversation without adding methodological value.

**Why 2 Phase 1 + 3 Phase 2 for drifting topics**: Phase 2 is the novel
measurement. More Phase 2 triggers provide better statistical power for drift
detection. Phase 1 gets 2 (not 0) because we need a within-topic Phase 1 vs
Phase 2 comparison. 2 is the minimum for this comparison.

### Cue Summary Design

The trigger prompt receives a summary of prior cues so the LLM can generate
triggers that are semantically distinct from them.

**Current implementation**: Summaries include principle descriptions
(e.g., "Persuader's approach: People defer to credible experts").

**Known issue — principle leakage**: Including principle descriptions in the
summary risks biasing trigger generation toward scenarios where that principle
is obviously applicable. **Required fix**: Replace with neutral descriptions
(e.g., "Approach A was effective" / "Approach B was ineffective") that convey
outcome without revealing the principle.

### Phase Filtering of Cue Summaries

Phase 1 triggers see only Phase 1 cue summaries. Phase 2 triggers see only
Phase 2 cue summaries.

**Justification**: The trigger generator needs to produce scenarios distinct from
the cues the model will have seen *in that phase*. Phase 2 triggers should not
rehash Phase 1 scenarios.

### Semantic Distance Constraint

The prompt instructs: "Do NOT reuse nouns, verbs, or phrases from the cues.
The trigger should stand alone as a natural conversational turn."

**Justification**: If triggers share surface language with cues, models can
shortcut via lexical matching rather than genuine preference recall. This
prompt-level constraint is a first line of defense; Stage 4 (semantic filtering)
provides a second.

### Related Cue IDs

**Current implementation**: Phase 1 triggers store only Phase 1 cue IDs.
Phase 2 triggers store only Phase 2 cue IDs.

**Known issue**: In the assembled conversation, the model sees *all* prior cues
(Phase 1, drift event, Phase 2) before a Phase 2 trigger. The `related_cue_ids`
field misrepresents the model's actual evidence. **Required fix**: For Phase 2
triggers, `related_cue_ids` should include Phase 1 cues, drift event cues, and
Phase 2 cues — everything the model would have seen up to that point.

### Temperature: 0.7

Same as cue generation. Triggers must be natural and coherent.

---

## Stage 4: Semantic Filtering

**Script**: `04_semantic_filter.py`
**Input**: `cues.json`, `triggers.json`
**Output**: `data/filtered/triggers_filtered.json`, `cues_filtered.json`

### Purpose

Remove trigger-cue pairs where surface similarity is high enough that a model
could answer correctly via lexical/semantic retrieval rather than genuine
preference memory.

### Method

For each trigger, compute similarity against its related cues using two metrics:

1. **BM25 (lexical)**: Token-overlap score using Okapi BM25. Captures word reuse.
2. **MPNet cosine (semantic)**: Sentence embedding cosine similarity using
   `all-mpnet-base-v2`. Captures paraphrase-level similarity.

A trigger is removed if **either** score exceeds its threshold (OR logic).

**Justification for OR**: A trigger that is lexically similar OR semantically
similar can be shortcut. AND logic would only remove triggers that are *both*
lexically and semantically similar, leaving single-channel shortcuts unfiltered.

### Thresholds

| Metric | Threshold | Justification |
|--------|-----------|---------------|
| BM25 | 15.0 | BM25 scores are unbounded; 15.0 corresponds to moderate overlap (~3-4 shared content words in short texts). Determined empirically: scores below 15 showed minimal word reuse in manual inspection of 50 pairs. |
| MPNet cosine | 0.65 | On the [0, 1] scale, 0.65 is the conventional boundary between "related" and "paraphrase" for MPNet (Reimers & Gurevych, 2019). Manual inspection confirmed pairs above 0.65 described substantially similar scenarios. |

**Sensitivity analysis requirement**: Before publication, run filtering at
thresholds {10, 12.5, 15, 17.5, 20} for BM25 and {0.55, 0.60, 0.65, 0.70, 0.75}
for cosine. Report the number of triggers removed at each threshold and the
impact on downstream metrics.

### What Happens to Removed Triggers

Removed triggers are saved to `triggers_removed.json` with their scores for
inspection. The filtered cue set only retains cues referenced by surviving triggers.

### Pipeline Integration

**Current status**: This step exists as a standalone script but is not integrated
into the main generation pipeline. **Required fix**: Make this step mandatory
and report filtering statistics (count removed, score distributions) as part of
the generation log.

---

## Stage 5: Conversation Assembly

**Script**: `05_assemble_conversations.py`
**Input**: `profiles.json`, `cues_filtered.json`, `triggers_filtered.json`
**Output**: `data/conversations/conversations.json`

### Purpose

Embed cues, drift events, and triggers into long conversations (400-600 turns)
with distractor chitchat, simulating a realistic long-term conversational
interaction.

### Conversation Structure

For each of the 20 users, one conversation is assembled:

```
[Phase 1 cues interleaved across topics]
  ↓ distractor chitchat between each cue
[Phase 1 triggers]
  ↓ distractor gap proportional to time_gap
[Drift events / erosion cues]
  ↓ distractor chitchat
[Phase 2 cues interleaved across topics]
  ↓ distractor chitchat
[Phase 2 triggers]
```

### Topic Interleaving

Cues are placed in round-robin order across shuffled topics, not grouped by topic.

**Justification**: Grouping all cues for one topic together would make recall
trivial (the model just needs short-term memory). Interleaving creates cross-topic
interference, requiring the model to selectively retrieve topic-relevant cues
from among many interleaved topics.

### Distractor Generation

Distractors are short chitchat exchanges (2-4 turns) about everyday topics
unrelated to the benchmark domains.

**Temperature: 0.9**

Highest in the pipeline because distractors should be maximally varied and
unpredictable. Quality is less critical than for cues/triggers since distractors
serve as noise, not signal. Temperature=0.9 is below the 1.0 threshold where
outputs become incoherent.

**Distractor Count by Time Gap**:

| Time Gap | Distractors | Justification |
|----------|-------------|---------------|
| One week | 3-6 | Short gap → few intervening conversations |
| Several weeks | 6-12 | Moderate gap → moderate interference |
| A few months | 12-20 | Long gap → substantial interference |
| Several months | 20-30 | Very long gap → heavy interference |

These ranges are calibrated so that "several weeks" (the most common time gap)
produces 6-12 distractors, yielding ~12-24 turns of interference. This is enough
to push the cue out of a model's immediate attention window without making the
conversation unreasonably long.

### Topic Avoidance in Distractors

Distractors are prompted to avoid the first 3 active topics by description.

**Why only 3**: Avoiding all 8 topics would over-constrain the distractor space,
producing repetitive chitchat. 3 topics is enough to prevent obvious topic
contamination while leaving room for natural variety.

### Drift Zone Positioning

**Current implementation**: `topic_drift_position` is computed (random uniform in
[0.3, 0.7]) but never used in segment assembly.

**Required fix**: Either implement position-aware placement (drift events appear
at the computed position in the conversation) or remove the dead code. The current
behavior places drift events after all Phase 1 cues, which is acceptable but
should be documented as the intended design.

### Random Seed

Conversation assembly is seeded (`--seed 42` default).

**Justification**: Unlike profile/cue/trigger generation (which depend on LLM
outputs and are inherently non-deterministic), conversation assembly involves
only local randomness (shuffling, distractor counts). Seeding this stage ensures
the conversation structure is reproducible given the same input data.

---

## Expected Dataset Statistics

| Metric | Stable Users | Drifting Users (per user, k drift topics) | Total (est.) |
|--------|-------------|------------------------------------------|--------------|
| Users | 8 | 12 | 20 |
| Phase 1 cues per topic | 3 | 3 | — |
| Phase 2 cues per topic | 0 | 3 (drifting topics only) | — |
| Drift event cues | 0 | 1 (event) or 3 (accumulation) per topic | — |
| Cues per user | 24 | 24 + 6k (event) or 24 + 6k (accumulation) | ~480-720 |
| Phase 1 triggers per topic | 3 | 2 (drifting) or 3 (stable) | — |
| Phase 2 triggers per topic | 0 | 3 (drifting topics only) | — |
| Triggers per user | 24 | 24 + 2k | ~480-570 |
| Conversation turns | 400-600 | 400-600 | ~8,000-12,000 |
