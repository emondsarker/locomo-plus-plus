# 01 — Research Question

## Primary Research Question

**Can large language models detect and adapt to temporal shifts in user persuasion
preferences during long-form conversational interactions?**

## Why This Matters

Existing long-term memory benchmarks (LoCoMo, MSC, LOCOMO-Plus) evaluate whether
models can *recall* static facts across conversation turns. This is necessary but
insufficient for real-world deployment. In practice, user preferences change:
a person who once responded to expert authority may, after a bad experience with a
financial advisor, shift toward peer-based social proof. A capable conversational
agent must detect this shift from behavioral signals and update its strategy.

This benchmark tests a harder capability than static recall: **adaptive cognitive
memory** — the ability to learn *what worked and what didn't* from observation,
detect when those patterns change, and generalize updated knowledge to new situations.

## What We Measure (Operationally)

We decompose the research question into four measurable sub-capabilities:

### 1. Preference Learning (Phase 1 Accuracy)
Can the model infer which persuasion principle is effective for a given user on a
given topic, based on observing prior successful and unsuccessful persuasion attempts?

**Operationalization**: Present the model with 3 cue dialogues (2 positive, 1 negative)
for a user-topic pair in Phase 1, then test with a new scenario in the same topic
domain. Score whether the model's response employs the effective principle.

### 2. Preference Retention Across Interference
Can the model maintain learned preferences when interleaved with cues from other
topics and distractor chitchat?

**Operationalization**: Cues and triggers are separated by 3-30 distractor exchanges
and cross-topic cues. The model must recall the correct principle despite interference.

### 3. Drift Detection (Phase 2 Accuracy)
After observing signals that a user's preferences have changed (via a life event or
gradual erosion), does the model update its strategy?

**Operationalization**: Present Phase 1 cues, then a drift signal (single event or
3 erosion cues), then Phase 2 cues showing a new effective principle. Test with
Phase 2 triggers. Score whether the response uses the *updated* principle.

### 4. Staleness Avoidance (Stale Rate)
Does the model avoid confidently applying an outdated strategy that no longer works?

**Operationalization**: For Phase 2 triggers, detect whether the model uses the
Phase 1 effective principle (which is now stale). This is scored as *worse* than
a generic wrong answer because it indicates the model remembered old preferences
but failed to detect the shift.

## How This Extends Prior Work

| Dimension | LoCoMo (Maharana et al., 2024) | LoCoMo++ (This Work) |
|-----------|-------------------------------|----------------------|
| Memory type | Static fact recall | Adaptive preference learning |
| What changes | Nothing (ground truth is fixed) | User preferences shift mid-conversation |
| Signal type | Explicit statements | Implicit behavioral outcomes |
| Evaluation | Did the model remember X? | Did the model *adapt* to a changed X? |
| Failure mode tested | Forgetting | Forgetting + staleness (using outdated knowledge) |
| Theoretical grounding | General QA | Cialdini's persuasion framework |

## Scope Limitations

This benchmark does **not** test:
- Whether the model can persuade effectively in general (we test memory, not rhetoric)
- Real human persuasion outcomes (all data is LLM-generated, not field data)
- More than 2 temporal phases (we test one drift per topic; real preferences may
  drift multiple times)
- Whether the model understands *why* preferences changed (we only test *that* it
  detects the change)

These limitations are documented in detail in [07_known_limitations.md](07_known_limitations.md).
