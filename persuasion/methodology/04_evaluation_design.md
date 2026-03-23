# 04 — Evaluation Design

## Evaluation Architecture

Model responses are evaluated by an LLM judge operating on per-trigger granularity.
Each trigger produces one judgment. Judgments are aggregated into metrics.

```
[Model Response] + [Trigger Metadata] + [Related Cues] → [LLM Judge] → [Label + Reasoning]
                                                                            ↓
                                                                      [Aggregate Metrics]
```

We use an LLM judge rather than exact-match or embedding similarity because:
1. Persuasion principle identification requires understanding *how* a response
   attempts to persuade, not just *what words* it uses. A response using authority
   might cite "recent research" without ever saying "authority."
2. Human evaluation is the gold standard but does not scale to the full dataset.
   We use LLM judging for the primary evaluation and human annotation for
   validation (see Human Evaluation below).

---

## Judge Prompt Design

The judge receives:
1. The user's name and topic
2. The current phase and phase context
3. Prior cue dialogues (evidence the model had access to)
4. The trigger scenario
5. The model's response
6. The effective, ineffective, and (for Phase 2) stale principles

### Phase Context

**Phase 1**: "The user's preferences are stable and based on interactions earlier
in the conversation."

**Phase 2**: "The user's preferences have SHIFTED during the conversation. The
Phase 1 effective principle ({stale_principle}) no longer works."

### Cue Evidence Provided to Judge

**Required fix (critical)**: The current implementation filters cue evidence by
phase — Phase 2 triggers only see Phase 2 cues. This is incorrect because:

1. The judge needs to see *what changed* to identify "stale" behavior. Without
   Phase 1 cues, the judge cannot distinguish "model uses old principle" from
   "model uses a random principle."
2. In the actual conversation, the model sees all cues from both phases. The
   judge's evidence should match the model's evidence.

**Correct design**: For Phase 2 triggers, the judge should receive:
- Phase 1 cues (labeled as "earlier evidence — preferences at that time")
- Drift event or erosion cues (labeled as "transition signals")
- Phase 2 cues (labeled as "current evidence — updated preferences")

For Phase 1 triggers, the judge receives only Phase 1 cues (no Phase 2 exists yet).

### Principle Identification

The judge is asked to identify which of Cialdini's 7 principles the model's
response primarily employs. The output field is `principle_detected`.

**Required fix**: The judge prompt must include the canonical list of 7 principle
names and require the output to use one of these exact strings. Without this
constraint, the judge may output variant names ("social proof" vs "social_proof"
vs "sp"), making aggregation unreliable.

### Stale Detection

**Required fix (critical)**: The current prompt says "Watch for 'stale' if the
model uses {stale_principle}!" This anchors the judge to over-detect staleness.

**Correct design**: Remove the explicit hint. Instead, provide the Phase 1 cues
as evidence and let the judge independently determine whether the response uses
an outdated strategy. The stale_principle field should be used only for *scoring*
(after the judge identifies the principle), not for *prompting* the judge.

Specifically:
- The judge identifies `principle_detected` from the response
- Post-processing compares `principle_detected` to `stale_principle`
- If they match AND the trigger is Phase 2, the label is "stale"
- This removes the judge's awareness of which specific principle is stale,
  eliminating anchoring bias

---

## Scoring Rubric

| Label | Score | Definition | When Applied |
|-------|-------|------------|-------------|
| correct | +2 | Response employs the effective principle for the current phase | Phase 1 or Phase 2 |
| partial | +1 | Response uses a neutral principle (neither effective nor ineffective), or gives relevant but strategically vague advice | Phase 1 or Phase 2 |
| wrong | 0 | Response uses the ineffective principle, gives a completely generic response, or contradicts observed evidence | Phase 1 or Phase 2 |
| stale | -1 | Response employs the Phase 1 effective principle on a Phase 2 trigger (outdated strategy) | Phase 2 only |

### Why These Scores?

**correct = +2**: The maximum score. The model correctly identified and applied
the current effective principle.

**partial = +1**: Half credit. The model showed some awareness but did not
converge on the correct strategy. This captures responses that are
"not wrong but not precisely right."

**wrong = 0**: No credit. The model either ignored context entirely or used
the wrong strategy. Zero (not negative) because a wrong answer does not
indicate the model *had* the right information and misapplied it.

**stale = -1**: Negative score. This is *worse* than wrong because it indicates
the model remembered the old preference but failed to detect the shift. A model
that remembers nothing scores 0 (wrong). A model that remembers the old
preference and confidently applies it despite contrary evidence scores -1.
The negative score captures this: **confident misapplication of outdated
knowledge is worse than ignorance.**

### Why Not 0/1 Binary Scoring?

Binary scoring (correct/incorrect) would collapse "partial" and "wrong" into
the same category. The partial label captures an important middle ground:
the model shows memory awareness but doesn't converge on the optimal strategy.
This distinction matters for understanding *how* models fail.

### Score Range

- Maximum possible: 2.0 (all correct)
- Minimum possible: -1.0 (all stale)
- A model that guesses randomly across 7 principles: ~0.29 expected score
  (1/7 chance of correct = +2, 1/7 chance of ineffective = 0, 5/7 chance of
  neutral = +1; weighted: (2/7 + 5/7)/1 ≈ 1.0 — but this overestimates because
  "partial" requires topical relevance, not just a non-matching principle)
- A model that ignores all cues and gives generic advice: expected ~0.5-1.0
  (mostly "partial" with some "wrong")

---

## Metric Definitions

### Primary Metrics

| Metric | Formula | Measures |
|--------|---------|----------|
| Correct % | correct_count / total × 100 | Overall accuracy |
| Avg Score | sum(label_scores) / total | Overall performance (range: -1 to 2) |
| Memory Awareness % | memory_aware_count / total × 100 | Does the response reference prior interactions? |
| Phase Awareness % | phase_aware_count / total × 100 | Does the response reflect *current* preferences? |

### Drift-Specific Metrics (Phase 2 Only)

| Metric | Formula | Measures |
|--------|---------|----------|
| Drift Detection Rate | phase2_correct / phase2_total × 100 | Can the model adapt to changed preferences? |
| Stale Rate | phase2_stale / phase2_total × 100 | How often does the model apply outdated knowledge? |

### Breakdown Metrics

- **Per-phase**: Phase 1 accuracy vs Phase 2 accuracy. The difference measures
  the "adaptation gap" — how much harder drift detection is than static recall.
- **Per-topic**: Accuracy by domain. Reveals whether certain domains are easier
  to learn (e.g., authority for taxes may be more stereotypical and thus easier).
- **Per-principle**: Accuracy by effective principle. Reveals whether certain
  principles are easier for models to identify from behavioral cues.

### Per-Topic Score Formula

**Current implementation** (evaluate.py line 118-120):
```python
avg_score = sum(LABEL_SCORES[l] * count[l] for l in LABEL_SCORES) / (n * 2) * 100
```

This normalizes to 0-100 by dividing by `n * 2` (the maximum possible score).
Note that stale labels make negative scores possible, so the range is actually
[-50, 100].

---

## Judge Model Selection

**Default**: The judge model should be a strong reasoning model (e.g., GPT-4o or
a comparable frontier model). It must be capable of:
1. Identifying which persuasion principle a response employs
2. Understanding the distinction between 7 related but distinct principles
3. Reasoning about temporal phases

**Temperature: 0.0** for the judge to maximize determinism and reproducibility.
Non-zero temperature would introduce variance in labels across runs.

**Independence from generation model**: The generation model (used in Stages 1-5)
and the judge model may differ. This is acceptable because:
1. The generation model creates the *data*; the judge evaluates *model responses*
   to that data. They serve different roles.
2. Using a stronger model for judging than generation is standard practice
   (analogous to using expert human annotators to evaluate crowd-sourced data).

---

## Human Evaluation

### Purpose

LLM judge labels are validated against human annotations on a random subset.

### Template

`evaluate.py` generates a `human_eval_template.json` containing:
- Trigger text, related cues, model response
- Fields for: label, principle_detected, memory_awareness, phase_awareness, notes
- For Phase 2 triggers: a note about the stale principle

**Required fix**: The human eval template currently includes the anchoring note
"WATCH FOR 'STALE': Using {stale_principle}..." This introduces the same
anchoring bias as the judge prompt. **Remove this hint from the human template.**
Annotators should receive the Phase 1 and Phase 2 cues and independently
determine whether the response is stale.

### Inter-Annotator Agreement

Report Cohen's kappa on the 4-label classification (correct/partial/wrong/stale)
for a sample of at least 50 triggers annotated by 2 independent annotators.

### Judge Calibration

Compare LLM judge labels to human labels. Report:
- Agreement rate (% matching labels)
- Confusion matrix (which labels does the judge confuse most?)
- Systematic biases (does the judge over-assign "partial"? under-detect "stale"?)
