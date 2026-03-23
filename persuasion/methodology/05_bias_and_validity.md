# 05 — Bias and Validity

This document catalogs every known threat to the benchmark's validity, categorizes
each by severity, and specifies concrete mitigations.

---

## Threat Model

We organize threats into four categories following the validity framework from
Messick (1995):

1. **Construct validity**: Does the benchmark measure what we claim it measures?
2. **Internal validity**: Are the results attributable to the model's capability,
   or to artifacts of the benchmark design?
3. **External validity**: Do the results generalize beyond this specific benchmark?
4. **Statistical conclusion validity**: Are the metrics computed correctly and
   with sufficient power?

---

## Construct Validity Threats

### T1: Principle Leakage in Trigger Generation (CRITICAL — must fix)

**Threat**: The trigger generation prompt receives cue summaries that include
principle descriptions (e.g., "Persuader's approach: People defer to credible
experts"). This leaks the principle name into the trigger context, potentially
biasing trigger scenarios to be more aligned with the effective principle.

**Impact**: Triggers may be easier than intended because they are inadvertently
shaped by knowledge of the correct answer. This inflates accuracy scores.

**Mitigation**: Replace principle descriptions in cue summaries with neutral
outcome labels:
```
Before: "Persuader's approach: People defer to credible experts → receptive"
After:  "Approach A → user was receptive"
        "Approach B → user was resistant"
```
The trigger generator needs to know what *happened* (to avoid duplicating scenarios)
but not *which principle* was used.

**Status**: FIXED. `03_generate_triggers.py:summarize_cues()` now uses neutral
"Approach A/B/C" labels instead of principle descriptions.

### T2: Judge Anchoring on "Stale" Label (CRITICAL — must fix)

**Threat**: The judge prompt explicitly warns "Watch for 'stale' if the model uses
{stale_principle}!" This anchors the judge to look for a specific principle and
classify it as stale, inflating the stale detection rate.

**Impact**: The stale rate metric is unreliable. The judge may label responses as
"stale" because it was primed to look for that label, not because the response
genuinely employs the outdated principle.

**Mitigation**: Remove the anchoring hint entirely. The judge should:
1. Identify the principle the response employs (without knowing which is stale)
2. Post-processing compares the detected principle to the stale_principle
3. If they match and the trigger is Phase 2, the label becomes "stale"

This separates principle identification (subjective judgment) from stale
classification (objective comparison), making the stale metric trustworthy.

**Status**: FIXED. Judge prompt no longer mentions stale. `evaluate.py:apply_stale_labels()`
handles stale detection in post-processing by comparing `principle_detected` to
`stale_principle`.

### T3: Phase-Split Evidence Breaks Drift Detection (CRITICAL — must fix)

**Threat**: Phase 2 judge prompts only show Phase 2 cues. Without Phase 1 cues,
the judge cannot see what *changed*, making "stale" detection structurally
impossible from the judge's perspective.

**Impact**: The judge cannot reliably distinguish between a model using an
outdated principle (stale) and a model using a random principle (wrong), because
it has no Phase 1 baseline to compare against.

**Mitigation**: For Phase 2 triggers, include all cues (Phase 1, drift, Phase 2)
in the judge's evidence, clearly labeled by phase. The judge needs the full
timeline to evaluate whether the model adapted.

**Status**: FIXED. `judge_prompt.py:format_judge_prompt()` passes all cues
(Phase 1 + drift + Phase 2) for Phase 2 triggers. `03_generate_triggers.py`
stores all-phase `related_cue_ids` for Phase 2 triggers.

---

## Internal Validity Threats

### T4: Batch Ordering Effects in Profile Generation (HIGH)

**Threat**: Early batches of profiles have maximum demographic diversity.
Later batches are constrained by "Names already used: ..." lists, potentially
biasing later profiles toward different demographics or personality types.

**Impact**: If profiles are ordered by user_id (they are), systematic differences
between early and late profiles could confound per-user analysis.

**Mitigation options** (choose one):
1. **Pre-generate names**: Generate 20 diverse names in a single call before
   profile generation. Assign names to profiles after generation.
2. **Shuffle profile order**: After generation, randomly shuffle profiles before
   assigning user_ids. This breaks the correlation between generation order and
   user_id.
3. **Document and accept**: Note the ordering effect in limitations. Since profiles
   are analyzed per-user and per-topic (not sequentially), the impact on aggregate
   metrics is minimal.

**Recommended**: Option 2 (shuffle) — lowest effort, sufficient mitigation.

**Status**: FIXED. `01_generate_profiles.py` shuffles profiles with `random.shuffle()`
before assigning user_ids.

### T5: Topic Ordering Effects in Cue/Trigger Generation (MEDIUM)

**Threat**: Cues and triggers are generated in TOPICS list order (personal_finance
first, lifestyle last). If the LLM's generation quality drifts over a long session,
early topics may receive higher-quality outputs.

**Impact**: Topic-level metric differences could partly reflect generation order
rather than genuine difficulty differences.

**Mitigation**: Shuffle topic order per user during cue and trigger generation.
This distributes any ordering effect randomly across topics.

**Status**: NOT FIXED. Topic order is still deterministic (TOPICS list order).
Accepted risk — documented as limitation. Impact is minimal because topics are
analyzed independently, not sequentially.

### T6: Principle Distribution Imbalance (MEDIUM)

**Threat**: The generation prompt requests "diverse" principle assignments but does
not enforce balanced distribution. LLMs tend toward stereotypical mappings (authority
for taxes, social_proof for social_relationships), which could create unbalanced
representation.

**Impact**: Per-principle accuracy metrics may have very different sample sizes,
making some principles' metrics unreliable. Models may also exploit stereotypical
mappings (correctly guessing "authority for taxes" without remembering cues).

**Mitigation**:
1. After profile generation, compute and report the distribution of effective
   principles across topics.
2. Flag if any principle appears as effective for >40% of users on any single topic.
3. If imbalanced, regenerate with explicit distribution constraints
   (e.g., "each principle should appear as effective for at most 5 users on any
   given topic").

**Status**: NOT ENFORCED programmatically, but profile generation prompt was
strengthened and drifting topic count is clamped to MAX_DRIFTING_TOPICS=4.
Distribution should be monitored in the dataset report (see 08_dataset_report.md).

### T7: Semantic Filtering Not Integrated (HIGH)

**Threat**: Stage 4 (semantic filtering) exists as a script but is not run as part
of the default pipeline. If skipped, triggers with high cue similarity remain in the
dataset, allowing models to shortcut via lexical matching.

**Impact**: Accuracy scores may be inflated by models that retrieve similar text
rather than remembering preferences.

**Mitigation**: Make semantic filtering a mandatory pipeline step. Report:
- Number of triggers removed
- Distribution of BM25 and cosine scores (histogram)
- Examples of removed triggers for manual inspection

**Status**: DONE. Semantic filter was run on the generated dataset. Results:
526 kept, 14 removed (2.6%). See 08_dataset_report.md for details. The filter
remains a standalone script (not auto-invoked by a pipeline runner) but was
executed as part of the generation sequence.

### T8: Silent Failures Produce Hollow Data (MEDIUM)

**Threat**: If LLM calls fail during cue/trigger generation, the error is caught
and an empty list is used. This can produce users with 0 cues or 0 triggers for
some topics.

**Impact**: Triggers without supporting cues test nothing (the model had no
evidence to learn from). These would produce "wrong" labels that reflect missing
data, not model failure.

**Mitigation**:
1. After generation, validate that every user-topic pair has the expected number
   of cues and triggers. Flag gaps.
2. Retry failed LLM calls up to 3 times before accepting failure.
3. Remove placeholder fallback strings that could leak into the dataset.

**Status**: PARTIALLY FIXED. Placeholder strings removed from `02_generate_cues.py`
(returns None/empty instead of fake strings). Retry logic exists in `utils.py:call_llm()`
(2 retries). Post-generation validation confirmed 0 placeholder strings in the
final dataset. Full per-user-topic gap checking not yet automated.

---

## External Validity Threats

### T9: LLM-Generated Data (Inherent Limitation)

**Threat**: All profiles, cues, triggers, and distractors are generated by an LLM.
The dialogue patterns, persuasion strategies, and user reactions may not reflect
real human behavior.

**Impact**: Models might perform well on LLM-generated persuasion patterns but
poorly on real human interactions (or vice versa).

**Mitigation**: This is an inherent limitation of synthetic benchmarks. We mitigate
partially by:
1. Grounding in Cialdini's empirically-validated framework (not ad-hoc categories)
2. Requiring implicit outcomes (natural reactions, not labels)
3. Human validation of a subset of generated dialogues for naturalness

This limitation should be prominently stated in any publication.

### T10: Single Generation Model (LOW)

**Threat**: All data is generated by one model (e.g., a small LLM). The generated
dialogues may reflect that model's stylistic and cultural biases.

**Impact**: Models from the same family as the generator may have an advantage
(familiar patterns). Models from different families may face distribution shift.

**Mitigation**:
1. Use a different model family for generation vs evaluation where possible.
2. Report which model generated the dataset.
3. Future work: regenerate with multiple models and compare benchmark stability.

### T11: English Only (LOW for this study)

**Threat**: All data is in English. Persuasion principles may operate differently
across languages and cultures.

**Impact**: Results do not generalize to non-English settings.

**Mitigation**: Stated as a scope limitation. Cross-lingual extension is future work.

---

## Statistical Conclusion Validity Threats

### T12: Small Sample Size for Drift Metrics (MEDIUM)

**Threat**: With 12 drifting users and 1-4 drifting topics each, the number of
Phase 2 triggers is approximately 24-48. Per-user drift detection rate may be
based on as few as 3 Phase 2 triggers.

**Impact**: Per-user drift metrics may have high variance. A model that gets 2/3
Phase 2 triggers correct (67%) vs 1/3 (33%) differs by a single trigger.

**Mitigation**:
1. Report aggregate drift metrics (across all users) as the primary metric.
2. Report per-user drift metrics with confidence intervals.
3. Do not over-interpret per-user differences.

### T13: Score Scale Asymmetry (LOW)

**Threat**: The scoring scale is asymmetric: correct=+2, stale=-1. The mean score
for a random model depends on the distribution of labels, making the scale harder
to interpret than a symmetric one.

**Impact**: Comparing average scores across models requires understanding the
label distribution, not just the scores.

**Mitigation**: Always report label distributions (% correct, partial, wrong, stale)
alongside average scores. The average score is a convenience metric, not the
primary result.

---

## Summary Table

| ID | Threat | Severity | Category | Status |
|----|--------|----------|----------|--------|
| T1 | Principle leakage in trigger prompts | CRITICAL | Construct | Must fix |
| T2 | Judge anchoring on "stale" | CRITICAL | Construct | Must fix |
| T3 | Phase-split evidence in judge | CRITICAL | Construct | Must fix |
| T4 | Batch ordering in profiles | HIGH | Internal | Fix or document |
| T5 | Topic ordering in cues/triggers | MEDIUM | Internal | Fix recommended |
| T6 | Principle distribution imbalance | MEDIUM | Internal | Add validation |
| T7 | Semantic filter not integrated | HIGH | Internal | Integrate into pipeline |
| T8 | Silent failures produce hollow data | MEDIUM | Internal | Add retry + validation |
| T9 | LLM-generated data | Inherent | External | Document as limitation |
| T10 | Single generation model | LOW | External | Document |
| T11 | English only | LOW | External | Document |
| T12 | Small sample for drift metrics | MEDIUM | Statistical | Report with CIs |
| T13 | Score scale asymmetry | LOW | Statistical | Report distributions |

### Pre-Generation Checklist

Before generating the final dataset, these CRITICAL and HIGH items must be resolved:

- [x] T1: Remove principle names from trigger cue summaries → `03_generate_triggers.py` uses neutral "Approach A/B" labels
- [x] T2: Remove stale anchoring from judge prompt; move to post-processing → `judge_prompt.py` no longer mentions stale; `evaluate.py:apply_stale_labels()` handles it
- [x] T3: Include all-phase cues in judge evidence for Phase 2 triggers → `judge_prompt.py:format_judge_prompt()` passes full cue timeline; `03_generate_triggers.py` stores all-phase `related_cue_ids` for Phase 2
- [x] T4: Shuffle profiles after generation → `01_generate_profiles.py` shuffles before assigning user_ids
- [ ] T7: Integrate semantic filtering into pipeline with reporting
- [x] T8: Remove placeholder fallback strings → `02_generate_cues.py` returns None/empty on failure instead of fake strings
