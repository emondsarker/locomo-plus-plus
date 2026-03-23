# 06 — Reproducibility

## The Reproducibility Problem

LLM-generated benchmarks are inherently non-deterministic. The same prompt given
to the same model at different times may produce different outputs due to:
1. Model version updates (even within the same model name)
2. Temperature sampling
3. Server-side batching effects on floating-point operations

Full bitwise reproducibility of generated data is therefore impossible. Instead,
we aim for **structural reproducibility**: given the same code, parameters, and
model version, the generated dataset should have the same statistical properties
(distribution of principles, cue counts, trigger counts, conversation lengths).

---

## What Must Be Recorded

Every dataset generation run must record the following metadata:

### Model Information
- **Generation model**: Exact model name and version (e.g., `haiku-4-5-20251001`)
- **Judge model**: Exact model name and version
- **API endpoint**: CLI version and tool used
- **Date of generation**: UTC timestamp

### Parameters
- **Random seed**: For all stages that use `random.seed()`
- **Temperature**: Per-stage (profile=0.8, cues=0.7, triggers=0.7, distractors=0.9, judge=0.0)
- **Batch sizes**: Profile batch=5, cue positive_per_topic=2, cue negative_per_topic=1, triggers_per_topic=3
- **Filtering thresholds**: BM25=15.0, cosine=0.65

### Counts (Post-Generation Validation)
- Number of profiles generated (expected: 20)
- Number of stable vs drifting profiles (expected: 8 + 12)
- Drift type distribution (event vs accumulation count)
- Total cues generated (expected: ~480-720 depending on drift topics)
- Total triggers generated (expected: ~480-570)
- Triggers removed by semantic filter (with score distributions)
- Conversation turn counts (min, max, mean, std)
- LLM call failure count per stage

---

## Seeding Protocol

### Current State

Only `05_assemble_conversations.py` accepts a `--seed` argument. Stages 1-4 are
not seeded.

### Required Changes

Add `--seed` argument to all generation scripts. At the top of each `main()`:
```python
random.seed(args.seed)
```

This controls:
- Topic shuffling order
- Drift topic selection (for drifting users)
- Distractor count selection
- Drift zone positioning

This does NOT control:
- LLM output content (inherently non-deterministic)
- LLM call ordering effects

**Justification**: Seeding the local randomness ensures that given identical LLM
outputs, the same dataset structure is produced. This isolates non-determinism to
the LLM calls themselves.

---

## Version Pinning

### Python Dependencies

`requirements.txt` must pin exact versions:
```
rank-bm25==0.2.2
sentence-transformers==2.2.2
numpy==1.24.0
```

**Justification**: BM25 tokenization and MPNet embeddings are deterministic given
identical library versions. Version drift could change semantic filter results.

### Model Versioning

Record the exact model identifier, not the alias. For example:
- Use the full model identifier (e.g., `haiku-4-5-20251001`) not `haiku`
- Use `gpt-4o-2024-08-06` not `gpt-4o`

**Justification**: Model aliases resolve to different versions over time. A
benchmark generated with "haiku" in March 2026 and "haiku" in June 2026 may use
different model versions.

---

## Generation Logging

Each pipeline run should produce a `generation_log.json` with:

```json
{
  "timestamp": "2026-03-17T12:00:00Z",
  "git_commit": "abc123",
  "model": {
    "generation": "haiku-4-5-20251001",
    "judge": "sonnet-4-6"
  },
  "parameters": {
    "seed": 42,
    "temperatures": {"profiles": 0.8, "cues": 0.7, "triggers": 0.7, "distractors": 0.9, "judge": 0.0},
    "batch_sizes": {"profiles": 5, "positive_per_topic": 2, "negative_per_topic": 1, "triggers_per_topic": 3},
    "filter_thresholds": {"bm25": 15.0, "cosine": 0.65},
    "user_composition": {"total": 20, "stable": 8, "drifting": 12}
  },
  "results": {
    "profiles_generated": 20,
    "stable_count": 8,
    "drifting_count": 12,
    "drift_type_distribution": {"event": 6, "accumulation": 6},
    "total_cues": 612,
    "total_triggers": 528,
    "triggers_removed_by_filter": 23,
    "conversations_assembled": 20,
    "turn_count_stats": {"min": 387, "max": 623, "mean": 492, "std": 58},
    "llm_call_failures": {"profiles": 0, "cues": 2, "triggers": 1, "distractors": 0}
  },
  "validation": {
    "profile_errors": [],
    "empty_cue_topics": [],
    "empty_trigger_topics": []
  }
}
```

**Status**: Not yet implemented. Must be added before final generation.

---

## Artifact Preservation

After final generation, the following artifacts must be preserved and versioned:

1. **Code snapshot**: Git commit hash of the generation code
2. **Generated data**: `profiles.json`, `cues.json`, `triggers.json`,
   `conversations.json`, `triggers_filtered.json`, `triggers_removed.json`
3. **Generation log**: `generation_log.json`
4. **Filter statistics**: Score distributions for BM25 and cosine

These artifacts constitute the "frozen benchmark." Model evaluations should
reference the frozen benchmark version, not regenerate from scratch.

---

## Reproduction Checklist

For anyone reproducing the benchmark:

1. Clone the repository at the exact git commit
2. Install Python dependencies from pinned `requirements.txt`
3. Verify LLM CLI version
4. Run the pipeline with the documented seed and parameters
5. Compare generation log statistics to the published values
6. Expect structural similarity but not identical content
   (same counts, similar distributions, different specific dialogues)
