# 08 — Dataset Generation Report

This document records the exact outputs of the dataset generation pipeline,
including statistics, anomalies, and validation results. It serves as the
provenance record for the frozen benchmark.

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| Generation model | LLM (small model via CLI, `--model haiku`) |
| Random seed | 42 (all stages) |
| Profile batch size | 5 |
| Stable users | 8 |
| Drifting users | 12 |
| Max drifting topics/user | 4 (enforced post-generation) |
| Positive cues per topic | 2 |
| Negative cues per topic | 1 |
| Triggers per non-drifting topic | 3 |
| Triggers per drifting topic | 5 (2 Phase 1 + 3 Phase 2) |
| BM25 filter threshold | 15.0 |
| Cosine filter threshold | 0.65 |
| Distractor temperature | 0.9 |
| Conversation assembly seed | 42 |

---

## Pipeline Execution Summary

### Stage 1: Profile Generation

- **Output**: 20 profiles
- **Composition**: 8 stable + 12 drifting (7 event, 5 accumulation)
- **Drifting topics per user**: min=2, max=3, mean=2.5
- **Validation errors**: 0
- **Clamping applied**: 0 profiles needed drifting topic reduction
  (all were within the 1-4 limit after prompt strengthening)

### Stage 2: Cue Generation

- **Output**: 633 cues
- **Phase distribution**:
  - Phase 1: 484 (76.5%)
  - Phase 2: 92 (14.5%)
  - Drift events: 15 (2.4%)
  - Erosion cues: 42 (6.6%)
- **Cues per user**: min=23, max=43, mean=31.6
- **LLM call errors**: 5 JSON parse failures across 20 users
  (users 02, 03, 08, 11, 19 each lost 1 cue from a failed call)
- **Placeholder strings in output**: 0

### Stage 3: Trigger Generation

- **Output**: 540 triggers
- **Phase distribution**:
  - Phase 1: 450 (83.3%)
  - Phase 2: 90 (16.7%)
- **Phase 2 triggers with stale_principle set**: 90/90 (100%)
- **LLM call errors**: 2 retries (users 09, 18), both recovered

### Stage 4: Semantic Filtering

- **Kept**: 526 triggers (97.4%)
- **Removed**: 14 triggers (2.6%)
- **Removed triggers — avg BM25**: 13.04
- **Removed triggers — avg cosine**: 0.618
- **Filtered cues retained**: 627 (6 cues dropped because their only
  referencing trigger was removed)
- **Embedding model**: `all-mpnet-base-v2` (sentence-transformers)

### Stage 5: Conversation Assembly

- **Output**: 20 conversations
- **Turn counts**: min=1,281, max=1,835, mean=1,614, total=32,283
- **Triggers embedded**: 526 (Phase 1: 449, Phase 2: 77)
- **Temporal ordering violations**: 0/130 checked
  (all Phase 1 triggers precede drift events; all Phase 2 triggers follow)
- **Distractor generation errors**: Frequent JSON parse failures from Haiku
  (~3-13 per conversation). Handled gracefully — conversations have slightly
  fewer distractors than planned but are structurally complete.
- **Turn type distribution**:
  - Distractors: ~89%
  - Cues: ~8%
  - Triggers: ~2%
  - Drift/erosion signals: ~1%

---

## File Manifest

| File | Records | Size |
|------|---------|------|
| `data/profiles/profiles.json` | 20 profiles | 39 KB |
| `data/cues/cues.json` | 633 cues | 622 KB |
| `data/triggers/triggers.json` | 540 triggers (pre-filter) | 421 KB |
| `data/filtered/triggers_filtered.json` | 526 triggers (post-filter) | 436 KB |
| `data/filtered/cues_filtered.json` | 627 cues (post-filter) | 618 KB |
| `data/filtered/triggers_removed.json` | 14 removed triggers | — |
| `data/conversations/conversations.json` | 20 conversations | 6,899 KB |
| **Total** | | **8.8 MB** |

---

## Known Anomalies

### 1. Phase 2 trigger count is 77 in conversations, not 90

90 Phase 2 triggers were generated, but after semantic filtering (14 removed)
and conversation assembly, 77 Phase 2 triggers appear in the final conversations.
The 13 missing Phase 2 triggers were among the 14 removed by the semantic filter.
This is expected behavior — the filter is more aggressive on Phase 2 triggers
because they share topic vocabulary with Phase 1 cues.

**Impact**: Drift detection metrics are computed over 77 Phase 2 triggers
rather than 90. This is still sufficient for aggregate statistics but reduces
per-user Phase 2 sample sizes.

### 2. Uneven cue counts across users

Users range from 23 to 43 cues. The variation comes from:
- Stable users: ~24 cues (8 topics × 3 cues)
- Drifting users with 2 drift topics: ~36 cues (6×3 + 2×9)
- Drifting users with 3 drift topics: ~42 cues (5×3 + 3×9)
- JSON parse failures reducing some users by 1 cue

This variation is inherent to the design (drifting users need more cues)
and does not indicate a data quality issue.

### 3. Distractor generation errors

Haiku frequently returned malformed JSON for distractor generation (~3-13
errors per conversation). The assembly script handled these gracefully by
requesting additional batches. Conversations are slightly shorter on
distractors than planned but remain in the 1,281-1,835 turn range, well
within the target of 400+ turns.

---

## Validation Checks Passed

- [x] All 20 profiles pass schema validation
- [x] All drifting topics have phase_1.effective ≠ phase_2.effective
- [x] All Phase 2 triggers have stale_principle set
- [x] Phase 2 related_cue_ids include all phases (P1 + drift + P2)
- [x] Zero placeholder strings in cue dialogues
- [x] Zero temporal ordering violations in conversations
- [x] All 8 topics covered in every conversation
- [x] Semantic filter removed 14 high-overlap triggers
- [x] Speaker balance: ~55% P / ~45% U (reasonable)
