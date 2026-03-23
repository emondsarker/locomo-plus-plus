# 09 — Next Steps

Status as of generation completion: the benchmark dataset is fully generated
and validated. What remains is model evaluation and analysis.

---

## Phase A: Response Collection (Required)

### A1. Build the response collection script

**What**: A script (`scripts/06_collect_responses.py`) that feeds each
conversation to a target model and records its response at each trigger point.

**How it works**:
1. For each conversation, iterate through turns in order
2. At each trigger turn, present all preceding turns as context and ask the
   model: "How would you respond to this person's situation?"
3. Collect the model's response (its persuasion attempt)
4. Output: `data/responses/{model_name}_responses.json` mapping
   `trigger_id → response_text`

**Design decisions to make**:
- **Context window handling**: Conversations are 1,200-1,800 turns long.
  Most models cannot fit this in a single context window. Options:
  - (a) Feed the full conversation (tests true long-context memory)
  - (b) Feed a fixed window of the last N turns (tests bounded memory)
  - (c) Feed a summary + recent turns (tests retrieval-augmented memory)
  - **Recommendation**: Option (a) for models that support it (e.g., models
    with 200K+ context). Option (b) as an ablation with
    N = {1000, 2000, 4000, 8000} tokens to measure memory decay curves.

- **System prompt**: The model should be told it is a persuader/advisor
  who has been having an ongoing conversation with this person. It should
  NOT be told about Cialdini's principles or the benchmark structure.

- **Response format**: Free-form text. The model should respond naturally,
  not in a structured format. The judge will analyze the response.

### A2. Select target models

**Minimum viable set** (for a thesis):
- 1 long-context model (200K+ context frontier model)
- 1 shorter-context model (GPT-4o with 128K)
- 1 open-weight model (Llama 3 70B or Mistral Large)

**Extended set** (for a stronger paper):
- Multiple capability tiers from the same family (e.g., small vs medium vs
  large) to test whether model capability affects drift detection
- A RAG-augmented baseline (model + vector store of conversation chunks)

### A3. Run response collection

For each target model, run the response collection script. This produces
one response file per model.

**Estimated cost/time**: 526 triggers × ~1,500 avg context turns per trigger.
For a 200K context model, each call processes ~50-100K tokens of context.
Budget ~500 API calls per model.

---

## Phase B: Evaluation (Required)

### B1. Run LLM judge

```bash
cd persuasion/eval
python evaluate.py \
  --responses ../data/responses/sonnet_responses.json \
  --triggers ../data/filtered/triggers_filtered.json \
  --cues ../data/cues/cues.json \
  --output results/sonnet/ \
  --judge-model sonnet
```

**Judge model choice**: Use a strong frontier model at
temperature=0.0. The judge model should ideally be different from the
target model to avoid self-evaluation bias.

**Output**: `judgments.json` (per-trigger labels) + `metrics.json` (aggregates)

### B2. Compute baselines

Four baselines are needed (see methodology/07_known_limitations.md §L6):

1. **Random**: For each trigger, randomly assign one of 7 principles.
   No model call needed — compute analytically or via simulation.

2. **Majority-class**: For each topic, use the most common effective
   principle across all users. Tests whether stereotypical mappings suffice.

3. **Phase-blind**: Always use Phase 1 effective principle for all triggers.
   Phase 1 accuracy = 100% by construction. Phase 2 accuracy reveals
   how much a "never adapts" strategy costs.

4. **No-memory**: Give the model only the trigger (no conversation history).
   Tests how much performance comes from general knowledge vs. cue memory.

### B3. Human evaluation (subset)

Annotate 50+ triggers (stratified by phase, topic, model) with 2 annotators.
Report Cohen's kappa and judge-human agreement.

Use the human eval template: `eval/results/human_eval_template.json`

---

## Phase C: Analysis (Required for thesis/paper)

### C1. Primary results table

| Model | Overall Acc | Avg Score | Memory % | Phase Aware % | Drift Det. Rate | Stale Rate |
|-------|-------------|-----------|----------|---------------|-----------------|------------|
| Random baseline | X% | X | — | — | — | — |
| Phase-blind baseline | X% | X | — | — | — | — |
| Model A | X% | X | X% | X% | X% | X% |
| Model B | X% | X | X% | X% | X% | X% |

### C2. Phase 1 vs Phase 2 accuracy gap

The "adaptation gap" = Phase 1 accuracy − Phase 2 accuracy.
This is the headline metric: how much harder is drift detection than
static preference recall?

### C3. Per-topic and per-principle breakdowns

- Which topics are hardest? (Likely taxes_legal is easiest due to
  authority stereotyping)
- Which principles are hardest to detect from behavioral cues?
- Is there a principle × topic interaction?

### C4. Error analysis

For each model, sample 10-20 "stale" responses and 10-20 "wrong" responses.
Qualitatively analyze:
- What did the model actually say?
- Did it show awareness of the drift event but still use the old principle?
- Did it ignore the cues entirely?

### C5. Context length ablation (if doing option b)

Plot accuracy vs. context window size. Does drift detection degrade faster
than Phase 1 accuracy as context shrinks?

---

## Phase D: Documentation (Required for submission)

### D1. Update methodology docs

- Record actual generation stats in 08_dataset_report.md (done)
- Record evaluation results when available
- Update 07_known_limitations.md with any new limitations discovered

### D2. Write the paper/thesis chapter

Structure:
1. Introduction (why temporal drift matters)
2. Related work (LoCoMo, MSC, memory benchmarks)
3. Benchmark design (draw from methodology/01-04)
4. Experimental setup (models, baselines, judge)
5. Results (Phase A-C outputs)
6. Analysis and discussion
7. Limitations (draw from methodology/07)
8. Conclusion

---

## Priority Order

```
A1 (build response script)     ← NEXT immediate step
A2 (select models)             ← can decide while A1 is being built
A3 (run responses)             ← blocked on A1 + A2
B1 (run judge)                 ← blocked on A3
B2 (compute baselines)         ← can start in parallel with A3
B3 (human eval)                ← blocked on B1 (need judge output for comparison)
C1-C5 (analysis)               ← blocked on B1 + B2
D1-D2 (writeup)                ← blocked on C
```

The critical path is: **A1 → A3 → B1 → C → D**.
Baselines (B2) can be computed in parallel at any time.
