# Methodology

This directory contains the complete methodological specification for LoCoMo++,
a benchmark for evaluating temporal preference drift detection in LLM persuasion memory.

Every design decision is justified. Every constant is explained. Every known threat
to validity is documented with its mitigation.

## Document Index

| Document | Purpose |
|----------|---------|
| [01_research_question.md](01_research_question.md) | What we measure, why it matters, and how it extends prior work |
| [02_theoretical_framework.md](02_theoretical_framework.md) | Cialdini's principles, temporal drift model, and phase structure |
| [03_dataset_construction.md](03_dataset_construction.md) | Step-by-step generation pipeline with justification for every parameter |
| [04_evaluation_design.md](04_evaluation_design.md) | Judge prompt design, scoring rubric, and metric definitions |
| [05_bias_and_validity.md](05_bias_and_validity.md) | Threat model, known biases, and concrete mitigations |
| [06_reproducibility.md](06_reproducibility.md) | Seeding, versioning, and exact reproduction protocol |
| [07_known_limitations.md](07_known_limitations.md) | Honest accounting of what this benchmark cannot claim |
| [08_dataset_report.md](08_dataset_report.md) | Frozen dataset statistics, anomalies, and validation results |
| [09_next_steps.md](09_next_steps.md) | Response collection, evaluation, analysis, and writeup plan |
