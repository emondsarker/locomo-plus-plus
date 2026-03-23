# 02 — Theoretical Framework

## Persuasion Principles: Cialdini's Framework

We ground persuasion preferences in Cialdini's 7 principles of influence
(Cialdini, 2021), the most widely cited taxonomy in persuasion psychology.
We use this framework rather than constructing our own because:

1. **Established validity**: These principles have been empirically validated across
   cultures and contexts over four decades of research.
2. **Mutual exclusivity**: The 7 principles describe distinct psychological mechanisms.
   A persuasion attempt can be classified as primarily employing one principle, making
   labeling tractable.
3. **Sufficient granularity**: 7 principles create a choice space large enough that
   random guessing yields ~14.3% accuracy, but small enough that an LLM judge can
   reliably distinguish between them.
4. **Practical relevance**: These principles map directly to real persuasion contexts
   (sales, negotiation, health interventions), making the benchmark ecologically valid.

### The 7 Principles

| Principle | Mechanism | Example in Context |
|-----------|-----------|-------------------|
| Reciprocity | Obligation to return favors | "I helped you with X, now consider Y" |
| Commitment & Consistency | Desire to act consistently with prior statements | "You said you value health, so this aligns" |
| Social Proof | Following peer behavior | "Most people in your situation choose X" |
| Authority | Deference to expertise | "The research from Harvard shows X" |
| Liking | Persuasion through rapport/similarity | "I totally get where you're coming from" |
| Scarcity | Valuing what is rare/urgent | "This opportunity closes Friday" |
| Unity | Shared identity and belonging | "As fellow parents, we know X matters" |

### Why Not Other Frameworks?

- **Elaboration Likelihood Model (Petty & Cacioppo)**: Describes processing *routes*
  (central vs peripheral) rather than discrete strategies. Not suitable for
  classification-based evaluation.
- **Fogg Behavior Model**: Focuses on motivation/ability/triggers rather than
  influence *tactics*. Less granular for our purposes.
- **Custom taxonomy**: Would lack external validity and make results harder to
  interpret or compare.

## Topic Domains

We define 8 life domains where persuasion preferences plausibly vary within a single
person. The selection criteria:

1. **Psychological distinctiveness**: A person who trusts expert authority for medical
   decisions may trust peer opinions for lifestyle choices. The domains must be
   psychologically separable enough that cross-domain preference variation is realistic.
2. **Practical relevance**: Each domain represents a context where LLM-based
   persuasion agents are plausibly deployed (financial advisors, health coaches,
   career mentors).
3. **Breadth**: 8 domains provide sufficient coverage to test cross-topic interference
   without making the preference matrix sparse.

| Domain | Justification |
|--------|---------------|
| Personal Finance | High-stakes, authority-sensitive domain |
| Health & Fitness | Mixes expert trust with peer influence |
| Career | Long-term commitment decisions |
| Taxes & Legal | Compliance-oriented, authority-heavy |
| Technology | Trend-driven, social-proof-sensitive |
| Social Relationships | Rapport-dependent, liking-heavy |
| Education | Self-improvement, varied influence paths |
| Lifestyle | Low-stakes, unity/liking-driven |

### Why 8 Domains?

- Fewer than 5 would not generate sufficient cross-topic interference in assembled
  conversations, reducing the benchmark's ability to test selective recall.
- More than 10 would thin the cue distribution per topic (with 20 users and 3 cues
  per topic, each domain already has ~60 cues total; more domains would reduce this).
- 8 is the same order of magnitude as Cialdini's 7 principles, creating a roughly
  square preference matrix (7 principles x 8 topics) that avoids sparse assignments.

## Temporal Drift Model

### Phase Structure

Each user-topic pair has either 1 or 2 phases:

- **Phase 1 (Initial)**: The user's starting preferences, established through
  initial cue dialogues. Every user-topic pair has a Phase 1.
- **Phase 2 (Post-Drift)**: Updated preferences after a preference shift. Only
  present for drifting topics. The Phase 2 effective principle must differ from
  Phase 1 effective.

We use exactly 2 phases (not 3+) because:
1. A single drift is the minimal structure needed to test whether the model can
   detect *any* change. Multiple drifts would increase complexity without testing
   a fundamentally different capability.
2. With 400-600 turn conversations and 8 topics, adding more phases would compress
   the cue-to-trigger distance within each phase, weakening the memory test.
3. Two phases create a clean experimental contrast: Phase 1 accuracy measures
   learning, Phase 2 accuracy measures adaptation.

### Drift Types

We model two psychologically distinct mechanisms of preference change:

**Event-based drift**: A single, identifiable life experience causes an abrupt
preference shift. Modeled as one natural utterance revealing the experience.

- *Justification*: Real preference changes often follow discrete events (job loss,
  health scare, financial loss). The model receives a clear signal that something
  changed.
- *Example*: "I lost a lot following what everyone was doing with meme stocks."
  (signals shift away from social_proof in finance)

**Accumulation-based drift**: Gradual erosion through repeated failures of a
principle. Modeled as 3 negative cue dialogues showing increasing resistance.

- *Justification*: Some preference changes are gradual (growing skepticism of
  authority after repeated bad advice). The model must aggregate weak signals
  rather than detect a single event.
- *Why 3 erosion cues*: Fewer than 3 provides insufficient signal for gradual
  erosion. More than 3 would make the erosion phase disproportionately long
  relative to the learning phases.

### Why Not Continuous Drift?

We model drift as a discrete phase transition rather than continuous change because:
1. Discrete phases allow clean evaluation (the correct answer is unambiguous at
   any point in the conversation).
2. Continuous drift would require a continuous scoring function, which is harder
   to validate and interpret.
3. The 2-phase model is already novel relative to existing benchmarks; continuous
   drift is a natural future extension.

## User Population Design

### Composition: 8 Stable + 12 Drifting

| Property | Count | Justification |
|----------|-------|---------------|
| Stable users | 8 (40%) | Control group. Tests that the model does not hallucinate drift where none exists. Without stable users, we cannot distinguish "good at detecting drift" from "always assumes drift." |
| Drifting users | 12 (60%) | Treatment group. Provides sufficient statistical power for drift detection metrics. With 1-4 drifting topics each, yields ~24-48 Phase 2 triggers total. |

### Why 60/40 Split (Not 50/50)?

A 50/50 split would provide equal power for both conditions, but drift detection
is the *novel* contribution. Weighting toward drifting users maximizes statistical
power for the novel metric while retaining enough stable users to serve as controls.
The 60/40 ratio is not critical; the key constraint is that stable users exist at all.

### Drifting Topics Per User: 1-4 out of 8

- **Minimum 1**: Every drifting user must have at least one drifting topic to
  justify their classification.
- **Maximum 4**: At most half of topics drift. If >4 topics drift, the "drifting"
  label becomes the default and the model could exploit a prior that "everything
  changes." Limiting to 4 ensures most topics per drifting user are still stable,
  requiring the model to selectively detect which topics shifted.
- **Uniform random in [1, 4]**: We do not fix the count because variability prevents
  the model from learning a fixed schema ("drifting users always have exactly 2
  drifting topics").

### Drift Type Assignment

Each drifting user is assigned exactly one drift type (event or accumulation) that
applies to all their drifting topics. This is a simplification: in reality, a person
might experience event-based drift on one topic and accumulation on another. We use
a single type per user because:

1. The user's backstory motivates the drift type (e.g., "recently divorced" suggests
   event-based; "growing disillusionment with experts" suggests accumulation). A
   single type per user keeps the backstory coherent.
2. The benchmark already tests both types across the population. Per-user type mixing
   would increase dataset complexity without testing a fundamentally different
   model capability.
