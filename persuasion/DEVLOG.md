# Persuasion Benchmark — Development Log

## Origin

This benchmark was inspired by the **LoCoMo-Plus** paper (Li et al., 2026), which introduced "beyond-factual cognitive memory" evaluation for LLM agents. LoCoMo-Plus tests whether models can retain and apply **implicit constraints** (causal, state, goal, value) across long conversations — going beyond simple factual recall.

We extend this idea into a new domain: **persuasion strategy adaptation**.

## Core Idea

People respond to different persuasion principles (Cialdini's 7) depending on the **topic**. For example, someone might respond to authority when it comes to taxes but prefer social proof for personal finance decisions.

A model acting as a persuader should **learn from past interactions** which principle works for a specific user on a specific topic, and apply that knowledge in future persuasion attempts.

### What this tests that LoCoMo-Plus doesn't

| Aspect | LoCoMo-Plus | This Benchmark |
|--------|-------------|----------------|
| Memory type | Implicit constraints (state, goal, value, causal) | Learned persuasion patterns (what works per user per topic) |
| Cue structure | Single cue → single trigger | Multiple cues per topic → test if model learns the pattern |
| Repeated cues | No deliberate repetition | Deliberate repetition to test learning from experience |
| Evaluation | Constraint consistency (binary) | Principle alignment (3-level: correct/partial/wrong) |
| Roles | General conversation | Persuader (model) + Persuadee (user) |
| What's tested | Does model remember implicit info? | Does model learn *what works* from observation and apply it? |

## Cialdini's 7 Principles

1. **Reciprocity** — People feel obligated to return favors
2. **Commitment & Consistency** — People want to act consistently with prior commitments
3. **Social Proof** — People follow what others are doing
4. **Authority** — People defer to credible experts
5. **Liking** — People are persuaded by those they like or relate to
6. **Scarcity** — People value what is rare or limited
7. **Unity** — People are persuaded by shared identity/belonging

## Pipeline

### Steps

1. **Generate user profiles** — Each has a hidden preference map: topic → effective/ineffective principle
2. **Generate cue dialogues** — Persuasion attempts with implicit success/failure outcomes (2 positive + 1 negative per topic per user)
3. **Generate trigger scenarios** — New situations in the same topic domain, semantically distant from cues
4. **Semantic filtering** — BM25 + MPNet to remove high-similarity cue-trigger pairs
5. **Assemble long conversations** — Embed cues + triggers into long dialogues with distractors
6. **Evaluation** — LLM judge + human annotation

### Scale

- 20 profiles (10 batches × 2 profiles)
- ~480 cue dialogues (20 profiles × 8 topics × 3 cues)
- ~480 trigger scenarios (20 profiles × 8 topics × 3 triggers)
- ~490 `claude -p` calls total

## Technical Decisions

### Using `claude -p` instead of OpenAI API

**Decision:** Use Claude CLI (`claude -p`) for all generation instead of the OpenAI SDK.

**Reasoning:**
- No API key management or billing setup needed
- Uses existing Claude Code subscription
- Cost: $0 additional (covered by Claude Code plan)
- Trade-off: Slower per call (~10-20s each since each spins up a full session) vs API (~2-3s)

**Implementation:** `utils.py:call_llm()` writes the prompt to a temp file and pipes it to `claude -p --model <model>`. The `CLAUDECODE` env var is unset to avoid nested session errors.

### Using Haiku for generation

**Decision:** Default to `--model haiku` for all generation.

**Reasoning:**
- Much faster than Sonnet/Opus via CLI
- Quality is sufficient for generating dialogue snippets and profiles (verified in trial run)
- Can upgrade to `--model sonnet` for higher quality if needed

### Batched generation with resume support

**Decision:** 10 batches of 2 profiles each, with per-batch save and `--resume-from-batch` flag.

**Reasoning:**
- Full run takes ~1-2 hours
- Each batch saves independently to `data/batches/batch_XX/`
- If interrupted: `python scripts/run_pipeline.py --resume-from-batch 4`
- Merge happens at the end

## Trial Run Results

Ran `trial_run.py` with Haiku, generating 2 profiles + cues + triggers for 2 topics. Quality was good:

**Profiles generated:**
- Diane: 54yo school librarian, community-rooted, trusts people she knows personally
- Raj: 29yo startup engineer, data-driven, respects credentials

**Quality observations:**
- Cue dialogues are natural and outcomes are implicit (not stated)
- Positive cue example: Diane convinced via commitment/consistency — *"It would be a little hypocritical of me to keep putting it off"*
- Negative cue example: Diane resists scarcity — *"The more someone tells me I need to hurry, the more I think I should slow down"*
- Triggers are semantically distant from cues (different specific situations within same topic)
- Profile preferences are psychologically coherent with backstories

## Running the Pipeline

```bash
# From the persuasion/ directory:
cd /home/emon/personal/long-context-benchmark/persuasion

# Trial run (8 calls, ~2 min)
python scripts/trial_run.py

# Full pipeline (490 calls, ~1-2 hours)
python scripts/run_pipeline.py

# Resume if interrupted
python scripts/run_pipeline.py --resume-from-batch 4

# Use sonnet for higher quality
python scripts/run_pipeline.py --model sonnet

# After generation: filter and assemble
python scripts/04_semantic_filter.py
python scripts/05_assemble_conversations.py
```

## File Structure

```
persuasion/
├── DEVLOG.md                        ← this file
├── plan.md                          ← full methodology
├── requirements.txt                 ← pip deps (for filter step only)
├── scripts/
│   ├── utils.py                     ← shared helpers, claude -p wrapper
│   ├── trial_run.py                 ← small trial (2 profiles, 2 topics)
│   ├── run_pipeline.py              ← full batched pipeline
│   ├── 01_generate_profiles.py      ← standalone profile generation
│   ├── 02_generate_cues.py          ← standalone cue generation
│   ├── 03_generate_triggers.py      ← standalone trigger generation
│   ├── 04_semantic_filter.py        ← BM25 + MPNet filtering
│   └── 05_assemble_conversations.py ← build long conversations
├── eval/
│   ├── judge_prompt.py              ← LLM judge prompts
│   └── evaluate.py                  ← run evaluation + human eval template
└── data/
    ├── trial/                       ← trial run output
    ├── batches/                     ← per-batch intermediate output
    ├── profiles/                    ← merged profiles
    ├── cues/                        ← merged cues
    ├── triggers/                    ← merged triggers
    ├── filtered/                    ← post-filter output
    └── conversations/               ← final assembled conversations
```

## Methodology Comparison: LoCoMo-Plus vs Persuasion Benchmark

This section documents exactly how LoCoMo-Plus builds their Level-2 cognitive memory dataset (Section 4 of the paper), exactly how we build ours, and the reasoning behind each decision.

---

### Step 1: Generating the Raw Content

#### LoCoMo-Plus: Implicit Cue Dialogue Generation

**What they do:** Use GPT-4o to generate 3-5 turn dialogue segments that embed implicit constraints into natural conversation. Each cue belongs to one of 4 constraint types:
- **Causal**: If X happened, then Y follows (e.g., "My knee's been acting up since the marathon" → later the model should know not to suggest running)
- **State**: Something changed about the person (e.g., "I finally went vegetarian" → later the model should remember this when suggesting restaurants)
- **Goal**: The person expressed an aspiration (e.g., "I'm saving up for a house" → later the model should factor this into financial advice)
- **Value**: A deep preference or belief (e.g., "I just don't think people should meddle in others' business" → later the model should respect this boundary)

**How they do it:** They prompt GPT-4o with the constraint type, the specific information to encode, and instructions to make it implicit. The model generates a short dialogue where the information comes out naturally in conversation rather than being stated as a fact. For example, instead of "I am allergic to nuts", a character might say "Last time I had pad thai at that place I had to use my EpiPen — they must've used peanut oil."

**Why they do it this way:** The entire point is to test whether models can extract and retain **implicit** information. If the cues stated things explicitly ("I prefer authority-based arguments"), any model could trivially retrieve them with keyword matching. By making information emerge through natural conversational behavior, they force models to actually *understand* what was said and remember the *implication*, not just the words.

#### Our Adaptation: Profile Generation + Cue Dialogue Generation

**What we do:** We split this into two separate steps.

**Step 1a — Profile Generation** (`01_generate_profiles.py`):
We first generate 20 user profiles, each with a hidden preference map: for each of 8 topic domains, which of Cialdini's 7 principles is most effective and which is least effective for this person. The profile also includes a name, age, occupation, and personality backstory.

*How it works:* We prompt Claude Haiku with all 7 principles, all 8 topics, and ask it to generate profiles in batches of 2 (via `run_pipeline.py`). The prompt requires that preferences feel psychologically coherent with the backstory — e.g., a data-driven engineer might respond to authority (credentials, evidence) but resist unity (group belonging appeals). We pass `diversity_hint` containing names already generated so later batches don't duplicate. Each profile is validated: all 8 topics must be present, effective ≠ ineffective, all principle names must be valid.

*Why we do this separately:* LoCoMo-Plus doesn't need explicit profiles because their constraints are one-off (a single cue encodes a single piece of information). We need profiles because our benchmark tests whether a model can learn a **pattern** — the same person responds consistently to the same principle across multiple situations within a topic. The profile defines the ground truth that the model must learn from observation.

**Step 1b — Cue Dialogue Generation** (`02_generate_cues.py`):
For each profile, for each of the 8 topics, we generate **3 cue dialogues**: 2 positive cues (effective principle → persuadee is receptive) and 1 negative cue (ineffective principle → persuadee resists).

*How it works:* We use separate prompts for positive and negative cues. Each prompt includes the user's name, backstory, the specific topic with description, the principle to use, and the desired outcome. The prompt explicitly instructs:
- The principle must be woven naturally — never named
- The outcome must be shown implicitly through the persuadee's reaction, not stated
- Each dialogue must be a different specific scenario within the topic (e.g., for personal_finance: one about investing, one about saving, one about debt)
- Dialogues are 3-5 turns alternating between P (Persuader) and U (Persuadee)

For a positive cue, the persuadee might say things like *"That makes sense, I'll try it"* or *"It would be a little hypocritical of me to keep putting it off."* For a negative cue: *"The more someone tells me I need to hurry, the more I think I should slow down."*

*Why 3 cues per topic (2 positive, 1 negative):* LoCoMo-Plus uses single cues — one cue, one trigger. We deliberately use multiple cues because we're testing a different cognitive capability: **learning from repeated experience**. A single cue tests recall. Multiple cues test whether the model can observe a pattern (this approach works, this one doesn't) and generalize. The 2:1 positive:negative ratio gives the model two examples of what works and one of what doesn't, which is a realistic learning signal — in real advisory relationships, you'd see multiple successful interactions with occasional failures.

---

### Step 2: Quality Verification

#### LoCoMo-Plus: Memory-Worthy Verification (Human QA)

**What they do:** After generating cues, human annotators review each one to verify:
- The implicit information is genuinely implicit (not too obvious)
- The information is memory-worthy (a human reader would need to remember it to handle a future situation correctly)
- The information isn't too obscure (a careful reader should be able to pick it up)

**Why they do it:** LLM-generated cues can fail in subtle ways. A cue might be technically implicit but so vague that no reader would extract the constraint. Or it might encode the information so directly that it's effectively explicit. This human QA step ensures the benchmark items are of consistent quality. Without it, you might evaluate a model's performance on items that aren't actually testing what you think they're testing.

#### Our Adaptation: Deferred to End + Trial Run Validation

**What we do:** We do NOT have an intermediate human verification step in the pipeline. Instead, we rely on:
1. **Trial run quality check** — Before running the full pipeline, we ran `trial_run.py` with 2 profiles and 2 topics and manually inspected the outputs. We verified that cue dialogues were natural, outcomes were implicit, and profiles were psychologically coherent.
2. **Prompt constraints** — Our prompts explicitly instruct the LLM to make outcomes implicit and never name principles. This pushes quality control into the generation step.
3. **End-stage human evaluation** — When we evaluate model responses, human annotators also assess whether the cues themselves were well-formed.

**Why we differ:** Pragmatism. LoCoMo-Plus is building a published benchmark dataset intended for broad reuse — they need rigorous intermediate QA. We're iterating on a new benchmark design — adding a blocking human review step after every generation batch would slow iteration from hours to days. The trade-off: our dataset may contain some cues that are too explicit or too vague. A production version should add this step back.

---

### Step 3: Constructing Test Queries

#### LoCoMo-Plus: Cue-Trigger Query Construction

**What they do:** For each verified cue, they use GPT-4o to generate a trigger query — a question or situation that tests whether the model retained the implicit information from the cue. The key design principle is **semantic disconnect**: the trigger must test the same underlying knowledge but use completely different surface language.

*Example:* If the cue was about a character mentioning their knee hurts after a marathon (implying they have a knee injury), the trigger might be "What outdoor activities would you recommend for the weekend?" — the model should know not to suggest running, but the trigger never mentions knees, marathons, or injuries.

**Why semantic disconnect matters:** Without it, models can "cheat" by using lexical retrieval. If the cue says "I'm saving for a house" and the trigger asks "Are you still saving for a house?", a simple keyword search finds the answer. Semantic disconnect forces the model to actually understand and retain the *meaning* of the cue, not just match surface words. This is the central methodological contribution of LoCoMo-Plus.

#### Our Adaptation: Trigger Scenario Generation (`03_generate_triggers.py`)

**What we do:** For each profile, for each topic, we generate 3 trigger scenarios — new situations where the persuadee brings up a problem or decision in the same topic domain.

*How it works:* The trigger prompt includes:
- The user's name and backstory
- The topic domain
- **Summaries of all related cues** — not the full dialogues, but a condensed version showing the approach used and whether the persuadee was receptive or resistant. This is given to the LLM so it can deliberately generate triggers that use different surface language.
- Explicit constraints: "Do NOT reuse nouns, verbs, or phrases from the cues", "Do NOT reference the prior conversations", "The trigger should stand alone as a natural conversational turn"

Each trigger is a single utterance from the persuadee — e.g., *"My knees have been aching something awful lately, and my neighbor swears by this yoga class at the rec center, but I've never done anything like that before and I'm not sure it's really for someone like me."* The model being tested must respond to this with a persuasion attempt, and the evaluation checks whether it uses the right principle.

The trigger also includes a `time_gap` field (one week / several weeks / a few months / several months) and `related_cue_ids` linking back to the cues it relates to.

*Why 3 triggers per topic:* LoCoMo-Plus generates 1 trigger per cue. We generate 3 per topic because we want to test robustness — can the model apply the learned pattern across multiple different surface-level scenarios? This also lets us analyze variance: if a model gets 2/3 triggers right for a topic, that's different from 0/3 or 3/3.

*Why we pass cue summaries to the trigger prompt:* This is a deliberate technique from LoCoMo-Plus. By telling the generator what the cues contained, it can actively avoid overlapping language. Without this, the LLM might independently generate scenarios that happen to use similar words to the cues, defeating the semantic disconnect requirement. We summarize rather than pass full text to keep prompts focused.

---

### Step 4: Ensuring Semantic Disconnect

#### LoCoMo-Plus: Semantic Filtering (BM25 + MPNet)

**What they do:** Even with careful prompting, some generated trigger-cue pairs end up with high surface similarity. They apply a dual automated filter:
1. **BM25 (lexical overlap)** — A classical information retrieval scoring method that measures word-level overlap. Higher scores mean more shared vocabulary. They remove pairs above a threshold.
2. **MPNet (neural semantic similarity)** — Using `all-mpnet-base-v2` sentence embeddings, they compute cosine similarity between the trigger text and the cue dialogue. Higher scores mean the sentences express similar meaning, even if they use different words. They remove pairs above a threshold.

Pairs that exceed EITHER threshold are removed. This is intentionally conservative: even if the words are different (passes BM25) but the meaning is too similar (fails MPNet), the pair is dropped.

**Why both methods:** BM25 catches lexical shortcuts (shared keywords, phrases, named entities). MPNet catches semantic shortcuts (paraphrases, synonymous phrasing). Using only BM25 would miss pairs like "saving for a house" / "putting money aside for a home". Using only MPNet would miss pairs where a single shared keyword (like a specific name or place) makes retrieval trivial even though overall semantic similarity is low.

#### Our Adaptation: Identical Approach (`04_semantic_filter.py`)

**What we do:** We use the exact same dual-filter strategy with the same tools:
- BM25 via `rank-bm25` library (BM25Okapi variant)
- MPNet via `sentence-transformers` library (`all-mpnet-base-v2` model)
- Default thresholds: BM25 > 15 OR cosine > 0.65 → remove

*How it works:* For each trigger, we find its related cues via `related_cue_ids`. We compute BM25 scores by tokenizing trigger text and cue dialogue text (lowercased, whitespace-split). We compute MPNet cosine similarity by encoding both texts and taking the dot product of L2-normalized embeddings. We take the max score across all related cues for each trigger. Triggers exceeding either threshold are moved to `triggers_removed.json` for inspection; the rest go to `triggers_filtered.json`.

*Why we copied this exactly:* This is the most directly transferable part of the LoCoMo-Plus methodology. The filtering step is domain-agnostic — it doesn't matter whether you're filtering constraints about food preferences or persuasion strategy cues. The thresholds may need tuning for our domain (persuasion dialogues might have systematically different similarity distributions than general conversation), but the approach itself is sound. We kept the same defaults as a starting point and can adjust after inspecting what gets filtered.

---

### Step 5: Validating Trigger Difficulty

#### LoCoMo-Plus: Cue Memory Elicitation Validation (Human QA)

**What they do:** A second round of human annotation. For each trigger, annotators verify:
- The trigger actually REQUIRES recalling the cue to answer correctly (it's not answerable from common sense alone)
- The trigger isn't so hard that even a careful human reader who DID remember the cue would struggle
- The "correct" answer space is clear and unambiguous

*Example of what fails this check:* If the cue establishes that someone is vegetarian, and the trigger asks "What food should we bring to the party?", a model might suggest salad as the safe option even without remembering the vegetarian detail — just from common sense about party food. That trigger doesn't actually test memory.

**Why this step exists:** Semantic filtering (step 4) only checks surface similarity. This step checks functional validity — does this trigger actually test what we think it tests? A trigger can be semantically distant from the cue (passes step 4) but still answerable without memory (fails step 5).

#### Our Adaptation: LLM Judge + End-Stage Human Eval

**What we do:** We don't have a dedicated pre-evaluation human validation step. Instead:

1. **Prompt-level mitigation:** Our trigger prompt instructs that triggers should be "underspecified — multiple persuasion approaches could seem reasonable without knowing the user's preferences." This is our prompt-level way of ensuring triggers aren't trivially answerable. If only one persuasion approach makes sense for a scenario regardless of user preferences, the trigger doesn't test memory.

2. **LLM Judge** (`eval/judge_prompt.py` + `eval/evaluate.py`): When we evaluate model responses, the judge prompt checks for `memory_awareness` — whether the response shows any sign of recalling prior interactions. This retroactively tells us which triggers actually required memory.

3. **Human evaluation template:** `evaluate.py --human-eval-only` generates a JSON template where human annotators see the trigger, the related cues, and the model's response side by side. They annotate: correct/partial/wrong, which principle was detected, whether memory was shown, and free-text notes. This is where issues with trigger quality surface.

**Why we differ:** Same reasoning as step 2. We're prioritizing iteration speed over intermediate quality gates. The risk: some of our triggers might be answerable without memory (a savvy model might always default to "authority" for tax-related questions regardless of user preference). We can detect this in analysis — if a trigger has high accuracy across all models, it's probably not testing memory.

---

### Step 6: Building the Full Conversations

#### LoCoMo-Plus: Insertion into LoCoMo Dialogues

**What they do:** They take the original LoCoMo dataset (long multi-session conversations between two people) and surgically insert the verified cue-trigger pairs at controlled positions. The existing conversation provides naturalistic distractor content between cues and triggers.

*How they structure it:* A cue is placed at some early point in the conversation. Many turns of unrelated conversation follow (these already exist in the LoCoMo data). Then the trigger is placed later. The distance between cue and trigger varies to test short-term vs long-term retention. The existing conversation provides realistic interference — the model must hold onto the cue information while processing many unrelated exchanges.

**Why they use existing conversations:** LoCoMo already has high-quality, naturalistic long dialogues (they were originally generated for factual memory testing). Reusing them ensures the distractor content is natural and diverse. It also provides a direct comparison with LoCoMo's Level-1 factual memory tasks — same conversations, different evaluation targets.

#### Our Adaptation: Full Conversation Assembly (`05_assemble_conversations.py`)

**What we do:** We generate entire conversations from scratch, one per user profile.

*How it works:*

1. **Layout planning:** For each profile, we take all their cues (grouped by topic) and triggers. We plan a conversation layout that interleaves topics using round-robin ordering. Within each round, we cycle through all active topics and place one cue per topic.

2. **Cue placement:** Cues are placed in round-robin order across topics. This means cues for topic A are interleaved with cues for topics B, C, D... creating cross-topic interference. Between every cue, we insert a small distractor gap (2-5 exchanges).

3. **Trigger placement:** After all cues are placed, triggers are appended with larger distractor gaps proportional to the `time_gap` field:
   - "one week": 3-6 distractor exchanges
   - "several weeks": 6-12 distractor exchanges
   - "a few months": 12-20 distractor exchanges
   - "several months": 20-30 distractor exchanges

4. **Distractor generation:** We prompt Claude Haiku to generate casual chitchat exchanges (weather, weekend plans, food, movies) that explicitly avoid the topic domains used in cues. These are generated in bulk for efficiency, then distributed across the conversation.

5. **Turn assembly:** Each segment (cue dialogue, distractor exchange, trigger utterance) is parsed into individual turns with `turn_id`, `speaker` (P or U), `text`, and `meta` (marking each turn as cue/trigger/distractor with associated IDs).

*Why we generate from scratch instead of inserting into existing conversations:*
- **Role structure:** Our benchmark requires a specific P (Persuader) and U (Persuadee) role structure. LoCoMo conversations don't have this.
- **Multiple cue chains:** We need multiple cue→trigger chains per topic per profile (3 cues + 3 triggers per topic × 8 topics = potentially 48+ task-relevant segments per conversation). This is much denser than LoCoMo-Plus's insertion approach.
- **Cross-topic interference:** We specifically design the cue ordering to create maximum interference — cues for different topics are interleaved so that when the model encounters a trigger for topic A, it has seen cues for topics B, C, D in between.
- **Trade-off:** Our distractor content may be less naturalistic than existing LoCoMo dialogues, since it's LLM-generated chitchat rather than organic conversation. But we gain precise control over conversation structure.

---

### Evaluation Comparison

#### LoCoMo-Plus Evaluation

**What they measure:** Constraint consistency — for each trigger, does the model's response fall within the valid answer space defined by the constraint? This is binary: consistent or inconsistent.

**How they measure it:** An LLM judge (GPT-4o) receives the cue constraint, the trigger query, and the model's response, and determines whether the response is consistent with the constraint. They also compute traditional metrics (F1, exact match, ROUGE-L, BERTScore) for factual questions.

**Key findings:**
- Best models: o1, GPT-4o (65-70% constraint consistency)
- Worst area: Value constraints (models struggle with implicit preference/belief retention)
- Performance degrades with cue-trigger distance
- All models significantly worse on Level-2 (cognitive) than Level-1 (factual)

#### Our Evaluation (`eval/evaluate.py` + `eval/judge_prompt.py`)

**What we measure:** Three things:
1. **Principle alignment** (3-level): Did the model use the effective principle (correct=2), a neutral one (partial=1), or the ineffective one (wrong=0)?
2. **Memory awareness** (binary): Does the response show any sign of recalling prior interactions?
3. **Principle detection**: Which of the 7 principles did the model actually employ?

**How we measure it:** Our LLM judge prompt provides the judge with:
- The user's effective and ineffective principles for this topic
- The full text of related cue dialogues (so the judge can see what the model had access to)
- The trigger scenario and the model's response
- Detailed criteria for each label (correct/partial/wrong)

The judge returns structured JSON: `{label, principle_detected, memory_awareness, reason}`.

We compute aggregate metrics: overall accuracy, per-topic breakdown (which topics are hardest?), per-principle breakdown (which principles are hardest to detect/apply?), and memory awareness rate.

**Why 3-level scoring instead of binary:** In persuasion, there's a meaningful middle ground. If the effective principle is "authority" and the ineffective is "scarcity", a model that uses "social proof" isn't correct but also isn't making the worst choice. Binary scoring would lump "used a reasonable alternative" with "used the exact worst approach." The 3-level scoring captures this gradient. It also gives us richer signal for analysis — a model scoring mostly "partial" is qualitatively different from one scoring mostly "wrong".

---

### Summary: What We Follow, Extend, and Simplify

**Followed faithfully:**
- Cue-trigger semantic disconnect as the core design principle (the defining contribution of LoCoMo-Plus)
- Implicit information encoding — outcomes shown through natural conversational reactions, never stated
- Dual automated filtering with BM25 + MPNet (same tools, same approach, same default thresholds)
- Long conversation context with distractor interference to test retention over distance

**Extended:**
- **Repeated cues** (2-3 per topic vs 1): Tests experiential pattern learning, not just single-fact recall
- **Structured preference space** (7 principles × 8 topics per user): Enables systematic quantitative evaluation and per-dimension analysis
- **Role-based interaction** (Persuader/Persuadee): Creates a task-oriented context where memory must inform action, not just answer a question
- **3-level evaluation** (correct/partial/wrong): Captures the gradient between right, neutral, and wrong responses
- **Per-topic and per-principle metrics**: Enables analysis of which domains and principles are hardest for models

**Simplified:**
- **No intermediate human verification** (LoCoMo-Plus has 2 human QA stages during generation; we defer to end): Trades quality gates for faster iteration. Risk: some items may be trivially answerable or poorly formed.
- **Generated vs existing base conversations** (we build from scratch; they insert into LoCoMo): We gain structural control but potentially lose naturalness in distractor content.

---

### Open Questions / What LoCoMo-Plus Findings Predict for Us

From the paper's key findings, here's what we might expect:

- **Distance degradation**: Model performance dropped with longer cue-trigger gaps. Our `time_gap` field and variable distractor counts should let us replicate this analysis. We predict the same pattern: "several months" gaps should produce lower scores than "one week" gaps.

- **Constraint type difficulty**: Value constraints were hardest for models. Our analog: which persuasion principles are hardest to detect and apply? We hypothesize that subtle principles like "unity" (shared identity) and "commitment_consistency" may be harder than explicit ones like "authority" (credentials) or "scarcity" (urgency).

- **Pattern learning vs single recall**: LoCoMo-Plus only tests single-cue recall. Our repeated cues create a genuinely new question: can models learn from multiple observations? If a model sees two positive cues for authority on taxes and one negative cue for social proof on taxes, can it synthesize that into "use authority, not social proof"? No existing benchmark tests this.

- **Graph-based approaches**: LoCoMo-Plus explicitly notes they didn't test graph methods (e.g., knowledge graphs, structured memory). Our relational structure (user→topic→principle, with multiple edges per relationship) is particularly well-suited for graph-based memory. This could be a novel contribution — comparing flat context models vs graph-augmented models on structured preference learning.

## What's Missing / TODO

- [ ] **Model runner script** — Takes assembled conversations, runs a target model at each trigger point, collects responses
- [ ] **Full pipeline run** — Execute `run_pipeline.py` end to end
- [ ] **Semantic filtering** — Run step 4 after generation completes
- [ ] **Conversation assembly** — Run step 5
- [ ] **Baseline evaluation** — Run at least one model against the benchmark
- [ ] **Human evaluation** — Use the generated template from `evaluate.py --human-eval-only`
- [ ] **Graph-based approach** — LoCoMo-Plus paper didn't test graph methods; could be interesting to try here since relationships between topics and principles are inherently relational
