# Persuasion Memory Benchmark — Plan

## Core Idea

People respond to different persuasion principles depending on the **topic**. A model acting as a persuader should learn from past interactions which principle works for a specific user on a specific topic, and apply that knowledge in future attempts.

This tests a form of **adaptive cognitive memory**: the model must retain not just facts, but *what worked and what didn't* across prior persuasion attempts, and generalize to new but similar situations.

## Cialdini's 7 Principles

| # | Principle | Description |
|---|-----------|-------------|
| 1 | Reciprocity | People feel obligated to return favors |
| 2 | Commitment & Consistency | People want to act consistently with prior commitments |
| 3 | Social Proof | People follow what others are doing |
| 4 | Authority | People defer to credible experts |
| 5 | Liking | People are persuaded by those they like or relate to |
| 6 | Scarcity | People value what is rare or limited |
| 7 | Unity | People are persuaded by shared identity/belonging |

## Topic Domains

We define a set of topic domains where persuasion applies differently per person:

- **Personal Finance** (investing, saving, budgeting)
- **Health & Fitness** (diet, exercise, medical decisions)
- **Career** (job changes, skill development, negotiation)
- **Taxes & Legal** (tax filing, compliance, legal advice)
- **Technology Adoption** (new tools, software, devices)
- **Social/Relationships** (conflict resolution, parenting, community)
- **Education** (courses, certifications, learning methods)
- **Lifestyle** (travel, hobbies, purchases)

## User Profiles (Persuadee Profiles)

Each persuadee has a **hidden preference matrix** mapping topics to their most/least effective persuasion principles. This is the ground truth the model must learn from cues.

Example profile:
```json
{
  "user_id": "user_01",
  "name": "Jordan",
  "preference_map": {
    "personal_finance": {"effective": "social_proof", "ineffective": "authority"},
    "health_fitness": {"effective": "authority", "ineffective": "scarcity"},
    "career": {"effective": "commitment_consistency", "ineffective": "liking"},
    "taxes_legal": {"effective": "authority", "ineffective": "social_proof"},
    "technology": {"effective": "scarcity", "ineffective": "unity"},
    "social_relationships": {"effective": "liking", "ineffective": "reciprocity"},
    "education": {"effective": "social_proof", "ineffective": "scarcity"},
    "lifestyle": {"effective": "unity", "ineffective": "authority"}
  }
}
```

---

## Pipeline (6 Steps)

### Step 1: Generate User Profiles

Generate 20 diverse persuadee profiles, each with a unique topic→principle preference map. Ensure:
- No two profiles are identical
- Each principle appears as "effective" for at least some topics across the pool
- Profiles have a short backstory (occupation, personality sketch) that makes preferences coherent

**Script:** `scripts/01_generate_profiles.py`

### Step 2: Generate Cue Dialogues (Persuasion Attempts)

For each profile, generate **cue dialogues** — short exchanges between a Persuader (P) and the Persuadee (U). Each cue shows a persuasion attempt on a specific topic using a specific principle. The outcome (success/failure) is **implicit** — shown through the persuadee's reaction, not stated explicitly.

**Types of cues:**
- **Positive cue**: Persuader uses the effective principle → persuadee is convinced / receptive
- **Negative cue**: Persuader uses an ineffective principle → persuadee resists / deflects / is unmoved

Each profile gets multiple cues across different topics, with **deliberate repetition** of similar topic areas so the model has multiple data points to learn from.

Target: ~8-12 cues per profile (mix of positive and negative), ~200 cue dialogues total.

**Script:** `scripts/02_generate_cues.py`

### Step 3: Generate Trigger Queries (Test Persuasion Scenarios)

For each profile, generate trigger scenarios where the persuader faces a **new persuasion situation** on a topic the model has seen cues for. The trigger:
- Presents a new situation in the same or very similar topic domain
- Does NOT mention what worked before
- Does NOT name any persuasion principle
- Requires the model to recall past cue outcomes and choose the right approach

**Semantic disconnect**: The trigger uses different surface language than the cues. E.g., a cue about "investing in index funds" and a trigger about "starting a retirement account" (same domain: personal finance, but different framing).

**Evaluation criteria**: Does the model's persuasion response use the principle that was shown to be effective for this user on this topic? Does it avoid the principle shown to be ineffective?

Target: 3-5 triggers per profile, ~80 trigger queries total.

**Script:** `scripts/03_generate_triggers.py`

### Step 4: Semantic Filtering

Filter out trigger-cue pairs where surface similarity is too high:
- BM25 scoring between trigger text and cue dialogues
- Sentence-transformer (MPNet) cosine similarity
- Remove pairs above threshold (configurable, default: BM25 > 15 or cosine > 0.65)

**Script:** `scripts/04_semantic_filter.py`

### Step 5: Assemble Long Conversations

Embed cue dialogues and triggers into long conversation trajectories:
- Interleave cues from the same profile across the conversation
- Insert **distractor turns** (unrelated chitchat) between cues and triggers
- Place multiple cue→trigger chains for the same topic at varying distances
- Include **cross-topic interference** (cues from different topics between a topic's cue and its trigger)

Structure per conversation:
```
[Cue: health topic, authority works]      ← turn 5-8
[distractor turns]                        ← turns 9-30
[Cue: finance topic, social proof works]  ← turns 31-35
[distractor turns]                        ← turns 36-60
[Cue: health topic, scarcity fails]       ← turns 61-65   (reinforcing cue)
[distractor turns]                        ← turns 66-100
[Trigger: new health persuasion scenario] ← turn 101      (model should use authority)
[more cues and triggers...]
```

Key design choices:
- **Repeated cues** for the same topic reinforce the pattern (the model gets 2-3 chances to observe what works)
- **Varying gap distances** test short-term vs long-term retention
- **Cross-topic cues** between same-topic cue and trigger create interference

**Script:** `scripts/05_assemble_conversations.py`

### Step 6: Human Evaluation

Evaluate model responses on triggers using a rubric:

| Score | Label | Criteria |
|-------|-------|----------|
| 2 | **Correct** | Uses the effective principle for this user+topic; avoids ineffective ones |
| 1 | **Partial** | Uses a neutral principle (neither the effective nor ineffective one) |
| 0 | **Wrong** | Uses the ineffective principle, or gives generic response ignoring all prior context |

Additional qualitative annotations:
- Did the response show awareness of past interactions?
- Did it adapt its persuasion strategy based on learned user preferences?
- Would a human persuader who observed the same cues choose a similar approach?

---

## Data Format

### Profile JSON
```json
{
  "user_id": "user_01",
  "name": "Jordan",
  "backstory": "A 34-year-old software engineer who values data and peer experiences...",
  "preference_map": { ... }
}
```

### Cue Dialogue JSON
```json
{
  "cue_id": "cue_001",
  "user_id": "user_01",
  "topic": "personal_finance",
  "principle_used": "social_proof",
  "outcome": "positive",
  "dialogue": "P: A lot of people in your income bracket have been moving to index funds...\nU: Really? That many? Maybe I should look into that too.",
  "implicit_signal": "persuadee shows receptivity through interest and willingness to follow"
}
```

### Trigger JSON
```json
{
  "trigger_id": "trig_001",
  "user_id": "user_01",
  "topic": "personal_finance",
  "effective_principle": "social_proof",
  "ineffective_principle": "authority",
  "trigger_scenario": "U: I'm thinking about whether to open a Roth IRA or just keep my savings in a high-yield account. What do you think?",
  "time_gap": "several_weeks",
  "related_cue_ids": ["cue_001", "cue_007"]
}
```

### Assembled Conversation JSON
```json
{
  "conversation_id": "conv_01",
  "user_id": "user_01",
  "turns": [
    {"turn_id": 1, "speaker": "U", "text": "..."},
    {"turn_id": 2, "speaker": "P", "text": "...", "meta": {"type": "cue", "cue_id": "cue_001"}},
    ...
    {"turn_id": 101, "speaker": "U", "text": "...", "meta": {"type": "trigger", "trigger_id": "trig_001"}}
  ],
  "triggers": [
    {
      "trigger_id": "trig_001",
      "turn_id": 101,
      "effective_principle": "social_proof",
      "ineffective_principle": "authority",
      "related_cue_turn_ids": [5, 61]
    }
  ]
}
```

---

## File Structure

```
persuasion/
├── plan.md                          ← this file
├── scripts/
│   ├── 01_generate_profiles.py      ← generate persuadee profiles
│   ├── 02_generate_cues.py          ← generate cue dialogues
│   ├── 03_generate_triggers.py      ← generate trigger scenarios
│   ├── 04_semantic_filter.py        ← BM25 + MPNet filtering
│   ├── 05_assemble_conversations.py ← build long conversations
│   └── utils.py                     ← shared helpers
├── data/
│   ├── profiles/                    ← generated user profiles
│   ├── cues/                        ← cue dialogues
│   ├── triggers/                    ← trigger queries
│   ├── filtered/                    ← post-filter pairs
│   └── conversations/               ← final assembled conversations
├── eval/
│   ├── judge_prompt.py              ← LLM judge prompts
│   ├── evaluate.py                  ← run evaluation
│   └── human_eval_template.json     ← template for human annotators
└── requirements.txt
```

---

## Execution Order

```bash
# 1. Generate profiles
python scripts/01_generate_profiles.py --num-profiles 20 --output data/profiles/

# 2. Generate cue dialogues
python scripts/02_generate_cues.py --profiles data/profiles/ --output data/cues/

# 3. Generate trigger queries
python scripts/03_generate_triggers.py --profiles data/profiles/ --cues data/cues/ --output data/triggers/

# 4. Semantic filtering
python scripts/04_semantic_filter.py --cues data/cues/ --triggers data/triggers/ --output data/filtered/

# 5. Assemble long conversations
python scripts/05_assemble_conversations.py --profiles data/profiles/ --filtered data/filtered/ --output data/conversations/

# 6. Human evaluation (manual, using the template)
# → data/conversations/ contains the final benchmark
```

---

## Key Differences from LoCoMo-Plus

| Aspect | LoCoMo-Plus | Persuasion Benchmark |
|--------|-------------|---------------------|
| Memory type | Implicit constraints (state, goal, value, causal) | Learned persuasion patterns (what works per user per topic) |
| Cue structure | Single cue → single trigger | Multiple cues per topic → test if model learns the pattern |
| Repeated cues | No deliberate repetition | Deliberate repetition to test learning from experience |
| Evaluation | Constraint consistency (binary) | Principle alignment (3-level: correct/partial/wrong) |
| Roles | General conversation | Persuader (model) + Persuadee (user) |
| Ground truth | Valid response space | Known effective principle per user×topic |
| What's tested | Does model remember implicit info? | Does model learn *what works* from observation and apply it? |
