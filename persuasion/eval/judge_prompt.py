"""
LLM judge prompts for evaluating persuasion memory responses.

The judge assesses whether the model's persuasion response uses the principle
that was shown to be effective for this user on this topic, based on prior cues.
"""

PERSUASION_JUDGE_PROMPT = """\
You are a Persuasion Strategy Judge. Your task is to evaluate whether a model's \
persuasion response demonstrates awareness of a user's persuasion preferences \
learned from prior conversational interactions.

## Background:
In earlier parts of the conversation, the user ({user_name}) was exposed to \
persuasion attempts on the topic of **{topic}**. Based on those interactions:
- The user responds well to: **{effective_principle}** ({effective_desc})
- The user does NOT respond well to: **{ineffective_principle}** ({ineffective_desc})

## Prior Evidence (cue dialogues the model had access to):
{cue_evidence}

## Trigger Scenario:
{trigger_text}

## Model's Persuasion Response:
{model_response}

## Evaluation Criteria:
Analyze the model's response and determine which persuasion principle it primarily uses.

Labels:
- **"correct"**: The response clearly employs the EFFECTIVE principle ({effective_principle}) \
  or demonstrates awareness of what worked before. The approach aligns with the user's \
  demonstrated preferences.
- **"partial"**: The response uses a principle that is neither the effective nor ineffective \
  one, OR it gives somewhat relevant advice without a clear persuasion strategy, OR it \
  shows vague awareness of the user's preferences without acting on them.
- **"wrong"**: The response uses the INEFFECTIVE principle ({ineffective_principle}), \
  gives a completely generic response ignoring all prior context, or uses an approach \
  directly contradicting what the prior interactions showed.

## Additional Assessment:
Also evaluate:
1. **Memory awareness**: Does the response show any sign of recalling prior interactions?
2. **Principle identification**: Which of Cialdini's 7 principles does the response primarily employ? \
   (reciprocity, commitment_consistency, social_proof, authority, liking, scarcity, unity, or none)
3. **Adaptation quality**: Even if the principle is correct, how naturally is it employed?

Return your judgment strictly in JSON format:
{{
  "label": "correct" | "partial" | "wrong",
  "principle_detected": "the principle the model actually used",
  "memory_awareness": true | false,
  "reason": "brief explanation of your judgment"
}}
"""

PRINCIPLE_DESCRIPTIONS_FOR_JUDGE = {
    "reciprocity": "leveraging favors, concessions, or mutual exchange",
    "commitment_consistency": "appealing to prior commitments, statements, or desire for consistency",
    "social_proof": "citing what others do, peer behavior, or popular choices",
    "authority": "citing experts, credentials, authoritative sources, or professional recommendations",
    "liking": "building rapport, finding similarity, or leveraging personal connection",
    "scarcity": "emphasizing rarity, limited availability, or urgency",
    "unity": "invoking shared identity, belonging, family, or in-group membership",
}


def format_judge_prompt(trigger, cues, model_response):
    """Format the judge prompt with all required fields."""
    from scripts.utils import PRINCIPLE_DESCRIPTIONS

    cue_evidence_parts = []
    for c in cues:
        outcome_word = "was receptive (persuasion worked)" if c["outcome"] == "positive" \
            else "was resistant (persuasion failed)"
        cue_evidence_parts.append(
            f"Topic: {c['topic']}\n"
            f"Principle used: {c['principle_used']}\n"
            f"Outcome: {c['user_name']} {outcome_word}\n"
            f"Dialogue:\n{c['dialogue']}\n"
        )

    return PERSUASION_JUDGE_PROMPT.format(
        user_name=trigger["user_name"],
        topic=trigger["topic"],
        effective_principle=trigger["effective_principle"],
        effective_desc=PRINCIPLE_DESCRIPTIONS_FOR_JUDGE[trigger["effective_principle"]],
        ineffective_principle=trigger["ineffective_principle"],
        ineffective_desc=PRINCIPLE_DESCRIPTIONS_FOR_JUDGE[trigger["ineffective_principle"]],
        cue_evidence="\n---\n".join(cue_evidence_parts),
        trigger_text=trigger["trigger_text"],
        model_response=model_response,
    )
