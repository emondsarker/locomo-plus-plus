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
persuasion attempts on the topic of **{topic}**.

### CURRENT PHASE: {phase}
{phase_context}

## Prior Evidence (cue dialogues the model had access to):
{cue_evidence}

## Trigger Scenario:
{trigger_text}

## Model's Persuasion Response:
{model_response}

## Evaluation Criteria:
Analyze the model's response and determine which persuasion principle it primarily uses, \
and whether it is appropriate for the CURRENT phase.

Labels:
- **"correct"**: The response clearly employs the EFFECTIVE principle for THIS PHASE ({effective_principle}) \
  or demonstrates awareness of the current preferences. The approach matches the user's demonstrated \
  current-phase behavior.
- **"partial"**: The response uses a neutral principle (neither effective nor ineffective for this phase), \
  OR gives somewhat relevant advice without a clear persuasion strategy, OR shows vague awareness.
- **"wrong"**: The response uses the INEFFECTIVE principle for THIS PHASE ({ineffective_principle}), \
  gives a completely generic response ignoring all prior context, or contradicts current-phase evidence.

## Cialdini's 7 Principles (use one of these exact names for principle_detected):
reciprocity, commitment_consistency, social_proof, authority, liking, scarcity, unity

## Additional Assessment:
1. **Memory awareness**: Does the response show awareness of prior interactions?
2. **Phase awareness**: Does it recognize the user's CURRENT preferences (not outdated ones)?
3. **Principle identification**: Which of the 7 principles above does the response primarily employ?
4. **Adaptation quality**: How naturally is the principle employed?

Return your judgment strictly in JSON format:
{{
  "label": "correct" | "partial" | "wrong",
  "principle_detected": "one of the 7 principle names listed above",
  "memory_awareness": true | false,
  "phase_awareness": true | false,
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
    """Format the judge prompt with all required fields, including phase awareness.

    Design decisions (see methodology/04_evaluation_design.md):
    - The judge does NOT know which principle is "stale." It identifies the
      principle used; post-processing in evaluate.py determines staleness.
      This prevents anchoring bias (methodology/05_bias_and_validity.md §T2).
    - For Phase 2 triggers, ALL cues (Phase 1 + drift + Phase 2) are included
      so the judge can see what changed (methodology/05_bias_and_validity.md §T3).
    """
    from scripts.utils import PRINCIPLE_DESCRIPTIONS

    # Determine phase context — no stale hints (T2 fix)
    phase = trigger.get("phase", 1)
    if phase == 1:
        phase_label = "Phase 1 (Initial Preferences)"
        phase_context = (
            "The user's preferences are based on interactions earlier in the conversation."
        )
    elif phase == 2:
        phase_label = "Phase 2 (Updated Preferences)"
        phase_context = (
            "The user's preferences have SHIFTED during the conversation. "
            "Earlier cues reflect old preferences; later cues reflect updated preferences. "
            "Evaluate whether the response matches the user's CURRENT (updated) preferences."
        )
    else:
        phase_label = f"Phase {phase}"
        phase_context = ""

    # Build cue evidence — include ALL phases for Phase 2 triggers (T3 fix)
    if phase == 2:
        # Phase 2: include Phase 1 cues, drift signals, and Phase 2 cues
        # so the judge can see the full preference timeline
        evidence_cues = cues  # all cues for this user-topic
    else:
        # Phase 1: only Phase 1 cues (Phase 2 doesn't exist yet)
        evidence_cues = [c for c in cues if c.get("phase") == 1]
        if not evidence_cues:
            evidence_cues = [c for c in cues if c.get("phase") not in ["drift_event", "erosion"]]

    cue_evidence_parts = []
    for c in evidence_cues:
        cue_phase = c.get("phase", 1)

        if cue_phase == "drift_event":
            cue_evidence_parts.append(
                f"[TRANSITION SIGNAL]\n"
                f"Topic: {c['topic']}\n"
                f"Dialogue:\n{c['dialogue']}\n"
            )
        elif cue_phase == "erosion":
            cue_evidence_parts.append(
                f"[EROSION — principle losing effectiveness]\n"
                f"Topic: {c['topic']}\n"
                f"Dialogue:\n{c['dialogue']}\n"
            )
        else:
            phase_label_for_cue = f"Phase {cue_phase}"
            if phase == 2 and cue_phase == 1:
                phase_label_for_cue = "Phase 1 (earlier preferences)"
            elif phase == 2 and cue_phase == 2:
                phase_label_for_cue = "Phase 2 (current preferences)"

            outcome_word = "was receptive (persuasion worked)" if c.get("outcome") == "positive" \
                else "was resistant (persuasion failed)"
            cue_evidence_parts.append(
                f"[{phase_label_for_cue}]\n"
                f"Topic: {c['topic']}\n"
                f"Principle used: {c.get('principle_used', 'N/A')}\n"
                f"Outcome: {c['user_name']} {outcome_word}\n"
                f"Dialogue:\n{c['dialogue']}\n"
            )

    cue_evidence = "\n---\n".join(cue_evidence_parts) if cue_evidence_parts else "[No cues available]"

    return PERSUASION_JUDGE_PROMPT.format(
        user_name=trigger["user_name"],
        topic=trigger["topic"],
        phase=phase_label,
        phase_context=phase_context,
        effective_principle=trigger["effective_principle"],
        ineffective_principle=trigger["ineffective_principle"],
        cue_evidence=cue_evidence,
        trigger_text=trigger["trigger_text"],
        model_response=model_response,
    )
