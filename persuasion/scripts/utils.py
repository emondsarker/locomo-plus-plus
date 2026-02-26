"""Shared utilities for the persuasion benchmark pipeline."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

PRINCIPLES = [
    "reciprocity",
    "commitment_consistency",
    "social_proof",
    "authority",
    "liking",
    "scarcity",
    "unity",
]

PRINCIPLE_DESCRIPTIONS = {
    "reciprocity": "People feel obligated to return favors or concessions.",
    "commitment_consistency": "People want to act consistently with their prior commitments and statements.",
    "social_proof": "People follow what others like them are doing.",
    "authority": "People defer to credible experts and authoritative sources.",
    "liking": "People are persuaded by those they like, relate to, or find similar to themselves.",
    "scarcity": "People value what is rare, limited, or about to become unavailable.",
    "unity": "People are persuaded by shared identity, belonging, and in-group membership.",
}

TOPICS = [
    "personal_finance",
    "health_fitness",
    "career",
    "taxes_legal",
    "technology",
    "social_relationships",
    "education",
    "lifestyle",
]

TOPIC_DESCRIPTIONS = {
    "personal_finance": "Investing, saving, budgeting, retirement planning, debt management.",
    "health_fitness": "Diet, exercise routines, medical decisions, wellness habits.",
    "career": "Job changes, skill development, salary negotiation, career pivots.",
    "taxes_legal": "Tax filing strategies, legal compliance, financial regulations.",
    "technology": "Adopting new tools, software, devices, platforms.",
    "social_relationships": "Conflict resolution, parenting approaches, community involvement.",
    "education": "Courses, certifications, learning methods, academic decisions.",
    "lifestyle": "Travel, hobbies, major purchases, daily routines.",
}

# Drift configuration
STABLE_USERS_COUNT = 8  # Users with no drift
DRIFTING_USERS_COUNT = 12  # Users with drift (mix of event/accumulation)
MIN_DRIFTING_TOPICS = 1  # Min topics that drift per drifting user
MAX_DRIFTING_TOPICS = 4  # Max topics that drift per drifting user
DRIFT_TYPES = ["event", "accumulation"]  # "event" = 1 life event, "accumulation" = 3 erosion cues


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_all_json_in_dir(directory):
    """Load and merge all JSON files in a directory."""
    all_data = []
    for fpath in sorted(Path(directory).glob("*.json")):
        all_data.extend(load_json(fpath))
    return all_data


def call_llm(prompt, model="haiku", temperature=None, max_tokens=None,
             timeout=300, retries=2):
    """Call Claude CLI via `claude -p`. Returns the text response.

    Args:
        prompt: The prompt text.
        model: Model to use — "haiku", "sonnet", or "opus". Default: haiku.
        temperature/max_tokens: Accepted for interface compat, ignored by CLI.
        timeout: Seconds before timing out a single attempt. Default: 300.
        retries: Number of retry attempts on failure. Default: 2.
    """
    import time as _time

    # Write prompt to a temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # avoid nested session error
        cmd = f"cat {prompt_file} | claude -p --model {model}"

        last_err = None
        for attempt in range(1 + retries):
            if attempt > 0:
                wait = 5 * attempt
                print(f"\n      Retry {attempt}/{retries} in {wait}s...", end="", flush=True)
                _time.sleep(wait)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True,
                    env=env,
                )

                if result.returncode != 0:
                    last_err = RuntimeError(f"claude -p failed (exit {result.returncode}): {result.stderr[:500]}")
                    continue

                output = result.stdout.strip()
                if not output:
                    last_err = RuntimeError(f"claude -p returned empty output. stderr: {result.stderr[:500]}")
                    continue

                return output

            except subprocess.TimeoutExpired:
                last_err = RuntimeError(f"claude -p timed out after {timeout}s")
                continue

        raise last_err

    finally:
        os.unlink(prompt_file)


def parse_json_from_response(text):
    """Extract JSON from an LLM response that may contain markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)
