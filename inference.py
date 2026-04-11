"""
inference.py — Baseline agent for the Dataset Curator OpenEnv environment.

Mandatory technical rules
--------------------------
* Uses OpenAI client for all LLM calls (not HuggingFace client).
* Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
* Emits EXACTLY:
    [START] task=<name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Usage
-----
    # Start the environment server first (docker or local):
    #   docker run -p 7860:7860 dataset-curator
    #   — or —
    #   python -m server.app

    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    export ENV_URL="http://localhost:7860"          # where the server is running
    export TASK="html_strip"                        # html_strip | pii_redact | quality_audit

    python inference.py
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (all via env vars)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
TASK: str = os.getenv("TASK", "all")
BENCHMARK: str = "dataset-curator"
MAX_STEPS: int = 30
SUCCESS_THRESHOLD: float = 0.60   # episode score ≥ 0.60 → success

# ---------------------------------------------------------------------------
# OpenAI client (mandatory — must use OpenAI client, not HuggingFace SDK)
# ---------------------------------------------------------------------------
llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "NONE")

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_str = error.replace("\n", " ") if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rwd_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rwd_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def _unwrap(data: Dict[str, Any]) -> Dict[str, Any]:
    """OpenEnv wraps responses as {"observation":{...},"done":bool,"reward":float}.
    Flatten so callers can do obs.get("progress") directly."""
    if "observation" in data and isinstance(data["observation"], dict):
        flat = dict(data["observation"])
        if "done"   in data: flat["done"]   = data["done"]
        if "reward" in data: flat["reward"] = data["reward"]
        return flat
    return data


def env_reset(task: str) -> Dict[str, Any]:
    """Call POST /reset and return the unwrapped observation dict."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task},
        timeout=30,
    )
    resp.raise_for_status()
    return _unwrap(resp.json())


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Call POST /step and return the unwrapped observation dict.

    OpenEnv /step body format: {"action": { ...fields... }}
    """
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return _unwrap(resp.json())


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------
_SYSTEM_PROMPTS = {
    "html_strip": textwrap.dedent("""
        You are a data-cleaning agent. Your task is to strip HTML from web-scraped records.

        AVAILABLE ACTIONS — respond with a single JSON object, no other text:
          {"action_type": "edit_record",   "record_id": "<id>", "text": "<cleaned text>"}
          {"action_type": "reject_record", "record_id": "<id>"}
          {"action_type": "keep_record",   "record_id": "<id>"}
          {"action_type": "submit"}

        RULES:
        - Remove ALL HTML tags (<p>, <div>, <script>, <style>, <!--comments-->, etc.).
        - Decode HTML entities (&amp; → &, &lt; → <, &nbsp; → space, &copy; → ©, etc.).
        - Preserve meaningful text content and structure (use \\n for block boundaries).
        - Only reject if the record is entirely non-recoverable garbage.
        - Call submit when the buffer is empty or you are done.

        Respond with ONLY valid JSON. No markdown fences. No explanation.
    """).strip(),

    "pii_redact": textwrap.dedent("""
        You are a privacy-compliance agent. Your task is to redact Personally Identifiable
        Information (PII) from records before they enter a training dataset.

        AVAILABLE ACTIONS — respond with a single JSON object, no other text:
          {"action_type": "edit_record",   "record_id": "<id>", "text": "<redacted text>"}
          {"action_type": "keep_record",   "record_id": "<id>"}
          {"action_type": "reject_record", "record_id": "<id>"}
          {"action_type": "submit"}

        PII TO REDACT (replace each occurrence with [REDACTED]):
        - Email addresses   (e.g. user@example.com)
        - Phone numbers     (any format: +1-800-555-0147, (773) 555-0293, etc.)
        - Physical addresses (street, city, postcode — not generic company names)

        RULES:
        - If a record has NO PII, call keep_record (do not alter clean records).
        - Do NOT over-redact: preserve names, dates, product IDs, and other non-PII.
        - Reject only if the record is entirely PII with no salvageable content.
        - Call submit when done.

        Respond with ONLY valid JSON. No markdown fences. No explanation.
    """).strip(),

    "quality_audit": textwrap.dedent("""
        You are a dataset quality auditor reviewing instruction-response pairs for a
        fine-tuning dataset. Your task is to classify each record as keep, fix, or reject.

        AVAILABLE ACTIONS — respond with a single JSON object, no other text:
          {"action_type": "keep_record",   "record_id": "<id>"}
          {"action_type": "edit_record",   "record_id": "<id>", "text": "<fixed response only>"}
          {"action_type": "reject_record", "record_id": "<id>"}
          {"action_type": "submit"}

        CLASSIFICATION GUIDE:
          keep   — Response is factually correct, appropriate length, no boilerplate.
          fix    — Response has the right idea but contains: hallucinated facts, AI-refusal
                   preamble ("As an AI, I cannot..."), unnecessary caveats, or excessive padding.
                   Provide ONLY the corrected response text (not the instruction).
          reject — Response is entirely an AI refusal, completely off-topic, or so badly
                   wrong that fixing it would require rewriting from scratch.

        If you choose edit_record, put the CORRECTED RESPONSE TEXT in the "text" field.

        Respond with ONLY valid JSON. No markdown fences. No explanation.
    """).strip(),
}


# ---------------------------------------------------------------------------
# LLM decision function
# ---------------------------------------------------------------------------

def _format_record(record: Dict[str, Any], task: str) -> str:
    """Format a record for the LLM prompt."""
    rid = record.get("id", "?")
    if task == "quality_audit":
        instruction = record.get("instruction", "")
        response = record.get("response", "")
        return (
            f"Record ID: {rid}\n"
            f"INSTRUCTION: {instruction}\n"
            f"RESPONSE: {response}"
        )
    else:
        text = record.get("text", "")
        return f"Record ID: {rid}\nTEXT:\n{text}"


def get_agent_action(
    obs: Dict[str, Any],
    task: str,
    history: List[str],
) -> Dict[str, Any]:
    """Ask the LLM what to do next and parse its JSON response."""
    current = obs.get("current_record")
    progress = obs.get("progress", 0)

    if current is None or progress == 0:
        return {"action_type": "submit", "episode_id": obs.get("episode_id",""), "metadata": {}}

    record_str = _format_record(current, task)
    history_block = "\n".join(history[-6:]) if history else "None yet."

    user_msg = textwrap.dedent(f"""
        Task: {task}
        Records remaining in buffer: {progress}

        CURRENT RECORD TO PROCESS:
        {record_str}

        Recent action history:
        {history_block}

        Choose your next action and respond with a single JSON object.
    """).strip()

    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,      # low temperature for deterministic grading tasks
            max_tokens=512,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip accidental markdown fences
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        action = json.loads(raw)

        # Inject episode_id and metadata (required by the env spec)
        action["episode_id"] = obs.get("episode_id", "")
        action.setdefault("metadata", {})
        return action

    except Exception as exc:
        # Fallback: skip the current record
        return {
            "action_type": "reject_record",
            "record_id": (current or {}).get("id", ""),
            "episode_id": obs.get("episode_id", ""),
            "metadata": {},
        }


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False
    history: List[str] = []

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ---- Reset -------------------------------------------------------
        obs = env_reset(task)
        total_records = obs.get("metadata", {}).get("total_records", 1)

        # ---- Step loop ---------------------------------------------------
        for step in range(1, MAX_STEPS + 1):
            done: bool = obs.get("done", False)
            if done:
                break

            # Get LLM decision
            action = get_agent_action(obs, task, history)

            # Execute action
            obs = env_step(action)

            reward: float = obs.get("reward") or 0.0
            done = obs.get("done", False)
            error: Optional[str] = obs.get("last_error")

            rewards.append(reward)
            steps_taken = step

            # Build a compact action string for logging
            action_str = json.dumps(
                {k: v for k, v in action.items() if k != "episode_id"},
                separators=(",", ":"),
            )

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action.get('action_type')} "
                f"record={action.get('record_id', '-')} → reward={reward:.2f}"
            )

            if done:
                break

        # ---- Compute final score -----------------------------------------
        # Normalise by maximum possible reward (1.0 per record + 0.10 completion bonus)
        max_possible = total_records * 1.0 + 0.10
        raw_score = sum(rewards)
        score = min(max(raw_score / max_possible, 0.0), 1.0)
        # Clamp to strictly within (0, 1) for validator
        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run all three tasks sequentially (or just the one specified by $TASK)
    tasks_to_run = (
        ["html_strip", "pii_redact", "quality_audit"]
        if TASK == "all"
        else [TASK]
    )

    for t in tasks_to_run:
        run_task(t)
