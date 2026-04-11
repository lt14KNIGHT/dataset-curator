"""
server/environment.py — Dataset Curator Environment.

HTTP is stateless: every /reset and /step call creates a FRESH Python instance.
Episode state lives in the module-level _EPISODES dict, keyed by episode_id.
The client must echo the episode_id back in every action payload.
"""

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from models import DatasetCuratorAction, DatasetCuratorObservation
from server.dataset import VALID_TASKS, get_task_records
from server.graders import grade_action

# ---------------------------------------------------------------------------
# Module-level episode store (survives across HTTP requests)
# ---------------------------------------------------------------------------
_EPISODES: Dict[str, Dict[str, Any]] = {}

_MAX_STEPS_PER_EPISODE = 50   # safety ceiling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_record_by_id(state: Dict, record_id: str) -> Optional[Dict]:
    """Find a record anywhere in the full record list by id."""
    for r in state["records"]:
        if r["id"] == record_id:
            return r
    return None


def _remove_from_buffer(state: Dict, record_id: str) -> None:
    """Remove the processed record from the buffer."""
    state["buffer"] = [r for r in state["buffer"] if r["id"] != record_id]


def _build_obs(
    state: Dict,
    episode_id: str,
    reward: float,
    error: Optional[str],
    done: bool,
) -> DatasetCuratorObservation:
    current = state["buffer"][0] if state["buffer"] else None
    return DatasetCuratorObservation(
        episode_id=episode_id,
        current_record=current,
        progress=len(state["buffer"]),
        last_error=error,
        done=done,
        reward=reward,
        metadata={
            "task": state["task"],
            "step": state["step_count"],
            "total_reward": round(state["total_reward"], 4),
            "decisions_made": len(state["decisions"]),
            "total_records": len(state["records"]),
        },
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DatasetCuratorEnv(Environment):
    """
    Real-world OpenEnv environment for training AI agents to curate,
    clean, and audit fine-tuning datasets.

    Three task tracks (set via reset kwargs):
        html_strip     — strip HTML markup from web-scraped strings
        pii_redact     — redact PII (phone, email, address)
        quality_audit  — classify records as keep / fix / reject
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DatasetCuratorObservation:
        task = kwargs.get("task", "html_strip")
        if task not in VALID_TASKS:
            task = "html_strip"

        eid = episode_id or str(uuid.uuid4())
        records = get_task_records(task)

        _EPISODES[eid] = {
            "task": task,
            "records": records,
            "buffer": list(records),   # shrinks as agent processes records
            "decisions": {},           # record_id → {"action": ..., "score": ...}
            "step_count": 0,
            "total_reward": 0.0,
            "done": False,
            "last_error": None,
        }

        state = _EPISODES[eid]
        return _build_obs(state, eid, reward=0.0, error=None, done=False)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: DatasetCuratorAction) -> DatasetCuratorObservation:  # type: ignore[override]
        eid = action.episode_id

        # ---- Validate episode --------------------------------------------
        if eid not in _EPISODES:
            return DatasetCuratorObservation(
                episode_id=eid,
                current_record=None,
                progress=0,
                last_error="Unknown episode_id — call /reset first.",
                done=True,
                reward=0.0,
                metadata={},
            )

        state = _EPISODES[eid]

        if state["done"]:
            return _build_obs(state, eid, reward=0.0, error="Episode already done.", done=True)

        # Safety ceiling
        state["step_count"] += 1
        if state["step_count"] > _MAX_STEPS_PER_EPISODE:
            state["done"] = True
            return _build_obs(state, eid, reward=0.0, error="Max steps exceeded.", done=True)

        action_type: str = action.action_type
        record_id: Optional[str] = action.record_id
        submitted_text: Optional[str] = action.text
        reward = 0.0
        error: Optional[str] = None

        # ---- Dispatch on action type ------------------------------------

        if action_type == "read_record":
            # Free look — no reward, no penalty.
            if record_id is None:
                error = "read_record requires record_id."
            elif _get_record_by_id(state, record_id) is None:
                error = f"Record '{record_id}' not found."
            # reward stays 0.0

        elif action_type in ("edit_record", "keep_record", "reject_record"):
            if record_id is None:
                error = f"{action_type} requires record_id."
            else:
                record = _get_record_by_id(state, record_id)
                if record is None:
                    error = f"Record '{record_id}' not found."
                elif record_id in state["decisions"]:
                    error = f"Record '{record_id}' already processed."
                else:
                    # Map action_type → grader action label
                    grader_action = {
                        "edit_record": "edit",
                        "keep_record": "keep",
                        "reject_record": "reject",
                    }[action_type]

                    if action_type == "edit_record" and not submitted_text:
                        error = "edit_record requires a non-empty 'text' field."
                        reward = 0.0
                    else:
                        reward = grade_action(
                            task=state["task"],
                            record=record,
                            agent_action=grader_action,
                            submitted_text=submitted_text,
                        )
                        state["decisions"][record_id] = {
                            "action": grader_action,
                            "score": reward,
                            "text": submitted_text,
                        }
                        _remove_from_buffer(state, record_id)

        elif action_type == "submit":
            # End the episode.
            completion_ratio = len(state["decisions"]) / max(len(state["records"]), 1)
            reward = round(0.10 * completion_ratio, 4)   # up to 0.10 bonus
            state["done"] = True

        else:
            error = (
                f"Unknown action_type '{action_type}'. "
                "Valid: read_record, edit_record, keep_record, reject_record, submit."
            )

        state["last_error"] = error
        state["total_reward"] += reward

        # Auto-complete when buffer is empty (agent processed everything)
        if not state["buffer"] and action_type != "submit":
            state["done"] = True

        done = state["done"]
        return _build_obs(state, eid, reward=reward, error=error, done=done)

    # ------------------------------------------------------------------
    # state property (required by OpenEnv spec)
    # ------------------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        # Instance has no state — all state lives in _EPISODES.
        return {}
