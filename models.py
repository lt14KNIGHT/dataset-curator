"""
models.py — Action and Observation types for the Dataset Curator environment.

Rules (from openenv.core source):
  - Action MUST extend openenv.core.env_server.types.Action
  - Observation MUST extend openenv.core.env_server.types.Observation
  - Do NOT re-declare fields already on the base:
      Observation base: done, reward, metadata
      Action base:      metadata
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation


class DatasetCuratorAction(Action):
    """
    Represents one agent step in the Dataset Curator environment.

    action_type options:
      read_record   – Inspect the current record (free look; no reward change).
      edit_record   – Submit a cleaned / fixed version of the text.
      keep_record   – Mark the record as high quality; leave it unchanged.
      reject_record – Flag the record as unfixable / low quality.
      submit        – End the episode and trigger final grading.
    """

    action_type: str  # read_record | edit_record | keep_record | reject_record | submit
    record_id: Optional[str] = None   # required for read/edit/keep/reject
    text: Optional[str] = None        # required for edit_record
    episode_id: str = ""              # must be echoed back on every /step call


class DatasetCuratorObservation(Observation):
    """
    What the agent sees after each action.

    Inherited from base: done (bool), reward (float|None), metadata (dict)
    """

    episode_id: str
    current_record: Optional[Dict[str, Any]] = None  # None when buffer is empty
    progress: int = 0                                 # records remaining in buffer
    last_error: Optional[str] = None                 # feedback from previous action
