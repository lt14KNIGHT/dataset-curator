"""
server/app.py — Expose the Dataset Curator environment as a FastAPI app.

create_app() wires up the OpenEnv HTTP spec:
  POST /reset  → env.reset()  → DatasetCuratorObservation
  POST /step   → env.step()   → DatasetCuratorObservation
  GET  /state  → env.state    → dict

Additional endpoints required by OpenEnv Phase 2 validator:
  GET /health    → {"status": "healthy"}
  GET /metadata  → {"name": ..., "description": ...}
  GET /schema    → {"action": ..., "observation": ..., "state": ...}
  POST /mcp      → {"jsonrpc": "2.0", ...}
"""

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

from models import DatasetCuratorAction, DatasetCuratorObservation
from server.environment import DatasetCuratorEnv


def _env_factory() -> DatasetCuratorEnv:
    """Called by OpenEnv for every incoming request to get a fresh instance."""
    return DatasetCuratorEnv()


app = create_app(
    _env_factory,
    DatasetCuratorAction,
    DatasetCuratorObservation,
    env_name="dataset-curator",
    max_concurrent_envs=64,
)


# ── Extra endpoints required by OpenEnv Phase 2 validator ────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata():
    return {
        "name": "dataset-curator",
        "description": (
            "An RL environment where agents learn to curate, clean, and audit "
            "fine-tuning datasets across three tasks: HTML stripping, PII redaction, "
            "and quality auditing of instruction-response pairs."
        ),
        "version": "1.0.0",
        "tasks": [
            {"name": "html_strip",     "difficulty": "easy",   "grader": True},
            {"name": "pii_redact",     "difficulty": "medium", "grader": True},
            {"name": "quality_audit",  "difficulty": "hard",   "grader": True},
        ],
    }


@app.get("/schema")
async def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["read_record", "edit_record", "keep_record", "reject_record", "submit"],
                },
                "record_id": {"type": "string"},
                "text":      {"type": "string"},
                "episode_id":{"type": "string"},
                "metadata":  {"type": "object"},
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "episode_id":     {"type": "string"},
                "current_record": {"type": "object"},
                "progress":       {"type": "integer"},
                "last_error":     {"type": "string"},
                "done":           {"type": "boolean"},
                "reward":         {"type": "number"},
                "metadata":       {"type": "object"},
            },
        },
        "state": {
            "type": "object",
            "description": "Episode state is stored server-side keyed by episode_id",
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Minimal JSON-RPC 2.0 endpoint for MCP compatibility."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "dataset-curator",
            "description": "Dataset curation RL environment",
            "tools": ["reset", "step", "state"],
        },
    })


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
