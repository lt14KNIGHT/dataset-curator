# 🧹 Dataset Curator — OpenEnv RL Environment

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://lt14knight-dataset-curator.hf.space)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Hackathon-orange)](https://huggingface.co/spaces/LT14KNIGHT/dataset-curator)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An RL training environment where AI agents learn to curate, clean, and audit fine-tuning datasets — replacing expensive human annotation pipelines.**

---

## 🎯 The Problem

Fine-tuning LLMs requires thousands of high-quality labeled examples. But raw datasets are dirty:

| Problem | Impact |
|---------|--------|
| HTML tags in web-scraped text | Noisy, unusable training data |
| Personal info (emails, phones, addresses) | Privacy violations / GDPR risk |
| AI refusals and hallucinations in responses | Degrades fine-tuned model quality |

Human annotators fix this manually — it's slow and expensive. **Dataset Curator trains agents to do it automatically.**

---

## 🎮 Live Demo

**HF Space (API):** https://lt14knight-dataset-curator.hf.space

```bash
# Try it right now
curl -X POST https://lt14knight-dataset-curator.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "html_strip"}'
```

---

## 📋 Tasks

| # | Task | Difficulty | Description |
|---|------|-----------|-------------|
| 1 | `html_strip` | 🟢 Easy | Remove HTML tags, decode entities, recover clean text |
| 2 | `pii_redact` | 🟡 Medium | Replace emails/phones/addresses with `[REDACTED]` |
| 3 | `quality_audit` | 🔴 Hard | Classify instruction-response pairs: keep / fix / reject |

---

## ⚡ Action Space

| Action | Fields | Effect |
|--------|--------|--------|
| `read_record` | `record_id` | Inspect record — free look, no reward |
| `edit_record` | `record_id`, `text` | Submit cleaned/fixed version |
| `keep_record` | `record_id` | Mark as high quality |
| `reject_record` | `record_id` | Flag as unfixable |
| `submit` | — | End episode, collect completion bonus |

---

## 👁️ Observation Space

```json
{
  "episode_id": "f819379f-e9ac-458c-979f-...",
  "current_record": {
    "id": "hs_001",
    "task": "html_strip",
    "difficulty": "easy",
    "text": "<p>The <b>quick</b> brown fox <a href='...'>jumps</a></p>"
  },
  "progress": 4,
  "last_error": null,
  "done": false,
  "reward": 0.97,
  "metadata": {
    "task": "html_strip",
    "step": 1,
    "total_reward": 0.97,
    "decisions_made": 1,
    "total_records": 5
  }
}
```

---

## 🏆 Reward Design — Partial Credit at Every Step

### html_strip
```
reward = 0.40 × (no HTML tags remain)
       + 0.60 × (similarity to reference clean text)
```

### pii_redact
```
reward = 0.60 × (fraction of PII items removed)
       + 0.40 × (precision — non-PII content preserved)
```

### quality_audit
| Agent decision | Ground truth | Score |
|---------------|-------------|-------|
| keep | keep | **1.00** |
| reject | reject | **1.00** |
| edit | fix | **0.40 + 0.60 × fix quality** |
| reject | fix | **0.25** — recognised it was bad |
| edit | reject | **0.10** |
| keep | reject | **0.00** |

> Partial credit at every step means the agent always gets a learning signal — not just binary win/lose.

---

## 🚀 Quick Start

### Run locally with Docker

```bash
# 1. Clone and build
git clone https://github.com/lt14KNIGHT/dataset-curator.git
cd dataset-curator
docker build -t dataset-curator .
docker run -p 7860:7860 dataset-curator

# 2. Run baseline agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"
export TASK="html_strip"   # or pii_redact | quality_audit | all

python inference.py
```

### Expected output
```
[START] task=html_strip env=dataset-curator model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"edit_record","record_id":"hs_001"} reward=0.97 done=false error=null
[STEP] step=2 action={"action_type":"edit_record","record_id":"hs_002"} reward=0.84 done=false error=null
[STEP] step=3 action={"action_type":"edit_record","record_id":"hs_003"} reward=0.91 done=false error=null
[STEP] step=4 action={"action_type":"edit_record","record_id":"hs_004"} reward=0.88 done=false error=null
[STEP] step=5 action={"action_type":"edit_record","record_id":"hs_005"} reward=0.58 done=true error=null
[END] success=true steps=5 score=0.85 rewards=0.97,0.84,0.91,0.88,0.58
```

### Run interactive Gradio demo (in Colab)

```python
import os
os.environ["HF_TOKEN"] = "hf_..."
os.environ["ENV_URL"]  = "http://localhost:7860"

!cd dataset-curator && python demo.py
# Opens a public gradio.live URL
```

---

## 🏗️ Architecture

```
dataset-curator/
├── inference.py        ← Baseline agent (mandatory)
├── models.py           ← Action + Observation types
├── demo.py             ← Gradio interactive UI
├── Dockerfile          ← python:3.11-slim, port 7860
├── openenv.yaml        ← OpenEnv spec
└── server/
    ├── app.py          ← FastAPI via create_app()
    ├── environment.py  ← reset() / step() / state
    ├── dataset.py      ← 15 embedded records (5 per task)
    └── graders.py      ← Deterministic graders 0.0–1.0
```

### Key design decisions

**Stateless HTTP** — Each request creates a fresh `Environment` instance. Episode state lives in a module-level `_EPISODES` dict keyed by `episode_id`. The client echoes `episode_id` back on every `/step` call.

**Deterministic grading** — All graders use reference-based scoring (no LLM judges, no subjective similarity). Scores are reproducible across runs.

**Partial credit** — Multi-level rewards mean the agent learns from every step, not just episode completion.

**No external APIs** — All 15 records are embedded in `dataset.py`. The environment runs fully offline.

---

## 📊 Baseline Results

| Task | Agent | Score | Steps |
|------|-------|-------|-------|
| html_strip | Qwen2.5-72B | ~0.85 | 5 |
| pii_redact | Qwen2.5-72B | ~0.78 | 5 |
| quality_audit | Qwen2.5-72B | ~0.72 | 5 |

---

## 🔧 Environment Spec

- **Runtime:** FastAPI + Uvicorn on port 7860
- **Concurrency:** 64 simultaneous episodes
- **State:** Module-level dict, stateless HTTP
- **Records:** 15 total, fully embedded
- **Baseline runtime:** < 5 minutes on vCPU=2, 8GB RAM
- **Framework:** OpenEnv (`openenv-core>=0.2.0`)
- **Agent LLM:** Qwen/Qwen2.5-72B-Instruct via HuggingFace Router

---

## 👤 Who Would Use This

- **ML teams** replacing human annotation with trained cleanup agents
- **RL researchers** studying agents on real-world NLP tasks  
- **Dataset maintainers** auditing instruction-tuning corpora like OpenHermes, FLAN, Alpaca

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
