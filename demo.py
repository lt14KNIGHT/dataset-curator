"""
demo.py — Interactive Gradio UI for the Dataset Curator OpenEnv environment.

Two modes in one app:
  🧑 Manual mode  — YOU pick actions (click buttons). Great for live demos.
  🤖 Agent  mode  — The LLM agent runs autonomously. Watch it work in real time.

Run:
    # In Colab or locally:
    pip install gradio requests openai
    python demo.py

    # In Colab — paste this instead of !python demo.py:
    import subprocess; subprocess.Popen(["python","demo.py"])
    # Then click the public URL that appears in the output.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from typing import Optional

import gradio as gr
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL   = os.getenv("ENV_URL",      "http://localhost:7860")
API_BASE  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL     = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN  = os.getenv("HF_TOKEN",     "")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7861"))  # different port from env server

# ── Boot the FastAPI env server if not already running ────────────────────────
_server_proc: Optional[subprocess.Popen] = None

def _ensure_server_running() -> str:
    global _server_proc
    try:
        r = requests.post(f"{ENV_URL}/reset", json={}, timeout=3)
        if r.status_code == 200:
            return "✅ Environment server already running."
    except Exception:
        pass

    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "server.app"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    for _ in range(15):
        time.sleep(1)
        try:
            r = requests.post(f"{ENV_URL}/reset", json={}, timeout=2)
            if r.status_code == 200:
                return "✅ Environment server started."
        except Exception:
            pass
    return "❌ Could not start environment server. Check that port 7860 is free."


# ── Environment helpers ───────────────────────────────────────────────────────
def _unwrap(data: dict) -> dict:
    """
    OpenEnv create_app wraps responses as:
      {"observation": {...obs fields...}, "done": bool, "reward": float}
    We flatten it so callers can do obs.get("progress") directly.
    Falls back gracefully if the response is already flat.
    """
    if "observation" in data and isinstance(data["observation"], dict):
        flat = dict(data["observation"])
        if "done"   in data: flat["done"]   = data["done"]
        if "reward" in data: flat["reward"] = data["reward"]
        return flat
    return data

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=10)
    r.raise_for_status()
    return _unwrap(r.json())

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=10)
    r.raise_for_status()
    return _unwrap(r.json())


# ── LLM agent helpers ─────────────────────────────────────────────────────────
_SYSTEM = {
    "html_strip": (
        "You clean HTML from text. Given a record, respond with ONLY a JSON object:\n"
        '{"action_type":"edit_record","record_id":"<id>","text":"<clean text>"}\n'
        "Remove ALL tags. Decode entities. Preserve text. No markdown fences."
    ),
    "pii_redact": (
        "You redact PII. Replace emails, phones, addresses with [REDACTED].\n"
        "If no PII: {\"action_type\":\"keep_record\",\"record_id\":\"<id>\"}\n"
        "Else: {\"action_type\":\"edit_record\",\"record_id\":\"<id>\",\"text\":\"<redacted>\"}\n"
        "No markdown fences."
    ),
    "quality_audit": (
        "You audit instruction-response pairs.\n"
        "keep → {\"action_type\":\"keep_record\",\"record_id\":\"<id>\"}\n"
        "fix  → {\"action_type\":\"edit_record\",\"record_id\":\"<id>\",\"text\":\"<fixed response>\"}\n"
        "reject → {\"action_type\":\"reject_record\",\"record_id\":\"<id>\"}\n"
        "Reject if AI-refusal boilerplate. Fix if hallucination. Keep if correct."
    ),
}

def llm_decide(record: dict, task: str, token: str) -> dict:
    import json
    if not token:
        return {"action_type": "reject_record", "record_id": record.get("id", "?"), "metadata": {}}
    client = OpenAI(base_url=API_BASE, api_key=token)
    if task == "quality_audit":
        content = (
            f"Record ID: {record['id']}\n"
            f"INSTRUCTION: {record.get('instruction','')}\n"
            f"RESPONSE: {record.get('response','')}"
        )
    else:
        content = f"Record ID: {record['id']}\nTEXT:\n{record.get('text','')}"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM[task]},
                {"role": "user",   "content": content},
            ],
            temperature=0.1, max_tokens=400,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except Exception as e:
        return {"action_type": "reject_record", "record_id": record.get("id","?"), "metadata": {}, "_error": str(e)}


# ── Formatting helpers ────────────────────────────────────────────────────────
def _record_html(record: dict, task: str) -> str:
    if not record:
        return "<div style='color:#64748b;padding:20px;text-align:center'>No record loaded.</div>"
    rid   = record.get("id", "?")
    diff  = record.get("difficulty", "")
    diff_color = {"easy": "#86efac", "medium": "#fde68a", "hard": "#f87171"}.get(diff, "#94a3b8")

    if task == "quality_audit":
        inst = record.get("instruction", "")
        resp = record.get("response", "")
        body = f"""
          <div style='margin-bottom:8px'>
            <span style='color:#94a3b8;font-size:11px;font-weight:600'>INSTRUCTION</span>
            <div style='background:#1e2433;border-radius:6px;padding:10px;margin-top:4px;font-size:13px;color:#e2e8f0'>{inst}</div>
          </div>
          <div>
            <span style='color:#94a3b8;font-size:11px;font-weight:600'>RESPONSE</span>
            <div style='background:#1e2433;border-radius:6px;padding:10px;margin-top:4px;font-size:13px;color:#e2e8f0;white-space:pre-wrap'>{resp}</div>
          </div>
        """
    else:
        text = record.get("text", "")
        body = f"""
          <div>
            <span style='color:#94a3b8;font-size:11px;font-weight:600'>RAW TEXT</span>
            <div style='background:#1e2433;border-radius:6px;padding:10px;margin-top:4px;font-size:12px;color:#e2e8f0;white-space:pre-wrap;word-break:break-all'>{text}</div>
          </div>
        """

    return f"""
    <div style='background:#161d2e;border-radius:10px;padding:14px;border:1px solid #2d3748'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
        <span style='font-weight:700;color:#7dd3fc;font-size:13px'>Record: {rid}</span>
        <span style='background:#0f1a10;color:{diff_color};font-size:10px;padding:2px 10px;border-radius:20px;font-weight:600'>{diff.upper()}</span>
      </div>
      {body}
    </div>
    """

def _score_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#86efac" if pct >= 70 else "#fde68a" if pct >= 40 else "#f87171"
    return f"""
    <div style='background:#161d2e;border-radius:8px;padding:10px;margin-top:8px'>
      <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
        <span style='font-size:11px;color:#94a3b8;font-weight:600'>EPISODE SCORE</span>
        <span style='font-size:13px;font-weight:700;color:{color}'>{pct}%</span>
      </div>
      <div style='background:#0d1117;border-radius:4px;height:10px;overflow:hidden'>
        <div style='width:{pct}%;height:100%;background:{color};border-radius:4px;transition:width .4s'></div>
      </div>
    </div>
    """

def _log_entry(step: int, action_type: str, rid: str, reward: float, done: bool) -> str:
    icon  = {"edit_record":"✏️","keep_record":"✅","reject_record":"🗑️","submit":"🏁"}.get(action_type,"❓")
    color = "#86efac" if reward >= 0.7 else "#fde68a" if reward >= 0.3 else "#f87171"
    done_badge = " <span style='color:#7dd3fc;font-size:9px'>DONE</span>" if done else ""
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:6px 8px;border-radius:6px;background:#1e2433;margin-bottom:4px'>"
        f"<span style='font-size:12px'>{icon} <b style='color:#e2e8f0'>Step {step}</b>"
        f" <span style='color:#64748b;font-size:11px'>{action_type} {rid}</span>{done_badge}</span>"
        f"<span style='font-weight:700;color:{color};font-size:13px'>{reward:+.2f}</span>"
        f"</div>"
    )


# ── Episode state (lives in a simple dict so Gradio state can hold it) ────────
def fresh_state():
    return {
        "episode_id": None,
        "task": "html_strip",
        "obs": None,
        "step": 0,
        "rewards": [],
        "log_html": "",
        "done": False,
    }


# ── Gradio callbacks ──────────────────────────────────────────────────────────
def do_reset(task, state):
    obs = env_reset(task)
    state = fresh_state()
    state["task"]       = task
    state["episode_id"] = obs.get("episode_id", "")
    state["obs"]        = obs
    state["done"]       = obs.get("done", False)

    record = obs.get("current_record") or {}
    prog   = obs.get("progress", 0)

    status = f"🎬 Episode started · task={task} · {prog} records in buffer"
    return (
        state,
        _record_html(record, task),
        _score_bar(0),
        f"<div style='color:#7dd3fc;font-size:12px;padding:6px'>{status}</div>",
        "",                            # error_display — clear any previous error
        gr.update(interactive=True),   # edit btn
        gr.update(interactive=True),   # keep btn
        gr.update(interactive=True),   # reject btn
    )


def _apply_action(state, action_dict):
    """Send action, update state, return (state, record_html, score_bar, log_html, error_msg)."""
    action_dict["episode_id"] = state["episode_id"]
    obs   = env_step(action_dict)
    reward = obs.get("reward") or 0.0
    done   = obs.get("done", False)
    error  = obs.get("last_error") or ""

    state["step"]    += 1
    state["obs"]      = obs
    state["done"]     = done
    state["rewards"].append(reward)

    atype = action_dict.get("action_type", "?")
    rid   = action_dict.get("record_id", "")
    state["log_html"] += _log_entry(state["step"], atype, rid, reward, done)

    record = obs.get("current_record") or {}
    prog   = obs.get("progress", 0)
    total  = obs.get("metadata", {}).get("total_records", 1)
    total_r = obs.get("metadata", {}).get("total_reward", 0)
    max_possible = total * 1.0 + 0.10
    score  = min(max(total_r / max_possible, 0), 1)

    status_msg = (
        f"✅ Done! Final score: {score:.0%}" if done
        else f"Step {state['step']} · reward={reward:+.2f} · {prog} records left"
    )
    status_html = (
        f"<div style='color:{'#86efac' if done else '#7dd3fc'};font-size:12px;padding:6px'>"
        f"{status_msg}</div>"
    )
    err_html = f"<div style='color:#f87171;font-size:11px;padding:4px'>{error}</div>" if error else ""

    btns_active = gr.update(interactive=not done)
    return (
        state,
        _record_html(record, task=state["task"]),
        _score_bar(score),
        status_html + state["log_html"],
        err_html,
        btns_active, btns_active, btns_active,
    )


def do_edit(edited_text, state):
    if state["done"] or not state["episode_id"]:
        return (state,) + ("",) * 6
    record = (state["obs"] or {}).get("current_record") or {}
    rid    = record.get("id", "")
    text   = edited_text.strip() or record.get("text", "")
    return _apply_action(state, {"action_type": "edit_record", "record_id": rid, "text": text, "metadata": {}})


def do_keep(state):
    if state["done"] or not state["episode_id"]:
        return (state,) + ("",) * 6
    record = (state["obs"] or {}).get("current_record") or {}
    rid    = record.get("id", "")
    return _apply_action(state, {"action_type": "keep_record", "record_id": rid, "metadata": {}})


def do_reject(state):
    if state["done"] or not state["episode_id"]:
        return (state,) + ("",) * 6
    record = (state["obs"] or {}).get("current_record") or {}
    rid    = record.get("id", "")
    return _apply_action(state, {"action_type": "reject_record", "record_id": rid, "metadata": {}})


def do_submit(state):
    if state["done"] or not state["episode_id"]:
        return (state,) + ("",) * 6
    return _apply_action(state, {"action_type": "submit", "record_id": "", "metadata": {}})


def do_agent_run(task, hf_token, state, progress=gr.Progress()):
    """Run the full LLM agent autonomously, yielding updates for each step."""
    obs   = env_reset(task)
    state = fresh_state()
    state["task"]       = task
    state["episode_id"] = obs.get("episode_id","")
    state["obs"]        = obs

    record_html = _record_html(obs.get("current_record") or {}, task)
    score_bar   = _score_bar(0)
    log_html    = ""
    status      = f"<div style='color:#7dd3fc;font-size:12px;padding:6px'>🤖 Agent running on {task}…</div>"
    yield state, record_html, score_bar, status + log_html, "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

    token  = hf_token.strip() or HF_TOKEN
    total  = obs.get("metadata",{}).get("total_records",1)

    for step_n in range(1, 25):
        if obs.get("done"):
            break
        record = obs.get("current_record")
        if not record:
            break

        progress(step_n / (total + 1), desc=f"Step {step_n}")

        action = llm_decide(record, task, token)
        action["episode_id"] = state["episode_id"]
        action.setdefault("metadata", {})

        obs    = env_step(action)
        reward = obs.get("reward") or 0.0
        done   = obs.get("done", False)
        state["step"]    = step_n
        state["obs"]     = obs
        state["done"]    = done
        state["rewards"].append(reward)

        atype  = action.get("action_type","?")
        rid    = action.get("record_id","")
        log_html += _log_entry(step_n, atype, rid, reward, done)
        state["log_html"] = log_html

        total_r = obs.get("metadata",{}).get("total_reward",0)
        max_p   = total * 1.0 + 0.10
        score   = min(max(total_r / max_p, 0), 1)

        record_html = _record_html(obs.get("current_record") or {}, task)
        score_bar   = _score_bar(score)
        status_txt  = (
            f"✅ Agent finished! Score: {score:.0%}" if done
            else f"🤖 Step {step_n} · {obs.get('progress',0)} records left · reward={reward:+.2f}"
        )
        status = f"<div style='color:{'#86efac' if done else '#7dd3fc'};font-size:12px;padding:6px'>{status_txt}</div>"

        yield state, record_html, score_bar, status + log_html, "", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

        if done:
            break
        time.sleep(0.4)   # small pause so the UI updates are visible


# ── Build UI ──────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0d1117 !important; font-family: 'Segoe UI', system-ui, sans-serif; }
.gr-button { font-weight: 600 !important; border-radius: 8px !important; }
footer { display: none !important; }
"""

def build_demo():
    with gr.Blocks(css=CSS, title="Dataset Curator — OpenEnv Demo") as demo:

        state = gr.State(fresh_state())

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div style='background:#161d2e;border-radius:12px;padding:18px 20px;margin-bottom:4px;border:1px solid #2d3748'>
          <div style='display:flex;align-items:center;gap:12px'>
            <span style='font-size:28px'>🧹</span>
            <div>
              <h1 style='color:#7dd3fc;font-size:20px;margin:0'>Dataset Curator</h1>
              <p style='color:#64748b;font-size:12px;margin:0'>OpenEnv RL environment — clean, redact, and audit fine-tuning datasets</p>
            </div>
          </div>
        </div>
        """)

        with gr.Row():
            # ── Left panel ─────────────────────────────────────────────────
            with gr.Column(scale=3):
                task_dd = gr.Dropdown(
                    choices=["html_strip", "pii_redact", "quality_audit"],
                    value="html_strip",
                    label="Task",
                    info="html_strip=easy · pii_redact=medium · quality_audit=hard",
                )
                reset_btn = gr.Button("▶ New Episode", variant="primary", size="sm")

                record_display = gr.HTML(
                    "<div style='color:#64748b;padding:30px;text-align:center;border:1px dashed #2d3748;border-radius:10px'>Click <b>New Episode</b> to start.</div>"
                )

                gr.HTML("<div style='color:#94a3b8;font-size:11px;font-weight:600;margin:8px 0 4px'>YOUR EDITED TEXT (for edit_record)</div>")
                edit_box = gr.Textbox(
                    lines=4,
                    placeholder="Paste your cleaned / fixed version here, then click ✏️ Submit Edit…",
                    show_label=False,
                )

                with gr.Row():
                    edit_btn   = gr.Button("✏️ Submit Edit",   variant="secondary", interactive=False)
                    keep_btn   = gr.Button("✅ Keep Record",   variant="secondary", interactive=False)
                    reject_btn = gr.Button("🗑️ Reject Record", variant="secondary", interactive=False)

                submit_btn = gr.Button("🏁 Submit Episode", size="sm")

            # ── Right panel ────────────────────────────────────────────────
            with gr.Column(scale=2):
                score_display  = gr.HTML(_score_bar(0))
                status_display = gr.HTML(
                    "<div style='color:#64748b;font-size:12px;padding:6px'>Waiting to start…</div>"
                )

                gr.HTML("<hr style='border-color:#1e2a3a;margin:8px 0'>")
                gr.HTML("<div style='color:#94a3b8;font-size:11px;font-weight:600;margin-bottom:6px'>🤖 AUTO-AGENT (LLM plays for you)</div>")
                hf_token_box = gr.Textbox(
                    placeholder="Paste your HF_TOKEN here for auto-agent…",
                    label="HF Token", type="password", lines=1,
                )
                agent_btn = gr.Button("🚀 Run LLM Agent", variant="primary")

                gr.HTML("<hr style='border-color:#1e2a3a;margin:8px 0'>")
                gr.HTML("<div style='color:#94a3b8;font-size:11px;font-weight:600;margin-bottom:6px'>📋 ACTION LOG</div>")
                log_display = gr.HTML(
                    "<div style='color:#475569;font-size:11px;padding:4px'>No actions yet.</div>"
                )

        error_display = gr.HTML("")

        # ── Task explainer ──────────────────────────────────────────────────
        with gr.Accordion("📖 Task Guide & Scoring", open=False):
            gr.HTML("""
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:10px 0'>
              <div style='background:#1e2433;border-radius:8px;padding:12px;border-left:3px solid #f59e0b'>
                <div style='color:#fde68a;font-weight:700;margin-bottom:6px'>✂️ HTML Strip</div>
                <div style='color:#94a3b8;font-size:11px'>Remove all HTML tags. Decode entities (&amp;amp;→&amp;). Preserve clean text.<br><br>
                <b style='color:#fbbf24'>Scoring:</b><br>
                +40% no tags remain<br>+60% similarity to reference</div>
              </div>
              <div style='background:#1e2433;border-radius:8px;padding:12px;border-left:3px solid #3b82f6'>
                <div style='color:#7dd3fc;font-weight:700;margin-bottom:6px'>🔒 PII Redact</div>
                <div style='color:#94a3b8;font-size:11px'>Replace emails, phones, addresses with [REDACTED]. Leave everything else intact. Use keep_record if no PII.<br><br>
                <b style='color:#60a5fa'>Scoring:</b><br>
                +60% PII recall<br>+40% non-PII preserved</div>
              </div>
              <div style='background:#1e2433;border-radius:8px;padding:12px;border-left:3px solid #a855f7'>
                <div style='color:#d8b4fe;font-weight:700;margin-bottom:6px'>🔍 Quality Audit</div>
                <div style='color:#94a3b8;font-size:11px'>keep → correct answer, no fluff<br>fix → hallucination or AI bloat<br>reject → AI refusal or gibberish<br><br>
                <b style='color:#c084fc'>Scoring:</b><br>
                Perfect match → 1.00<br>edit on fix → 0.40 + quality</div>
              </div>
            </div>
            """)

        # ── Wire up events ──────────────────────────────────────────────────
        OUTPUTS = [state, record_display, score_display, status_display, error_display, edit_btn, keep_btn, reject_btn]

        reset_btn.click(do_reset,   inputs=[task_dd, state], outputs=OUTPUTS)

        edit_btn.click(do_edit,     inputs=[edit_box, state], outputs=OUTPUTS)
        keep_btn.click(do_keep,     inputs=[state],           outputs=OUTPUTS)
        reject_btn.click(do_reject, inputs=[state],           outputs=OUTPUTS)
        submit_btn.click(do_submit, inputs=[state],           outputs=OUTPUTS)

        agent_btn.click(
            do_agent_run,
            inputs=[task_dd, hf_token_box, state],
            outputs=OUTPUTS,
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting environment server…")
    msg = _ensure_server_running()
    print(msg)

    demo = build_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=True,          # ← produces a public gradio.live URL (works in Colab)
        show_error=True,
    )
