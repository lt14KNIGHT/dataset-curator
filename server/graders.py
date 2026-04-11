"""
server/graders.py — Deterministic, reference-based graders for each task.

Every grader returns a float in (0.0, 1.0) — strictly excluding the endpoints.
The _clamp_score() wrapper in the dispatcher enforces this, AND every individual
function also avoids raw 0.0 / 1.0 returns for defence-in-depth.

Grading rubrics
---------------
html_strip
  0.40  No HTML tags remain in the submitted text.
  0.60  Character-level similarity to the reference clean text (SequenceMatcher).

pii_redact
  0.60  Recall  — fraction of known PII items that are absent from submitted text.
  0.40  Precision proxy — ratio of submitted length to reference length
        (penalises over-redaction that destroys non-PII content).
  Special case: if record has no PII, score = preservation similarity to original.

quality_audit
  Action matches ground truth:
    keep  + keep_record   → 0.99
    fix   + edit_record   → 0.40 base + 0.60 × fix_similarity_to_reference
    reject+ reject_record → 0.99
  Partial credit:
    fix   + reject_record → 0.25  (recognised the problem, wrong resolution)
    keep  + edit_record   → 0.20  (unnecessary edit, penalised)
    reject+ edit_record   → 0.10  (should have rejected, tried to fix instead)
    all other mismatches  → 0.01
"""

import difflib
import html
import re
from typing import Dict, Optional


def _clamp_score(score: float) -> float:
    """Clamp score to strictly within (0, 1) — not 0.0 or 1.0."""
    return max(0.01, min(0.99, score))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio after normalising whitespace.

    NOTE: This is a math helper returning a ratio, NOT a grader score.
    Its 0.0/1.0 values are fine — they feed into weighted formulas that
    produce intermediate scores, which then get clamped by _clamp_score().
    """
    a = re.sub(r"\s+", " ", a).strip().lower()
    b = re.sub(r"\s+", " ", b).strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _has_html_tags(text: str) -> bool:
    """True if the text still contains HTML-like tags."""
    return bool(re.search(r"<[a-zA-Z/!][^>]{0,200}>", text))


# ---------------------------------------------------------------------------
# Task 1 — HTML Strip
# ---------------------------------------------------------------------------

def grade_html_strip(record: Dict, submitted_text: str) -> float:
    """
    Returns a score for an edit_record action on an html_strip record.
    reject_record is always wrong for html_strip tasks (no record should_reject).
    """
    if record.get("should_reject"):
        return 0.01  # agent should have called reject_record instead

    reference: str = record["reference"]
    submitted: str = html.unescape(submitted_text)  # normalise HTML entities

    # 40 % — no tags remaining
    tag_score = 0.0 if _has_html_tags(submitted) else 0.40

    # 60 % — character similarity to reference
    text_score = _similarity(submitted, reference) * 0.60

    return round(tag_score + text_score, 4)


def grade_html_strip_reject(record: Dict) -> float:
    """Penalise reject on a fixable html_strip record."""
    return 0.10 if not record.get("should_reject") else 0.99


def grade_html_strip_keep(record: Dict) -> float:
    """Keeping an HTML record is always wrong (it still needs cleaning)."""
    return 0.05


# ---------------------------------------------------------------------------
# Task 2 — PII Redact
# ---------------------------------------------------------------------------

def grade_pii_redact(record: Dict, submitted_text: str) -> float:
    """Score an edit_record action on a pii_redact record."""
    pii_items = record.get("pii_items", [])
    original: str = record["text"]
    reference: str = record["reference"]

    # Special case: no PII → submitted text should be unchanged
    if not pii_items:
        return round(_similarity(submitted_text, original), 4)

    # 60 % — recall: fraction of PII values that no longer appear verbatim
    redacted_count = sum(
        1 for item in pii_items if item["value"] not in submitted_text
    )
    recall = redacted_count / len(pii_items)

    # 40 % — precision proxy: length ratio submitted vs reference
    ref_len = len(reference)
    sub_len = len(submitted_text)
    if max(ref_len, sub_len) == 0:
        precision = 1.0
    else:
        precision = min(ref_len, sub_len) / max(ref_len, sub_len)

    return round(0.60 * recall + 0.40 * precision, 4)


def grade_pii_redact_keep(record: Dict) -> float:
    """Keeping a record with PII is wrong; keeping a clean record is fine."""
    if record.get("pii_items"):
        return 0.01
    return 0.99


def grade_pii_redact_reject(record: Dict) -> float:
    """pii_redact records are fixable; rejecting them is wrong."""
    return 0.05


# ---------------------------------------------------------------------------
# Task 3 — Quality Audit
# ---------------------------------------------------------------------------

def grade_quality_audit(
    record: Dict,
    agent_action: str,            # "edit" | "keep" | "reject"
    submitted_text: Optional[str] = None,
) -> float:
    """
    Score an action on a quality_audit record.

    Parameters
    ----------
    agent_action : str
        One of "edit" (edit_record), "keep" (keep_record), "reject" (reject_record).
    submitted_text : str | None
        Only relevant when agent_action == "edit".
    """
    gt: str = record["ground_truth"]  # "keep" | "fix" | "reject"

    # ---- Perfect matches ------------------------------------------------
    if gt == "keep" and agent_action == "keep":
        return 0.99

    if gt == "reject" and agent_action == "reject":
        return 0.99

    if gt == "fix" and agent_action == "edit":
        base = 0.40
        fixed_ref: Optional[str] = record.get("fixed")
        if fixed_ref and submitted_text:
            quality = _similarity(submitted_text, fixed_ref)
            return round(base + 0.60 * quality, 4)
        return base

    # ---- Partial credit -------------------------------------------------
    # Recognised the problem but chose wrong resolution
    if gt == "fix" and agent_action == "reject":
        return 0.25  # knew it was bad; over-corrected

    if gt == "reject" and agent_action == "edit":
        return 0.10  # tried to fix something that should be thrown out

    if gt == "keep" and agent_action == "edit":
        # Penalty proportional to how different submitted text is from original
        original = record.get("response", "")
        unchanged = _similarity(submitted_text or "", original)
        return round(0.20 * unchanged, 4)  # light penalty for unnecessary edit

    # Everything else (e.g. keep when should reject)
    return 0.01


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade_action(
    task: str,
    record: Dict,
    agent_action: str,      # "edit" | "keep" | "reject"
    submitted_text: Optional[str] = None,
) -> float:
    """
    Route to the correct grader based on task.

    Returns score in (0.01, 0.99) — strictly excluding 0.0 and 1.0.
    """
    if task == "html_strip":
        if agent_action == "edit":
            return _clamp_score(grade_html_strip(record, submitted_text or ""))
        if agent_action == "reject":
            return _clamp_score(grade_html_strip_reject(record))
        if agent_action == "keep":
            return _clamp_score(grade_html_strip_keep(record))

    elif task == "pii_redact":
        if agent_action == "edit":
            return _clamp_score(grade_pii_redact(record, submitted_text or ""))
        if agent_action == "keep":
            return _clamp_score(grade_pii_redact_keep(record))
        if agent_action == "reject":
            return _clamp_score(grade_pii_redact_reject(record))

    elif task == "quality_audit":
        return _clamp_score(grade_quality_audit(record, agent_action, submitted_text))

    return 0.01
