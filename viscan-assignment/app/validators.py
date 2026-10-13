"""Business-rule validation. Sits BETWEEN the LLM and everything that acts on its output.

The LLM only *proposes* (intent, entities, findings). Nothing it says is trusted
until it passes these deterministic checks.
"""
from __future__ import annotations
import re
from .vocab import (FINDING_VOCAB, POLARITIES, CERTAINTIES, REPORT_TYPES, find_sites, describe_fdi)

MAX_INPUT_CHARS = 1500
ALLOWED_INTENTS = {"open_report", "summarize_report", "dictate", "unknown"}
ALLOWED_RECENCY = {"latest", None}
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,59}$")
PATIENT_ID_RE = re.compile(r"^P\d{3}$")


class InputError(ValueError):
    pass


def clean_text(text: str | None, max_len: int = MAX_INPUT_CHARS) -> str:
    if text is None or not str(text).strip():
        raise InputError("Empty input.")
    t = _CTRL.sub(" ", str(text)).strip()
    if len(t) > max_len:
        raise InputError(f"Input too long (max {max_len} characters).")
    return t


def validate_patient_id(pid: str | None) -> str | None:
    if pid is None:
        return None
    if not PATIENT_ID_RE.match(pid):
        raise InputError("Invalid patient_id format (expected e.g. P001).")
    return pid


def validate_interpretation(raw: dict) -> tuple[dict, list[str]]:
    """Coerce whatever the LLM returned into a safe, closed schema."""
    warnings: list[str] = []
    raw = raw if isinstance(raw, dict) else {}
    intent = raw.get("intent")
    if intent not in ALLOWED_INTENTS:
        warnings.append(f"Unrecognised intent {intent!r}; treated as unknown.")
        intent = "unknown"
    name = raw.get("patient_name")
    if name is not None:
        name = str(name).strip()
        if not name or not _NAME_OK.match(name):
            warnings.append("Patient name failed validation and was ignored.")
            name = None
    rtype = raw.get("report_type")
    if rtype is not None:
        rtype = str(rtype).upper()
        if rtype not in REPORT_TYPES:
            warnings.append(f"Unsupported report type {rtype!r} ignored.")
            rtype = None
    recency = raw.get("recency") if raw.get("recency") in ALLOWED_RECENCY else None
    return {
        "intent": intent, "patient_name": name, "report_type": rtype, "recency": recency,
        "pronoun": raw.get("pronoun") if raw.get("pronoun") in ("female", "male", "neutral") else None,
    }, warnings


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def validate_findings(proposed: list, transcript: str) -> tuple[list[dict], list[dict]]:
    """Reject any proposed finding the clinician did not actually say.

    A finding is accepted only if:
      1. it is in the closed vocabulary,
      2. its evidence span literally occurs in the transcript,
      3. the evidence text itself supports that finding term,
      4. any anatomical site is valid FDI and is stated in the evidence.
    """
    accepted, rejected = [], []
    t = _norm(transcript)
    for p in proposed or []:
        reasons: list[str] = []
        if not isinstance(p, dict):
            rejected.append({"claim": str(p)[:200], "reasons": ["malformed"]}); continue
        name = p.get("finding")
        ev = _norm(str(p.get("evidence", "")))
        if name not in FINDING_VOCAB:
            reasons.append("finding not in supported vocabulary")
        if not ev or ev not in t:
            reasons.append("evidence not found in the dictated transcript")
        elif name in FINDING_VOCAB and not any(rx.search(ev) for rx in FINDING_VOCAB[name]):
            reasons.append("evidence text does not support this finding")
        if p.get("polarity", "present") not in POLARITIES:
            reasons.append("invalid polarity")
        if p.get("certainty", "stated") not in CERTAINTIES:
            reasons.append("invalid certainty")
        site = p.get("site")
        if site:
            fdi = str(site.get("fdi", ""))
            if not re.fullmatch(r"[1-4][1-8]", fdi):
                reasons.append("invalid FDI tooth code")
            elif fdi not in {s["fdi"] for s in find_sites(ev)}:
                reasons.append("site not stated in the evidence")
        if reasons:
            rejected.append({"claim": {"finding": name, "site": site, "evidence": p.get("evidence")}, "reasons": reasons})
        else:
            f = dict(p)
            f.setdefault("polarity", "present"); f.setdefault("certainty", "stated")
            if site:
                f["site"] = {"fdi": site["fdi"], "description": describe_fdi(site["fdi"])}
            accepted.append(f)
    return accepted, rejected
