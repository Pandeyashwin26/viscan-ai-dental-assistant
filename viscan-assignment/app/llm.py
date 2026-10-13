"""LLM adapters. They only PROPOSE structured data; validators.py decides what is accepted.

RuleBasedLLM  - offline, deterministic, default (no API key needed, used in tests)
AnthropicLLM  - real tool-calling via the Anthropic API; falls back to rules on any failure
"""
from __future__ import annotations
import logging, os, re
from .dictation import extract_rules
from .vocab import FINDING_VOCAB, SITE_RE, FDI_RE

log = logging.getLogger("viscan.llm")

_VERBS = r"(?:open|show|get|pull up|bring up|find|fetch|display|load|summari[sz]e|summary of|give me)"
_TYPES = [("OPG", r"\bopg\b|panoramic|orthopantomogram"), ("BITEWING", r"bite[\s-]?wing"),
          ("CBCT", r"\bcbct\b|cone[\s-]?beam"), ("IOPA", r"\biopa\b|periapical (?:x-?ray|radiograph|image)")]
_RECENCY = re.compile(r"\b(latest|most recent|newest|last|current)\b", re.I)
_PRONOUN = re.compile(r"\b(her|she|his|him|he|their|they|this patient|same patient)\b", re.I)
_TERMINATORS = r"(?:['’]s\b|\s+(?:latest|last|most recent|newest|current|opg|panoramic|bite|cbct|cone|iopa|periapical|report|reports|scan|x-?ray)\b|[,.?!]|$)"
_STOP = {"the", "a", "an", "patient", "me", "up", "report", "reports", "scan", "x-ray", "xray", "latest", "last",
         "most", "recent", "newest", "current", "her", "his", "their", "she", "he", "they", "this", "same", "please"}
_NAME = r"(?P<name>[a-z][a-z .'\-]*?)"


def _extract_name(t: str) -> str | None:
    m = (re.search(rf"\b(?:for|of)\s+(?:patient\s+)?{_NAME}{_TERMINATORS}", t, re.I)
         or re.search(rf"\b{_VERBS}\s+(?:me\s+)?(?:the\s+)?(?:patient\s+)?{_NAME}{_TERMINATORS}", t, re.I))
    if not m:
        return None
    toks = [w for w in m.group("name").split() if w.lower() not in _STOP]
    return " ".join(w.capitalize() for w in toks) or None


class RuleBasedLLM:
    name = "rule-based-offline"

    def interpret(self, text: str) -> dict:
        t = text.strip()
        has_verb = bool(re.search(rf"\b{_VERBS}\b", t, re.I))
        rtype = next((k for k, rx in _TYPES if re.search(rx, t, re.I)), None)
        pron = _PRONOUN.search(t)
        gender = None
        if pron:
            w = pron.group(1).lower()
            gender = "female" if w in ("her", "she") else "male" if w in ("his", "him", "he") else "neutral"
        if re.search(r"\bsummari[sz]e|summary of\b", t, re.I):
            intent = "summarize_report"
        elif has_verb:
            intent = "open_report"
        elif SITE_RE.search(t) or FDI_RE.search(t) or any(rx.search(t) for ps in FINDING_VOCAB.values() for rx in ps):
            intent = "dictate"
        else:
            intent = "unknown"
        return {"intent": intent, "patient_name": _extract_name(t) if intent != "dictate" else None,
                "report_type": rtype, "recency": "latest" if _RECENCY.search(t) else None, "pronoun": gender}

    def propose_findings(self, transcript: str) -> dict:
        return extract_rules(transcript)


class AnthropicLLM(RuleBasedLLM):
    """Tool-calling adapter. Output is still untrusted and goes through validators."""

    def __init__(self):
        import anthropic  # optional dependency
        self.client = anthropic.Anthropic()
        self.model = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"

    def _tool_call(self, system: str, user: str, tool: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=800, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}], timeout=15)
        return next(b.input for b in resp.content if b.type == "tool_use")

    def interpret(self, text: str) -> dict:
        tool = {"name": "route_request", "description": "Classify a dental assistant request.",
                "input_schema": {"type": "object", "properties": {
                    "intent": {"type": "string", "enum": ["open_report", "summarize_report", "dictate", "unknown"]},
                    "patient_name": {"type": ["string", "null"]},
                    "report_type": {"type": ["string", "null"], "enum": ["OPG", "BITEWING", "CBCT", "IOPA", None]},
                    "recency": {"type": ["string", "null"], "enum": ["latest", None]},
                    "pronoun": {"type": ["string", "null"], "enum": ["female", "male", "neutral", None]}},
                    "required": ["intent"]}}
        try:
            return self._tool_call("Extract intent and entities only. Never answer clinical questions. "
                                   "Copy the patient name exactly as spoken; use null if absent.", text, tool)
        except Exception as e:  # network, timeout, schema...
            log.warning("LLM interpret failed (%s); falling back to rules", e)
            return super().interpret(text)

    def propose_findings(self, transcript: str) -> dict:
        tool = {"name": "record_findings", "description": "Record ONLY observations explicitly dictated.",
                "input_schema": {"type": "object", "properties": {"findings": {"type": "array", "items": {
                    "type": "object", "properties": {
                        "finding": {"type": "string"},
                        "site": {"type": ["object", "null"], "properties": {"fdi": {"type": "string"}}},
                        "polarity": {"type": "string", "enum": ["present", "absent"]},
                        "certainty": {"type": "string", "enum": ["stated", "uncertain"]},
                        "evidence": {"type": "string", "description": "verbatim quote from the transcript"}},
                    "required": ["finding", "evidence"]}}}, "required": ["findings"]}}
        try:
            out = self._tool_call("You transcribe dental radiology dictation into findings. Do not infer, diagnose "
                                  "or add anything the clinician did not say. Quote the evidence verbatim. Mark "
                                  "hedged language as uncertain.", transcript, tool)
            return {"findings": out.get("findings", []), "unparsed": []}
        except Exception as e:
            log.warning("LLM propose_findings failed (%s); falling back to rules", e)
            return super().propose_findings(transcript)


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("LLM_PROVIDER", "anthropic") == "anthropic":
        try:
            return AnthropicLLM()
        except Exception as e:
            log.warning("Anthropic adapter unavailable (%s); using rule-based", e)
    return RuleBasedLLM()
