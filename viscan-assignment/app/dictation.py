"""Clinical dictation -> structured draft. Extraction is separate from validation."""
from __future__ import annotations
import re
import uuid
from .vocab import FINDING_VOCAB, UNCERTAIN_RE, NEGATION_RE, find_sites
from .validators import validate_findings

_SPLIT = re.compile(r"[.;\n]+|\s+(?:and|also|then)\s+(?=(?:the\s+)?(?:upper|lower|maxillary|mandibular|tooth|teeth|#))", re.I)


def split_clauses(text: str) -> list[str]:
    return [c.strip(" ,") for c in _SPLIT.split(text) if c and c.strip(" ,")]


def _find_terms(clause: str) -> list[str]:
    work, found = clause, []
    for name, patterns in FINDING_VOCAB.items():
        for rx in patterns:
            m = rx.search(work)
            if m:
                found.append(name)
                work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
                break
    return found


def extract_rules(transcript: str) -> dict:
    """Offline, deterministic proposer (also the fallback when no LLM key is set)."""
    findings, unparsed = [], []
    for clause in split_clauses(transcript):
        sites, terms = find_sites(clause), _find_terms(clause)
        if not terms:
            unparsed.append({"text": clause, "reason": "no supported finding recognised; not added to report"})
            continue
        if len(sites) > 1 and len(terms) > 1:
            unparsed.append({"text": clause, "reason": "ambiguous site/finding pairing; clinician to enter manually"})
            continue
        polarity = "absent" if NEGATION_RE.search(clause) else "present"
        certainty = "uncertain" if UNCERTAIN_RE.search(clause) else "stated"
        for site in (sites or [None]):
            for term in terms:
                findings.append({"finding": term, "site": site, "polarity": polarity,
                                 "certainty": certainty, "evidence": clause})
    return {"findings": findings, "unparsed": unparsed}


def _sentence(f: dict) -> str:
    where = f" at tooth {f['site']['fdi']} ({f['site']['description']})" if f.get("site") else " (site not stated)"
    if f["polarity"] == "absent":
        s = f"No {f['finding']} noted{where}."
    elif f["certainty"] == "uncertain":
        s = f"Possible {f['finding']}{where} - UNCERTAIN, clinician to confirm."
    else:
        s = f"{f['finding'].capitalize()}{where}."
    return s[0].upper() + s[1:]


def render_draft(findings: list[dict], unparsed: list[dict]) -> str:
    lines = ["DRAFT - NOT A CLINICAL RECORD. Requires clinician review and sign-off.", "",
             "Dictated observations:"]
    lines += [f"{i}. {_sentence(f)}" for i, f in enumerate(findings, 1)] or ["(no supported findings extracted)"]
    if unparsed:
        lines += ["", "Dictated statements NOT converted (clinician to review):"]
        lines += [f"- \"{u['text']}\" - {u['reason']}" for u in unparsed]
    lines += ["", "Structures or findings that were not dictated are not commented on."]
    return "\n".join(lines)


def build_draft(transcript: str, llm, patient: dict | None = None) -> dict:
    raw = llm.propose_findings(transcript)                      # 1. LLM proposes
    if not isinstance(raw, dict):
        raw = {"findings": [], "unparsed": [{"text": transcript, "reason": "invalid model output; clinician to review manually"}]}
    accepted, rejected = validate_findings(raw.get("findings", []), transcript)  # 2. rules decide
    for i, f in enumerate(accepted, 1):
        f["id"] = f"F{i}"; f["needs_review"] = True
    unparsed = raw.get("unparsed", [])
    contradictions = []
    for index, left in enumerate(accepted):
        for right in accepted[index + 1:]:
            if left.get("finding") == right.get("finding") and left.get("site") == right.get("site") and left.get("polarity") != right.get("polarity"):
                contradictions.append({"finding": left["finding"], "site": left.get("site"), "claims": [left, right]})
    return {
        "draft_id": f"D-{uuid.uuid4().hex[:8].upper()}",
        "report_status": "DRAFT_REQUIRES_CLINICIAN_REVIEW",
        "clinician_reviewed": False,
        "patient": {"id": patient["id"], "name": patient["name"]} if patient else None,
        "transcript": transcript,
        "findings": accepted,
        "unparsed_statements": unparsed,
        "rejected_claims": rejected,
        "contradictions": contradictions,
        "draft_report": render_draft(accepted, unparsed),
        "generated_by": getattr(llm, "name", "unknown"),
    }
