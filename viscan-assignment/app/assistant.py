"""Orchestrator: LLM interpretation -> business-rule validation -> deterministic tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from .store import Store, public_patient
from .validators import (InputError, clean_text, validate_interpretation, validate_patient_id)
from .dictation import build_draft

_GENDER = {"female": "F", "male": "M"}


@dataclass
class Session:
    patient_id: str | None = None
    report_id: str | None = None
    pending_text: str | None = None


def _resp(status: str, message: str, **kw) -> dict:
    return {"status": status, "message": message, **kw}


class Assistant:
    def __init__(self, store: Store, llm):
        self.store, self.llm = store, llm

    # ---- tools (deterministic) -------------------------------------------------
    def _report_payload(self, r: dict) -> dict:
        return {k: r[k] for k in ("id", "patient_id", "type", "date", "clinician", "status", "findings", "impression")}

    def _pick_report(self, patient: dict, interp: dict, session: Session, warnings: list) -> dict:
        reports = self.store.reports_for(patient["id"], interp["report_type"])
        if not reports:
            avail = sorted({r["type"] for r in self.store.reports_for(patient["id"])})
            what = interp["report_type"] or "any"
            return _resp("no_report", f"No {what} report on file for {patient['name']}. Available types: {', '.join(avail) or 'none'}.",
                         patient=public_patient(patient), available_types=avail, warnings=warnings)
        r = reports[0]
        session.patient_id, session.report_id = patient["id"], r["id"]
        return _resp("ok", f"{r['type']} report for {patient['name']} dated {r['date']}.",
                     patient=public_patient(patient), report=self._report_payload(r),
                     sources=[{"report_id": r["id"], "fields": ["type", "date"]}], warnings=warnings)

    def _summary(self, r: dict, patient: dict) -> dict:
        bullets = "; ".join(f.rstrip(".") for f in r["findings"])
        text = (f"{r['type']} dated {r['date']} for {patient['name']} (reported by {r['clinician']}). "
                f"Findings: {bullets}. Impression: {r['impression']}")
        return _resp("ok", text, patient=public_patient(patient),
                     sources=[{"report_id": r["id"], "fields": ["findings", "impression"],
                               "evidence": r["findings"] + [r["impression"]]}], warnings=[])

    # ---- identity resolution ---------------------------------------------------
    def _resolve(self, interp: dict, session: Session, explicit_id: str | None, text: str):
        """Returns (patient, early_response). Never guesses between candidates."""
        if explicit_id:
            p = self.store.get_patient(explicit_id)
            return (p, None) if p else (None, _resp("no_patient", f"Unknown patient_id {explicit_id}."))
        warnings: list[str] = []
        if interp["patient_name"]:
            res = self.store.find_patients(interp["patient_name"])
            cands = [public_patient(p) for p in res["candidates"]]
            if res["status"] == "ambiguous":
                session.pending_text = text
                return None, _resp("ambiguous_patient", f"Several patients match '{interp['patient_name']}'. Which one did you mean? "
                                   "Re-send with patient_id.", candidates=cands)
            if res["status"] == "fuzzy":
                session.pending_text = text
                return None, _resp("needs_confirmation", f"No exact match for '{interp['patient_name']}'. Did you mean one of these?", candidates=cands)
            if res["status"] == "none":
                return None, _resp("no_patient", f"No patient found matching '{interp['patient_name']}'.")
            p = res["candidates"][0]
            p = dict(p)
            p["_warnings"] = ([f"Similar-sounding patient also on file: {s['name']} (DOB {s['dob']}). Verify DOB {p['dob']} before acting."
                               for s in res["similar"]])
            return p, None
        # follow-up reference ("her report", "his scan")
        if session.patient_id:
            p = self.store.get_patient(session.patient_id)
            want = _GENDER.get(interp.get("pronoun") or "")
            if want and p["sex"] != want:
                return None, _resp("needs_patient", "That pronoun does not match the patient currently open. Which patient do you mean?")
            return p, None
        return None, _resp("needs_patient", "Which patient do you mean? No patient is currently selected.")

    # ---- public entry points ---------------------------------------------------
    def handle(self, session: Session, text: str | None, patient_id: str | None = None) -> dict:
        try:
            patient_id = validate_patient_id(patient_id)
            if not (text or "").strip() and patient_id and session.pending_text:
                text = session.pending_text          # user is resolving an earlier ambiguity
            text = clean_text(text)
        except InputError as e:
            return _resp("invalid_input", str(e))
        interp, warnings = validate_interpretation(self.llm.interpret(text))
        out = self._route(session, text, interp, patient_id)
        out.setdefault("warnings", [])
        out["warnings"] = warnings + out["warnings"]
        out["interpretation"] = interp
        return out

    def _route(self, session: Session, text: str, interp: dict, patient_id: str | None) -> dict:
        intent = interp["intent"]
        if intent == "dictate":
            return self.dictate(session, text, patient_id)
        if intent == "unknown":
            return _resp("unknown_intent", "I can open a patient's report, summarize it, or turn a dictation into a draft report.")
        patient, early = self._resolve(interp, session, patient_id, text)
        if early:
            return early
        extra = patient.pop("_warnings", []) if isinstance(patient, dict) else []
        session.pending_text = None
        if intent == "open_report":
            out = self._pick_report(patient, interp, session, extra)
            return out
        # summarize_report: reuse the report already on screen when unambiguous
        if not interp["patient_name"] and not interp["report_type"] and session.report_id:
            r = self.store.get_report(session.report_id)
        else:
            reports = self.store.reports_for(patient["id"], interp["report_type"])
            r = reports[0] if reports else None
        if not r:
            return _resp("no_report", f"No matching report on file for {patient['name']}.", patient=public_patient(patient))
        session.patient_id, session.report_id = patient["id"], r["id"]
        out = self._summary(r, patient)
        out["warnings"] = extra
        return out

    def dictate(self, session: Session, transcript: str | None, patient_id: str | None = None) -> dict:
        try:
            patient_id = validate_patient_id(patient_id)
            transcript = clean_text(transcript, max_len=4000)
        except InputError as e:
            return _resp("invalid_input", str(e))
        pid = patient_id or session.patient_id
        patient = self.store.get_patient(pid) if pid else None
        draft = build_draft(transcript, self.llm, patient)
        warnings = [] if patient else ["No patient linked to this draft. Select a patient before saving."]
        if draft["rejected_claims"]:
            warnings.append(f"{len(draft['rejected_claims'])} model-proposed claim(s) rejected as unsupported by the dictation.")
        return _resp("ok", "Draft created. Clinician review required.", draft=draft, warnings=warnings,
                     sources=[{"type": "dictation_transcript", "text": transcript}])
