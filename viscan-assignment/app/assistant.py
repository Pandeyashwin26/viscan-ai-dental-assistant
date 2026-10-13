"""Orchestrator: LLM interpretation -> business-rule validation -> deterministic tools."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from .store import Store, public_patient
from .booking import BookingStore, DuplicateBooking, SlotUnavailable, resolve_relative_date
from .validators import (InputError, clean_text, validate_interpretation, validate_patient_id)
from .dictation import build_draft

_GENDER = {"female": "F", "male": "M"}


@dataclass
class Session:
    patient_id: str | None = None
    report_id: str | None = None
    pending_text: str | None = None
    pending_booking: dict | None = None


def _resp(status: str, message: str, **kw) -> dict:
    return {"status": status, "message": message, **kw}


class Assistant:
    def __init__(self, store: Store, llm, booking_store: BookingStore | None = None):
        self.store, self.llm = store, llm
        self.booking_store = booking_store or BookingStore()

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
        if interp.get("previous") and len(reports) < 2:
            return _resp("no_report", f"No previous {interp['report_type'] or 'any'} report on file for {patient['name']}.",
                         patient=public_patient(patient), warnings=warnings)
        r = reports[1] if interp.get("previous") else reports[0]
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

    def _report_question(self, r: dict, patient: dict, text: str) -> dict:
        question = text.lower()
        if re.search(r"\b(compare|comparison)\b", question):
            reports = self.store.reports_for(patient["id"], r["type"])
            if len(reports) < 2:
                return _resp("missing_comparison", "A previous report of the same type is not available.",
                             patient=public_patient(patient), report_id=r["id"], warnings=[])
            previous = next(report for report in reports if report["id"] != r["id"])
            answer = (f"Current ({r['id']}): {'; '.join(r['findings'])}. "
                      f"Previous ({previous['id']}): {'; '.join(previous['findings'])}.")
            return _resp("ok", answer, patient=public_patient(patient), report_id=r["id"],
                         sources=[{"report_id": r["id"], "fields": ["findings"], "evidence": r["findings"]},
                                  {"report_id": previous["id"], "fields": ["findings"], "evidence": previous["findings"]}], warnings=[])
        if re.search(r"\b(findings?|observations?)\b", question):
            fields, evidence = ["findings"], r["findings"]
            answer = "Findings: " + "; ".join(evidence)
        elif re.search(r"\b(anatomical site|tooth|teeth|site)\b", question):
            fields, evidence = ["findings"], r["findings"]
            sites = sorted(set(re.findall(r"\b(?:tooth\s*)?([1-4][1-8])\b", " ".join(evidence))))
            answer = "Documented anatomical sites: " + (", ".join(sites) if sites else "No tooth site is stated.")
        elif re.search(r"\b(uncertain|uncertainty)\b", question):
            fields, evidence = ["findings", "impression"], r["findings"] + [r["impression"]]
            uncertain = [item for item in evidence if re.search(r"\b(possible|uncertain|may|could)\b", item, re.I)]
            answer = "Uncertain content: " + "; ".join(uncertain) if uncertain else "No uncertainty is documented."
        elif re.search(r"\bmissing\b", question):
            fields, evidence = ["id", "type", "date", "findings", "impression"], []
            answer = "The stored report does not include symptoms, treatment recommendations, or image files."
        elif re.search(r"\b(impression|conclusion)\b", question):
            fields, evidence = ["impression"], [r["impression"]]
            answer = "Impression: " + r["impression"]
        elif re.search(r"\b(date|when)\b", question):
            fields, evidence = ["date"], [r["date"]]
            answer = f"The report date is {r['date']}."
        elif re.search(r"\b(clinician|reported by|who)\b", question):
            fields, evidence = ["clinician"], [r["clinician"]]
            answer = f"The report was reported by {r['clinician']}."
        else:
            return _resp("unsupported_question",
                         "I can answer questions about the report's findings, impression, date, or clinician.",
                         patient=public_patient(patient), report_id=r["id"], warnings=[])
        return _resp("ok", answer, patient=public_patient(patient), report_id=r["id"],
                     sources=[{"report_id": r["id"], "fields": fields, "evidence": evidence}], warnings=[])

    def _booking(self, session: Session, interp: dict, patient: dict | None) -> dict:
        if interp["intent"] == "booking_confirm":
            pending = session.pending_booking
            if not pending or not pending.get("slot"):
                return _resp("needs_booking_selection", "Select an appointment slot before confirming a booking.", warnings=[])
            patient = patient or (self.store.get_patient(pending.get("patient_id")) if pending.get("patient_id") else None)
            if not patient:
                return _resp("needs_patient", "Which patient should this appointment be booked for?", options=[pending["slot"]], warnings=[])
            try:
                booking = self.booking_store.book(pending["slot"]["id"], patient)
            except SlotUnavailable as exc:
                session.pending_booking = None
                return _resp("slot_unavailable", str(exc), warnings=[])
            except DuplicateBooking as exc:
                session.pending_booking = None
                return _resp("duplicate_booking", str(exc), warnings=[])
            session.pending_booking = None
            return _resp("booking_confirmed", "Appointment booked successfully.", booking=booking,
                         sources=[{"slot_id": booking["slot_id"], "fields": ["date", "time", "scan_type"]}], warnings=[])

        if interp["appointment_time"] and session.pending_booking and not interp["patient_name"]:
            matches = [slot for slot in session.pending_booking.get("options", [])
                       if slot["time"] == interp["appointment_time"]]
            if len(matches) == 1:
                slot = matches[0]
                session.pending_booking["slot"] = slot
                if patient:
                    session.pending_booking["patient_id"] = patient["id"]
                return _resp("booking_confirmation_required",
                             "Review the appointment details and reply 'confirm' to book.",
                             appointment=slot, patient=public_patient(patient) if patient else None,
                             sources=[{"slot_id": slot["id"], "fields": ["date", "time", "scan_type"]}], warnings=[])

        appointment_date = resolve_relative_date(interp["appointment_date"])
        options = self.booking_store.available(interp["scan_type"], appointment_date)
        if interp["appointment_time"]:
            options = [slot for slot in options if slot["time"] == interp["appointment_time"]]
        if not options:
            return _resp("no_available_slots", "No matching synthetic appointment slots are available.", warnings=[])
        session.pending_booking = {"options": options, "patient_id": patient["id"] if patient else None}
        if len(options) == 1:
            session.pending_booking["slot"] = options[0]
            return _resp("booking_confirmation_required",
                         "Review the appointment details and reply 'confirm' to book.",
                         appointment=options[0], patient=public_patient(patient) if patient else None,
                         sources=[{"slot_id": options[0]["id"], "fields": ["date", "time", "scan_type"]}], warnings=[])
        return _resp("booking_options", "Select one appointment slot, then confirm the booking.",
                     options=options, patient=public_patient(patient) if patient else None,
                     sources=[{"slot_id": slot["id"], "fields": ["date", "time", "scan_type"]} for slot in options], warnings=[])

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
        if intent in ("booking", "booking_confirm"):
            patient = None
            if patient_id:
                patient = self.store.get_patient(patient_id)
                if not patient:
                    return _resp("no_patient", f"Unknown patient_id {patient_id}.")
            elif interp["patient_name"]:
                patient, early = self._resolve(interp, session, None, text)
                if early:
                    return early
            elif session.patient_id:
                patient = self.store.get_patient(session.patient_id)
            out = self._booking(session, interp, patient)
            if patient:
                session.patient_id = patient["id"]
            return out
        if interp.get("report_id"):
            report = self.store.get_report(interp["report_id"])
            if not report:
                return _resp("no_report", f"No report found with id {interp['report_id']}.", warnings=[])
            if patient_id and report["patient_id"] != patient_id:
                return _resp("authorization_error", "That report does not belong to the selected patient.", warnings=[])
            patient = self.store.get_patient(report["patient_id"])
            session.patient_id, session.report_id = patient["id"], report["id"]
            return _resp("ok", f"{report['type']} report dated {report['date']}.", patient=public_patient(patient),
                         report=self._report_payload(report), sources=[{"report_id": report["id"], "fields": ["id", "type", "date"]}], warnings=[])
        patient, early = self._resolve(interp, session, patient_id, text)
        if early:
            return early
        extra = patient.pop("_warnings", []) if isinstance(patient, dict) else []
        session.pending_text = None
        if intent == "open_report":
            out = self._pick_report(patient, interp, session, extra)
            return out
        # Report follow-ups reuse the report already on screen when unambiguous.
        if not interp["patient_name"] and not interp["report_type"] and session.report_id:
            r = self.store.get_report(session.report_id)
        else:
            reports = self.store.reports_for(patient["id"], interp["report_type"])
            r = reports[0] if reports else None
        if not r:
            return _resp("no_report", f"No matching report on file for {patient['name']}.", patient=public_patient(patient))
        session.patient_id, session.report_id = patient["id"], r["id"]
        if intent == "report_question":
            out = self._report_question(r, patient, text)
            out["warnings"] = extra
            return out
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
