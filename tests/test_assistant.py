import pytest
from fastapi.testclient import TestClient
from app.assistant import Assistant, Session
from app.llm import RuleBasedLLM
from app.store import Store
from app.booking import BookingStore


@pytest.fixture()
def bot():
    return Assistant(Store(), RuleBasedLLM()), Session()


# 1. correct retrieval --------------------------------------------------------
def test_retrieves_latest_opg_for_exact_name(bot):
    a, s = bot
    out = a.handle(s, "Open Priya Nair's latest OPG report")
    assert out["status"] == "ok"
    assert out["report"]["id"] == "R-1002"            # newest of two OPGs
    assert out["report"]["type"] == "OPG"
    assert any("Priya Nayar" in w for w in out["warnings"])   # similar-name safety warning


def test_retrieves_previous_report_and_direct_report_id(bot):
    a, s = bot
    previous = a.handle(s, "Open Priya Nair's previous OPG report")
    assert previous["status"] == "ok" and previous["report"]["id"] == "R-1001"
    direct = a.handle(s, "Open report R-1002")
    assert direct["status"] == "ok" and direct["report"]["id"] == "R-1002"


# 2. ambiguous patient identity ------------------------------------------------
def test_first_name_only_is_ambiguous_and_returns_no_report(bot):
    a, s = bot
    out = a.handle(s, "Open Priya's latest OPG report")
    assert out["status"] == "ambiguous_patient"
    assert {c["id"] for c in out["candidates"]} == {"P001", "P002"}
    assert "report" not in out and s.patient_id is None   # nothing selected, nothing leaked


def test_ambiguity_resolved_with_patient_id_reuses_original_request(bot):
    a, s = bot
    a.handle(s, "Open Priya's latest OPG report")
    out = a.handle(s, "", patient_id="P002")
    assert out["status"] == "ok" and out["report"]["id"] == "R-2001"


def test_misheard_name_requires_confirmation(bot):
    a, s = bot
    out = a.handle(s, "Open Priya Nayr's latest OPG")
    assert out["status"] == "needs_confirmation" and "report" not in out


def test_missing_report_and_unknown_patient(bot):
    a, s = bot
    assert a.handle(s, "Show Rahul Mehta's CBCT")["status"] == "no_report"
    assert a.handle(s, "Open Zoya Khan's latest OPG")["status"] == "no_patient"


# 3. conversational follow-up ---------------------------------------------------
def test_followup_summary_uses_context_and_cites_source(bot):
    a, s = bot
    a.handle(s, "Open Anita Desai's latest OPG report")
    out = a.handle(s, "Summarize her report")
    assert out["status"] == "ok" and "Implant" in out["message"]
    assert out["sources"][0]["report_id"] == "R-5001"


def test_followup_without_context_asks_instead_of_guessing(bot):
    a, s = bot
    assert a.handle(s, "Summarize her report")["status"] == "needs_patient"


def test_followup_pronoun_mismatch_asks(bot):
    a, s = bot
    a.handle(s, "Open Karan Shah's latest IOPA")
    assert a.handle(s, "Summarize her report")["status"] == "needs_patient"


def test_report_question_is_grounded_in_selected_report(bot):
    a, s = bot
    a.handle(s, "Open Priya Nair's latest OPG report")
    out = a.handle(s, "What are the findings in her report?")
    assert out["status"] == "ok"
    assert "Horizontally impacted" in out["message"]
    assert out["sources"][0]["fields"] == ["findings"]


def test_unsupported_report_question_does_not_invent_an_answer(bot):
    a, s = bot
    a.handle(s, "Open Priya Nair's latest OPG report")
    out = a.handle(s, "What treatment should she receive?")
    assert out["status"] == "unknown_intent"


def test_report_chatbot_supports_sites_uncertainty_and_comparison(bot):
    a, s = bot
    a.handle(s, "Open Priya Nair's latest OPG report")
    site = a.handle(s, "Which anatomical site is mentioned in her report?")
    uncertainty = a.handle(s, "Does the report contain uncertainty?")
    comparison = a.handle(s, "Compare this report with the previous one")
    assert site["status"] == "ok" and "48" in site["message"]
    assert uncertainty["status"] == "ok" and "Radiolucency" in uncertainty["message"]
    assert comparison["status"] == "ok" and len(comparison["sources"]) == 2


def test_booking_requires_confirmation_and_persists_successful_booking():
    booking_store = BookingStore()
    a = Assistant(Store(), RuleBasedLLM(), booking_store)
    s = Session()
    options = a.handle(s, "Book an OPG scan for Priya Nair on 2026-10-15")
    assert options["status"] == "booking_options"
    assert not booking_store.bookings
    selected = a.handle(s, "Book the 11 AM slot")
    assert selected["status"] == "booking_confirmation_required"
    confirmed = a.handle(s, "Confirm booking")
    assert confirmed["status"] == "booking_confirmed"
    assert confirmed["booking"]["patient_id"] == "P001"
    assert len(booking_store.bookings) == 1


def test_booking_conflict_is_reported_without_false_success():
    booking_store = BookingStore()
    first, second = (Assistant(Store(), RuleBasedLLM(), booking_store) for _ in range(2))
    first_session, second_session = Session(), Session()
    first.handle(first_session, "Book an OPG scan for Priya Nair on 2026-10-15 at 09:00")
    first.handle(first_session, "Confirm booking")
    unavailable = second.handle(second_session, "Book an OPG scan for Rahul Mehta on 2026-10-15 at 09:00")
    assert unavailable["status"] == "no_available_slots"
    result = second.handle(second_session, "Confirm booking")
    assert result["status"] == "needs_booking_selection"
    assert len(booking_store.bookings) == 1


def test_tomorrow_resolves_from_supplied_reference_date():
    from datetime import date
    from app.booking import resolve_relative_date

    assert resolve_relative_date("tomorrow", date(2026, 10, 14)) == "2026-10-15"


# 4. dictation + unsupported clinical claims -------------------------------------
def test_dictation_structured_draft(bot):
    a, s = bot
    out = a.dictate(s, "Lower right third molar is horizontally impacted")
    d = out["draft"]
    assert d["clinician_reviewed"] is False and d["report_status"].startswith("DRAFT")
    f = d["findings"][0]
    assert f["finding"] == "horizontal impaction" and f["site"]["fdi"] == "48" and f["needs_review"]
    assert "requires clinician review" in d["draft_report"].lower()


def test_uncertainty_is_labelled(bot):
    a, s = bot
    d = a.dictate(s, "Possible caries on upper left first molar")["draft"]
    assert d["findings"][0]["certainty"] == "uncertain" and "UNCERTAIN" in d["draft_report"]


def test_does_not_invent_observations(bot):
    a, s = bot
    d = a.dictate(s, "Patient is comfortable and cooperative today")["draft"]
    assert d["findings"] == [] and d["unparsed_statements"]


class HallucinatingLLM(RuleBasedLLM):
    """Simulates a model that adds diagnoses the clinician never dictated."""
    name = "fake-hallucinating"
    def propose_findings(self, transcript):
        return {"findings": [
            {"finding": "horizontal impaction", "site": {"fdi": "48"}, "polarity": "present",
             "certainty": "stated", "evidence": "lower right third molar is horizontally impacted"},
            {"finding": "periapical abscess", "site": {"fdi": "48"}, "evidence": "lower right third molar is horizontally impacted"},
            {"finding": "dental caries", "site": {"fdi": "36"}, "evidence": "caries on 36"},
            {"finding": "dental caries", "site": {"fdi": "48"}, "evidence": "lower right third molar is horizontally impacted"},
        ], "unparsed": []}


def test_unsupported_clinical_claims_are_rejected():
    a, s = Assistant(Store(), HallucinatingLLM()), Session()
    d = a.dictate(s, "Lower right third molar is horizontally impacted")["draft"]
    assert [f["finding"] for f in d["findings"]] == ["horizontal impaction"]
    assert len(d["rejected_claims"]) == 3
    assert "abscess" not in d["draft_report"].lower()


# 5. input validation + API -----------------------------------------------------
def test_input_validation(bot):
    a, s = bot
    assert a.handle(s, "   ")["status"] == "invalid_input"
    assert a.handle(s, "x" * 5000)["status"] == "invalid_input"
    assert a.handle(s, "Open report", patient_id="'; DROP")["status"] == "invalid_input"


def test_api_end_to_end():
    from app.main import app
    c = TestClient(app)
    r = c.post("/chat", json={"session_id": "t1", "text": "Open Rahul Mehta's latest OPG report"}).json()
    assert r["status"] == "ok" and r["report"]["id"] == "R-3001"
    r = c.post("/chat", json={"session_id": "t1", "text": "Summarize his report"}).json()
    assert r["status"] == "ok" and r["sources"][0]["report_id"] == "R-3001"
    assert c.post("/transcribe", files={"file": ("a.wav", b"123")}).status_code == 501


def test_api_draft_can_be_edited_and_requires_review_before_finalize():
    from app.main import app
    c = TestClient(app)
    draft = c.post("/dictate", json={"session_id": "draft-lifecycle", "transcript": "Lower right third molar is horizontally impacted"}).json()["draft"]
    draft_id = draft["draft_id"]
    assert c.get(f"/drafts/{draft_id}").status_code == 200
    assert c.post(f"/drafts/{draft_id}/finalize", json={"clinician_reviewed": False}).status_code == 400
    edited = c.put(f"/drafts/{draft_id}", json={"draft_report": "Edited draft for review."})
    assert edited.status_code == 200 and edited.json()["clinician_reviewed"] is False
    finalized = c.post(f"/drafts/{draft_id}/finalize", json={"clinician_reviewed": True})
    assert finalized.status_code == 200 and finalized.json()["report_status"] == "FINAL"
    assert c.delete(f"/drafts/{draft_id}").status_code == 200


def test_api_key_protects_draft_routes(monkeypatch):
    from app.main import app

    monkeypatch.setenv("API_KEY", "test-secret")
    c = TestClient(app)
    assert c.get("/drafts/not-found").status_code == 401
    assert c.get("/drafts/not-found", headers={"X-API-Key": "test-secret"}).status_code == 404
    assert c.get("/health").status_code == 200
