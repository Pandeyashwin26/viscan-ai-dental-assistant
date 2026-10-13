import pytest
from fastapi.testclient import TestClient
from app.assistant import Assistant, Session
from app.llm import RuleBasedLLM
from app.store import Store


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
