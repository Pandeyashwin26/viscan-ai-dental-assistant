"""FastAPI demo API.  Run: uvicorn app.main:app --reload"""
from __future__ import annotations
import logging, os
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Any
from .assistant import Assistant, Session
from .llm import get_llm
from .store import Store
from . import stt

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Viscan voice-first dental assistant (synthetic data PoC)")
assistant = Assistant(Store(), get_llm())
SESSIONS: dict[str, Session] = {}
DRAFTS: dict[str, dict] = {}
MAX_AUDIO_BYTES = 10 * 1024 * 1024


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")


class ChatIn(BaseModel):
    session_id: str = Field("default", max_length=64, pattern=r"^[\w\-]+$")
    text: str = Field("", max_length=1500)
    patient_id: str | None = Field(None, max_length=8)


class DictateIn(BaseModel):
    session_id: str = Field("default", max_length=64, pattern=r"^[\w\-]+$")
    transcript: str = Field(..., max_length=4000)
    patient_id: str | None = Field(None, max_length=8)


class DraftEditIn(BaseModel):
    draft_report: str | None = Field(None, max_length=10000)
    findings: list[dict[str, Any]] | None = None


class FinalizeIn(BaseModel):
    clinician_reviewed: bool = False


@app.get("/health")
def health():
    return {"ok": True, "llm": assistant.llm.name}


@app.get("/")
def root():
    return {
        "name": "Viscan voice-first dental assistant",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(body: ChatIn):
    return assistant.handle(SESSIONS.setdefault(body.session_id, Session()), body.text, body.patient_id)


@app.post("/dictate", dependencies=[Depends(require_key)])
def dictate(body: DictateIn):
    result = assistant.dictate(SESSIONS.setdefault(body.session_id, Session()), body.transcript, body.patient_id)
    if result.get("status") == "ok":
        DRAFTS[result["draft"]["draft_id"]] = result["draft"]
    return result


@app.get("/drafts/{draft_id}")
def get_draft(draft_id: str):
    draft = DRAFTS.get(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found.")
    return draft


@app.put("/drafts/{draft_id}")
def edit_draft(draft_id: str, body: DraftEditIn):
    draft = DRAFTS.get(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found.")
    if draft["report_status"] == "FINAL":
        raise HTTPException(409, "Finalized drafts cannot be edited.")
    if body.draft_report is not None:
        draft["draft_report"] = body.draft_report
    if body.findings is not None:
        draft["findings"] = body.findings
    draft["clinician_reviewed"] = False
    draft["report_status"] = "DRAFT_REQUIRES_CLINICIAN_REVIEW"
    return draft


@app.post("/drafts/{draft_id}/finalize")
def finalize_draft(draft_id: str, body: FinalizeIn):
    draft = DRAFTS.get(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found.")
    if not body.clinician_reviewed:
        raise HTTPException(400, "Clinician review is required before finalization.")
    draft["clinician_reviewed"] = True
    draft["report_status"] = "FINAL"
    return draft


@app.delete("/drafts/{draft_id}")
def discard_draft(draft_id: str):
    if draft_id not in DRAFTS:
        raise HTTPException(404, "Draft not found.")
    del DRAFTS[draft_id]
    return {"status": "discarded", "draft_id": draft_id}


@app.post("/transcribe", dependencies=[Depends(require_key)])
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too large (max 10 MB)")
    try:
        return {"transcript": stt.transcribe(data, file.filename or "audio.wav")}
    except stt.STTNotConfigured as e:
        raise HTTPException(501, str(e))
    except Exception:
        raise HTTPException(502, "STT provider failed; send a typed transcript instead.")
