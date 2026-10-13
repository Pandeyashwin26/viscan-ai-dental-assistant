"""FastAPI demo API.  Run: uvicorn app.main:app --reload"""
from __future__ import annotations
import logging, os
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .assistant import Assistant, Session
from .llm import get_llm
from .store import Store
from . import stt

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Viscan voice-first dental assistant (synthetic data PoC)")
assistant = Assistant(Store(), get_llm())
SESSIONS: dict[str, Session] = {}
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


@app.get("/health")
def health():
    return {"ok": True, "llm": assistant.llm.name}


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(body: ChatIn):
    return assistant.handle(SESSIONS.setdefault(body.session_id, Session()), body.text, body.patient_id)


@app.post("/dictate", dependencies=[Depends(require_key)])
def dictate(body: DictateIn):
    return assistant.dictate(SESSIONS.setdefault(body.session_id, Session()), body.transcript, body.patient_id)


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
