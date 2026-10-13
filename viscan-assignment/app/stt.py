"""Speech-to-text boundary. Voice input = audio -> transcript -> same pipeline as typed text."""
from __future__ import annotations
import io, os


class STTNotConfigured(RuntimeError):
    pass


def transcribe(audio: bytes, filename: str = "audio.wav") -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise STTNotConfigured("Set OPENAI_API_KEY to enable Whisper STT, or send a typed transcript instead.")
    from openai import OpenAI  # optional dependency
    f = io.BytesIO(audio); f.name = filename
    return OpenAI().audio.transcriptions.create(model=os.getenv("STT_MODEL", "whisper-1"), file=f).text
