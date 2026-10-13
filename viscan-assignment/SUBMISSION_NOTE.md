Subject: AI Engineer take-home: dental voice assistant PoC

Hi Hiring Team,

Repository: https://github.com/Pandeyashwin26/viscan-ai-dental-assistant
Demo video: to be added by candidate. A documented offline demo is available with `python demo.py`; generated outputs are in `docs/sample_outputs.md`.

**What works:** report retrieval from typed/spoken requests with exact/ambiguous/fuzzy/missing handling, previous-report and direct report-ID lookup (incl. similar names Priya Nair/Nayar,
Rahul Mehta/Mehra); dictation to structured, editable JSON + draft lifecycle with uncertainty labels and mandatory clinician review; "Summarize her report"
follow-up using session context, with cited sources; grounded report questions for findings, impression, date, clinician, anatomical sites, uncertainty, missing information, and previous-report comparison; confirmation-gated synthetic scan booking with conflict handling; LLM output validated by deterministic rules (21 automated tests, including ambiguity, unsupported claims, grounding, booking, and draft lifecycle); optional Claude tool-calling and Whisper STT.

**Incomplete:** rule-based extractor has a limited vocabulary; in-memory sessions; no real-audio accuracy testing.

**With more time:** phonetic/Indic name matching, streaming STT with a dental-term lexicon, an eval set of labelled dictations to measure hallucination and recall, persistent sessions with per-clinician auth and audit logs, and LLM-written summaries with claim-level grounding checks.

Regards,
Ashwin Pandey
