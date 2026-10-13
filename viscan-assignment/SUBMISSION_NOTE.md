Subject: AI Engineer take-home: dental voice assistant PoC

Hi <name>,

Repo/ZIP: <link>. Demo: <video link> (also `python demo.py`, outputs in docs/sample_outputs.md).

**What works:** report retrieval from typed/spoken requests with exact/ambiguous/fuzzy/missing handling (incl. similar names Priya Nair/Nayar,
Rahul Mehta/Mehra); dictation to structured, editable JSON + draft with uncertainty labels and mandatory clinician review; "Summarize her report"
follow-up using session context, with cited sources; LLM output validated by deterministic rules (14 automated tests, including ambiguity and unsupported claims); optional Claude tool-calling and Whisper STT.

**Incomplete:** bonus feature not built; rule-based extractor has a limited vocabulary; in-memory sessions; no real-audio accuracy testing.

**With more time:** phonetic/Indic name matching, streaming STT with a dental-term lexicon, an eval set of labelled dictations to measure hallucination and recall, persistent sessions with per-clinician auth and audit logs, LLM-written summaries with claim-level grounding checks, and the report-aware chatbot.

Regards,
Ashwin Pandey
