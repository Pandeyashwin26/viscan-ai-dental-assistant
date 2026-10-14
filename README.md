# Voice-first dental assistant: proof of concept (synthetic data only)

Retrieves patient reports from spoken/typed requests, turns clinical dictation into an
editable structured draft, and answers context-dependent follow-ups with cited sources.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q              # 23 tests, fully offline
python demo.py                   # scripted demo; writes docs/sample_outputs.md
uvicorn app.main:app --reload    # API at http://127.0.0.1:8000/docs
python -m app.cli                # interactive CLI
```
Optional: `cp .env.example .env` and export `ANTHROPIC_API_KEY` to use a real LLM (tool calling)
and `OPENAI_API_KEY` for Whisper STT. **With no keys, everything still runs** using the offline rule-based adapter.

## Render deployment
This repository includes [`render.yaml`](./render.yaml) with the build command, Uvicorn start
command, and `/health` health check. In Render, connect the GitHub repository and deploy the
blueprint. No provider key is required for the offline demo; add `API_KEY` and optionally
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as Render environment variables if needed. After deployment,
verify `https://<your-service>.onrender.com/health` and `/docs`; a public URL is not included in this
repository and deployment must still be performed in the Render dashboard.

## API examples
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"Open Priya Nair'"'"'s latest OPG report"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"Summarize her report"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"What are the findings in her report?"}'
curl -s localhost:8000/dictate -H 'content-type: application/json' \
  -d '{"transcript":"Lower right third molar is horizontally impacted"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"booking","text":"Book an OPG scan for Priya Nair on 2026-10-15"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"booking","text":"Book the 11 AM slot"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"booking","text":"Confirm booking"}'
# draft lifecycle: GET/PUT /drafts/{draft_id}, POST /drafts/{draft_id}/finalize, DELETE /drafts/{draft_id}
# ambiguity: first re-send with the chosen patient_id (original request is replayed)
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo2","text":"Open Priya'"'"'s latest OPG"}'
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"demo2","text":"","patient_id":"P002"}'
```

## Provider choices
| Layer | Choice | Why |
|---|---|---|
| LLM | Anthropic Claude via tool calling (`LLM_MODEL`, default `claude-sonnet-5-5`); offline rule-based fallback | Forced tool-use gives schema-constrained output; fallback keeps tests and demo deterministic and free |
| STT | OpenAI Whisper (`/transcribe`), optional | Audio becomes a transcript, then follows the identical text pipeline. Typed transcript is accepted, per the brief |
| Backend | FastAPI | Typed request validation, auto docs |
| Bonus | Grounded report-aware chatbot | Answers only supported questions from the selected report and cites source fields |
| Bonus | Synthetic scan booking | Lists slots, requires explicit confirmation, rechecks availability, and prevents conflicts |

## Design in one paragraph
The LLM only **proposes** (intent/entities, candidate findings). `validators.py` then enforces closed
schemas; `store.py` does identity resolution and retrieval deterministically. Findings must be in a
closed vocabulary, quote the clinician verbatim as evidence, and have a valid FDI site stated in that
evidence, or they are rejected and listed under `rejected_claims`. See `ARCHITECTURE.md`.

## Production considerations
- **Privacy:** synthetic data only here. In production: PHI never sent to a provider without a BAA/DPA or zero-retention; redact names before LLM calls (the LLM only needs intent and a name string, never report bodies); encrypt at rest/in transit; do not log transcripts or report text; audio is not stored.
- **Authorization:** demo uses an optional `X-API-Key`. Production needs per-clinician auth (OIDC), tenant/clinic scoping on every query, patient-level access checks, and an audit trail of who opened which report.
- **Latency:** retrieval and validation are local (<10 ms). The LLM call is the only slow step (~0.5-2 s), with a 15 s timeout then rule-based fallback. Whisper adds ~1-3 s per short clip; use streaming STT for live dictation.
- **Cost:** one small tool-call per utterance (about 300-600 tokens in/out), well under a cent per request. Report summaries are deterministic (no tokens). Cache identical interpretations; use the cheapest model that passes the eval set.
- **Failure handling:** 1500/4000-char input caps, control-char stripping, `patient_id` format check, 10 MB audio cap, LLM/STT timeouts with fallback or clear 501/502, ambiguous/missing/unknown cases return explicit statuses instead of guessing.
- **Grounded chatbot:** report questions are limited to findings, impression, date, and clinician. Unsupported treatment or diagnostic questions return `unsupported_question` or `unknown_intent`; no new clinical claims are generated.
- **Booking:** `data/appointments.json` contains synthetic slots. Booking is two-step (`booking_confirmation_required` then `confirm booking`) and uses a thread lock for the commit/recheck boundary.

## Known limitations
- Rule-based extractor covers a limited vocabulary (15 finding types, FDI sites by name or number); free-form speech beyond that lands in `unparsed_statements` for the clinician, by design.
- Sessions are in-memory (single process); no persistence or multi-tenant isolation.
- Name matching is edit-distance based, not phonetic; no Indic-name or accent handling; no handling of two patients with identical full names beyond returning both as ambiguous.
- Summaries are extractive templates, not LLM prose; chatbot questions are intentionally limited to grounded report fields.
- Drafts and bookings are in-memory at runtime; restart loses new bookings/drafts. The JSON appointment file is a reproducible seed, not a production database.
- No evaluation set beyond unit tests; no real STT quality testing on dental terminology.
