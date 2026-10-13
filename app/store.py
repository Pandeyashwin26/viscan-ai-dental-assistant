"""Deterministic data access + patient identity resolution (no LLM here)."""
from __future__ import annotations
import json, re
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TITLES = {"dr", "mr", "mrs", "ms", "miss", "patient"}


def norm_name(s: str) -> str:
    s = re.sub(r"['’]s\b", "", s.lower())
    return " ".join(t for t in re.sub(r"[^a-z ]", " ", s).split() if t not in _TITLES)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def public_patient(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "dob": p["dob"], "last_visit": p["last_visit"]}


class Store:
    def __init__(self, patients_path: Path | None = None, reports_path: Path | None = None):
        self.patients = json.loads((patients_path or DATA_DIR / "patients.json").read_text())
        self.reports = json.loads((reports_path or DATA_DIR / "reports.json").read_text())

    def get_patient(self, pid: str) -> dict | None:
        return next((p for p in self.patients if p["id"] == pid), None)

    def get_report(self, rid: str) -> dict | None:
        return next((r for r in self.reports if r["id"] == rid), None)

    def find_patients(self, name: str) -> dict:
        """Returns {status: exact|ambiguous|fuzzy|none, candidates, similar}.

        exact     -> exactly one patient matches the spoken name
        ambiguous -> several patients match (e.g. first name only)
        fuzzy     -> no exact match but close matches exist; must confirm
        none      -> nothing close
        """
        q = norm_name(name)
        toks = q.split()
        if not toks:
            return {"status": "none", "candidates": [], "similar": []}
        full = {p["id"]: norm_name(p["name"]) for p in self.patients}
        if len(toks) >= 2:
            exact = [p for p in self.patients if full[p["id"]] == q]
            if len(exact) == 1:
                similar = [p for p in self.patients if p["id"] != exact[0]["id"] and _ratio(full[p["id"]], q) >= 0.85]
                return {"status": "exact", "candidates": exact, "similar": similar}
            fuzzy = [p for p in self.patients if _ratio(full[p["id"]], q) >= 0.8]
        else:
            t = toks[0]
            exact = [p for p in self.patients if t in full[p["id"]].split()]
            if len(exact) == 1:
                return {"status": "exact", "candidates": exact, "similar": []}
            if len(exact) > 1:
                return {"status": "ambiguous", "candidates": exact, "similar": []}
            fuzzy = [p for p in self.patients if any(_ratio(part, t) >= 0.8 for part in full[p["id"]].split())]
        return {"status": "fuzzy" if fuzzy else "none", "candidates": fuzzy, "similar": []}

    def reports_for(self, pid: str, report_type: str | None = None) -> list[dict]:
        rs = [r for r in self.reports if r["patient_id"] == pid and (not report_type or r["type"] == report_type)]
        return sorted(rs, key=lambda r: r["date"], reverse=True)
