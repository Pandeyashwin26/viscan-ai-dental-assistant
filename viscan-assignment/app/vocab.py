"""Closed clinical vocabulary used for deterministic validation.

Anything the LLM (or the rule-based extractor) proposes must map onto this
vocabulary AND be backed by a span of the clinician's own words.
"""
from __future__ import annotations
import re

QUADRANT = {("upper", "right"): 1, ("upper", "left"): 2, ("lower", "left"): 3, ("lower", "right"): 4}
JAW_ALIAS = {"maxillary": "upper", "mandibular": "lower"}
TOOTH_NUM = {
    "central incisor": 1, "lateral incisor": 2, "canine": 3, "first premolar": 4,
    "second premolar": 5, "first molar": 6, "second molar": 7, "third molar": 8,
}
_NUM_TO_NAME = {v: k for k, v in TOOTH_NUM.items()}
_QUAD_TO_NAME = {v: k for k, v in QUADRANT.items()}

SITE_RE = re.compile(
    r"\b(upper|lower|maxillary|mandibular)\s+(right|left)\s+"
    r"(central incisor|lateral incisor|canine|first premolar|second premolar|first molar|second molar|third molar)s?\b",
    re.I,
)
FDI_RE = re.compile(r"(?:\btooth\s+(?:number\s+)?|\bteeth\s+|#\s*)([1-4][1-8])\b", re.I)


def describe_fdi(fdi: str) -> str:
    q, n = int(fdi[0]), int(fdi[1])
    jaw, side = _QUAD_TO_NAME[q]
    return f"{jaw} {side} {_NUM_TO_NAME[n]}"


def find_sites(text: str) -> list[dict]:
    sites: list[dict] = []
    for m in SITE_RE.finditer(text):
        jaw = JAW_ALIAS.get(m.group(1).lower(), m.group(1).lower())
        fdi = f"{QUADRANT[(jaw, m.group(2).lower())]}{TOOTH_NUM[m.group(3).lower()]}"
        sites.append({"fdi": fdi, "description": describe_fdi(fdi)})
    for m in FDI_RE.finditer(text):
        fdi = m.group(1)
        sites.append({"fdi": fdi, "description": describe_fdi(fdi)})
    seen, out = set(), []
    for s in sites:
        if s["fdi"] not in seen:
            seen.add(s["fdi"]); out.append(s)
    return out


# Ordered: specific patterns first so generic "impacted" does not double count.
FINDING_VOCAB: dict[str, list[re.Pattern]] = {
    k: [re.compile(p, re.I) for p in v]
    for k, v in {
        "horizontal impaction": [r"horizontally impacted", r"horizontal impaction"],
        "mesioangular impaction": [r"mesioangularly impacted", r"mesioangular impaction"],
        "distoangular impaction": [r"distoangularly impacted", r"distoangular impaction"],
        "vertical impaction": [r"vertically impacted", r"vertical impaction"],
        "impaction (orientation not stated)": [r"\bimpacted\b", r"\bimpaction\b"],
        "periapical radiolucency": [r"periapical radiolucenc\w+", r"periapical lesion"],
        "dental caries": [r"\bcaries\b", r"carious (?:lesion|cavity|tooth)", r"\bdecay\b"],
        "root canal treatment": [r"root canal (?:treated|treatment|filling)", r"endodontically treated"],
        "restoration": [r"\brestoration\b", r"\brestored\b", r"\bfilling\b"],
        "crown": [r"\bcrowned\b", r"\bcrown\b"],
        "bone loss": [r"bone loss", r"bone resorption"],
        "fracture": [r"\bfracture[d]?\b"],
        "retained root": [r"retained root"],
        "implant": [r"\bimplant\b"],
        "missing tooth": [r"\bmissing\b", r"\babsent\b"],
    }.items()
}

UNCERTAIN_RE = re.compile(
    r"(\b(possible|possibly|probable|probably|suspected|suspicious|likely|questionable|"
    r"may be|might be|appears?|seems?|cannot exclude|can't exclude)\b|\?)", re.I)
NEGATION_RE = re.compile(r"\b(no|not|without|negative for|free of|nil)\b", re.I)
POLARITIES = {"present", "absent"}
CERTAINTIES = {"stated", "uncertain"}
REPORT_TYPES = {"OPG", "BITEWING", "CBCT", "IOPA"}
