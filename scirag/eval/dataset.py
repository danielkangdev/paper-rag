"""Golden set format and the evidence-matching rule.

Each question is labelled with (paper, verbatim quote) pairs instead of chunk
ids, so the labels survive any change to chunking parameters:

    {"qid": "g001", "question": "...", "answerable": true,
     "evidence": [{"paper_id": "arxiv:2310.11511", "quote": "reflection tokens"}]}

A retrieved chunk hits an evidence item when it belongs to that paper and its
normalized text contains the normalized quote (lowercase, whitespace collapsed,
end-of-line hyphenation removed).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden.jsonl"

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_WS_RE = re.compile(r"\s+")


class Evidence(BaseModel):
    paper_id: str
    quote: str


class GoldenItem(BaseModel):
    qid: str
    question: str
    answerable: bool
    evidence: list[Evidence] = Field(default_factory=list)
    gold_answer: str | None = None
    category: str | None = None


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("­", "")  # soft hyphen
    return _WS_RE.sub(" ", text).strip().lower()


def chunk_hits(paper_id: str, text: str, ev: Evidence) -> bool:
    return paper_id == ev.paper_id and normalize(ev.quote) in normalize(text)


def evidence_hit_vector(paper_id: str, text: str, evidence: list[Evidence]) -> list[bool]:
    norm = normalize(text)
    return [paper_id == e.paper_id and normalize(e.quote) in norm for e in evidence]


def load_golden(path: str | Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldenItem.model_validate(json.loads(line)))
    return items


def save_golden(items: list[GoldenItem], path: str | Path = GOLDEN_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
