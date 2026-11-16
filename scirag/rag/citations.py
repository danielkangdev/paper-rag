"""Citation parsing, validation and rendering.

The LLM cites evidence as ``[E1]``..``[En]``. Validation:
  1. removes tags that point outside the evidence list (hallucinated ids),
  2. strips ungrounded citation styles (``[3]``, ``(Lewis et al., 2020)``),
  3. reports sentences that make a claim without any citation.
Rendering turns ``[E2]`` into ``[Paper title, §Section, p.N]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chunk.models import Hit

_TAG_RE = re.compile(r"\[E(\d+)\]")
_NUM_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]")
_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*[A-Z][A-Za-zÀ-ſ.\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z\-]+))?,?\s*(?:19|20)\d{2}[a-z]?\s*\)"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def cited_indices(answer: str) -> list[int]:
    """1-based evidence indices cited in ``answer``, in order of first appearance."""
    return list(dict.fromkeys(int(m) for m in _TAG_RE.findall(answer)))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


@dataclass
class CitationCheck:
    text: str
    valid: list[int] = field(default_factory=list)
    invalid: list[int] = field(default_factory=list)
    stripped_styles: int = 0
    uncited_sentences: list[str] = field(default_factory=list)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def validate(answer: str, n_evidence: int) -> CitationCheck:
    found = cited_indices(answer)
    valid = [i for i in found if 1 <= i <= n_evidence]
    invalid = [i for i in found if not 1 <= i <= n_evidence]

    text = _TAG_RE.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_evidence else "", answer)
    n_styles = len(_NUM_CITE_RE.findall(text)) + len(_AUTHOR_YEAR_RE.findall(text))
    text = _AUTHOR_YEAR_RE.sub("", _NUM_CITE_RE.sub("", text))
    text = _tidy(text)

    uncited = [s for s in split_sentences(text) if not _TAG_RE.search(s) and _is_claim(s)]
    return CitationCheck(text=text, valid=valid, invalid=invalid, stripped_styles=n_styles,
                         uncited_sentences=uncited)


_REFUSAL_RE = re.compile(
    r"\b(do(es)? not (contain|provide|include|mention|say|specify|discuss|address)|no information|"
    r"cannot (be )?answer|insufficient (evidence|information)|not mentioned|not specified|"
    r"not (found|available) in the (indexed )?(papers|evidence|passages))\b",
    re.IGNORECASE,
)


_SOURCE_WORD_RE = re.compile(r"\b(papers?|evidence|passages?|context|sources?|answer)\b", re.IGNORECASE)


def is_refusal_sentence(sentence: str) -> bool:
    """A sentence that only says the evidence lacks the answer (no factual claim)."""
    return (
        bool(_REFUSAL_RE.search(sentence))
        and bool(_SOURCE_WORD_RE.search(sentence))
        and not _TAG_RE.search(sentence)
    )


def _is_claim(sentence: str) -> bool:
    return len(sentence.split()) >= 4 and not is_refusal_sentence(sentence)


def render(answer: str, evidence: list[Hit]) -> str:
    def _sub(m: re.Match) -> str:
        i = int(m.group(1))
        return evidence[i - 1].cite_label() if 1 <= i <= len(evidence) else m.group(0)

    rendered = _TAG_RE.sub(_sub, answer)
    # Collapse identical adjacent labels: "[A][A]" -> "[A]"
    return re.sub(r"(\[[^\[\]]+\])(\s*\1)+", r"\1", rendered)
