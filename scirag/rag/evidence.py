"""Deterministic selection of the passages the LLM is allowed to cite.

Model score (reranker, else RRF) carries most of the weight, lexical overlap
with the question breaks ties, and a per-paper cap keeps one paper from
crowding out the rest.
"""

from __future__ import annotations

import re

from ..chunk.models import Hit

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from how in into is it of on or "
    "should that the this to use used uses using what when where which who why with would".split()
)


def _tokens(text: str) -> set[str]:
    toks = set(_TOKEN_RE.findall(text.lower()))
    return (toks - _STOPWORDS) or toks


def _model_score(h: Hit) -> float:
    for v in (h.score_rerank, h.score_rrf, h.score_dense, h.score_sparse):
        if v is not None:
            return float(v)
    return 0.0


def select_evidence(question: str, hits: list[Hit], *, max_chunks: int = 5,
                    max_per_paper: int = 3) -> list[Hit]:
    q = _tokens(question)
    scored = []
    for rank, h in enumerate(hits):
        overlap = len(q & _tokens(f"{h.title} {h.section} {h.text}")) / len(q) if q else 0.0
        scored.append((_model_score(h) + 0.2 * overlap, -rank, h))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected: list[Hit] = []
    counts: dict[str, int] = {}
    for _, _, h in scored:
        if counts.get(h.paper_id, 0) >= max_per_paper:
            continue
        selected.append(h)
        counts[h.paper_id] = counts.get(h.paper_id, 0) + 1
        if len(selected) >= max_chunks:
            break
    return selected
