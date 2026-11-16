"""Reciprocal Rank Fusion (pure function)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Duplicate ids within one list only count at their best rank. Ties are
    broken by first appearance, so the result is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    order: dict[str, int] = {}
    for lst in ranked_lists:
        seen: set[str] = set()
        for rank, doc_id in enumerate(lst, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
            order.setdefault(doc_id, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
