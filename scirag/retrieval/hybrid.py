"""Hybrid retrieval: dense + FTS5 fused with RRF, optional per-paper cap.

Degradation is implicit in the fusion: if one side fails (and reports a flag)
RRF simply fuses the remaining list.
"""

from __future__ import annotations

from .. import runtime
from ..chunk.models import Hit
from . import dense, sparse
from .rrf import rrf_fuse


def hydrate(scored: list[tuple[str, float]], field: str) -> list[Hit]:
    """Turn (chunk_id, score) pairs into Hits using SQLite, preserving order."""
    chunks = runtime.sqlite_store().get_chunks([cid for cid, _ in scored])
    hits = []
    for cid, score in scored:
        c = chunks.get(cid)
        if c is None:  # stale vector point for a deleted paper
            continue
        hits.append(
            Hit(
                chunk_id=c.chunk_id, paper_id=c.paper_id, title=c.title, section=c.section,
                page=c.page, chunk_index=c.chunk_index, text=c.text, **{field: score},
            )
        )
    return hits


def diversify_by_paper(hits: list[Hit], max_per_paper: int) -> list[Hit]:
    """Stable re-order: at most ``max_per_paper`` hits per paper first, overflow after."""
    if max_per_paper <= 0:
        return hits
    head, tail, counts = [], [], {}
    for h in hits:
        n = counts.get(h.paper_id, 0)
        (head if n < max_per_paper else tail).append(h)
        counts[h.paper_id] = n + 1
    return head + tail


def hybrid_candidates(
    query: str, *, top_k_dense: int, top_k_sparse: int, rrf_k: int
) -> tuple[list[Hit], list[str]]:
    dense_hits, f1 = dense.search(query, top_k_dense)
    sparse_hits, f2 = sparse.search(query, top_k_sparse)
    fused = rrf_fuse([[c for c, _ in dense_hits], [c for c, _ in sparse_hits]], k=rrf_k)
    hits = hydrate(fused, "score_rrf")
    dmap, smap = dict(dense_hits), dict(sparse_hits)
    for h in hits:
        h.score_dense = dmap.get(h.chunk_id)
        h.score_sparse = smap.get(h.chunk_id)
    return hits, f1 + f2
