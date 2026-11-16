"""The four retrieval modes used by the CLI, API, MCP tools, QA and ablation.

    dense          bge-small embeddings in Qdrant
    sparse         SQLite FTS5 / bm25
    hybrid         dense + sparse fused with RRF, per-paper cap
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder, per-paper cap
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .. import runtime
from ..chunk.models import Hit
from ..config import get_settings
from . import dense, sparse
from .hybrid import diversify_by_paper, hybrid_candidates, hydrate

log = logging.getLogger("scirag.modes")

MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    degraded: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "degraded": self.degraded,
            "latency_ms": round(self.latency_ms, 1),
            "hits": [h.model_dump() for h in self.hits],
        }


def _rerank(query: str, hits: list[Hit]) -> tuple[list[Hit], list[str]]:
    if not hits:
        return hits, []
    try:
        scores = runtime.reranker().score(query, [h.text for h in hits])
    except Exception as exc:
        log.warning("reranker unavailable, keeping RRF order: %s", exc)
        return hits, ["rerank_unavailable"]
    for h, s in zip(hits, scores):
        h.score_rerank = float(s)
    return sorted(hits, key=lambda h: -(h.score_rerank or 0.0)), []


def search(query: str, mode: str = "hybrid_rerank", top_k: int = 10) -> SearchResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    s = get_settings()
    r = s.retrieve
    t0 = time.perf_counter()
    degraded: list[str] = []

    if mode == "dense":
        scored, degraded = dense.search(query, top_k)
        hits = hydrate(scored, "score_dense")
    elif mode == "sparse":
        scored, degraded = sparse.search(query, top_k)
        hits = hydrate(scored, "score_sparse")
    else:
        hits, degraded = hybrid_candidates(
            query, top_k_dense=r.top_k_dense, top_k_sparse=r.top_k_sparse, rrf_k=r.rrf_k
        )
        if mode == "hybrid_rerank":
            if s.reranker.enabled:
                head = hits[: s.reranker.candidates]
                reranked, flags = _rerank(query, head)
                hits = reranked + hits[s.reranker.candidates :]
                degraded += flags
            else:
                degraded.append("rerank_disabled")
        hits = diversify_by_paper(hits, r.max_per_paper)

    hits = hits[:top_k]
    return SearchResult(
        query=query, mode=mode, hits=hits, degraded=list(dict.fromkeys(degraded)),
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
