"""Dense retrieval: embed the query, nearest neighbours from Qdrant.

Never raises: failures return ``[]`` plus a degradation flag so hybrid
retrieval can continue on FTS5 alone.
"""

from __future__ import annotations

import logging

from .. import runtime

log = logging.getLogger("scirag.dense")


def search(query: str, top_k: int) -> tuple[list[tuple[str, float]], list[str]]:
    try:
        qvec = runtime.embedder().encode_query(query)
    except Exception as exc:
        log.warning("embedding unavailable: %s", exc)
        return [], ["embedding_unavailable"]
    try:
        return runtime.vector_store().search(qvec, top_k), []
    except Exception as exc:
        log.warning("vector store unavailable: %s", exc)
        return [], ["vector_store_unavailable"]
