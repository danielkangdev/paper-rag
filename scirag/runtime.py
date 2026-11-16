"""Process-wide lazily built backends (store, vector index, models).

Everything heavy is constructed on first use so that importing scirag is cheap
and tests can swap settings with ``config.set_settings`` + ``reset()``.
"""

from __future__ import annotations

import atexit
import threading
from typing import Any

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _get(name: str, factory):
    obj = _CACHE.get(name)
    if obj is not None:
        return obj
    with _LOCK:
        obj = _CACHE.get(name)
        if obj is None:
            obj = factory()
            _CACHE[name] = obj
    return obj


def sqlite_store():
    from .config import get_settings
    from .storage.sqlite import SqliteStore

    return _get("sqlite", lambda: SqliteStore(get_settings().resolve(get_settings().paths.sqlite_path)))


def vector_store():
    from .config import get_settings
    from .storage.qdrant import QdrantStore

    s = get_settings()
    return _get(
        "qdrant",
        lambda: QdrantStore(
            url=s.qdrant.url,
            path=str(s.resolve(s.qdrant.path)),
            collection=s.qdrant.collection,
            dim=s.embedding.dim,
        ),
    )


def embedder():
    from .retrieval.embedder import build_embedder

    return _get("embedder", build_embedder)


def reranker():
    from .retrieval.reranker import build_reranker

    return _get("reranker", build_reranker)


def _close_at_exit() -> None:
    reset()


atexit.register(_close_at_exit)


def reset() -> None:
    with _LOCK:
        vs = _CACHE.get("qdrant")
        if vs is not None:
            try:
                vs.close()
            except Exception:
                pass
        _CACHE.clear()
