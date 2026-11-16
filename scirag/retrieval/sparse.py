"""Sparse retrieval with SQLite FTS5 (porter stemming, bm25 ranking).

The user query is turned into an OR of quoted terms. If FTS5 still rejects it,
the query is reduced to plain ASCII alphanumeric barewords and retried once;
if that fails too, sparse returns ``[]`` and hybrid continues dense-only.
"""

from __future__ import annotations

import logging
import re

from .. import runtime

log = logging.getLogger("scirag.sparse")

_STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from how in into is it its of on or "
    "should that the their this to was were what when where which who why with would".split()
)
_FTS_KEYWORDS = frozenset({"and", "or", "not", "near"})


def build_match_query(query: str) -> str:
    terms = []
    for raw in query.split():
        term = re.sub(r"[^\w\-]", "", raw).strip("-_")
        if term and term.lower() not in _STOPWORDS:
            terms.append('"' + term.replace('"', "") + '"')
    return " OR ".join(dict.fromkeys(terms))


def sanitize_query(query: str) -> str:
    """Strict fallback: lowercase ASCII alphanumeric barewords only."""
    terms = [
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) > 1 and t not in _STOPWORDS and t not in _FTS_KEYWORDS
    ]
    return " OR ".join(dict.fromkeys(terms))


def search(query: str, top_k: int) -> tuple[list[tuple[str, float]], list[str]]:
    store = runtime.sqlite_store()
    q = build_match_query(query)
    if q:
        try:
            return store.fts_match(q, top_k), []
        except Exception as exc:
            log.warning("FTS5 query failed (%s); retrying with sanitized query", exc)
    flags = ["sparse_query_retried"] if q else []
    q2 = sanitize_query(query)
    if not q2:
        return [], flags
    try:
        return store.fts_match(q2, top_k), flags
    except Exception as exc:
        log.warning("FTS5 unavailable: %s", exc)
        return [], [*flags, "sparse_unavailable"]
