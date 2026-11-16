"""Cross-encoder reranking.

``CrossEncoderReranker`` wraps a sentence-transformers CrossEncoder (default
``cross-encoder/ms-marco-MiniLM-L-6-v2``; ``BAAI/bge-reranker-base`` also works
but was ~15x slower on CPU) and returns sigmoid scores in [0, 1].
``LexicalReranker`` is a dependency-free fake for tests/CI.
"""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are does do how is in of on or the to what when which why with".split())


class LexicalReranker:
    """Fraction of (non-stopword) query terms present in the passage."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(_TOKEN_RE.findall(query.lower())) - _STOP
        out = []
        for t in texts:
            toks = set(_TOKEN_RE.findall(t.lower()))
            out.append(len(q & toks) / len(q) if q else 0.0)
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=max_length, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        raw = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        vals = [float(x) for x in raw]
        if any(v < 0.0 or v > 1.0 for v in vals):  # logits -> probabilities
            vals = [1.0 / (1.0 + math.exp(-v)) for v in vals]
        return vals


def build_reranker():
    from ..config import get_settings

    r = get_settings().reranker
    if r.provider == "lexical":
        return LexicalReranker()
    if r.provider == "cross-encoder":
        return CrossEncoderReranker(r.model_name)
    raise ValueError(f"unknown reranker provider: {r.provider}")
