"""Text embedders.

* ``SentenceTransformerEmbedder`` — real model (default ``BAAI/bge-small-en-v1.5``, CPU-friendly).
* ``HashEmbedder`` — deterministic hashed bag-of-words vectors for tests/CI: no
  downloads, but similar texts still get similar vectors.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [self.query_instruction + text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return vec.tolist()


def build_embedder():
    from ..config import get_settings

    e = get_settings().embedding
    if e.provider == "hash":
        return HashEmbedder(e.dim)
    if e.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(e.model_name, e.batch_size, e.query_instruction)
    raise ValueError(f"unknown embedding provider: {e.provider}")
