"""Typed configuration.

Load order (later wins):
  1. ``config/default.yaml`` (or the file named by ``SCIRAG_CONFIG``)
  2. ``$VAR`` placeholders expanded from the environment (``.env`` is loaded first)
  3. ``SCIRAG__<SECTION>__<KEY>`` environment overrides
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class Paths(BaseModel):
    data_dir: str = "./data"
    pdf_dir: str = "./data/pdfs"
    sqlite_path: str = "./data/scirag.sqlite"


class Embedding(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    batch_size: int = 32
    query_instruction: str = ""


class Reranker(BaseModel):
    enabled: bool = True
    provider: str = "cross-encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 30


class Qdrant(BaseModel):
    url: str | None = None
    path: str = "./data/qdrant"
    collection: str = "chunks"


class Chunk(BaseModel):
    target_tokens: int = 500
    overlap_tokens: int = 50
    encoding: str = "cl100k_base"


class Retrieve(BaseModel):
    top_k_dense: int = 30
    top_k_sparse: int = 30
    rrf_k: int = 60
    max_per_paper: int = 0


class Abstain(BaseModel):
    enabled: bool = True
    threshold_low: float = 0.05
    threshold_high: float = 0.3
    dense_threshold_low: float = 0.55
    dense_threshold_high: float = 0.65
    aggregate: str = "max"  # max | mean_top3
    message: str = "I could not find evidence for this question in the indexed papers."


class QA(BaseModel):
    max_evidence: int = 5
    max_per_paper: int = 3


class LLM(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    chat_model: str | None = None
    timeout_sec: float = 60
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 700


class Security(BaseModel):
    allow_writes: bool = False


class Settings(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    embedding: Embedding = Field(default_factory=Embedding)
    reranker: Reranker = Field(default_factory=Reranker)
    qdrant: Qdrant = Field(default_factory=Qdrant)
    chunk: Chunk = Field(default_factory=Chunk)
    retrieve: Retrieve = Field(default_factory=Retrieve)
    abstain: Abstain = Field(default_factory=Abstain)
    qa: QA = Field(default_factory=QA)
    llm: LLM = Field(default_factory=LLM)
    security: Security = Field(default_factory=Security)

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return os.environ.get(value[1:]) or None
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    for key, val in os.environ.items():
        if not key.startswith("SCIRAG__"):
            continue
        parts = key[len("SCIRAG__"):].lower().split("__")
        if len(parts) != 2:
            continue
        section, field = parts
        raw.setdefault(section, {})[field] = yaml.safe_load(val) if val != "" else None
    return raw


def load_settings(path: str | Path | None = None) -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        pass
    cfg_path = Path(path or os.environ.get("SCIRAG_CONFIG") or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = _apply_env_overrides(_expand_env(raw))
    return Settings.model_validate(raw)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def set_settings(settings: Settings | None) -> None:
    """Replace the process-wide settings (tests) and drop cached backends."""
    global _SETTINGS
    _SETTINGS = settings
    from . import runtime

    runtime.reset()
