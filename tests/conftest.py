"""Shared fixtures: an isolated data dir, fake models, and tiny generated PDFs."""

from __future__ import annotations

from pathlib import Path

import pytest

from scirag import config
from scirag.config import Settings

PAPER_A = {
    "title": "Reflective Retrieval for Language Models",
    "sections": [
        ("Abstract", ["We present a reflective retrieval method. The model emits reflection tokens "
                      "that decide whether to retrieve passages and critique its own output."]),
        ("1 Introduction", ["Large language models hallucinate facts. Retrieval augmentation "
                            "grounds generation in external passages from a corpus.",
                            "Our method retrieves on demand rather than for every query."]),
        ("2 Method", ["Reflection tokens are special vocabulary items. The retrieve token is "
                      "predicted before generation and triggers retrieval only when needed.",
                      "A critic model labels training data with relevance and support tokens."]),
        ("3 Experiments", ["We evaluate on open-domain question answering and long-form "
                           "generation. Reflective retrieval improves factual precision by a "
                           "large margin over standard retrieval augmented generation."]),
        ("References", ["[1] A. Author. Some paper. 2020."]),
    ],
}

PAPER_B = {
    "title": "Sparse Lexical Ranking with BM25",
    "sections": [
        ("Abstract", ["BM25 is a lexical ranking function based on term frequency and inverse "
                      "document frequency. It remains a strong baseline for zero-shot retrieval."]),
        ("1 Introduction", ["Sparse retrieval matches exact terms. Dense retrieval learns "
                            "embeddings. Hybrid systems combine both rankings."]),
        ("2 Saturation", ["The k1 parameter controls term frequency saturation and b controls "
                          "document length normalization in BM25 scoring."]),
    ],
}


def make_pdf(path: Path, paper: dict) -> Path:
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 60
    page.insert_textbox(fitz.Rect(50, y, 550, y + 40), paper["title"], fontsize=16)
    y += 50
    for heading, paras in paper["sections"]:
        if y > 650:
            page = doc.new_page()
            y = 60
        page.insert_textbox(fitz.Rect(50, y, 550, y + 30), heading, fontsize=12)
        y += 34
        for para in paras:
            rect = fitz.Rect(50, y, 550, y + 90)
            page.insert_textbox(rect, para, fontsize=10)
            y += 90
    doc.set_metadata({"title": paper["title"]})
    doc.save(str(path))
    doc.close()
    return path


def make_settings(tmp: Path, **overrides) -> Settings:
    s = Settings.model_validate({
        "paths": {"data_dir": str(tmp), "pdf_dir": str(tmp / "pdfs"),
                  "sqlite_path": str(tmp / "scirag.sqlite")},
        "embedding": {"provider": "hash", "dim": 64},
        "reranker": {"provider": "lexical", "candidates": 20},
        "qdrant": {"url": None, "path": str(tmp / "qdrant"), "collection": "test"},
        "abstain": {"threshold_low": 0.2, "threshold_high": 0.4,
                    "dense_threshold_low": 0.05, "dense_threshold_high": 0.1},
        "llm": {"api_key": None, "chat_model": None, "max_retries": 0},
    })
    for dotted, val in overrides.items():
        section, key = dotted.split("__")
        setattr(getattr(s, section), key, val)
    return s


@pytest.fixture
def settings(tmp_path):
    s = make_settings(tmp_path)
    (tmp_path / "pdfs").mkdir()
    config.set_settings(s)
    yield s
    config.set_settings(None)


@pytest.fixture
def corpus(settings, tmp_path):
    """Two ingested papers; returns their paper ids."""
    from scirag.ingest.pipeline import ingest_local_pdf

    pdfs = tmp_path / "pdfs"
    a = ingest_local_pdf(make_pdf(pdfs / "a.pdf", PAPER_A))
    b = ingest_local_pdf(make_pdf(pdfs / "b.pdf", PAPER_B))
    assert a["status"] == b["status"] == "indexed"
    return a["paper_id"], b["paper_id"]


def fake_chat(messages: list[dict]) -> str:
    """Deterministic stand-in LLM: cites the first two evidence passages."""
    return ("The retrieve token is predicted before generation and triggers retrieval [E1]. "
            "A critic model labels the training data [E2].")
