"""End to end on two generated PDFs with fake embedder/reranker/LLM."""

import pytest
from fastapi.testclient import TestClient

from scirag import runtime
from scirag.ingest.pipeline import ingest_local_pdf
from scirag.rag import qa
from scirag.retrieval.modes import MODES, search

from .conftest import fake_chat


def test_ingest_writes_both_stores_with_section_and_page(corpus, settings):
    counts = runtime.sqlite_store().counts()
    assert counts["paper"] == 2 and counts["chunk"] > 0
    assert counts["chunk_fts"] == counts["chunk"]
    assert runtime.vector_store().count() == counts["chunk"]
    for c in runtime.sqlite_store().all_chunks():
        assert c.section and c.page >= 1


def test_reingest_is_idempotent(corpus, settings, tmp_path):
    before = runtime.sqlite_store().counts()
    pdf = tmp_path / "pdfs" / "a.pdf"
    assert ingest_local_pdf(pdf)["status"] == "skipped"
    assert ingest_local_pdf(pdf, force=True)["status"] == "indexed"
    assert runtime.sqlite_store().counts() == before
    assert runtime.vector_store().count() == before["chunk"]


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_return_results(corpus, mode):
    res = search("BM25 term frequency saturation", mode=mode, top_k=3)
    assert res.hits and not res.degraded
    assert res.hits[0].paper_id == corpus[1]


def test_ask_answers_with_rendered_citations(corpus):
    ans = qa.answer("How does the retrieve token trigger retrieval?", chat_fn=fake_chat)
    assert not ans["abstained"] and ans["llm_called"]
    assert ans["citations"] and ans["citations"][0]["chunk_id"]
    assert "[Reflective Retrieval for Language Models, §" in ans["answer"]
    assert "[E1]" not in ans["answer"]


def test_unanswerable_question_abstains_without_llm(corpus):
    called = []
    ans = qa.answer("What is the stock price of NVIDIA tomorrow?", chat_fn=lambda m: called.append(m) or "x")
    assert ans["abstained"] and not called
    assert ans["answer"] == runtime_message()


def runtime_message():
    from scirag.config import get_settings

    return get_settings().abstain.message


def test_api_endpoints(corpus, settings, monkeypatch):
    from scirag.api.server import app
    from scirag.rag import llm

    monkeypatch.setattr(llm, "chat", lambda messages, **kw: fake_chat(messages))
    client = TestClient(app)
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["components"]["sqlite"]["paper"] == 2
    r = client.post("/search", json={"query": "reflection tokens", "mode": "hybrid", "top_k": 2})
    assert r.status_code == 200 and len(r.json()["hits"]) == 2
    r = client.post("/ask", json={"question": "How does the retrieve token trigger retrieval?"})
    assert r.status_code == 200 and r.json()["citations"]
    assert client.post("/ingest", json={"pdf_path": "a.pdf"}).status_code == 403
    settings.security.allow_writes = True
    r = client.post("/ingest", json={"pdf_path": "a.pdf"})
    assert r.status_code == 200 and r.json()["status"] == "skipped"
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_env_override(monkeypatch):
    from scirag.config import load_settings

    monkeypatch.setenv("SCIRAG__RETRIEVE__RRF_K", "10")
    monkeypatch.setenv("SCIRAG__SECURITY__ALLOW_WRITES", "true")
    s = load_settings()
    assert s.retrieve.rrf_k == 10 and s.security.allow_writes is True


@pytest.mark.slow
def test_real_models_smoke(tmp_path):
    """Real bge-small + MiniLM cross-encoder (downloads models; skipped in CI)."""
    from scirag import config

    from .conftest import PAPER_A, make_pdf, make_settings

    s = make_settings(tmp_path)
    s.embedding.provider, s.embedding.dim = "sentence-transformers", 384
    s.reranker.provider = "cross-encoder"
    config.set_settings(s)
    try:
        (tmp_path / "pdfs").mkdir()
        ingest_local_pdf(make_pdf(tmp_path / "pdfs" / "a.pdf", PAPER_A))
        res = search("what do reflection tokens do", mode="hybrid_rerank", top_k=2)
        assert res.hits and res.hits[0].score_rerank is not None and not res.degraded
    finally:
        config.set_settings(None)
