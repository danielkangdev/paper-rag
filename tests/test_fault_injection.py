"""Each dependency is made to fail; the request must still succeed and say how it degraded."""

import sqlite3

import pytest

from scirag import runtime
from scirag.rag import qa
from scirag.retrieval import modes
from scirag.storage.sqlite import SqliteStore

from .conftest import fake_chat

Q = "How does the retrieve token trigger retrieval?"


def _boom(*_a, **_k):
    raise RuntimeError("injected failure")


def test_qdrant_down_uses_fts5_only(corpus, monkeypatch):
    monkeypatch.setattr(runtime, "vector_store", _boom)
    res = modes.search(Q, mode="hybrid_rerank", top_k=5)
    assert res.hits, "sparse results should still come back"
    assert "vector_store_unavailable" in res.degraded
    assert all(h.score_dense is None and h.score_sparse is not None for h in res.hits)
    ans = qa.answer(Q, chat_fn=fake_chat)
    assert not ans["abstained"] and "vector_store_unavailable" in ans["degraded"]


def test_embedding_load_failure_uses_fts5_only(corpus, monkeypatch):
    monkeypatch.setattr(runtime, "embedder", _boom)
    res = modes.search(Q, mode="hybrid", top_k=5)
    assert res.hits and "embedding_unavailable" in res.degraded
    assert all(h.score_sparse is not None for h in res.hits)


def test_fts5_error_retries_with_sanitized_query(corpus, monkeypatch):
    real = SqliteStore.fts_match
    calls = []

    def flaky(self, q, k):
        calls.append(q)
        if len(calls) == 1:
            raise sqlite3.OperationalError("fts5: syntax error")
        return real(self, q, k)

    monkeypatch.setattr(SqliteStore, "fts_match", flaky)
    res = modes.search(Q, mode="sparse", top_k=5)
    assert res.hits and res.degraded == ["sparse_query_retried"]
    assert '"' not in calls[1]  # second attempt is plain barewords


def test_fts5_down_falls_back_to_dense_only(corpus, monkeypatch):
    def always_fail(self, q, k):
        raise sqlite3.OperationalError("database disk image is malformed")

    monkeypatch.setattr(SqliteStore, "fts_match", always_fail)
    res = modes.search(Q, mode="hybrid_rerank", top_k=5)
    assert res.hits
    assert "sparse_unavailable" in res.degraded
    assert all(h.score_sparse is None and h.score_dense is not None for h in res.hits)


def test_reranker_failure_keeps_rrf_order(corpus, monkeypatch):
    expected = [h.chunk_id for h in modes.search(Q, mode="hybrid", top_k=5).hits]
    monkeypatch.setattr(runtime, "reranker", _boom)
    res = modes.search(Q, mode="hybrid_rerank", top_k=5)
    assert res.degraded == ["rerank_unavailable"]
    assert [h.chunk_id for h in res.hits] == expected
    assert all(h.score_rerank is None for h in res.hits)


@pytest.mark.parametrize("exc", [TimeoutError("timed out"), RuntimeError("500 from provider")])
def test_llm_failure_returns_snippets_with_citations(corpus, exc):
    def failing_chat(_messages):
        raise exc

    ans = qa.answer(Q, chat_fn=failing_chat)
    assert "llm_unavailable" in ans["degraded"]
    assert not ans["abstained"]
    assert ans["citations"] and all(c["label"].startswith("[") for c in ans["citations"])
    assert ans["citations"][0]["label"] in ans["answer"]


def test_llm_retries_then_raises(settings, monkeypatch):
    from scirag.rag import llm

    settings.llm.api_key, settings.llm.chat_model, settings.llm.max_retries = "k", "m", 2
    attempts = []

    class FakeCompletions:
        def create(self, **_kw):
            attempts.append(1)
            raise TimeoutError("timeout")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "_client", lambda: FakeClient())
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    with pytest.raises(llm.LLMUnavailable):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 3


def test_both_retrievers_empty_abstains_without_llm(corpus, monkeypatch):
    monkeypatch.setattr(runtime, "vector_store", _boom)

    def no_rows(self, q, k):
        return []

    monkeypatch.setattr(SqliteStore, "fts_match", no_rows)
    called = []
    ans = qa.answer(Q, chat_fn=lambda m: called.append(m) or "x")
    assert ans["abstained"] and not ans["llm_called"] and not called
    assert "no_results" in ans["degraded"]
    assert ans["decision"]["decision"] == "no_evidence"
