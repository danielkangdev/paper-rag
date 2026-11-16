"""Grounded question answering.

hybrid_rerank retrieval -> abstention decision -> evidence selection -> LLM
-> citation validation -> render ``[Title, §Section, p.N]`` labels.

If the LLM is unavailable the answer degrades to the top evidence passages
with their citations. ``abstain=False`` / ``check_citations=False`` exist for
the baseline configuration in ``eval/run_answer_eval.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..chunk.models import Hit
from ..config import get_settings
from ..retrieval.modes import search
from . import abstention, citations, llm, prompts
from .evidence import select_evidence

ChatFn = Callable[[list[dict]], str]


def _citation_payload(evidence: list[Hit], indices: list[int]) -> list[dict]:
    out = []
    for i in indices:
        if 1 <= i <= len(evidence):
            h = evidence[i - 1]
            out.append({"tag": f"E{i}", "label": h.cite_label(), "chunk_id": h.chunk_id,
                        "paper_id": h.paper_id, "title": h.title, "section": h.section, "page": h.page})
    return out


def _snippet_answer(evidence: list[Hit], max_chars: int = 300) -> str:
    lines = ["The language model is unavailable; here are the most relevant passages:"]
    for h in evidence[:3]:
        text = h.text if len(h.text) <= max_chars else h.text[:max_chars].rsplit(" ", 1)[0] + "…"
        lines.append(f"- {text} {h.cite_label()}")
    return "\n".join(lines)


def answer(
    question: str,
    *,
    top_k: int = 10,
    abstain: bool = True,
    check_citations: bool = True,
    chat_fn: ChatFn | None = None,
) -> dict:
    s = get_settings()
    t0 = time.perf_counter()
    res = search(question, mode="hybrid_rerank", top_k=top_k)
    degraded = list(res.degraded)
    a = s.abstain
    decision = abstention.decide(
        res.hits, enabled=abstain and a.enabled, threshold_low=a.threshold_low,
        threshold_high=a.threshold_high, dense_threshold_low=a.dense_threshold_low,
        dense_threshold_high=a.dense_threshold_high, aggregate=a.aggregate,
    )
    if not res.hits:
        degraded.append("no_results")

    base = {"question": question, "decision": decision.to_dict(), "degraded": degraded,
            "retrieved": [h.chunk_id for h in res.hits]}

    if decision.abstain:
        return {**base, "answer": a.message, "abstained": True, "llm_called": False,
                "citations": [], "evidence": [], "uncited_sentences": [], "invalid_citations": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    evidence = select_evidence(question, res.hits, max_chunks=s.qa.max_evidence,
                               max_per_paper=s.qa.max_per_paper)
    # weak_evidence answers normally but is flagged low-confidence. (A prompt hint
    # telling the model the evidence was weak made it refuse 7/45 answerable questions.)
    base["confidence"] = "low" if decision.decision == abstention.WEAK else "high"
    messages = prompts.build_messages(question, evidence)
    try:
        raw = (chat_fn or llm.chat)(messages)
    except Exception:
        degraded.append("llm_unavailable")
        cites = _citation_payload(evidence, list(range(1, min(3, len(evidence)) + 1)))
        return {**base, "degraded": degraded, "answer": _snippet_answer(evidence), "abstained": False,
                "llm_called": True, "citations": cites,
                "evidence": [h.model_dump() for h in evidence], "uncited_sentences": [],
                "invalid_citations": [], "raw_answer": None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    if check_citations:
        chk = citations.validate(raw, len(evidence))
        text, used, invalid, uncited = chk.text, chk.valid, chk.invalid, chk.uncited_sentences
    else:
        text = raw
        found = citations.cited_indices(raw)
        used = found
        invalid = [i for i in found if not 1 <= i <= len(evidence)]
        uncited = []

    return {
        **base,
        "degraded": degraded,
        "answer": citations.render(text, evidence),
        "answer_tagged": text,
        "raw_answer": raw,
        "abstained": False,
        "llm_called": True,
        "citations": _citation_payload(evidence, used),
        "invalid_citations": invalid,
        "uncited_sentences": uncited,
        "evidence": [h.model_dump() for h in evidence],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
