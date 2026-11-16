"""Retrieval ablation over the four modes on the answerable golden questions.

    python -m scirag.eval.run_ablation [--golden eval/golden.jsonl] [--k 10]

Writes eval/results/ablation_<date>.json and .md. Reports Recall@5, Recall@10,
MRR, nDCG@10, p50/p95 latency per mode, and a paired bootstrap 95% CI
(1,000 resamples) for the Recall@10 gain of hybrid_rerank over dense.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

from .. import runtime
from ..config import PROJECT_ROOT, get_settings
from ..retrieval.modes import MODES, search
from .dataset import GOLDEN_PATH, evidence_hit_vector, load_golden
from .metrics import mean, mrr, ndcg_at_k, paired_bootstrap_ci, percentile, recall_at_k
from .report import write_ablation_markdown

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def evaluate_mode(mode: str, items, k: int) -> dict:
    search(items[0].question, mode=mode, top_k=k)  # warm-up: model loading is not query latency
    per_q = []
    for it in items:
        res = search(it.question, mode=mode, top_k=k)
        hits = [evidence_hit_vector(h.paper_id, h.text, it.evidence) for h in res.hits]
        n = len(it.evidence)
        per_q.append({
            "qid": it.qid,
            "recall@5": recall_at_k(hits, n, 5),
            "recall@10": recall_at_k(hits, n, 10),
            "mrr": mrr(hits),
            "ndcg@10": ndcg_at_k(hits, n, 10),
            "latency_ms": res.latency_ms,
            "degraded": res.degraded,
            "top_chunks": [h.chunk_id for h in res.hits],
        })
    lat = [q["latency_ms"] for q in per_q]
    return {
        "mode": mode,
        "recall@5": mean([q["recall@5"] for q in per_q]),
        "recall@10": mean([q["recall@10"] for q in per_q]),
        "mrr": mean([q["mrr"] for q in per_q]),
        "ndcg@10": mean([q["ndcg@10"] for q in per_q]),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "n_degraded": sum(1 for q in per_q if q["degraded"]),
        "per_question": per_q,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(GOLDEN_PATH))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    s = get_settings()
    items = [it for it in load_golden(args.golden) if it.answerable and it.evidence]
    counts = runtime.sqlite_store().counts()
    results = {m: evaluate_mode(m, items, args.k) for m in args.modes}

    out = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "golden": Path(args.golden).resolve().relative_to(PROJECT_ROOT).as_posix()
        if Path(args.golden).resolve().is_relative_to(PROJECT_ROOT) else Path(args.golden).name,
        "n_questions": len(items),
        "n_evidence": sum(len(it.evidence) for it in items),
        "corpus": {"papers": counts["paper"], "chunks": counts["chunk"]},
        "config": {
            "embedding": s.embedding.model_name if s.embedding.provider != "hash" else "hash",
            "reranker": s.reranker.model_name if s.reranker.provider != "lexical" else "lexical",
            "chunk_tokens": s.chunk.target_tokens, "chunk_overlap": s.chunk.overlap_tokens,
            "top_k_dense": s.retrieve.top_k_dense, "top_k_sparse": s.retrieve.top_k_sparse,
            "rrf_k": s.retrieve.rrf_k, "rerank_candidates": s.reranker.candidates,
            "max_per_paper": s.retrieve.max_per_paper, "k": args.k,
        },
        "modes": results,
    }
    if "dense" in results and "hybrid_rerank" in results:
        a = [q["recall@10"] for q in results["dense"]["per_question"]]
        b = [q["recall@10"] for q in results["hybrid_rerank"]["per_question"]]
        out["bootstrap_recall@10_hybrid_rerank_vs_dense"] = paired_bootstrap_ci(a, b, n_resamples=1000)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"ablation_{dt.date.today().isoformat()}{('_' + args.tag) if args.tag else ''}"
    json_path = RESULTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md_path = write_ablation_markdown(out, RESULTS_DIR / f"{stem}.md")
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
