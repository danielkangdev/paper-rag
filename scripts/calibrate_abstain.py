"""Calibrate abstention thresholds on the golden set.

    python scripts/calibrate_abstain.py [--write-config]

For every golden question, run hybrid_rerank retrieval and record the top-1
reranker score and the top-1 dense cosine. For each signal, choose
``threshold_low`` maximizing (unanswerable refused) − (answerable refused)
subject to false refusals <= --max-false-refusal; ties go to the lower
threshold. ``threshold_high`` (the weak-evidence tier, which only adds a hint
to the prompt) is the 25th percentile of answerable scores.

The questions used for calibration are the same ones the answer eval reports
on, so the in-sample rates are optimistic. Leave-one-out rates (threshold
re-picked without the held-out question) are reported alongside.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scirag.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from scirag.eval.dataset import GOLDEN_PATH, load_golden
from scirag.eval.metrics import percentile
from scirag.rag.abstention import aggregate_scores
from scirag.retrieval.modes import search

OUT = PROJECT_ROOT / "eval" / "results" / "abstain_calibration.json"
AGGS = ("max", "mean_top3")


def rates(pos: list[float], neg: list[float], t: float) -> tuple[float, float]:
    return (sum(s < t for s in neg) / len(neg), sum(s < t for s in pos) / len(pos))


def pick_threshold(pos: list[float], neg: list[float], max_fr: float) -> float:
    values = sorted(set(pos + neg))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1e-6]
    best_t, best_key = 0.0, None
    for t in candidates:
        succ, fr = rates(pos, neg, t)
        if fr > max_fr:
            continue
        key = (succ - fr, -fr, -t)
        if best_key is None or key > best_key:
            best_t, best_key = t, key
    return best_t


def loocv(pos: list[float], neg: list[float], max_fr: float) -> dict:
    refused_neg = refused_pos = 0
    for i in range(len(neg)):
        t = pick_threshold(pos, neg[:i] + neg[i + 1:], max_fr)
        refused_neg += neg[i] < t
    for i in range(len(pos)):
        t = pick_threshold(pos[:i] + pos[i + 1:], neg, max_fr)
        refused_pos += pos[i] < t
    return {"abstain_success": refused_neg / len(neg), "false_refusal": refused_pos / len(pos)}


def calibrate(pos: list[float], neg: list[float], max_fr: float) -> dict:
    t = pick_threshold(pos, neg, max_fr)
    succ, fr = rates(pos, neg, t)
    return {
        "threshold_low": round(t, 4),
        "threshold_high": round(max(t, percentile(pos, 25)), 4),
        "abstain_success": succ,
        "false_refusal": fr,
        "loocv": loocv(pos, neg, max_fr),
        "answerable_scores": sorted(round(x, 4) for x in pos),
        "unanswerable_scores": sorted(round(x, 4) for x in neg),
    }


def _write_config(cal: dict) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    for key, val in (
        ("threshold_low", cal["rerank"]["threshold_low"]),
        ("threshold_high", cal["rerank"]["threshold_high"]),
        ("dense_threshold_low", cal["dense"]["threshold_low"]),
        ("dense_threshold_high", cal["dense"]["threshold_high"]),
    ):
        text = re.sub(rf"(\n  {key}: )[0-9.]+", rf"\g<1>{val}", text)
    text = re.sub(r"(\n  aggregate: )\w+", rf"\g<1>{cal['aggregate']}", text)
    DEFAULT_CONFIG_PATH.write_text(text, encoding="utf-8")


def _j(c: dict) -> float:
    return c["loocv"]["abstain_success"] - c["loocv"]["false_refusal"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(GOLDEN_PATH))
    ap.add_argument("--max-false-refusal", type=float, default=0.05)
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args(argv)

    keys = [f"{sig}_{agg}" for sig in ("rerank", "dense") for agg in AGGS]
    signals: dict[str, dict[str, list[float]]] = {k: {"pos": [], "neg": []} for k in keys}
    per_q = []
    for it in load_golden(args.golden):
        res = search(it.question, mode="hybrid_rerank", top_k=10)
        row = {"qid": it.qid, "answerable": it.answerable, "degraded": res.degraded}
        side = "pos" if it.answerable else "neg"
        for sig, field in (("rerank", "score_rerank"), ("dense", "score_dense")):
            scores = [getattr(h, field) for h in res.hits if getattr(h, field) is not None] or [0.0]
            for agg in AGGS:
                v = aggregate_scores(scores, agg)
                signals[f"{sig}_{agg}"][side].append(v)
                row[f"{sig}_{agg}"] = round(v, 4)
        per_q.append(row)
        print(f"{it.qid} {'ans' if it.answerable else 'NEG'} " +
              " ".join(f"{k}={row[k]:.3f}" for k in keys), file=sys.stderr)

    variants = {k: calibrate(v["pos"], v["neg"], args.max_false_refusal) for k, v in signals.items()}
    # Pick the aggregation by leave-one-out quality of the reranker signal; ties -> max.
    agg = max(AGGS, key=lambda a: (_j(variants[f"rerank_{a}"]), a == "max"))
    cal = {"aggregate": agg, "rerank": variants[f"rerank_{agg}"], "dense": variants[f"dense_{agg}"],
           "variants": variants, "max_false_refusal": args.max_false_refusal, "per_question": per_q}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"chosen aggregate: {agg}")
    for name, c in variants.items():
        print(f"{name}: low={c['threshold_low']} high={c['threshold_high']} "
              f"in-sample abstain_success={c['abstain_success']:.3f} false_refusal={c['false_refusal']:.3f} "
              f"| LOOCV {c['loocv']['abstain_success']:.3f} / {c['loocv']['false_refusal']:.3f}")
    if args.write_config:
        _write_config(cal)
        print(f"updated thresholds in {DEFAULT_CONFIG_PATH}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
