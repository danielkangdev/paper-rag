"""Markdown rendering of eval results, and README table injection.

    python -m scirag.eval.report --readme   # refresh the README tables from the latest JSON

Every number in the README comes from eval/results/*.json via this module.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
README = PROJECT_ROOT / "README.md"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def ablation_table(res: dict) -> str:
    lines = [
        "| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 latency | p95 latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, m in res["modes"].items():
        lines.append(
            f"| `{mode}` | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | "
            f"{m['ndcg@10']:.3f} | {m['latency_p50_ms']:.0f} ms | {m['latency_p95_ms']:.0f} ms |"
        )
    b = res.get("bootstrap_recall@10_hybrid_rerank_vs_dense")
    if b:
        lines.append("")
        lines.append(
            f"Recall@10, `hybrid_rerank` − `dense`: **{b['mean_diff']:+.3f}** "
            f"(paired bootstrap 95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}], "
            f"{b['n_resamples']} resamples over {b['n_questions']} questions)."
        )
    c = res["corpus"]
    cfg = res["config"]
    lines.append("")
    lines.append(
        f"{res['n_questions']} answerable questions / {res['n_evidence']} evidence quotes; "
        f"corpus {c['papers']} papers / {c['chunks']} chunks; embedding `{cfg['embedding']}`, "
        f"reranker `{cfg['reranker']}`, CPU. Source: `eval/results/{res.get('_file', 'ablation_*.json')}`."
    )
    return "\n".join(lines)


def write_ablation_markdown(res: dict, path: Path) -> Path:
    path.write_text(f"# Retrieval ablation ({res['created_at']})\n\n{ablation_table(res)}\n", encoding="utf-8")
    return path


def answer_table(res: dict) -> str:
    a, b = res["configs"]["A"], res["configs"]["B"]
    rows = [
        ("Citation existence rate (cited ids exist in the evidence given)", "citation_existence", True),
        ("Citation precision (cited chunk contains a gold quote)", "citation_precision", True),
        ("Abstention success (unanswerable questions refused)", "abstain_success", True),
        ("False refusal (answerable questions refused)", "false_refusal", False),
        ("Ungrounded answer rate (LLM judge: ≥1 unsupported sentence)", "ungrounded_rate", False),
    ]
    lines = ["| Metric | A: baseline (no abstention, no citation check) | B: full system |",
             "| --- | ---: | ---: |"]
    for label, key, _ in rows:
        lines.append(f"| {label} | {_pct(a.get(key))} | {_pct(b.get(key))} |")
    n_neg = res["n_unanswerable"]
    lines.append(f"| LLM calls made for the {n_neg} unanswerable questions | "
                 f"{n_neg - a['system_abstentions']} | {n_neg - b['system_abstentions']} |")
    lines.append("")
    lines.append(
        f"Sentences the judge left unlabelled (excluded, never counted as unsupported): "
        f"A {a.get('unjudged_sentences', 0)}, B {b.get('unjudged_sentences', 0)}. "
        f"{res['n_answerable']} answerable + {res['n_unanswerable']} unanswerable questions; "
        f"generator `{res['chat_model']}`, judge `{res['judge_model']}`. "
        f"Source: `eval/results/{res.get('_file', 'answer_eval_*.json')}`."
    )
    return "\n".join(lines)


def latest(prefix: str) -> tuple[dict, str] | None:
    files = sorted(p for p in RESULTS_DIR.glob(f"{prefix}_*.json") if re.search(r"\d{4}-\d{2}-\d{2}(\.json)$", p.name))
    if not files:
        return None
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    data["_file"] = files[-1].name
    return data, files[-1].name


def _inject(text: str, marker: str, body: str) -> str:
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    return pattern.sub(lambda _m: f"{start}\n{body}\n{end}", text)


def update_readme() -> None:
    text = README.read_text(encoding="utf-8")
    abl = latest("ablation")
    if abl:
        text = _inject(text, "ablation", ablation_table(abl[0]))
    ans = latest("answer_eval")
    if ans:
        text = _inject(text, "answer-eval", answer_table(ans[0]))
    v1 = sorted(RESULTS_DIR.glob("answer_eval_*_v1_weakhint.json"))
    if v1:
        d = json.loads(v1[-1].read_text(encoding="utf-8"))
        b = d["configs"]["B"]
        text = _inject(text, "answer-v1", (
            f"First run (`eval/results/{v1[-1].name}`): with a prompt hint on weak evidence, B wrongly "
            f"refused {_pct(b['false_refusal'])} of answerable questions; the `{d['judge_model']}` judge "
            f"rated almost every sentence supported. Both were fixed before the run above (hint removed; "
            f"judge switched, refusal sentences post-checked); see `eval/judge_prompt.md`."))
    caps = sorted(RESULTS_DIR.glob("ablation_*_cap2.json"))
    if abl and caps:
        text = _inject(text, "cap", cap_comparison(json.loads(caps[-1].read_text(encoding="utf-8")),
                                                   abl[0], caps[-1].name))
    cal_path = RESULTS_DIR / "abstain_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        text = _inject(text, "abstain", calibration_summary(cal))
    README.write_text(text, encoding="utf-8")


def cap_comparison(capped: dict, uncapped: dict, capped_file: str) -> str:
    lines = ["| Mode | Recall@10, cap 2 per paper | Recall@10, no cap | nDCG@10, cap 2 | nDCG@10, no cap |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for mode in ("hybrid", "hybrid_rerank"):
        c, u = capped["modes"][mode], uncapped["modes"][mode]
        lines.append(f"| `{mode}` | {c['recall@10']:.3f} | {u['recall@10']:.3f} | "
                     f"{c['ndcg@10']:.3f} | {u['ndcg@10']:.3f} |")
    lines.append("")
    lines.append(f"Source: `eval/results/{capped_file}` vs `eval/results/{uncapped['_file']}`.")
    return "\n".join(lines)


def calibration_summary(cal: dict) -> str:
    r = cal["rerank"]
    loo = r.get("loocv", {})
    agg = {"max": "top-1", "mean_top3": "mean of top-3"}.get(cal.get("aggregate", "max"), cal.get("aggregate"))
    return (
        f"Signal: {agg} reranker score (chosen over the alternative by leave-one-out). "
        f"`threshold_low` = **{r['threshold_low']:.3f}** "
        f"(in-sample: refuses {_pct(r['abstain_success'])} of unanswerable, wrongly refuses "
        f"{_pct(r['false_refusal'])} of answerable; leave-one-out: {_pct(loo.get('abstain_success'))} / "
        f"{_pct(loo.get('false_refusal'))}). Source: `eval/results/abstain_calibration.json`."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true")
    args = ap.parse_args(argv)
    if args.readme:
        update_readme()
        print(f"updated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
