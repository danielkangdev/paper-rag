"""Answer-quality evaluation (needs an LLM API key).

    python -m scirag.eval.run_answer_eval [--limit N]

Every golden question (answerable and unanswerable) is answered twice:
  A = baseline: abstention off, citation validation off
  B = full system
Metrics per configuration:
  citation_existence  cited [E#] tags that point at a passage actually given
  citation_precision  valid citations (answerable questions) whose chunk contains a gold quote
  abstain_success     unanswerable questions refused (system abstention, or an
                      answer the judge finds has no factual claim)
  false_refusal       answerable questions refused (same definition)
  ungrounded_rate     answers with >= 1 sentence the judge marks unsupported
The judge rules/prompt live in eval/judge_prompt.md. Ten random judged answers
are written to eval/results/judge_spotcheck.md for manual review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import re
from pathlib import Path

from ..config import PROJECT_ROOT, get_settings
from ..rag import citations, llm, qa
from .dataset import GOLDEN_PATH, evidence_hit_vector, load_golden
from .report import answer_table

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
JUDGE_PROMPT_PATH = PROJECT_ROOT / "eval" / "judge_prompt.md"


def _judge_prompt() -> str:
    text = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    m = re.search(r"<!-- PROMPT:start -->\n(.*?)<!-- PROMPT:end -->", text, re.DOTALL)
    if not m:
        raise SystemExit("judge prompt markers missing in eval/judge_prompt.md")
    return m.group(1)


def judge(question: str, answer_tagged: str, evidence: list[dict], model: str | None) -> dict:
    sentences = citations.split_sentences(answer_tagged)
    if not sentences:
        return {"sentences": [], "refusal": True, "ungrounded": False, "unjudged": 0}
    ev_text = "\n\n".join(f"[E{i}] {e['text']}" for i, e in enumerate(evidence, 1))
    sent_text = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    prompt = (_judge_prompt().replace("{question}", question).replace("{evidence}", ev_text)
              .replace("{sentences}", sent_text).replace("{n}", str(len(sentences))))
    labels: dict[int, tuple[str, str]] = {}
    raws = []
    for _attempt in range(3):  # re-ask if the judge skipped sentences or broke the JSON
        raw = llm.chat([{"role": "user", "content": prompt}], model=model, temperature=0,
                       max_tokens=2000, json_mode=True)
        raws.append(raw)
        try:
            items = json.loads(raw).get("sentences", [])
            for pos, x in enumerate(items, 1):
                i = int(str(x.get("i", pos)).strip(". ")) if str(x.get("i", pos)).strip(". ").isdigit() else pos
                lab = str(x.get("label", "")).strip().lower()
                if lab in {"supported", "unsupported", "no_claim"} and 1 <= i <= len(sentences):
                    labels.setdefault(i, (lab, str(x.get("reason", ""))))
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass
        if len(labels) == len(sentences):
            break
    out = []
    for i, s in enumerate(sentences, 1):
        # Never default a missing label to "unsupported": report it as unjudged instead.
        label, reason = labels.get(i, ("unjudged", "judge gave no label after 3 attempts"))
        if citations.is_refusal_sentence(s) and label != "no_claim":
            # The judge sometimes calls "the papers do not contain X" supported.
            label, reason = "no_claim", f"refusal pattern (judge said {label}: {reason})"
        out.append({"sentence": s, "label": label, "reason": reason})
    return {
        "sentences": out,
        "refusal": all(x["label"] == "no_claim" for x in out),
        "ungrounded": any(x["label"] == "unsupported" for x in out),
        "unjudged": sum(x["label"] == "unjudged" for x in out),
        "judge_raw": raws,
    }


def run_config(item, *, baseline: bool, judge_model: str | None) -> dict:
    res = qa.answer(item.question, abstain=not baseline, check_citations=not baseline)
    evidence = res.get("evidence", [])
    if res["abstained"]:
        verdict = {"sentences": [], "refusal": True, "ungrounded": False, "unjudged": 0}
    else:
        verdict = judge(item.question, res.get("answer_tagged") or res["answer"], evidence, judge_model)

    cited = citations.cited_indices(res.get("answer_tagged") or "")
    valid = [i for i in cited if 1 <= i <= len(evidence)]
    precise = 0
    if item.answerable:
        for i in valid:
            e = evidence[i - 1]
            if any(evidence_hit_vector(e["paper_id"], e["text"], item.evidence)):
                precise += 1
    return {
        "qid": item.qid, "answerable": item.answerable, "question": item.question,
        "answer": res["answer"], "answer_tagged": res.get("answer_tagged"),
        "abstained": res["abstained"], "decision": res["decision"], "degraded": res["degraded"],
        "n_cited": len(cited), "n_valid_cited": len(valid), "n_precise_cited": precise,
        "refusal": res["abstained"] or verdict["refusal"], "judge": verdict,
        "evidence": [{k: e.get(k) for k in ("chunk_id", "paper_id", "section", "page", "text")}
                     for e in evidence],
    }


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    n_cited = sum(r["n_cited"] for r in rows)
    n_valid = sum(r["n_valid_cited"] for r in rows)
    n_valid_ans = sum(r["n_valid_cited"] for r in ans)
    return {
        "citation_existence": n_valid / n_cited if n_cited else None,
        "citation_precision": sum(r["n_precise_cited"] for r in ans) / n_valid_ans if n_valid_ans else None,
        "abstain_success": sum(r["refusal"] for r in neg) / len(neg) if neg else None,
        "false_refusal": sum(r["refusal"] for r in ans) / len(ans) if ans else None,
        "ungrounded_rate": sum(r["judge"]["ungrounded"] for r in rows) / len(rows),
        "ungrounded_rate_answerable": sum(r["judge"]["ungrounded"] for r in ans) / len(ans) if ans else None,
        "ungrounded_rate_unanswerable": sum(r["judge"]["ungrounded"] for r in neg) / len(neg) if neg else None,
        "system_abstentions": sum(r["abstained"] for r in rows),
        "unjudged_sentences": sum(r["judge"].get("unjudged", 0) for r in rows),
        "n_citations": n_cited,
    }


def write_spotcheck(rows: list[dict], path, n: int = 10, seed: int = 0) -> None:
    judged = [r for r in rows if r["judge"]["sentences"]]
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    lines = ["# Judge spot-check", "",
             f"{len(sample)} randomly sampled judged answers (seed={seed}). Check each label by hand "
             "against the evidence; rules are in `eval/judge_prompt.md`.", ""]
    for r in sample:
        lines += [f"## {r['qid']} ({r['config']}, {'answerable' if r['answerable'] else 'unanswerable'})",
                  "", f"**Q:** {r['question']}", "", "| # | Label | Sentence | Judge reason |",
                  "| --- | --- | --- | --- |"]
        for i, s in enumerate(r["judge"]["sentences"], 1):
            sent = s["sentence"].replace("|", "\\|")
            lines.append(f"| {i} | {s['label']} | {sent} | {s['reason'].replace('|', '/')} |")
        lines += ["", "<details><summary>Evidence given to the model</summary>", ""]
        for i, e in enumerate(r["evidence"], 1):
            lines.append(f"- **E{i}** {e['paper_id']} §{e['section']} p.{e['page']}: {e['text'][:600]}…")
        lines += ["", "</details>", "", "Human verdict: ☐ agree ☐ disagree", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(GOLDEN_PATH))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    ap.add_argument("--rejudge", help="re-run only the judge on the answers in this results JSON")
    args = ap.parse_args(argv)
    if not llm.is_configured():
        raise SystemExit("LLM not configured: set OPENAI_API_KEY / CHAT_MODEL in .env")

    s = get_settings()
    if args.rejudge:
        # Keep the generated answers fixed; only re-run the judge.
        prev = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
        rows = prev["rows"]
        for cfg_name, cfg_rows in rows.items():
            for n, r in enumerate(cfg_rows, 1):
                if not r["abstained"]:
                    r["judge"] = judge(r["question"], r.get("answer_tagged") or r["answer"],
                                       r["evidence"], args.judge_model)
                r["refusal"] = r["abstained"] or r["judge"]["refusal"]
                print(f"[{cfg_name} {n}/{len(cfg_rows)}] {r['qid']} ungrounded={r['judge']['ungrounded']} "
                      f"unjudged={r['judge'].get('unjudged', 0)}")
        meta = {k: prev[k] for k in ("chat_model", "n_answerable", "n_unanswerable", "abstain_thresholds")}
        meta["rejudged_from"] = Path(args.rejudge).name
    else:
        items = load_golden(args.golden)
        if args.limit:
            items = items[: args.limit]
        rows = {"A": [], "B": []}
        for n, it in enumerate(items, 1):
            for cfg_name in ("A", "B"):
                r = run_config(it, baseline=cfg_name == "A", judge_model=args.judge_model)
                r["config"] = cfg_name
                rows[cfg_name].append(r)
            print(f"[{n}/{len(items)}] {it.qid} A: ungrounded={rows['A'][-1]['judge']['ungrounded']} "
                  f"B: abstained={rows['B'][-1]['abstained']} ungrounded={rows['B'][-1]['judge']['ungrounded']}")
        meta = {
            "chat_model": s.llm.chat_model,
            "n_answerable": sum(it.answerable for it in items),
            "n_unanswerable": sum(not it.answerable for it in items),
            "abstain_thresholds": {"low": s.abstain.threshold_low, "high": s.abstain.threshold_high},
        }

    out = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        **meta,
        "judge_model": args.judge_model or s.llm.chat_model,
        "configs": {k: aggregate(v) for k, v in rows.items()},
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"answer_eval_{dt.date.today().isoformat()}" + ("_partial" if args.limit else "")
    path = RESULTS_DIR / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["_file"] = path.name
    md = RESULTS_DIR / f"{stem}.md"
    md.write_text(f"# Answer-quality eval ({out['created_at']})\n\n{answer_table(out)}\n", encoding="utf-8")
    write_spotcheck(rows["A"] + rows["B"], RESULTS_DIR / "judge_spotcheck.md")
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {path}, {md}, judge_spotcheck.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
