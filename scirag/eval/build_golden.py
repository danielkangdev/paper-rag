"""Convert the original chunk-id golden set into paper+quote labels.

    python -m scirag.eval.build_golden [--source eval/source/qa_set.golden.jsonl]

For every answerable question, ALL chunks of its relevant papers (not just the
ones our retriever finds, to avoid biasing labels toward our own system) are
shown to the LLM, which picks 1-3 short quotes (5-15 words) that support the
gold answer. A quote that is not an exact substring of the chunk it names is
snapped to the most similar 5-15 word window of that chunk, and kept only if
the similarity is >= 0.85, so every stored quote is verbatim source text.
Questions with no valid quote are retried once. Unanswerable questions get
``answerable=false, evidence=[]``. Every question and quote is printed for
manual spot-checking.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from pathlib import Path

from .. import runtime
from ..config import PROJECT_ROOT
from ..rag.llm import chat
from .dataset import GOLDEN_PATH, Evidence, GoldenItem, normalize, save_golden

SOURCE = PROJECT_ROOT / "eval" / "source" / "qa_set.golden.jsonl"
SNAP_MIN_RATIO = 0.85

PROMPT = """You are labelling a retrieval benchmark.

Question: {question}
Reference answer: {gold}

Below are passages from the relevant paper(s), each tagged [C<n>] with its paper id.
Pick 1 to 3 short quotes that together best support the reference answer.

Rules:
- Each quote must be copied EXACTLY, character for character, from ONE passage (5 to 15 words).
- Do not paraphrase, fix typos, join text across passages, or add ellipses.
- Prefer quotes that state the key fact itself, not passing mentions.
- If passages from two papers are given and both are relevant, include at least one quote from each.

Return JSON: {{"quotes": [{{"chunk": "C<n>", "quote": "..."}}]}}

Passages:
{passages}
"""


def _passages(chunks) -> str:
    return "\n\n".join(f"[C{i}] ({c.paper_id}, §{c.section}) {c.text}" for i, c in enumerate(chunks, 1))


def snap_quote(quote: str, text: str, min_ratio: float = SNAP_MIN_RATIO) -> tuple[str | None, float]:
    """Return the 5-15 word window of ``text`` most similar to ``quote``.

    LLMs copy imperfectly (a changed word, an extra clause). Instead of trusting
    the model's string, the stored quote is always an exact window of the source
    text; it is accepted only if it is a close match to what the model proposed.
    """
    words = text.split()
    target = normalize(quote)
    n = min(max(len(quote.split()), 5), 40)
    best, best_ratio = None, 0.0
    for length in range(max(5, n - 2), n + 3):
        for i in range(0, max(1, len(words) - length + 1)):
            window = " ".join(words[i : i + length])
            sm = difflib.SequenceMatcher(None, target, normalize(window), autojunk=False)
            if sm.real_quick_ratio() < best_ratio or sm.quick_ratio() < best_ratio:
                continue
            r = sm.ratio()
            if r > best_ratio:
                best, best_ratio = window, r
    if best is None or best_ratio < min_ratio:
        return None, best_ratio
    # Over-long quotes are cut to their first 15 words (still a verbatim span).
    return " ".join(best.split()[:15]).strip(" ,;:([\"'"), best_ratio


def _validate(raw: str, chunks) -> tuple[list[Evidence], list[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], ["invalid JSON"]
    good: list[Evidence] = []
    problems: list[str] = []
    for q in data.get("quotes", [])[:3]:
        quote = " ".join(str(q.get("quote", "")).split())
        if not quote:
            continue
        tag = str(q.get("chunk", "")).lstrip("Cc")
        candidates = [chunks[int(tag) - 1]] if tag.isdigit() and 1 <= int(tag) <= len(chunks) else chunks
        nq = normalize(quote)
        owner = next((c for c in candidates if nq in normalize(c.text)), None)
        if owner is not None and 5 <= len(quote.split()) <= 15:
            final = quote
        else:
            final, owner, best = None, None, 0.0
            for c in candidates:
                snapped, ratio = snap_quote(quote, c.text)
                if snapped and ratio > best:
                    final, owner, best = snapped, c, ratio
            if final is None:
                problems.append(f"no close verbatim match: {quote!r}")
                continue
            problems.append(f"snapped ({best:.2f}): {quote!r} -> {final!r}")
        if not 5 <= len(final.split()) <= 15:
            problems.append(f"length {len(final.split())}: {final!r}")
            continue
        if not any(e.paper_id == owner.paper_id and normalize(e.quote) == normalize(final) for e in good):
            good.append(Evidence(paper_id=owner.paper_id, quote=final))
    return good, problems


def build(source=SOURCE, model: str | None = None) -> list[GoldenItem]:
    store = runtime.sqlite_store()
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    items: list[GoldenItem] = []
    for row in rows:
        papers = row.get("relevant_paper_ids") or []
        item = GoldenItem(qid=row["qid"], question=row["question"], answerable=bool(papers),
                          gold_answer=row.get("gold_answer") if papers else None,
                          category=row.get("category"))
        if papers:
            chunks = [c for pid in papers for c in store.chunks_for_paper(pid)]
            if not chunks:
                raise SystemExit(f"{row['qid']}: papers {papers} are not ingested")
            messages = [{"role": "user", "content": PROMPT.format(
                question=row["question"], gold=row["gold_answer"], passages=_passages(chunks))}]
            evidence, problems = [], []
            for _attempt in range(2):
                raw = chat(messages, model=model, temperature=0, max_tokens=400, json_mode=True)
                evidence, problems = _validate(raw, chunks)
                if evidence:
                    break
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "Some quotes were invalid: "
                              + "; ".join(problems) + ". Copy quotes exactly from one passage."}]
            item.evidence = evidence
            print(f"\n{item.qid} [{', '.join(papers)}] {item.question}")
            for e in evidence:
                print(f"   - ({e.paper_id}) \"{e.quote}\"")
            for p in problems:
                print(f"   ! {p}")
            if not evidence:
                print("   !! NO VALID QUOTE — question will be skipped by retrieval eval")
        else:
            print(f"\n{item.qid} [unanswerable] {item.question}")
        items.append(item)
    return items


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", default=str(GOLDEN_PATH))
    ap.add_argument("--model", default=None, help="override CHAT_MODEL for labelling")
    args = ap.parse_args(argv)
    items = build(args.source, args.model)
    save_golden(items, args.out)
    n_ans = sum(it.answerable for it in items)
    n_ev = sum(len(it.evidence) for it in items)
    n_empty = sum(1 for it in items if it.answerable and not it.evidence)
    print(f"\nwrote {args.out}: {len(items)} questions, {n_ans} answerable, {n_ev} quotes, "
          f"{n_empty} answerable without a valid quote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
