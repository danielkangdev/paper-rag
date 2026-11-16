"""Command-line interface.

    scirag ingest 2310.11511 2005.11401      # arXiv ids
    scirag ingest --file data/corpus.txt     # one id per line (# comments ok)
    scirag ingest --pdf path/to/paper.pdf    # local PDF (CLI is trusted: no allow_writes check)
    scirag papers
    scirag search "what are reflection tokens" --mode hybrid_rerank
    scirag ask "How does Self-RAG decide when to retrieve?"
    scirag mcp                               # MCP server on stdio
    scirag serve                             # FastAPI on :8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .retrieval.modes import MODES


def _read_ids(path: str) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def cmd_ingest(args) -> int:
    from .ingest import pipeline

    if args.pdf:
        print(json.dumps(pipeline.ingest_local_pdf(args.pdf, title=args.title, force=args.force)))
        return 0
    ids = list(args.ids) + (_read_ids(args.file) if args.file else [])
    if not ids:
        print("nothing to ingest", file=sys.stderr)
        return 2
    failed = 0
    for i, aid in enumerate(ids, 1):
        t0 = time.perf_counter()
        try:
            r = pipeline.ingest_arxiv(aid, force=args.force)
            print(f"[{i}/{len(ids)}] {r['paper_id']}: {r['status']} chunks={r.get('chunks')} "
                  f"degraded={r.get('degraded')} ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(ids)}] {aid}: FAILED {exc}", file=sys.stderr)
    return 1 if failed else 0


def cmd_papers(_args) -> int:
    from . import service

    for p in service.list_papers():
        print(f"{p['paper_id']:<22} {p['n_chunks']:>4} chunks  {p['title'][:80]}")
    return 0


def cmd_search(args) -> int:
    from . import service

    res = service.search(args.query, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    print(f"mode={res['mode']} latency={res['latency_ms']}ms degraded={res['degraded']}")
    for i, h in enumerate(res["hits"], 1):
        scores = {k: round(v, 4) for k, v in h.items() if k.startswith("score_") and v is not None}
        print(f"{i:>2}. {h['paper_id']} §{h['section']} p.{h['page']} {scores}")
        print(f"    {h['text'][:200]}")
    return 0


def cmd_ask(args) -> int:
    from . import service

    res = service.ask(args.question, top_k=args.top_k)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    print(res["answer"])
    print(f"\n-- decision={res['decision']['decision']} score={res['decision']['score']} "
          f"degraded={res['degraded']} latency={res['latency_ms']}ms")
    for c in res["citations"]:
        print(f"   {c['tag']}: {c['label']} ({c['chunk_id']})")
    return 0


def cmd_mcp(_args) -> int:
    from .mcp.server import run_stdio

    return run_stdio()


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("scirag.api.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # avoid UnicodeEncodeError on non-UTF-8 consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="scirag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest arXiv papers or a local PDF")
    pi.add_argument("ids", nargs="*")
    pi.add_argument("--file")
    pi.add_argument("--pdf")
    pi.add_argument("--title")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(fn=cmd_ingest)

    sub.add_parser("papers", help="list indexed papers").set_defaults(fn=cmd_papers)

    ps = sub.add_parser("search", help="retrieve passages")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ps.add_argument("--top-k", type=int, default=5)
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(fn=cmd_search)

    pa = sub.add_parser("ask", help="answer a question with citations")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, default=10)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(fn=cmd_ask)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(fn=cmd_mcp)

    pv = sub.add_parser("serve", help="run the FastAPI server")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
