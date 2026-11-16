"""End-to-end demo against a running API (used by `make demo`; works on Windows too).

    python scripts/demo.py [--api http://localhost:8000]

Waits for /health, ingests 5 papers through /ingest, then asks one question.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

DEMO_IDS = ["2005.11401", "2310.11511", "2305.06983", "2104.08663", "2312.10997"]
QUESTION = "How does Self-RAG decide when to retrieve?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    with httpx.Client(base_url=args.api, timeout=args.timeout) as http:
        print("waiting for the API ...", flush=True)
        deadline = time.time() + args.timeout
        while True:
            try:
                if http.get("/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                print("API did not become healthy", file=sys.stderr)
                return 1
            time.sleep(2)

        for aid in DEMO_IDS:
            t0 = time.perf_counter()
            r = http.post("/ingest", json={"arxiv_id": aid})
            if r.status_code != 200:
                print(f"ingest {aid} failed: {r.status_code} {r.text}", file=sys.stderr)
                return 1
            d = r.json()
            print(f"ingest {aid}: {d['status']} chunks={d.get('chunks')} ({time.perf_counter() - t0:.1f}s)", flush=True)

        r = http.post("/ask", json={"question": QUESTION})
        r.raise_for_status()
        d = r.json()
        print(f"\nQ: {QUESTION}\n\n{d['answer']}\n")
        print(f"decision={d['decision']['decision']} degraded={d['degraded']} latency={d['latency_ms']}ms")
        for c in d["citations"]:
            print(f"  {c['tag']}: {c['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
