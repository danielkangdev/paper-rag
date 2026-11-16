"""arXiv metadata + PDF download."""

from __future__ import annotations

import re
from pathlib import Path

from ..chunk.models import Paper

_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_arxiv_id(identifier: str) -> str:
    m = _ARXIV_RE.search(identifier)
    if not m:
        raise ValueError(f"not an arXiv id: {identifier!r}")
    return m.group(1)


def paper_id_for(arxiv_id: str) -> str:
    return f"arxiv:{normalize_arxiv_id(arxiv_id)}"


def fetch_arxiv(identifier: str, pdf_dir: Path) -> Paper:
    """Fetch metadata and download the PDF (skipped if already on disk)."""
    import arxiv
    import httpx

    aid = normalize_arxiv_id(identifier)
    client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=5)
    results = list(client.results(arxiv.Search(id_list=[aid])))
    if not results:
        raise ValueError(f"arXiv id not found: {aid}")
    r = results[0]

    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{aid}.pdf"
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        url = r.pdf_url or f"https://arxiv.org/pdf/{aid}"
        with httpx.Client(timeout=120, follow_redirects=True) as http:
            resp = http.get(url)
            resp.raise_for_status()
            tmp = pdf_path.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(pdf_path)

    return Paper(
        paper_id=paper_id_for(aid),
        title=" ".join(r.title.split()),
        authors=[a.name for a in r.authors],
        year=r.published.year if r.published else None,
        abstract=" ".join((r.summary or "").split()),
        source="arxiv",
        pdf_path=str(pdf_path),
    )
