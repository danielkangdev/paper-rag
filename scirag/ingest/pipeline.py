"""Ingest pipeline: parse -> chunk -> embed -> write SQLite (+FTS5) and Qdrant.

Idempotent: chunk ids are deterministic and every write replaces the paper's
previous rows/points, so re-ingesting never duplicates data. If embedding or
Qdrant fails, the paper is still written to SQLite (sparse search keeps
working) and the result is marked ``degraded``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .. import runtime
from ..chunk.chunker import build_chunks
from ..chunk.models import Paper
from ..config import get_settings
from .pdf_parser import pdf_title, pdf_to_markdown

log = logging.getLogger("scirag.ingest")


def ingest_paper(paper: Paper, *, force: bool = False) -> dict:
    s = get_settings()
    store = runtime.sqlite_store()
    existing = store.get_paper(paper.paper_id)
    if existing and existing["n_chunks"] > 0 and not force:
        return {"paper_id": paper.paper_id, "status": "skipped", "reason": "already indexed",
                "chunks": existing["n_chunks"], "degraded": []}
    if not paper.pdf_path or not Path(paper.pdf_path).exists():
        raise FileNotFoundError(f"PDF not found for {paper.paper_id}: {paper.pdf_path}")

    md = pdf_to_markdown(paper.pdf_path)
    sections, chunks = build_chunks(
        paper.paper_id, md, title=paper.title,
        target_tokens=s.chunk.target_tokens, overlap_tokens=s.chunk.overlap_tokens,
        encoding=s.chunk.encoding,
    )
    if not chunks:
        raise ValueError(f"no text extracted from {paper.pdf_path}")

    degraded: list[str] = []
    vectors = None
    try:
        vectors = runtime.embedder().encode_documents([c.embed_text() for c in chunks])
    except Exception as exc:
        log.warning("embedding failed for %s: %s", paper.paper_id, exc)
        degraded.append("embedding_unavailable")

    store.replace_paper(paper, sections, chunks)

    if vectors is not None:
        try:
            vs = runtime.vector_store()
            vs.delete_paper(paper.paper_id)
            vs.upsert([(c.chunk_id, c.paper_id, v) for c, v in zip(chunks, vectors)])
        except Exception as exc:
            log.warning("qdrant write failed for %s: %s", paper.paper_id, exc)
            degraded.append("vector_store_unavailable")

    return {"paper_id": paper.paper_id, "status": "indexed", "title": paper.title,
            "sections": len(sections), "chunks": len(chunks), "degraded": degraded}


def ingest_arxiv(identifier: str, *, force: bool = False) -> dict:
    from .arxiv import fetch_arxiv, paper_id_for

    s = get_settings()
    pid = paper_id_for(identifier)
    existing = runtime.sqlite_store().get_paper(pid)
    if existing and existing["n_chunks"] > 0 and not force:
        return {"paper_id": pid, "status": "skipped", "reason": "already indexed",
                "chunks": existing["n_chunks"], "degraded": []}
    paper = fetch_arxiv(identifier, s.resolve(s.paths.pdf_dir))
    return ingest_paper(paper, force=True)


def ingest_local_pdf(pdf_path: str | Path, *, title: str | None = None, force: bool = False) -> dict:
    path = Path(pdf_path)
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    paper = Paper(
        paper_id=f"pdf:{digest}",
        title=title or pdf_title(path) or path.stem,
        source="pdf",
        pdf_path=str(path),
    )
    return ingest_paper(paper, force=force)


def delete_paper(paper_id: str) -> dict:
    removed = runtime.sqlite_store().delete_paper(paper_id)
    degraded: list[str] = []
    try:
        runtime.vector_store().delete_paper(paper_id)
    except Exception as exc:
        log.warning("qdrant delete failed for %s: %s", paper_id, exc)
        degraded.append("vector_store_unavailable")
    return {"paper_id": paper_id, "deleted": removed, "degraded": degraded}
