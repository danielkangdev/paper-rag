"""Shared service layer used by the CLI, the MCP tools and the FastAPI app."""

from __future__ import annotations

from pathlib import Path

from . import runtime
from .config import get_settings
from .ingest import pipeline
from .rag import qa
from .retrieval import modes


class WriteNotAllowed(PermissionError):
    pass


class PathNotAllowed(PermissionError):
    pass


def _require_writes() -> None:
    if not get_settings().security.allow_writes:
        raise WriteNotAllowed("write operations are disabled (set security.allow_writes: true)")


def resolve_pdf(pdf_path: str) -> Path:
    """Resolve ``pdf_path`` inside the configured PDF directory; reject anything outside it."""
    s = get_settings()
    root = s.resolve(s.paths.pdf_dir).resolve()
    candidate = Path(pdf_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise PathNotAllowed(f"only PDFs under {root} may be ingested")
    if candidate.suffix.lower() != ".pdf":
        raise PathNotAllowed("only .pdf files may be ingested")
    if not candidate.is_file():
        raise FileNotFoundError(f"no such PDF: {candidate.name}")
    return candidate


def ingest(*, arxiv_id: str | None = None, pdf_path: str | None = None, title: str | None = None,
           force: bool = False, check_writes: bool = True) -> dict:
    if check_writes:
        _require_writes()
    if bool(arxiv_id) == bool(pdf_path):
        raise ValueError("give exactly one of arxiv_id or pdf_path")
    if arxiv_id:
        return pipeline.ingest_arxiv(arxiv_id, force=force)
    return pipeline.ingest_local_pdf(resolve_pdf(pdf_path), title=title, force=force)


def delete_paper(paper_id: str, *, check_writes: bool = True) -> dict:
    if check_writes:
        _require_writes()
    return pipeline.delete_paper(paper_id)


def list_papers() -> list[dict]:
    return [
        {k: p[k] for k in ("paper_id", "title", "year", "n_chunks", "source")}
        for p in runtime.sqlite_store().list_papers()
    ]


def search(query: str, mode: str = "hybrid_rerank", top_k: int = 10) -> dict:
    return modes.search(query, mode=mode, top_k=top_k).to_dict()


def ask(question: str, top_k: int = 10) -> dict:
    return qa.answer(question, top_k=top_k)


def get_section(paper_id: str, section: str | int | None = None) -> dict:
    store = runtime.sqlite_store()
    if store.get_paper(paper_id) is None:
        raise KeyError(f"unknown paper_id: {paper_id}")
    if section is None or section == "":
        return {"paper_id": paper_id, "sections": store.list_sections(paper_id)}
    row = store.get_section(paper_id, section)
    if row is None:
        raise KeyError(f"section {section!r} not found in {paper_id}")
    return {"paper_id": paper_id, "section": row["name"], "idx": row["idx"],
            "page_start": row["page_start"], "text": row["text"]}


def health() -> dict:
    out: dict = {"status": "ok", "components": {}}
    try:
        out["components"]["sqlite"] = runtime.sqlite_store().counts()
    except Exception as exc:
        out["components"]["sqlite"] = f"error: {exc}"
        out["status"] = "degraded"
    try:
        out["components"]["qdrant"] = {"points": runtime.vector_store().count()}
    except Exception as exc:
        out["components"]["qdrant"] = f"error: {exc}"
        out["status"] = "degraded"
    from .rag.llm import is_configured

    out["components"]["llm_configured"] = is_configured()
    return out
