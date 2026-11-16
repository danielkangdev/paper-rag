"""Core data models shared by ingest, storage, retrieval and QA."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    paper_id: str  # e.g. "arxiv:2310.11511" or "pdf:<sha1>"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str | None = None
    source: str = "arxiv"
    pdf_path: str | None = None


class Section(BaseModel):
    section_id: str
    paper_id: str
    idx: int
    name: str
    level: int = 1
    page_start: int | None = None
    text: str = ""


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    section_idx: int
    page: int
    chunk_index: int  # position of the chunk within the paper
    text: str
    title: str = ""

    def embed_text(self) -> str:
        """Text sent to the embedder: a short context header plus the body."""
        return f"{self.title}\n{self.section}\n{self.text}"


class Hit(BaseModel):
    """A retrieved chunk plus the scores that produced its rank."""

    chunk_id: str
    paper_id: str
    title: str = ""
    section: str = ""
    page: int | None = None
    chunk_index: int | None = None
    text: str = ""
    score_dense: float | None = None
    score_sparse: float | None = None
    score_rrf: float | None = None
    score_rerank: float | None = None

    def short_title(self, max_len: int = 60) -> str:
        t = (self.title or self.paper_id).split(":")[0].strip()
        return t if len(t) <= max_len else t[: max_len - 1].rstrip() + "…"

    def cite_label(self) -> str:
        page = f"p.{self.page}" if self.page else "p.?"
        return f"[{self.short_title()}, §{self.section}, {page}]"
