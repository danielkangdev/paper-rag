"""FastAPI app: /ingest /search /ask /health (same service layer as the MCP tools)."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__, service

app = FastAPI(title="scirag", version=__version__,
              description="Hybrid-retrieval RAG over scientific papers with grounded citations.")


class IngestRequest(BaseModel):
    arxiv_id: str | None = None
    pdf_path: str | None = Field(None, description="Relative to the configured PDF directory")
    title: str | None = None
    force: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(10, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=50)


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    try:
        return service.ingest(arxiv_id=req.arxiv_id, pdf_path=req.pdf_path, title=req.title,
                              force=req.force)
    except (service.WriteNotAllowed, service.PathNotAllowed) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search")
def search(req: SearchRequest) -> dict:
    return service.search(req.query, mode=req.mode, top_k=req.top_k)


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return service.ask(req.question, top_k=req.top_k)


@app.get("/papers")
def papers() -> dict:
    return {"papers": service.list_papers()}
