"""The six MCP tools. Arguments are pydantic models; their JSON Schema is what
``tools/list`` advertises, and every call is validated against it.

``ingest_paper`` and ``delete_paper`` are refused unless
``security.allow_writes`` is true, and ``ingest_paper`` only reads PDFs from
the configured PDF directory (or downloads by arXiv id).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .. import service


class IngestPaperArgs(BaseModel):
    arxiv_id: str | None = Field(None, description="arXiv id, e.g. 2310.11511")
    pdf_path: str | None = Field(None, description="PDF path relative to the configured PDF directory")
    title: str | None = Field(None, description="Optional title for a local PDF")
    force: bool = Field(False, description="Re-ingest even if already indexed")

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.arxiv_id) == bool(self.pdf_path):
            raise ValueError("give exactly one of arxiv_id or pdf_path")
        return self


class ListPapersArgs(BaseModel):
    pass


class SearchPapersArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(5, ge=1, le=50)


class AskPaperArgs(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=50)


class GetSectionArgs(BaseModel):
    paper_id: str = Field(..., description="e.g. arxiv:2310.11511")
    section: str | None = Field(None, description="Section name or index; omit to list sections")


class DeletePaperArgs(BaseModel):
    paper_id: str


def _search(a: SearchPapersArgs) -> dict:
    res = service.search(a.query, mode=a.mode, top_k=a.top_k)
    for h in res["hits"]:
        h["text"] = h["text"][:800]
    return res


TOOLS: dict[str, tuple[type[BaseModel], str, Callable[[BaseModel], object]]] = {
    "ingest_paper": (IngestPaperArgs, "Ingest a paper by arXiv id or a PDF from the PDF directory (requires allow_writes).",
                     lambda a: service.ingest(arxiv_id=a.arxiv_id, pdf_path=a.pdf_path, title=a.title, force=a.force)),
    "list_papers": (ListPapersArgs, "List indexed papers.", lambda a: {"papers": service.list_papers()}),
    "search_papers": (SearchPapersArgs, "Retrieve passages with dense / sparse / hybrid / hybrid_rerank.", _search),
    "ask_paper": (AskPaperArgs, "Answer a question with citations [Title, §Section, p.N], or abstain.",
                  lambda a: service.ask(a.question, top_k=a.top_k)),
    "get_section": (GetSectionArgs, "Get the full text of one section, or list a paper's sections.",
                    lambda a: service.get_section(a.paper_id, a.section)),
    "delete_paper": (DeletePaperArgs, "Delete a paper and its chunks/vectors (requires allow_writes).",
                     lambda a: service.delete_paper(a.paper_id)),
}


def list_tools() -> list[dict]:
    out = []
    for name, (model, desc, _) in TOOLS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        out.append({"name": name, "description": desc, "inputSchema": schema})
    return out


def _content(payload: object, *, is_error: bool = False) -> dict:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}], "structuredContent": payload if isinstance(payload, dict) else {"result": payload}, "isError": is_error}


def call_tool(name: str, arguments: dict | None) -> dict:
    if name not in TOOLS:
        return _content({"error": "unknown_tool", "message": f"unknown tool: {name}"}, is_error=True)
    model, _, fn = TOOLS[name]
    try:
        args = model.model_validate(arguments or {})
    except ValidationError as exc:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _content({"error": "invalid_arguments", "details": errors}, is_error=True)
    try:
        return _content(fn(args))
    except service.WriteNotAllowed as exc:
        return _content({"error": "writes_disabled", "message": str(exc)}, is_error=True)
    except service.PathNotAllowed as exc:
        return _content({"error": "path_not_allowed", "message": str(exc)}, is_error=True)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        return _content({"error": "not_found_or_invalid", "message": str(exc).strip("'\"")}, is_error=True)
    except Exception as exc:  # never crash the server loop
        return _content({"error": "internal", "message": str(exc).splitlines()[0][:300]}, is_error=True)
