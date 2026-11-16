"""PDF -> markdown-ish text with ``<!-- page N -->`` markers (PyMuPDF).

Each text block becomes one paragraph: line breaks inside a block are joined,
and end-of-line hyphenation (``retrie-\\nval``) is undone. Headings are usually
their own block, so ``3.2 Training`` stays on its own line for the section
parser.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_HYPHEN_BREAK_RE = re.compile(r"(\w)-[ \t]*\n[ \t]*([a-z])")
_WS_RE = re.compile(r"[ \t]+")


def _clean_block(text: str) -> str:
    # NFKC folds ligatures (ﬁ -> fi) and other compatibility characters.
    text = unicodedata.normalize("NFKC", text).replace("\x00", "").replace("\r", "")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\n", " ")
    return _WS_RE.sub(" ", text).strip()


def pdf_to_markdown(pdf_path: str | Path) -> str:
    import pymupdf as fitz

    parts: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_no, page in enumerate(doc, start=1):
            parts.append(f"<!-- page {page_no} -->")
            # Content-stream order (not sort=True): keeps two-column papers column-wise.
            blocks = page.get_text("blocks") or []
            for b in blocks:
                if len(b) >= 7 and b[6] != 0:  # skip image blocks
                    continue
                cleaned = _clean_block(b[4])
                if cleaned:
                    parts.append(cleaned)
    return "\n\n".join(parts)


def pdf_title(pdf_path: str | Path) -> str | None:
    """Best-effort title from PDF metadata (used for local PDFs without arXiv metadata)."""
    import pymupdf as fitz

    with fitz.open(str(pdf_path)) as doc:
        title = (doc.metadata or {}).get("title") or ""
    title = title.strip()
    return title or None
