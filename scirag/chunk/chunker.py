"""Section-aware, token-bounded chunker.

* Chunks never cross a section boundary.
* Within a section, sentences are packed greedily up to ``target_tokens``;
  the next chunk starts with the trailing ~``overlap_tokens`` of the previous.
* Each chunk records the page its first sentence came from (from the
  ``<!-- page N -->`` markers written by the PDF parser).
* Sections after "References" are already dropped by the section parser; the
  References section itself is kept as a Section row but not chunked.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import Chunk, Section
from .section_parser import split_sections

_PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s*-->")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
_SKIP_SECTIONS = {"references", "acknowledgments", "acknowledgements"}

_ENCODER = None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding(encoding)
        except Exception:  # offline / not installed: rough estimate
            _ENCODER = False
    if _ENCODER is False:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text, disallowed_special=()))


@dataclass
class _Sentence:
    text: str
    page: int
    tokens: int


def _page_at(md: str, offset: int) -> int:
    page = 1
    for m in _PAGE_RE.finditer(md):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def _sentences(body: str, start_page: int, encoding: str, max_tokens: int) -> list[_Sentence]:
    out: list[_Sentence] = []
    page = start_page
    for para in body.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        m = _PAGE_RE.fullmatch(para)
        if m:
            page = int(m.group(1))
            continue
        para = _PAGE_RE.sub(" ", para)
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            n = count_tokens(sent, encoding)
            if n <= max_tokens:
                out.append(_Sentence(sent, page, n))
                continue
            # Very long "sentence" (flattened table, equations): split on words.
            words = sent.split()
            step = max(1, int(len(words) * max_tokens / n))
            for i in range(0, len(words), step):
                piece = " ".join(words[i : i + step])
                out.append(_Sentence(piece, page, count_tokens(piece, encoding)))
    return out


def _pack(sents: list[_Sentence], target: int, overlap: int) -> list[list[_Sentence]]:
    groups: list[list[_Sentence]] = []
    cur: list[_Sentence] = []
    cur_tokens = 0
    for s in sents:
        if cur and cur_tokens + s.tokens > target:
            groups.append(cur)
            tail: list[_Sentence] = []
            tail_tokens = 0
            for prev in reversed(cur):
                if tail_tokens + prev.tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev.tokens
            cur, cur_tokens = tail, tail_tokens
        cur.append(s)
        cur_tokens += s.tokens
    if cur:
        groups.append(cur)
    return groups


def chunk_id_for(paper_id: str, section_idx: int, ordinal: int) -> str:
    return hashlib.sha1(f"{paper_id}::{section_idx}::{ordinal}".encode()).hexdigest()[:20]


def build_chunks(
    paper_id: str,
    md: str,
    *,
    title: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding: str = "cl100k_base",
) -> tuple[list[Section], list[Chunk]]:
    sections: list[Section] = []
    chunks: list[Chunk] = []
    chunk_index = 0
    for raw in split_sections(md):
        start_page = _page_at(md, raw.start)
        body_text = _PAGE_RE.sub("", raw.body)
        body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()
        sections.append(
            Section(
                section_id=hashlib.sha1(f"{paper_id}::sec::{raw.idx}".encode()).hexdigest()[:16],
                paper_id=paper_id,
                idx=raw.idx,
                name=raw.name,
                level=raw.level,
                page_start=start_page,
                text=body_text,
            )
        )
        if raw.name.lower() in _SKIP_SECTIONS:
            continue
        sents = _sentences(raw.body, start_page, encoding, target_tokens)
        for ordinal, group in enumerate(_pack(sents, target_tokens, overlap_tokens)):
            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(paper_id, raw.idx, ordinal),
                    paper_id=paper_id,
                    section=raw.name,
                    section_idx=raw.idx,
                    page=group[0].page,
                    chunk_index=chunk_index,
                    text=" ".join(s.text for s in group),
                    title=title,
                )
            )
            chunk_index += 1
    return sections, chunks
