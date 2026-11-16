"""SQLite store: paper / section / chunk tables plus an FTS5 index over chunks.

The FTS5 table is kept in sync with ``chunk`` by triggers, so sparse retrieval
needs no separate rebuild step. One short-lived connection per call keeps it
safe to use from FastAPI worker threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..chunk.models import Chunk, Paper, Section

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper (
    paper_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    authors    TEXT NOT NULL DEFAULT '[]',
    year       INTEGER,
    abstract   TEXT,
    source     TEXT,
    pdf_path   TEXT,
    n_chunks   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS section (
    section_id TEXT PRIMARY KEY,
    paper_id   TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    name       TEXT NOT NULL,
    level      INTEGER NOT NULL DEFAULT 1,
    page_start INTEGER,
    text       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_section_paper ON section(paper_id);
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id    TEXT PRIMARY KEY,
    paper_id    TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    section     TEXT NOT NULL,
    section_idx INTEGER NOT NULL,
    page        INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_chunk_paper ON chunk(paper_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    paper_id UNINDEXED,
    title,
    text,
    tokenize = "porter unicode61 remove_diacritics 2"
);
CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_id, paper_id, title, text)
    VALUES (new.chunk_id, new.paper_id, new.title, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
    DELETE FROM chunk_fts WHERE chunk_id = old.chunk_id;
END;
CREATE TRIGGER IF NOT EXISTS chunk_au AFTER UPDATE ON chunk BEGIN
    DELETE FROM chunk_fts WHERE chunk_id = old.chunk_id;
    INSERT INTO chunk_fts(chunk_id, paper_id, title, text)
    VALUES (new.chunk_id, new.paper_id, new.title, new.text);
END;
"""

_CHUNK_COLS = "chunk_id, paper_id, section, section_idx, page, chunk_index, text, title"


class SqliteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- writes -----------------------------------------------------------
    def replace_paper(self, paper: Paper, sections: list[Section], chunks: list[Chunk]) -> None:
        """Atomically (re)write a paper and all of its sections/chunks."""
        with self.connect() as conn:
            conn.execute("DELETE FROM chunk WHERE paper_id = ?", (paper.paper_id,))
            conn.execute("DELETE FROM section WHERE paper_id = ?", (paper.paper_id,))
            conn.execute(
                "INSERT INTO paper(paper_id, title, authors, year, abstract, source, pdf_path, n_chunks) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(paper_id) DO UPDATE SET "
                "title=excluded.title, authors=excluded.authors, year=excluded.year, "
                "abstract=excluded.abstract, source=excluded.source, pdf_path=excluded.pdf_path, "
                "n_chunks=excluded.n_chunks",
                (
                    paper.paper_id, paper.title, json.dumps(paper.authors), paper.year,
                    paper.abstract, paper.source, paper.pdf_path, len(chunks),
                ),
            )
            conn.executemany(
                "INSERT INTO section(section_id, paper_id, idx, name, level, page_start, text) "
                "VALUES (?,?,?,?,?,?,?)",
                [(s.section_id, s.paper_id, s.idx, s.name, s.level, s.page_start, s.text) for s in sections],
            )
            conn.executemany(
                f"INSERT INTO chunk({_CHUNK_COLS}) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.paper_id, c.section, c.section_idx, c.page, c.chunk_index, c.text, c.title)
                    for c in chunks
                ],
            )

    def delete_paper(self, paper_id: str) -> bool:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunk WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM section WHERE paper_id = ?", (paper_id,))
            cur = conn.execute("DELETE FROM paper WHERE paper_id = ?", (paper_id,))
            return cur.rowcount > 0

    # ---- reads ------------------------------------------------------------
    def get_paper(self, paper_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM paper WHERE paper_id = ?", (paper_id,)).fetchone()
        return _paper_row(row) if row else None

    def list_papers(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper ORDER BY paper_id").fetchall()
        return [_paper_row(r) for r in rows]

    def list_sections(self, paper_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT section_id, idx, name, level, page_start FROM section "
                "WHERE paper_id = ? ORDER BY idx",
                (paper_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_section(self, paper_id: str, name_or_idx: str | int) -> dict | None:
        with self.connect() as conn:
            if isinstance(name_or_idx, int) or str(name_or_idx).isdigit():
                row = conn.execute(
                    "SELECT * FROM section WHERE paper_id = ? AND idx = ?",
                    (paper_id, int(name_or_idx)),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM section WHERE paper_id = ? AND lower(name) = lower(?) ORDER BY idx",
                    (paper_id, name_or_idx),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        "SELECT * FROM section WHERE paper_id = ? AND lower(name) LIKE lower(?) ORDER BY idx",
                        (paper_id, f"%{name_or_idx}%"),
                    ).fetchone()
        return dict(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunk WHERE chunk_id IN ({marks})", chunk_ids
            ).fetchall()
        return {r["chunk_id"]: Chunk(**dict(r)) for r in rows}

    def chunks_for_paper(self, paper_id: str) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunk WHERE paper_id = ? ORDER BY chunk_index",
                (paper_id,),
            ).fetchall()
        return [Chunk(**dict(r)) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunk ORDER BY paper_id, chunk_index"
            ).fetchall()
        return [Chunk(**dict(r)) for r in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("paper", "section", "chunk", "chunk_fts")
            }

    def fts_match(self, match_query: str, top_k: int) -> list[tuple[str, float]]:
        """Run a raw FTS5 MATCH. Raises sqlite3.OperationalError on bad syntax."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, -bm25(chunk_fts, 0.0, 0.0, 0.5, 1.0) AS score FROM chunk_fts "
                "WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts, 0.0, 0.0, 0.5, 1.0) LIMIT ?",
                (match_query, top_k),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]


def _paper_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["authors"] = json.loads(d.get("authors") or "[]")
    return d
