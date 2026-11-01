"""SQLite storage: reports, chunks (per chunk set) and embeddings (per embedding profile).

Vectors are stored as little-endian float32 BLOBs. With a few hundred chunks, exact search with
NumPy is instant, so there is no vector index (and no approximate-search surprises).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports(
  report_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL, program TEXT NOT NULL DEFAULT '', weakness TEXT NOT NULL DEFAULT '',
  substate TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '', disclosed_at TEXT NOT NULL DEFAULT '',
  severity TEXT NOT NULL DEFAULT '', vote_count INTEGER NOT NULL DEFAULT 0,
  body TEXT NOT NULL, body_hash TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT '', source_url TEXT NOT NULL DEFAULT '', cwe TEXT NOT NULL DEFAULT '',
  quality_flags TEXT NOT NULL DEFAULT '[]', quality_score REAL NOT NULL DEFAULT 0,
  when_to_try TEXT NOT NULL DEFAULT '', images TEXT NOT NULL DEFAULT '[]', injection_hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunk_sets(name TEXT PRIMARY KEY, config_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_set TEXT NOT NULL REFERENCES chunk_sets(name) ON DELETE CASCADE,
  report_id INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, section TEXT NOT NULL, has_code INTEGER NOT NULL, n_tokens INTEGER NOT NULL,
  header TEXT NOT NULL, text TEXT NOT NULL, meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS chunks_by_set ON chunks(chunk_set, report_id, ordinal);
CREATE TABLE IF NOT EXISTS embeddings(
  chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  profile TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY(chunk_id, profile)
);
"""

# Columns added after the first release: a database made by an older version is upgraded in place on open.
ADDED_COLUMNS = {
    "reports": {"source": "TEXT NOT NULL DEFAULT ''", "source_url": "TEXT NOT NULL DEFAULT ''", "cwe": "TEXT NOT NULL DEFAULT ''",
                "quality_flags": "TEXT NOT NULL DEFAULT '[]'", "quality_score": "REAL NOT NULL DEFAULT 0",
                "when_to_try": "TEXT NOT NULL DEFAULT ''", "images": "TEXT NOT NULL DEFAULT '[]'",
                "injection_hits": "INTEGER NOT NULL DEFAULT 0"},
    "chunks": {"meta_json": "TEXT NOT NULL DEFAULT '{}'"},
}


@dataclass(frozen=True)
class ReportRow:
    report_id: int
    title: str
    program: str
    weakness: str
    substate: str
    created_at: str
    disclosed_at: str
    severity: str
    vote_count: int
    body: str
    body_hash: str
    source: str = ""                    # 'hackerone_api' | 'gitlab' | 'dataset'
    source_url: str = ""                # where the text really came from (what citations link to)
    cwe: str = ""
    quality_flags: str = "[]"           # JSON list, e.g. ["sparse", "image_dependent"]
    quality_score: float = 0.0          # 0..1, used to prefer reports with steps, PoC and impact
    when_to_try: str = ""               # retrieval aid written for security questions
    images: str = "[]"                  # JSON list of {"kind", "ref"} the cleaner found (flag-only, never fetched)
    injection_hits: int = 0             # how many AI-directed instructions were removed from the text

    @property
    def flags(self) -> list[str]:
        return json.loads(self.quality_flags or "[]")

    @property
    def image_list(self) -> list[dict]:
        return json.loads(self.images or "[]")


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    chunk_set: str
    report_id: int
    ordinal: int
    section: str
    has_code: bool
    n_tokens: int
    header: str
    text: str
    meta: dict = field(default_factory=dict)      # snapshot of the report's metadata, stored with every chunk

    @property
    def embed_text(self) -> str:
        """What gets embedded and keyword-indexed: the label plus the content."""
        return f"{self.header}\n\n{self.text}"


@dataclass(frozen=True)
class NewChunk:
    report_id: int
    ordinal: int
    section: str
    has_code: bool
    n_tokens: int
    header: str
    text: str
    meta: dict = field(default_factory=dict)


class StoreError(RuntimeError):
    pass


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        for table, columns in ADDED_COLUMNS.items():
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns.items():
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---- meta ----
    def set_meta(self, key: str, value: object) -> None:
        with self.db:
            self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, json.dumps(value)))

    def get_meta(self, key: str, default: object = None) -> object:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # ---- reports ----
    def replace_reports(self, reports: Iterable[ReportRow]) -> int:
        """Replace the whole report set (chunks and embeddings go with it, by cascade)."""
        rows = list(reports)
        with self.db:
            self.db.execute("DELETE FROM reports")
            self.db.execute("DELETE FROM chunk_sets")
            self.db.executemany(
                "INSERT INTO reports(report_id,title,program,weakness,substate,created_at,disclosed_at,severity,vote_count,body,body_hash,"
                "source,source_url,cwe,quality_flags,quality_score,when_to_try,images,injection_hits)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(r.report_id, r.title, r.program, r.weakness, r.substate, r.created_at, r.disclosed_at, r.severity, r.vote_count,
                  r.body, r.body_hash, r.source, r.source_url, r.cwe, r.quality_flags, r.quality_score, r.when_to_try, r.images,
                  r.injection_hits) for r in rows])
        return len(rows)

    def reports(self) -> list[ReportRow]:
        return [ReportRow(**dict(r)) for r in self.db.execute("SELECT * FROM reports ORDER BY report_id")]

    def report(self, report_id: int) -> ReportRow | None:
        row = self.db.execute("SELECT * FROM reports WHERE report_id=?", (report_id,)).fetchone()
        return ReportRow(**dict(row)) if row else None

    def set_when_to_try(self, report_id: int, text: str) -> None:
        with self.db:
            self.db.execute("UPDATE reports SET when_to_try=? WHERE report_id=?", (text, report_id))

    def report_ids(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT report_id FROM reports")}

    # ---- chunks ----
    def replace_chunk_set(self, name: str, config: dict, chunks: Sequence[NewChunk]) -> int:
        with self.db:
            self.db.execute("DELETE FROM chunk_sets WHERE name=?", (name,))   # cascades to chunks + embeddings
            self.db.execute("INSERT INTO chunk_sets(name,config_json) VALUES(?,?)", (name, json.dumps(config, sort_keys=True)))
            self.db.executemany(
                "INSERT INTO chunks(chunk_set,report_id,ordinal,section,has_code,n_tokens,header,text,meta_json) VALUES(?,?,?,?,?,?,?,?,?)",
                [(name, c.report_id, c.ordinal, c.section, int(c.has_code), c.n_tokens, c.header, c.text,
                  json.dumps(c.meta, ensure_ascii=False, sort_keys=True)) for c in chunks])
        return len(chunks)

    def chunk_set_config(self, name: str) -> dict | None:
        row = self.db.execute("SELECT config_json FROM chunk_sets WHERE name=?", (name,)).fetchone()
        return json.loads(row["config_json"]) if row else None

    def chunk_sets(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT name FROM chunk_sets ORDER BY name")]

    def chunks(self, chunk_set: str) -> list[ChunkRow]:
        rows = self.db.execute("SELECT * FROM chunks WHERE chunk_set=? ORDER BY report_id, ordinal", (chunk_set,)).fetchall()
        return [ChunkRow(chunk_id=r["chunk_id"], chunk_set=r["chunk_set"], report_id=r["report_id"], ordinal=r["ordinal"],
                         section=r["section"], has_code=bool(r["has_code"]), n_tokens=r["n_tokens"],
                         header=r["header"], text=r["text"], meta=json.loads(r["meta_json"] or "{}")) for r in rows]

    # ---- embeddings ----
    def put_embeddings(self, profile: str, model: str, chunk_ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(chunk_ids) != len(vectors):
            raise StoreError("chunk_ids and vectors have different lengths")
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id,profile,model,dim,vec) VALUES(?,?,?,?,?)",
                [(int(cid), profile, model, int(v.shape[0]), vec_to_blob(v)) for cid, v in zip(chunk_ids, vectors)])

    def embedded_ids(self, chunk_set: str, profile: str) -> set[int]:
        rows = self.db.execute(
            "SELECT e.chunk_id FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id WHERE c.chunk_set=? AND e.profile=?",
            (chunk_set, profile))
        return {r[0] for r in rows}

    def embedding_models(self, chunk_set: str, profile: str) -> set[tuple[str, int]]:
        rows = self.db.execute(
            "SELECT DISTINCT e.model, e.dim FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id "
            "WHERE c.chunk_set=? AND e.profile=?", (chunk_set, profile))
        return {(r[0], r[1]) for r in rows}

    def matrix(self, chunk_set: str, profile: str) -> tuple[list[int], np.ndarray]:
        """(chunk_ids, N x dim float32 matrix) for every embedded chunk in the set."""
        rows = self.db.execute(
            "SELECT c.chunk_id, e.vec FROM chunks c JOIN embeddings e ON e.chunk_id=c.chunk_id "
            "WHERE c.chunk_set=? AND e.profile=? ORDER BY c.chunk_id", (chunk_set, profile)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r[0] for r in rows], np.vstack([blob_to_vec(r[1]) for r in rows]).astype(np.float32)

    def delete_embeddings(self, chunk_set: str, profile: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM embeddings WHERE profile=? AND chunk_id IN (SELECT chunk_id FROM chunks WHERE chunk_set=?)",
                            (profile, chunk_set))

    # ---- summary ----
    def stats(self) -> dict:
        one = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        per_set = {r["chunk_set"]: r["n"] for r in self.db.execute("SELECT chunk_set, COUNT(*) n FROM chunks GROUP BY chunk_set")}
        per_emb = [dict(r) for r in self.db.execute(
            "SELECT c.chunk_set, e.profile, e.model, e.dim, COUNT(*) n FROM embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id "
            "GROUP BY c.chunk_set, e.profile, e.model, e.dim ORDER BY 1,2")]
        sources = {r["source"] or "dataset": r["n"] for r in self.db.execute("SELECT source, COUNT(*) n FROM reports GROUP BY source")}
        flags: dict[str, int] = {}
        for r in self.reports():
            for f in r.flags:
                flags[f] = flags.get(f, 0) + 1
        return {"reports": one("SELECT COUNT(*) FROM reports"), "sources": sources, "quality_flags": dict(sorted(flags.items())),
                "chunks_per_set": per_set, "embeddings": per_emb}
