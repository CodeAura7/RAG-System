"""Shared test fixtures: a tiny in-memory corpus and fake model clients."""
from __future__ import annotations

import hashlib

import numpy as np

from hrag.store import NewChunk, ReportRow, Store


def report(rid: int, title: str, body: str, program: str = "acme", weakness: str = "XSS", substate: str = "resolved") -> ReportRow:
    return ReportRow(rid, title, program, weakness, substate, "2023-01-01T00:00:00Z", "2023-02-01T00:00:00Z", "", 0,
                     body, hashlib.sha256(body.encode()).hexdigest())


def unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeQueryEmbedder:
    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table
        self.calls: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        return self.table[text]


def corpus_store() -> tuple[Store, dict[int, int]]:
    """3 reports -> 4 chunks with hand-made 3-d vectors. Returns (store, {report_id: first chunk_id})."""
    store = Store(":memory:")
    store.replace_reports([
        report(1, "Log4Shell RCE", "x"), report(2, "Stored XSS in comments", "y"), report(3, "IDOR on invoices", "z"),
    ])
    chunks = [
        NewChunk(1, 0, "summary", False, 20, "HackerOne report #1: Log4Shell RCE | section: summary", "JNDI lookup in log4j CVE-2021-44228 leads to RCE"),
        NewChunk(2, 0, "steps", True, 20, "HackerOne report #2: Stored XSS in comments | section: steps", "post a comment containing a script tag payload"),
        NewChunk(2, 1, "impact", False, 20, "HackerOne report #2: Stored XSS in comments | section: impact", "script runs for every visitor who views the comment"),
        NewChunk(3, 0, "summary", False, 20, "HackerOne report #3: IDOR on invoices | section: summary", "change the invoice id in the URL to read other customers' invoices"),
    ]
    store.replace_chunk_set("c400", {"max_tokens": 400}, chunks)
    ids = [c.chunk_id for c in store.chunks("c400")]
    vectors = np.vstack([unit(1, 0, 0), unit(0, 1, 0), unit(0.05, 1, 0), unit(0, 0, 1)])
    store.put_embeddings("p", "fake-model", ids, vectors)
    return store, {c.report_id: c.chunk_id for c in reversed(store.chunks("c400"))}
