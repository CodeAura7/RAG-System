"""Retrieval, three modes:
  vector - semantic: exact cosine similarity over the stored embeddings;
  bm25   - keyword: in-memory BM25 (needs no embeddings, keeps identifiers like CVE-2021-44228 whole);
  hybrid - both, merged with Reciprocal Rank Fusion: score(chunk) = sum over the ranked lists of 1 / (k + rank), k = 60
           (Cormack, Clarke & Buettcher, SIGIR 2009).
Then an optional rerank, near-duplicate suppression and a cap on chunks per report. Ties are broken by chunk id so
results are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

import numpy as np

from .config import RetrievalSettings
from .keyword import Bm25Index
from .store import ChunkRow, Store


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


class RerankerLike(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]: ...


@dataclass
class Hit:
    rank: int
    chunk: ChunkRow
    score: float
    via: dict[str, int] = field(default_factory=dict)      # which legs found it, with their rank

    @property
    def report_id(self) -> int:
        return self.chunk.report_id


def reciprocal_rank_fusion(lists: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[int, float] = {}
    for ranked in lists:
        seen: set[int] = set()
        for position, item in enumerate(ranked, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


class Retriever:
    def __init__(self, store: Store, chunk_set: str, *, profile: str | None = None,
                 embedder: QueryEmbedder | None = None, reranker: RerankerLike | None = None):
        self.chunk_set = chunk_set
        self.profile = profile
        self.embedder = embedder
        self.reranker = reranker
        self.chunks = {c.chunk_id: c for c in store.chunks(chunk_set)}
        if not self.chunks:
            raise ValueError(f"chunk set '{chunk_set}' is empty - run `python -m hrag chunk` first")
        ids = sorted(self.chunks)
        self._bm25 = Bm25Index(ids, [self.chunks[i].embed_text for i in ids])
        self._vec_ids: list[int] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._row_of: dict[int, int] = {}
        if profile:
            self._vec_ids, self._matrix = store.matrix(chunk_set, profile)
            self._row_of = {cid: i for i, cid in enumerate(self._vec_ids)}

    # ---- legs ----
    def _vector_ranked(self, query: str, n: int) -> list[tuple[int, float]]:
        """(chunk_id, cosine similarity), best first."""
        if self.embedder is None or not len(self._vec_ids):
            raise ValueError("vector search needs an embedder and embedded chunks - run `python -m hrag embed` first")
        q = self.embedder.embed_query(query)
        if q.shape[0] != self._matrix.shape[1]:
            raise ValueError(f"query vector has {q.shape[0]} dimensions but stored vectors have {self._matrix.shape[1]} "
                             f"- the profile's model/dimensions changed; re-run `embed --force`")
        sims = self._matrix @ q
        order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), self._vec_ids[i]))[:n]
        return [(self._vec_ids[i], float(sims[i])) for i in order]

    def _bm25_ranked(self, query: str, n: int) -> list[tuple[int, float]]:
        """(chunk_id, BM25 score), best first."""
        return self._bm25.search(query, n)

    # ---- public ----
    def search(self, query: str, settings: RetrievalSettings, *, mode: str | None = None, top_k: int | None = None,
               rerank: bool | None = None) -> list[Hit]:
        mode = mode or settings.mode
        k = top_k or settings.top_k
        n = max(settings.candidates, k)
        use_rerank = settings.rerank if rerank is None else rerank
        scored: dict[str, list[tuple[int, float]]] = {}
        if mode in ("vector", "hybrid"):
            scored["vector"] = self._vector_ranked(query, n)
        if mode in ("bm25", "hybrid"):
            scored["bm25"] = self._bm25_ranked(query, n)
        if not scored:
            raise ValueError(f"unknown mode '{mode}': use vector, bm25 or hybrid")
        legs = {name: [cid for cid, _ in pairs] for name, pairs in scored.items()}

        if len(legs) == 1:      # a single leg keeps its own meaningful score (cosine or BM25)
            fused = next(iter(scored.values()))
        else:                   # hybrid: scores of different kinds cannot be added, so fuse by rank
            fused = reciprocal_rank_fusion(list(legs.values()), settings.rrf_k)
        fused = fused[:n]
        ranks = {name: {cid: i for i, cid in enumerate(ids, start=1)} for name, ids in legs.items()}
        hits = [Hit(0, self.chunks[cid], score, {name: r[cid] for name, r in ranks.items() if cid in r}) for cid, score in fused]

        if use_rerank and hits:
            if self.reranker is None:
                raise ValueError("rerank requested but no reranker is configured")
            order = self.reranker.rerank(query, [h.chunk.embed_text for h in hits], top_k=len(hits))
            hits = [replace(hits[i], score=s) for i, s in order]

        hits = self._suppress(hits, k, settings)
        return [replace(h, rank=i) for i, h in enumerate(hits, start=1)]

    def _suppress(self, hits: list[Hit], k: int, settings: RetrievalSettings) -> list[Hit]:
        """Walk the ranking best-first; skip near-duplicates and over-represented reports."""
        kept: list[Hit] = []
        per_report: dict[int, int] = {}
        for h in hits:
            if per_report.get(h.report_id, 0) >= settings.max_chunks_per_report:
                continue
            if settings.dedupe_cosine > 0 and self._too_similar(h, kept, settings.dedupe_cosine):
                continue
            kept.append(h)
            per_report[h.report_id] = per_report.get(h.report_id, 0) + 1
            if len(kept) == k:
                break
        return kept

    def _too_similar(self, hit: Hit, kept: list[Hit], threshold: float) -> bool:
        row = self._row_of.get(hit.chunk.chunk_id)
        if row is None:
            return False
        for other in kept:
            other_row = self._row_of.get(other.chunk.chunk_id)
            if other_row is not None and float(self._matrix[row] @ self._matrix[other_row]) >= threshold:
                return True
        return False
