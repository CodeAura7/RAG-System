"""Glue between the store and the models: build chunks, embed chunks."""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Protocol, Sequence

import numpy as np

from .chunking import chunk_report, make_header
from .config import ChunkSetCfg, EmbeddingProfile
from .store import NewChunk, Store

EMBED_SLICE = 128          # save progress after every slice, so a failure never loses finished work


class PipelineError(RuntimeError):
    pass


class DocEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None) -> np.ndarray: ...


def chunk_meta(r, d) -> dict:
    """The report's metadata, copied onto every chunk so a result can be filtered, explained and cited on its own."""
    return {"report_id": r.report_id, "title": r.title, "program": r.program, "weakness": r.weakness, "cwe": r.cwe,
            "severity": r.severity, "substate": r.substate, "source": r.source, "source_url": r.source_url,
            "disclosed_at": r.disclosed_at, "quality_flags": r.flags, "quality_score": r.quality_score,
            "images": len(r.image_list), "section": d.section, "ordinal": d.ordinal, "has_code": d.has_code}


def build_chunks(store: Store, cfg: ChunkSetCfg) -> dict:
    reports = store.reports()
    if not reports:
        raise PipelineError("no reports in the database - run `python -m hrag collect` first")
    new: list[NewChunk] = []
    for r in reports:
        for d in chunk_report(r.body, max_tokens=cfg.max_tokens, overlap_tokens=cfg.overlap_tokens, min_tokens=cfg.min_tokens):
            header = make_header(r.report_id, r.title, r.program, r.weakness, r.substate, d.section,
                                 severity=r.severity, when_to_try=r.when_to_try)
            new.append(NewChunk(r.report_id, d.ordinal, d.section, d.has_code, d.n_tokens, header, d.text, chunk_meta(r, d)))
    store.replace_chunk_set(cfg.name, asdict(cfg), new)
    tokens = [c.n_tokens for c in new]
    return {"chunk_set": cfg.name, "reports": len(reports), "chunks": len(new),
            "tokens_min": min(tokens, default=0), "tokens_median": int(np.median(tokens)) if tokens else 0,
            "tokens_max": max(tokens, default=0), "with_code": sum(c.has_code for c in new)}


def embed_chunks(store: Store, chunk_set: str, profile: EmbeddingProfile, embedder: DocEmbedder, *, force: bool = False,
                 progress: Callable[[int, int], None] | None = None) -> dict:
    chunks = store.chunks(chunk_set)
    if not chunks:
        raise PipelineError(f"chunk set '{chunk_set}' is empty - run `python -m hrag chunk --set {chunk_set}` first")
    if force:
        store.delete_embeddings(chunk_set, profile.name)
    existing = store.embedding_models(chunk_set, profile.name)
    if existing and not any(m == profile.model for m, _ in existing):
        raise PipelineError(f"profile '{profile.name}' already holds vectors from {sorted(existing)}, but it is now configured "
                            f"for model '{profile.model}'. Re-run with --force to re-embed.")
    done = store.embedded_ids(chunk_set, profile.name)
    todo = [c for c in chunks if c.chunk_id not in done]
    for start in range(0, len(todo), EMBED_SLICE):
        part = todo[start:start + EMBED_SLICE]
        vectors = embedder.embed_documents([c.embed_text for c in part])
        dims = {d for _, d in existing}
        if dims and vectors.shape[1] not in dims:
            raise PipelineError(f"new vectors have {vectors.shape[1]} dimensions but stored ones have {sorted(dims)}; use --force")
        store.put_embeddings(profile.name, profile.model, [c.chunk_id for c in part], vectors)
        existing = store.embedding_models(chunk_set, profile.name)
        if progress:
            progress(min(start + EMBED_SLICE, len(todo)), len(todo))
    return {"chunk_set": chunk_set, "profile": profile.name, "model": profile.model,
            "embedded_now": len(todo), "already_done": len(done), "total": len(chunks)}
