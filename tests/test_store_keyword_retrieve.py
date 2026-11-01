import numpy as np
import pytest

from hrag.config import RetrievalSettings
from hrag.keyword import Bm25Index, tokenize
from hrag.retrieve import Retriever, reciprocal_rank_fusion
from hrag.store import Store, blob_to_vec, vec_to_blob
from tests.helpers import FakeQueryEmbedder, corpus_store, report, unit

SETTINGS = RetrievalSettings(mode="hybrid", top_k=3, candidates=10, rrf_k=60, max_chunks_per_report=2, dedupe_cosine=0.0)


# ---------- tokenizer / BM25 (keyword search) ----------
def test_tokenizer_keeps_identifiers_whole_and_in_parts():
    toks = tokenize("Exploit CVE-2021-44228 via /api/v1/users")
    assert "cve-2021-44228" in toks and "44228" in toks and "api/v1/users" in toks and "api" in toks
    assert tokenize("/api/v1/users") == tokenize("api/v1/users")      # query and document tokenise identically
    assert "via" in toks and "the" not in tokenize("the bug")


def test_bm25_ranks_exact_identifier_first_and_ignores_unknown_terms():
    idx = Bm25Index([10, 20, 30], ["log4j rce CVE-2021-44228", "generic rce write-up", "idor on invoices"])
    assert idx.search("CVE-2021-44228", 3)[0][0] == 10
    assert idx.search("zzzunknown", 3) == []
    assert {d for d, _ in idx.search("rce", 3)[:2]} == {10, 20}
    assert idx.search("rce", 1) == idx.search("rce", 3)[:1]                  # n limits the result, order is stable


def test_bm25_prefers_the_rarer_term_and_shorter_document():
    idx = Bm25Index([1, 2, 3], ["ssrf ssrf ssrf in webhook", "ssrf plus a very long text " + "filler " * 40, "unrelated idor note"])
    assert [d for d, _ in idx.search("ssrf", 3)] == [1, 2]                  # more occurrences + shorter wins; no match for doc 3
    assert idx.search("webhook idor", 3)[0][0] in (1, 3)


# ---------- RRF (hybrid fusion) ----------
def test_rrf_known_answer_and_deterministic_ties():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    tie = reciprocal_rank_fusion([[5], [7]], k=60)
    assert [i for i, _ in tie] == [5, 7]                      # equal scores -> lower id first
    assert reciprocal_rank_fusion([[1, 1, 1]], k=60) == [(1, 1 / 61)]   # repeats count once
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=-1)


# ---------- store ----------
def test_vector_blob_roundtrip_and_store_stats():
    v = unit(1, 2, 3)
    assert np.allclose(blob_to_vec(vec_to_blob(v)), v)
    store, _ = corpus_store()
    s = store.stats()
    assert s["reports"] == 3 and s["chunks_per_set"] == {"c400": 4}
    assert s["embeddings"][0]["n"] == 4 and s["embeddings"][0]["dim"] == 3
    ids, matrix = store.matrix("c400", "p")
    assert matrix.shape == (4, 3) and len(ids) == 4


def test_replacing_a_chunk_set_drops_its_embeddings_and_replacing_reports_drops_everything():
    store, _ = corpus_store()
    store.replace_chunk_set("c400", {"max_tokens": 400}, [])
    assert store.stats()["embeddings"] == [] and store.chunks("c400") == []
    store2, _ = corpus_store()
    store2.replace_reports([report(9, "New", "body")])
    assert store2.chunk_sets() == [] and store2.stats()["embeddings"] == []


# ---------- retrieval ----------
def retriever(store, table):
    return Retriever(store, "c400", profile="p", embedder=FakeQueryEmbedder(table))


def test_vector_mode_orders_by_cosine():
    store, _ = corpus_store()
    r = retriever(store, {"q": unit(0, 1, 0.01)})
    hits = r.search("q", SETTINGS, mode="vector", top_k=3)
    assert [h.report_id for h in hits][:2] == [2, 2] and hits[-1].report_id in (1, 3)
    assert [h.rank for h in hits] == [1, 2, 3]


def test_bm25_finds_identifier_that_the_vector_leg_misses():
    store, _ = corpus_store()
    r = retriever(store, {"CVE-2021-44228": unit(0, 0, 1)})      # embedding points at report 3, keywords at report 1
    assert r.search("CVE-2021-44228", SETTINGS, mode="vector", top_k=1)[0].report_id == 3
    assert r.search("CVE-2021-44228", SETTINGS, mode="bm25", top_k=1)[0].report_id == 1
    hybrid = r.search("CVE-2021-44228", SETTINGS, mode="hybrid", top_k=2)
    assert {h.report_id for h in hybrid} == {1, 3}
    assert all(set(h.via) <= {"vector", "bm25"} and h.via for h in hybrid)
    assert {k for h in hybrid for k in h.via} == {"vector", "bm25"}     # hybrid really used both legs
    for h in hybrid:                                                    # and the score IS the rank fusion, not a cosine / BM25 score
        assert h.score == pytest.approx(sum(1 / (60 + rank) for rank in h.via.values()))
    assert hybrid[0].score > hybrid[1].score


def test_keyword_mode_needs_no_embeddings_and_hybrid_does():
    store, _ = corpus_store()
    keyword_only = Retriever(store, "c400")                                # no embedder, no profile
    assert keyword_only.search("CVE-2021-44228", SETTINGS, mode="bm25", top_k=1)[0].report_id == 1
    with pytest.raises(ValueError, match="vector search needs"):
        keyword_only.search("CVE-2021-44228", SETTINGS, mode="hybrid")
    with pytest.raises(ValueError, match="unknown mode"):
        keyword_only.search("x", SETTINGS, mode="fuzzy")


def test_per_report_cap_and_near_duplicate_suppression():
    store, _ = corpus_store()
    r = retriever(store, {"q": unit(0, 1, 0)})
    capped = r.search("q", SETTINGS.__class__(**{**SETTINGS.__dict__, "max_chunks_per_report": 1}), mode="vector", top_k=3)
    assert len({h.report_id for h in capped}) == len(capped)
    deduped = r.search("q", SETTINGS.__class__(**{**SETTINGS.__dict__, "dedupe_cosine": 0.95}), mode="vector", top_k=3)
    assert sum(1 for h in deduped if h.report_id == 2) == 1       # the two XSS chunks are ~identical vectors


def test_rerank_reorders_candidates():
    store, _ = corpus_store()

    class Flip:
        def rerank(self, query, documents, top_k):
            return [(len(documents) - 1 - i, 1.0 - i / 10) for i in range(len(documents))]

    r = Retriever(store, "c400", profile="p", embedder=FakeQueryEmbedder({"q": unit(1, 0, 0)}), reranker=Flip())
    plain = r.search("q", SETTINGS, mode="vector", top_k=4)
    flipped = r.search("q", SETTINGS, mode="vector", top_k=4, rerank=True)
    assert [h.chunk.chunk_id for h in flipped] == [h.chunk.chunk_id for h in reversed(plain)]


def test_helpful_errors():
    store, _ = corpus_store()
    with pytest.raises(ValueError, match="dimensions"):
        retriever(store, {"q": unit(1, 0, 0, 0)}).search("q", SETTINGS, mode="vector")
    with pytest.raises(ValueError, match="no reranker"):
        retriever(store, {"q": unit(1, 0, 0)}).search("q", SETTINGS, mode="vector", rerank=True)
    with pytest.raises(ValueError, match="empty"):
        Retriever(Store(":memory:"), "nope")
