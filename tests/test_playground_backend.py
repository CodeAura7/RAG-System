"""Tests for the RAG Playground backend. Like the rest of the project: tiny corpus, fake query embedder, fake LLM, no network."""
from __future__ import annotations

import dataclasses
import json
import math
import sqlite3

import numpy as np
import pytest

from hrag.config import load_config
from hrag.retrieve import Retriever
from hrag.store import NewChunk
from playground import backend as pb

from .helpers import FakeQueryEmbedder, corpus_store, unit

QUERIES = {
    "how does the xss payload run": unit(0.02, 1, 0),
    "log4j jndi": unit(1, 0.05, 0),
    "invoice id": unit(0, 0.1, 1),
}


class FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.prompts = reply, []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.reply


@pytest.fixture()
def be(tmp_path):
    """A Backend over a FILE copy of the shared test corpus (the backend opens its own connections), with a fake embedder."""
    store, first = corpus_store()
    # a second chunk set so configurations can be compared
    whole = [NewChunk(1, 0, "summary", False, 50, "HackerOne report #1: Log4Shell RCE | section: summary", "JNDI lookup in log4j CVE-2021-44228 leads to RCE"),
             NewChunk(2, 0, "steps", True, 50, "HackerOne report #2: Stored XSS in comments | section: steps", "post a comment containing a script tag payload; the script runs for every visitor"),
             NewChunk(3, 0, "summary", False, 50, "HackerOne report #3: IDOR on invoices | section: summary", "change the invoice id in the URL to read other customers' invoices")]
    store.replace_chunk_set("whole", {"max_tokens": 100000}, whole)
    ids = [c.chunk_id for c in store.chunks("whole")]
    store.put_embeddings("p", "fake-model", ids, np.vstack([unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]))
    path = tmp_path / "hrag.db"
    dest = sqlite3.connect(path)
    store.db.backup(dest)
    dest.close()
    cfg = dataclasses.replace(load_config(), db_path=path, eval_dir=tmp_path / "eval")
    emb = FakeQueryEmbedder(QUERIES)

    def factory(st, chunk_set, profile, need_vectors, need_rerank):
        return Retriever(st, chunk_set, profile="p" if need_vectors else None, embedder=emb if need_vectors else None)

    backend = pb.Backend(cfg, retriever_factory=factory, llm_factory=lambda name: FakeLLM("{}"))
    backend.fake_embedder = emb
    return backend


# ---------- retrieval transparency ----------

def test_supplied_chunks_are_exactly_what_search_returns_for_top_k(be):
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=2, pool=4)
    plain = be.retriever("c400", "p", need_vectors=True, need_rerank=False).search(
        run.question, dataclasses.replace(be.cfg.retrieval, mode="hybrid", top_k=2), top_k=2)
    assert [v.chunk_id for v in run.supplied] == [h.chunk.chunk_id for h in plain]
    assert len(run.views) > len(run.supplied) and all(not v.supplied for v in run.extra)
    assert [v.chunk_id for v in run.views[:2]] == [h.chunk.chunk_id for h in run.hits[:2]]


def test_leg_ranks_shown_equal_the_ranks_search_itself_reports(be):
    """Guards the two underscore calls: if the retriever changes, these numbers would silently mislead."""
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=3, pool=4)
    for v, h in zip(run.views, run.hits):
        assert v.vector_rank == h.via.get("vector") and v.bm25_rank == h.via.get("bm25")
        expected = sum(1.0 / (run.rrf_k + r) for r in (v.vector_rank, v.bm25_rank) if r)
        assert v.score == pytest.approx(expected)
        assert (v.rrf_vector or 0) + (v.rrf_bm25 or 0) == pytest.approx(expected)


def test_single_leg_modes_show_only_their_own_leg(be):
    vec = be.retrieve("log4j jndi", chunk_set="c400", mode="vector", k=2)
    assert all(v.bm25_rank is None and v.vector_rank for v in vec.views) and vec.views[0].vector_score == pytest.approx(vec.views[0].score)
    kw = be.retrieve("log4j jndi", chunk_set="c400", mode="bm25", k=2)
    assert all(v.vector_rank is None for v in kw.views) and kw.views[0].bm25_score == pytest.approx(kw.views[0].score)
    assert be.fake_embedder.calls == ["log4j jndi"]          # bm25 never embeds; vector embedded once


def test_one_paid_embedding_call_per_distinct_question_even_across_chunk_sets(be):
    for cs in ("c400", "whole", "c400"):
        be.retrieve("invoice id", chunk_set=cs, mode="hybrid", k=2)
    assert be.fake_embedder.calls == ["invoice id"]
    assert be.retrieve("invoice id", chunk_set="c400", mode="hybrid").timings["embed_cached"] is True


def test_timings_split_embedding_from_search(be):
    t = be.retrieve("log4j jndi", chunk_set="c400", mode="hybrid", k=2).timings
    assert t["total_ms"] >= t["search_ms"] >= 0 and t["embed_first_ms"] is not None and t["embed_cached"] is False


def test_empty_question_and_missing_database_are_refused_without_creating_files(be, tmp_path):
    with pytest.raises(pb.PlaygroundError):
        be.retrieve("   ", chunk_set="c400", mode="bm25")
    ghost = pb.Backend(dataclasses.replace(be.cfg, db_path=tmp_path / "nope" / "x.db"))
    with pytest.raises(pb.PlaygroundError, match="never creates"):
        ghost.overview()
    assert not (tmp_path / "nope").exists()


# ---------- answers and citations ----------

def test_answer_receives_only_the_supplied_chunks_and_shows_numbered_citations(be):
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=2, pool=4)
    shown = [v.chunk_id for v in run.supplied]
    extra = run.extra[0]
    llm = FakeLLM(json.dumps({"answer": f"A script tag runs for visitors [C{shown[0]}], see also [C{shown[1]}].",
                              "citations": [f"C{shown[0]}", f"C{extra.chunk_id}"], "abstained": False}))
    be._llm_factory = lambda name: llm
    ans = be.answer(run, "claude_sonnet")
    assert f"[C{extra.chunk_id}]" not in llm.prompts[0] and extra.text not in llm.prompts[0]
    assert all(f"[C{c}]" in llm.prompts[0] for c in shown)
    assert ans.display_text == "A script tag runs for visitors [1], see also [2]."
    assert ans.cited_valid == [shown[0], shown[1]]
    assert ans.cited_invalid == [extra.chunk_id]               # cited a chunk that was retrieved but never shown to the model


def test_renumber_handles_groups_unknown_ids_and_leaves_other_brackets_alone():
    assert pb.renumber_citations("x [C5, C7] y [C9] z [note] [1]", [7, 5]) == "x [2, 1] y [C9?] z [note] [1]"
    assert pb.inline_citation_ids("a [C5] b [C7, c5] c [C9]") == [5, 7, 9]


def test_safe_markdown_blocks_image_fetches():
    assert "![" not in pb.safe_markdown("look ![x](http://evil.example/p.png?q=secret)")


# ---------- metrics ----------

def _rec(labels: dict[int, str], **extra) -> dict:
    rows = [{"chunk_id": cid, "report_id": cid * 10, "rank": i, "label": lab} for i, (cid, lab) in enumerate(labels.items(), start=1)]
    return {"chunk_set": "c400", "mode": "hybrid", "embed_profile": "p", "retrieved": rows, "timings": {"total_ms": 12.5}, **extra}


def test_metrics_match_hand_computed_values():
    m = pb.record_metrics(_rec({1: "irrelevant", 2: "relevant", 3: "irrelevant", 4: "relevant", 5: "irrelevant"}))
    assert (m["P@1"], m["P@5"]) == (0.0, 0.4) and m["R@3"] == 0.5 and m["R@5"] == 1.0 and m["MRR"] == 0.5
    dcg, ideal = 1 / math.log2(3) + 1 / math.log2(5), 1 / math.log2(2) + 1 / math.log2(3)
    assert m["nDCG@5"] == pytest.approx(dcg / ideal) and m["judged@5"] == 1.0 and m["latency_ms"] == 12.5


def test_unjudged_chunks_are_not_counted_relevant_and_are_reported_as_such():
    m = pb.record_metrics(_rec({1: "relevant", 2: "unjudged", 3: "unjudged"}))
    assert m["P@3"] == pytest.approx(1 / 3) and m["judged@3"] == pytest.approx(1 / 3)


def test_nothing_judged_gives_undefined_not_zero():
    m = pb.record_metrics(_rec({1: "unjudged", 2: "unjudged"}))
    assert m["P@5"] is None and m["R@5"] is None and m["MRR"] is None and m["nDCG@5"] is None
    assert pb.record_metrics(_rec({1: "irrelevant"}))["R@5"] is None       # judged, but nothing relevant exists to find


def test_aggregate_groups_by_configuration_and_ignores_undefined_values():
    a = _rec({1: "relevant", 2: "irrelevant"})
    b = _rec({1: "irrelevant", 2: "relevant"})
    c = dict(_rec({1: "relevant"}), mode="bm25")
    rows = {r["configuration"]: r for r in pb.aggregate([a, b, c])}
    assert rows["c400 | hybrid | p"]["evaluations"] == 2 and rows["c400 | hybrid | p"]["MRR"] == 0.75
    assert rows["c400 | bm25 | p"]["MRR"] == 1.0


def test_answer_summary_counts_only_rated_answers():
    recs = [{"answer": {"accuracy": "correct", "faithfulness": pb.FAITHFULNESS[1]}},
            {"answer": {"accuracy": "incorrect", "faithfulness": "not rated"}},
            {"answer": None}]
    s = pb.answer_summary(recs)
    assert s["rated_accuracy"] == 2 and s["accuracy_correct_rate"] == 0.5 and s["rated_faithfulness"] == 1 and s["faithful_rate"] == 1.0


# ---------- saving ----------

def test_saved_record_round_trips_with_unicode_and_survives_a_torn_line(be, tmp_path):
    run = be.retrieve("log4j jndi", chunk_set="c400", mode="hybrid", k=2, pool=3)
    labels = {run.views[0].chunk_id: "relevant", run.views[1].chunk_id: "irrelevant"}
    log = pb.JudgmentLog(tmp_path / "e" / "j.jsonl")
    log.append(pb.build_record(run, labels, None, "not rated", "not rated", "naïve – “quotes” ✓"))
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"torn": ')                                   # a half-written line
    log.append(pb.build_record(run, {}, None, "not rated", "not rated", ""))
    recs = log.load()
    assert len(recs) == 2 and recs[0]["notes"] == "naïve – “quotes” ✓" and recs[0]["answer"] is None
    assert recs[0]["relevant_report_ids"] == [run.views[0].report_id]
    assert [r["label"] for r in recs[0]["retrieved"][:2]] == ["relevant", "irrelevant"] and recs[0]["retrieved"][0]["supplied"] is True


def test_saving_never_touches_the_database(be):
    before = be.cfg.db_path.read_bytes()
    run = be.retrieve("log4j jndi", chunk_set="c400", mode="hybrid", k=2)
    pb.judgment_log(be.cfg).append(pb.build_record(run, {}, None, "not rated", "not rated", ""))
    be.overview()
    assert be.cfg.db_path.read_bytes() == before


# ---------- comparing configurations ----------

def test_compare_configs_scores_against_reports_marked_relevant(be):
    run = be.retrieve("log4j jndi", chunk_set="c400", mode="hybrid", k=2, pool=4)
    top = run.views[0]
    rec = pb.build_record(run, {top.chunk_id: "relevant"}, None, "not rated", "not rated", "")
    rows = pb.compare_configs(be, [rec], chunk_sets=["c400", "whole"], modes=["bm25", "hybrid"], k=3)
    assert {(r["chunk_set"], r["mode"]) for r in rows} == {("c400", "bm25"), ("c400", "hybrid"), ("whole", "bm25"), ("whole", "hybrid")}
    assert all(r["questions"] == 1 and r["MRR"] == 1.0 for r in rows)      # report 1 (Log4Shell) is first everywhere here


def test_compare_configs_needs_at_least_one_relevant_mark(be):
    run = be.retrieve("log4j jndi", chunk_set="c400", mode="bm25")
    with pytest.raises(pb.PlaygroundError, match="relevant"):
        pb.compare_configs(be, [pb.build_record(run, {}, None, "not rated", "not rated", "")], chunk_sets=["c400"], modes=["bm25"], k=3)


# ---------- chunk explorer ----------

def test_chunk_checks_flag_split_code_fences_and_midsentence_starts(be):
    chunks = be.chunks("c400")
    xss = next(c for c in chunks if c.section == "steps")
    split = dataclasses.replace(xss, text="```js\nalert(1)", n_tokens=80)
    mid = dataclasses.replace(xss, text="and then the script runs.", n_tokens=80)
    assert any("fence" in f for f in pb.chunk_checks(split, 40))
    assert any("mid-sentence" in f for f in pb.chunk_checks(mid, 40))
    assert pb.chunk_checks(dataclasses.replace(xss, text="Post a comment.", n_tokens=80), 40) == []
    s = pb.chunk_set_summary([split, mid], 40)
    assert s["chunks"] == 2 and s["split_code_fence"] == 1 and s["flagged_any"] == 2
