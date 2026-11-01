import json
import math

import numpy as np
import pytest

import hrag.pipeline as pipeline
from hrag.answer import AnswerResult, answer_question, build_user_prompt, parse_citation, parse_json_object
from hrag.config import ChunkSetCfg, EmbeddingProfile, RetrievalSettings
from hrag.evaluate import (EvalError, Question, check_against_store, compare_rows, evaluate_answers, evaluate_retrieval,
                           load_questions, ndcg_at_k, paired_bootstrap, recall_at_k, reciprocal_rank, write_results)
from hrag.pipeline import PipelineError, build_chunks, embed_chunks
from hrag.questions import draft_questions
from hrag.retrieve import Retriever
from hrag.store import Store
from tests.helpers import FakeQueryEmbedder, corpus_store, report, unit

SETTINGS = RetrievalSettings(mode="hybrid", top_k=3, candidates=10, rrf_k=60, max_chunks_per_report=2, dedupe_cosine=0.0)


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply(system, user) if callable(self.reply) else self.reply


class CountingEmbedder:
    """Deterministic fake document embedder; can be told to fail on its Nth call."""
    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = 0, fail_on_call

    def embed_documents(self, texts, progress=None):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("provider down")
        rng = [np.random.default_rng(abs(hash(t)) % (2 ** 32)).normal(size=8) for t in texts]
        m = np.vstack(rng).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


# ---------- pipeline ----------
def reports_store():
    store = Store(":memory:")
    long_body = "Intro text about the bug. " * 30 + "\n\n## Steps To Reproduce\n```\ncurl -X POST http://x/login -d 'a=b'\n```\n\n## Impact\nAccount takeover."
    store.replace_reports([report(1, "Auth bypass", long_body), report(2, "Short one", "Short body " * 8)])
    return store


def test_build_chunks_summary_and_headers():
    store = reports_store()
    out = build_chunks(store, ChunkSetCfg("c100", 100, 10, 10))
    assert out["chunks"] > 2 and out["with_code"] >= 1
    chunks = store.chunks("c100")
    assert all(c.header.startswith("HackerOne report #") for c in chunks) and store.chunk_set_config("c100")["max_tokens"] == 100
    with pytest.raises(PipelineError):
        build_chunks(Store(":memory:"), ChunkSetCfg("x", 100, 0, 0))


def test_embed_resumes_after_failure_and_guards_model_changes(monkeypatch):
    monkeypatch.setattr(pipeline, "EMBED_SLICE", 2)
    store = reports_store()
    build_chunks(store, ChunkSetCfg("c100", 100, 10, 10))
    total = len(store.chunks("c100"))
    prof = EmbeddingProfile(name="p1", provider="openai", model="m1", api_key_name="K")
    with pytest.raises(RuntimeError):
        embed_chunks(store, "c100", prof, CountingEmbedder(fail_on_call=2))
    assert len(store.embedded_ids("c100", "p1")) == 2                  # the first slice was saved
    ok = CountingEmbedder()
    out = embed_chunks(store, "c100", prof, ok)
    assert out["already_done"] == 2 and out["embedded_now"] == total - 2 and len(store.embedded_ids("c100", "p1")) == total
    assert embed_chunks(store, "c100", prof, ok)["embedded_now"] == 0   # idempotent
    changed = EmbeddingProfile(name="p1", provider="openai", model="m2", api_key_name="K")
    with pytest.raises(PipelineError, match="--force"):
        embed_chunks(store, "c100", changed, ok)
    assert embed_chunks(store, "c100", changed, ok, force=True)["embedded_now"] == total


# ---------- answers ----------
def test_json_and_citation_parsing():
    assert parse_json_object('```json\n{"answer": "x"}\n```') == {"answer": "x"}
    assert parse_json_object('Sure! {"answer": "x", "citations": []} Hope that helps') == {"answer": "x", "citations": []}
    assert parse_json_object("no json here") is None and parse_json_object("[1,2]") is None
    assert [parse_citation(x) for x in ["C12", "[C13]", 14, "c15", "x", "C"]] == [12, 13, 14, 15, None, None]


def test_answer_question_happy_path_prompt_and_bad_shapes():
    store, _ = corpus_store()
    r = Retriever(store, "c400", profile="p", embedder=FakeQueryEmbedder({"q": unit(1, 0, 0)}))
    hits = r.search("q", SETTINGS, mode="vector", top_k=2)
    prompt = build_user_prompt("q", hits)
    assert "[C1]" in prompt and "Question: q" in prompt
    llm = ScriptedLLM('{"answer": "It is Log4Shell [C1]", "citations": ["C1", "[C1]", "C99"], "abstained": false}')
    res = answer_question("q", hits, llm)
    assert res.citations == [1, 99] and not res.abstained and not res.parse_error and "untrusted" in llm.calls[0][0]
    assert answer_question("q", hits, ScriptedLLM("plain prose")).parse_error
    assert answer_question("q", hits, ScriptedLLM('{"answer": "a", "citations": "C1"}')).citations == []
    none = ScriptedLLM("should not be called")
    assert answer_question("q", [], none).abstained and none.calls == []


# ---------- metrics ----------
def test_metrics_known_values():
    ranking, gold = [3, 1, 2], {1, 2}
    assert reciprocal_rank(ranking, gold) == 0.5 and reciprocal_rank([9, 8], gold) == 0.0
    assert recall_at_k(ranking, gold, 1) == 0.0 and recall_at_k(ranking, gold, 2) == 0.5 and recall_at_k(ranking, gold, 3) == 1.0
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, gold, 3) == pytest.approx(expected)


def test_paired_bootstrap_detects_real_difference_and_not_noise():
    same = paired_bootstrap([0.5] * 20, [0.5] * 20, n_boot=500)
    assert same["mean_diff"] == 0 and not same["significant"] and same["ties"] == 20
    better = paired_bootstrap([1.0] * 20, [0.2] * 20, n_boot=500)
    assert better["significant"] and better["a_better"] == 20
    with pytest.raises(EvalError):
        paired_bootstrap([1.0], [], n_boot=10)
    rows_a, rows_b = [{"id": "q1", "rr": 1.0}, {"id": "q2", "rr": 0.5}], [{"id": "q2", "rr": 0.5}, {"id": "q1", "rr": 0.0}]
    assert compare_rows(rows_a, rows_b, "rr")["mean_diff"] == pytest.approx(0.5)


# ---------- questions file ----------
def write_q(tmp_path, lines):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(x) if isinstance(x, dict) else x for x in lines), encoding="utf-8")
    return p


def test_question_validation(tmp_path):
    good = {"id": "q1", "question": "What?", "gold_report_ids": [1], "key_facts": ["a"]}
    qs = load_questions(write_q(tmp_path, [good, {"id": "q2", "question": "Unknown?", "answerable": False, "gold_report_ids": []}]))
    assert qs[0].qtype == "lookup" and qs[1].qtype == "unanswerable"
    for bad in ([good, good], [{"id": "q1", "question": "x", "gold_report_ids": []}],
                [{"id": "q1", "question": "x", "answerable": False, "gold_report_ids": [1]}],
                [{**good, "type": "weird"}], ["not json"]):
        with pytest.raises(EvalError):
            load_questions(write_q(tmp_path, bad))
    store, _ = corpus_store()
    assert check_against_store([Question("q", "x", "lookup", True, (1, 777))], store) == ["q: gold report 777 is not in the database"]


# ---------- end-to-end evaluation with fakes ----------
def qs():
    return [Question("q1", "CVE-2021-44228", "exact_identifier", True, (1,), ("uses JNDI", "leads to RCE")),
            Question("q2", "unrelated thing", "unanswerable", False, ()),
            Question("q3", "invoice id url", "lookup", True, (3,), ("change the id",))]


def retriever_for_eval():
    store, _ = corpus_store()
    table = {"CVE-2021-44228": unit(1, 0, 0), "unrelated thing": unit(0.3, 0.3, 0.3), "invoice id url": unit(0, 0, 1)}
    return Retriever(store, "c400", profile="p", embedder=FakeQueryEmbedder(table))


def test_retrieval_evaluation_end_to_end(tmp_path):
    rows, summary = evaluate_retrieval(qs(), retriever_for_eval(), SETTINGS, mode="vector", rerank=False)
    by_id = {r["id"]: r for r in rows}
    assert by_id["q1"]["rr"] == 1.0 and by_id["q3"]["hit@1"] == 1.0 and "rr" not in by_id["q2"]
    assert summary["n_answerable"] == 2 and summary["mean"]["rr"] == 1.0 and summary["by_type"]["lookup"]["n"] == 1
    csv_path, json_path = write_results(tmp_path, "retrieval-test", rows, summary, {"mode": "vector"})
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("id,type") and json.loads(json_path.read_text())["meta"]["mode"] == "vector"


def test_answer_evaluation_end_to_end():
    def answerer(system, user):
        if "unrelated thing" in user:
            return '{"answer": "The sources do not cover this.", "citations": [], "abstained": true}'
        if "invoice id url" in user:
            return '{"answer": "Wrong source used [C1]", "citations": ["C1"], "abstained": false}'
        return '{"answer": "JNDI lookup leads to RCE [C1]", "citations": ["C1"], "abstained": false}'

    judge = ScriptedLLM(lambda s, u: '{"facts_present": [true, false], "unsupported_claims": [], "faithful": true}'
                        if "JNDI" in u else "garbage")
    rows, summary = evaluate_answers(qs(), retriever_for_eval(), SETTINGS, ScriptedLLM(answerer), judge, mode="vector", rerank=False)
    by_id = {r["id"]: r for r in rows}
    assert by_id["q1"]["fact_recall"] == 0.5 and by_id["q1"]["faithful"] is True and by_id["q1"]["citation_precision"] == 1.0
    assert by_id["q2"]["abstain_correct"] is True
    assert by_id["q3"]["cited_gold"] is False and by_id["q3"]["citation_precision"] == 0.0 and by_id["q3"]["judge_error"] is True
    assert summary["correct_abstain_rate"] == 1.0 and summary["false_abstain_rate"] == 0.0
    assert summary["judge_errors"] == 1 and summary["cited_gold_rate"] == 0.5


def test_uncited_or_invented_citations_are_flagged():
    def answerer(system, user):
        return '{"answer": "No citation given", "citations": [], "abstained": false}' if "CVE" in user \
            else '{"answer": "Made up", "citations": ["C999"], "abstained": false}'

    judge = ScriptedLLM('{"facts_present": [true], "unsupported_claims": [], "faithful": true}')
    rows, _ = evaluate_answers([qs()[0], qs()[2]], retriever_for_eval(), SETTINGS, ScriptedLLM(answerer), judge, mode="vector", rerank=False)
    assert [r["citations_valid"] for r in rows] == [False, False]


# ---------- question drafting ----------
def test_draft_questions_are_flagged_for_review_and_filtered():
    store = Store(":memory:")
    store.replace_reports([report(i, f"Title {i}", "b") for i in range(1, 7)])
    from hrag.store import NewChunk
    chunks = [NewChunk(i, 0, "steps" if i % 2 else "impact", False, 30, f"hdr {i}", f"Details of bug {i}. CVE-2021-4422{i} appears here." if i == 2 else f"Details of bug {i}.")
              for i in range(1, 7)]
    store.replace_chunk_set("c400", {}, chunks)

    def llm(system, user):
        if "FIXED set" in system:
            return '{"questions": ["How does CVE-2099-0001 in AcmeCorp Router firmware work?", "  "]}'
        if "bug 3" in user:
            return '{"question": "What does this report say about bug 3?", "key_facts": ["x"]}'      # self-referential -> rejected
        return '{"question": "How is a vulnerability of kind %d exploited in practice?", "key_facts": ["a", "b"]}' % len(user)

    items = draft_questions(store, "c400", ScriptedLLM(llm), n_answerable=5, n_unanswerable=2, n_identifier=1, seed=3, model_name="fake")
    assert all(i["needs_review"] for i in items) and len({i["id"] for i in items}) == len(items)
    assert any(i["type"] == "exact_identifier" and i["gold_report_ids"] == [2] for i in items)
    assert not any("this report" in i["question"] for i in items)
    unanswerable = [i for i in items if not i["answerable"]]
    assert len(unanswerable) == 1 and unanswerable[0]["gold_report_ids"] == []


def test_every_chunk_gets_report_metadata_and_the_when_to_try_header():
    import json as _json
    from hrag.store import ReportRow
    body = "Intro text about the bug. " * 30 + "\n\n## Steps To Reproduce\n1. Send this request:\n```\ncurl -X POST http://x/login -d 'a=b'\n```\n\n## Impact\nAccount takeover."
    store = Store(":memory:")
    store.replace_reports([ReportRow(
        report_id=7, title="Auth bypass", program="gitlab", weakness="CWE-287 (Improper Authentication)", substate="", created_at="",
        disclosed_at="2021-01-01", severity="high", vote_count=0, body=body, body_hash="h", source="gitlab",
        source_url="https://gitlab.com/gitlab-org/gitlab/-/issues/7", cwe="CWE-287", quality_flags=_json.dumps(["no_impact"]),
        quality_score=0.8, when_to_try="Try this when a login endpoint trusts a client-supplied flag.",
        images=_json.dumps([{"kind": "video", "ref": "poc.mp4"}]))])
    build_chunks(store, ChunkSetCfg("c100", 100, 10, 10))
    chunks = store.chunks("c100")
    assert len(chunks) > 2
    for c in chunks:
        assert c.header.split("\n")[1] == "When to try this: Try this when a login endpoint trusts a client-supplied flag."
        assert "severity: high" in c.header.split("\n")[0] and c.embed_text.startswith(c.header)
        m = c.meta
        assert (m["report_id"], m["program"], m["source"], m["cwe"], m["severity"]) == (7, "gitlab", "gitlab", "CWE-287", "high")
        assert m["source_url"].endswith("/issues/7") and m["quality_flags"] == ["no_impact"] and m["quality_score"] == 0.8
        assert m["section"] == c.section and m["ordinal"] == c.ordinal and m["has_code"] == c.has_code
        assert m["images"] == 1 and m["disclosed_at"] == "2021-01-01"
