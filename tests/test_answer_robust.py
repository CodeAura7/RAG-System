"""Answer replies must be read reliably: unquoted ids, retries, provider-enforced JSON, and honest reporting.

Born from a real failure: gpt-4o-mini answered  "citations": [C34, C128]  (no quotes), the parser returned None, and the
Playground then displayed the raw reply AFTER renumbering the ids, so the broken reply looked like valid JSON.
"""
import json

import httpx
import pytest

from hrag.answer import ANSWER_SCHEMA, SYSTEM_PROMPT, AnswerResult, answer_question, parse_answer_reply
from hrag.config import LLMProfile, RetrievalSettings, load_config
from hrag.evaluate import evaluate_answers
from hrag.llm import LLM, LLMError
from hrag.retrieve import Retriever
import playground.backend as pb
from tests.helpers import FakeQueryEmbedder, corpus_store, unit
from tests.test_pipeline_answer_eval import SETTINGS, ScriptedLLM, qs, retriever_for_eval
from tests.test_playground_backend import FakeLLM, be  # noqa: F401  (pytest fixture)

ANSWER = "Command injection lets an attacker run commands [C1]."


def hits_for_chunks_1_and_2():
    store, _ = corpus_store()
    r = Retriever(store, "c400", profile="p", embedder=FakeQueryEmbedder({"q": unit(1, 0, 0)}))
    return r.search("q", RetrievalSettings(mode="vector", top_k=3, candidates=10, rrf_k=60, max_chunks_per_report=2, dedupe_cosine=0.0),
                    mode="vector", top_k=3)


# ---------- the tolerant reader ----------

@pytest.mark.parametrize("raw, how, ids", [
    ('{"answer": "x", "citations": ["C34", "C128"], "abstained": false}', "strict", [34, 128]),
    ('{"answer": "x", "citations": [34, 128], "abstained": false}', "strict", [34, 128]),
    ('Sure!\n```json\n{"answer": "x", "citations": ["C34"], "abstained": false}\n```', "strict", [34]),
    ('{"answer": "x", "citations": [], "abstained": true}', "strict", []),
    ('{"answer": "x", "citations": [C34, C128, C125, C123], "abstained": false}', "repaired", [34, 128, 125, 123]),   # the real failure
    ('{"answer": "x", "citations": [[C34], [C128]], "abstained": false}', "repaired", [34, 128]),
    ('{"answer": "x", "citations": ["C34",], "abstained": false,}', "repaired", [34]),
    ('{"answer": "The \\"citations\\": [C1] are here", "citations": [C34], "abstained": false}', "repaired", [34]),
    ('{"answer": "He said "run it" and left [C34]", "citations": [C34], "abstained": false}', "salvaged", [34]),
])
def test_parse_answer_reply_reads_the_shapes_models_actually_produce(raw, how, ids):
    data, got = parse_answer_reply(raw)
    assert got == how and data is not None and isinstance(data["answer"], str)
    assert [int(str(c).strip("[]Cc ")) for c in data["citations"]] == ids


@pytest.mark.parametrize("raw, how", [('{"answer": "Command injection lets an attacker run comm', "truncated"),
                                      ("I cannot find that in the sources.", "failed")])
def test_unreadable_replies_are_named_not_guessed(raw, how):
    assert parse_answer_reply(raw) == (None, how)


# ---------- answer_question ----------

def test_unquoted_ids_no_longer_fail_the_whole_answer():
    llm = ScriptedLLM('{"answer": "%s", "citations": [C1, C2], "abstained": false}' % ANSWER)
    res = answer_question("q", hits_for_chunks_1_and_2(), llm)
    assert not res.parse_error and res.parse_how == "repaired" and res.citations == [1, 2] and res.answer == ANSWER
    assert len(llm.calls) == 1


def test_a_clean_reply_costs_exactly_one_model_call():
    llm = ScriptedLLM('{"answer": "ok [C1]", "citations": ["C1"], "abstained": false}')
    res = answer_question("q", hits_for_chunks_1_and_2(), llm)
    assert res.parse_how == "strict" and len(llm.calls) == 1


def test_an_unreadable_reply_is_asked_again_once_and_then_read():
    replies = iter(["I think the answer is in C1", '{"answer": "ok [C1]", "citations": ["C1"], "abstained": false}'])
    llm = ScriptedLLM(lambda s, u: next(replies))
    res = answer_question("q", hits_for_chunks_1_and_2(), llm)
    assert not res.parse_error and res.citations == [1] and len(llm.calls) == 2
    assert "not valid JSON" in llm.calls[1][1] and "Question: q" in llm.calls[1][1]


def test_a_reply_that_stays_unreadable_is_a_parse_error_after_exactly_one_retry():
    llm = ScriptedLLM("plain prose")
    res = answer_question("q", hits_for_chunks_1_and_2(), llm)
    assert res.parse_error and res.parse_how == "failed" and res.raw == "plain prose" and len(llm.calls) == 2


def test_the_prompt_shows_quoted_ids_so_models_stop_copying_the_bracket_style():
    assert '"citations": ["C12", "C15"]' in SYSTEM_PROMPT and "untrusted" in SYSTEM_PROMPT


class SchemaAwareLLM:
    def __init__(self):
        self.seen = []

    def complete(self, system, user, json_schema=None):
        self.seen.append(json_schema)
        return '{"answer": "ok [C1]", "citations": ["C1"], "abstained": false}'


def test_models_that_accept_a_schema_are_given_it_and_plain_fakes_still_work():
    llm = SchemaAwareLLM()
    answer_question("q", hits_for_chunks_1_and_2(), llm)
    assert llm.seen == [ANSWER_SCHEMA]
    assert answer_question("q", hits_for_chunks_1_and_2(), ScriptedLLM('{"answer": "a", "citations": [], "abstained": false}')).parse_how == "strict"


# ---------- provider-enforced JSON (LLM client) ----------

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def anthropic_profile():
    return LLMProfile(name="a", provider="anthropic", model="claude-x", api_key_name="K", temperature=None)


def compat_profile(base="https://api.example/v1"):
    return LLMProfile(name="o", provider="openai_compat", base_url=base, model="m", api_key_name="K", temperature=None)


def test_anthropic_gets_output_config_format_only_when_a_schema_is_given():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}]})

    llm = LLM(anthropic_profile(), "k", client=client(handler))
    llm.complete("s", "u", json_schema=SCHEMA)
    llm.complete("s", "u")
    assert bodies[0]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert "output_config" not in bodies[1]


def test_openai_compatible_gets_a_strict_json_schema():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    LLM(compat_profile(), "k", client=client(handler)).complete("s", "u", json_schema=SCHEMA)
    assert bodies[0]["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": SCHEMA}}
    assert "provider" not in bodies[0]


def test_openrouter_is_told_to_route_only_to_providers_that_support_the_schema():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    LLM(compat_profile("https://openrouter.ai/api/v1"), "k", client=client(handler)).complete("s", "u", json_schema=SCHEMA)
    assert bodies[0]["provider"] == {"require_parameters": True}


def test_a_provider_that_rejects_the_schema_is_retried_once_without_it_and_remembered():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = LLM(compat_profile(), "k", client=client(handler))
    assert llm.complete("s", "u", json_schema=SCHEMA) == "{}"
    assert llm.complete("s", "u", json_schema=SCHEMA) == "{}"
    assert ["response_format" in b for b in bodies] == [True, False, False]       # tried once, then never again


def test_openrouter_style_404_no_endpoint_supports_it_also_falls_back():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(404, json={"error": {"message": "No endpoints found that support the requested parameters"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    assert LLM(compat_profile("https://openrouter.ai/api/v1"), "k", client=client(handler)).complete("s", "u", json_schema=SCHEMA) == "{}"
    assert ["response_format" in b for b in bodies] == [True, False]


def test_a_failure_with_the_schema_that_also_fails_without_it_is_not_blamed_on_the_schema():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "unknown model"}})

    llm = LLM(compat_profile(), "k", client=client(handler))
    for _ in range(2):
        with pytest.raises(LLMError, match="unknown model"):
            llm.complete("s", "u", json_schema=SCHEMA)
    assert ["response_format" in b for b in bodies] == [True, False, True, False]   # the schema is still tried on the next call


def test_a_400_that_has_nothing_to_do_with_a_schema_still_raises():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    with pytest.raises(LLMError):
        LLM(compat_profile(), "k", client=client(handler)).complete("s", "u")


# ---------- evaluation reports how replies were read ----------

def test_evaluation_counts_repaired_replies_instead_of_hiding_them():
    def answerer(system, user):
        if "unrelated thing" in user:
            return '{"answer": "The sources do not cover this.", "citations": [], "abstained": true}'
        return '{"answer": "JNDI lookup leads to RCE [C1]", "citations": [C1], "abstained": false}'

    judge = ScriptedLLM('{"facts_present": [true, true], "unsupported_claims": [], "faithful": true}')
    rows, summary = evaluate_answers(qs(), retriever_for_eval(), SETTINGS, ScriptedLLM(answerer), judge, mode="vector", rerank=False)
    assert {r["parse_how"] for r in rows} == {"strict", "repaired"}
    assert summary["parse_outcomes"] == {"strict": 1, "repaired": 2} and summary["parse_errors"] == 0


# ---------- Playground ----------

def test_a_failed_reply_never_ticks_citations_nobody_checked(be):  # noqa: F811
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=2, pool=4)
    be._llm_factory = lambda name: FakeLLM("sorry, see [C1, C2] nothing in JSON")
    ans = be.answer(run, "claude_sonnet")
    assert ans.result.parse_error and ans.cited_valid == [] and ans.inline_ids == []


def test_a_failed_reply_is_displayed_exactly_as_received_not_renumbered(be):  # noqa: F811
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=2, pool=4)
    shown = [v.chunk_id for v in run.supplied]
    raw = '{"answer": "x [C%d] and then the model ran out of tok' % shown[0]              # cut off inside the answer text
    be._llm_factory = lambda name: FakeLLM(raw)
    ans = be.answer(run, "claude_sonnet")
    assert ans.result.parse_error and ans.display_text == raw


def test_a_repaired_reply_is_ticked_and_the_page_is_told_it_was_repaired(be):  # noqa: F811
    run = be.retrieve("how does the xss payload run", chunk_set="c400", mode="hybrid", k=2, pool=4)
    shown = [v.chunk_id for v in run.supplied]
    be._llm_factory = lambda name: FakeLLM('{"answer": "Scripts run [C%d]", "citations": [C%d, C%d], "abstained": false}' % (shown[0], shown[0], shown[1]))
    ans = be.answer(run, "claude_sonnet")
    assert ans.cited_valid == shown and pb.answer_notice(ans.result)[0] == "info" and "repaired" in pb.answer_notice(ans.result)[1]


def test_a_clean_reply_gets_no_notice_and_an_unreadable_one_gets_a_warning():
    assert pb.answer_notice(AnswerResult("q", "a", parse_how="strict")) is None
    level, text = pb.answer_notice(AnswerResult("q", "garbage", parse_error=True, parse_how="failed", raw="garbage"))
    assert level == "warning" and "could not be checked" in text


# ---------- config ----------

def test_a_dedicated_answer_profile_has_room_for_a_full_json_answer():
    p = load_config().llm("openrouter_answer")
    assert p.model == "openai/gpt-4o-mini" and p.max_tokens >= 1500 and p.base_url == "https://openrouter.ai/api/v1"
