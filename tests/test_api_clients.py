"""API clients tested against a fake HTTP transport - no network, no keys."""
import json

import httpx
import numpy as np
import pytest

from hrag.config import EmbeddingProfile, LLMProfile, RerankProfile
from hrag.embeddings import Embedder, EmbeddingError
from hrag.http import ApiError, post_json
from hrag.llm import LLM, LLMError
from hrag.rerank import Reranker, RerankError


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def emb_profile(**kw):
    base = dict(name="t", provider="openai", model="m", api_key_name="K", batch_size=2, max_retries=2)
    base.update(kw)
    return EmbeddingProfile(**base)


def echo_embeddings(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append((str(request.url), dict(request.headers), body))
        data = [{"index": i, "embedding": [float(i + 1), 2.0, 2.0]} for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})  # out of order on purpose
    return handler


def test_openai_request_shape_and_normalisation():
    seen = []
    emb = Embedder(emb_profile(dimensions=3), "sk-test", client=make_client(echo_embeddings(seen)))
    out = emb.embed_documents(["a", "b", "c"])
    assert out.shape == (3, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    url, headers, body = seen[0]
    assert url == "https://api.openai.com/v1/embeddings"
    assert headers["authorization"] == "Bearer sk-test"
    assert body["dimensions"] == 3 and body["model"] == "m"
    assert len(seen) == 2                       # 3 texts, batch_size 2
    assert out[0][0] < out[0][1]                # index 0 got [1,2,2] despite reversed reply order


def test_voyage_sends_input_type_and_output_dimension():
    seen = []
    emb = Embedder(emb_profile(provider="voyage", dimensions=3), "vk", client=make_client(echo_embeddings(seen)))
    emb.embed_query("hello")
    emb.embed_documents(["doc"])
    assert seen[0][0] == "https://api.voyageai.com/v1/embeddings"
    assert seen[0][2]["input_type"] == "query" and seen[1][2]["input_type"] == "document"
    assert seen[0][2]["output_dimension"] == 3 and "dimensions" not in seen[0][2]


def test_prefixes_apply_to_queries_and_documents_separately():
    seen = []
    prof = emb_profile(provider="openai_compat", base_url="https://x.example/v1", query_prefix="Q:", doc_prefix="D:")
    emb = Embedder(prof, "k", client=make_client(echo_embeddings(seen)))
    emb.embed_query("what")
    emb.embed_documents(["thing"])
    assert seen[0][2]["input"] == ["Q:what"] and seen[1][2]["input"] == ["D:thing"]
    assert seen[0][0] == "https://x.example/v1/embeddings"


@pytest.mark.parametrize("reply", [
    {"data": []},                                                # wrong count
    {"data": [{"index": 0, "embedding": [1.0, "x", 2.0]}]},      # non-numeric
    {"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]},      # zero vector
    {"data": [{"index": 0, "embedding": [1.0, 2.0]}]},           # wrong length for dimensions=3
])
def test_bad_embedding_replies_are_rejected(reply):
    emb = Embedder(emb_profile(dimensions=3), "k", client=make_client(lambda r: httpx.Response(200, json=reply)))
    with pytest.raises(EmbeddingError):
        emb.embed_query("x")


def test_empty_key_refuses_to_send():
    with pytest.raises(EmbeddingError):
        Embedder(emb_profile(), "")


def test_retries_then_succeeds_and_error_never_contains_the_key():
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(429, text="slow down") if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    sleeps = []
    out = post_json("https://x/y", headers={"Authorization": "Bearer SECRET"}, payload={}, max_retries=3,
                    client=make_client(flaky), sleep=sleeps.append)
    assert out == {"ok": 1} and calls["n"] == 3 and len(sleeps) == 2

    with pytest.raises(ApiError) as err:
        post_json("https://x/y", headers={"Authorization": "Bearer SECRET"}, payload={}, max_retries=0,
                  client=make_client(lambda r: httpx.Response(401, text="bad key")), sleep=lambda s: None)
    assert "SECRET" not in str(err.value) and err.value.status == 401


def test_non_retriable_error_fails_fast():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="nope")

    with pytest.raises(ApiError):
        post_json("https://x/y", headers={}, payload={}, max_retries=5, client=make_client(handler), sleep=lambda s: None)
    assert calls["n"] == 1


def test_anthropic_request_and_text_extraction():
    seen = {}

    def handler(request):
        seen["url"], seen["headers"], seen["body"] = str(request.url), dict(request.headers), json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]})

    prof = LLMProfile(name="a", provider="anthropic", model="claude-x", api_key_name="K", temperature=0.0)
    assert LLM(prof, "ak", client=make_client(handler)).complete("sys", "usr") == "hi there"
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "ak" and seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "sys" and seen["body"]["messages"][0]["content"] == "usr"


def test_openai_compat_chat_params_are_configurable():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    prof = LLMProfile(name="o", provider="openai_compat", base_url="https://api.example/v1", model="m",
                      api_key_name="K", temperature=None, max_tokens_param="max_completion_tokens", max_tokens=77)
    assert LLM(prof, "k", client=make_client(handler)).complete("s", "u") == "ok"
    assert seen["body"]["max_completion_tokens"] == 77
    assert "temperature" not in seen["body"] and "max_tokens" not in seen["body"]


def test_unset_model_is_refused():
    prof = LLMProfile(name="o", provider="openai_compat", base_url="https://x/v1", model="SET_ME", api_key_name="K")
    with pytest.raises(LLMError):
        LLM(prof, "k")


def test_rerank_orders_by_score_and_validates_indexes():
    def good(request):
        return httpx.Response(200, json={"data": [{"index": 0, "relevance_score": 0.2}, {"index": 2, "relevance_score": 0.9}]})

    rr = Reranker(RerankProfile(name="r", provider="voyage", model="rerank-2.5", api_key_name="K"), "k", client=make_client(good))
    assert rr.rerank("q", ["a", "b", "c"], top_k=2) == [(2, 0.9), (0, 0.2)]

    def bad(request):
        return httpx.Response(200, json={"data": [{"index": 9, "relevance_score": 0.2}]})

    with pytest.raises(RerankError):
        Reranker(RerankProfile(name="r", provider="voyage", model="m", api_key_name="K"), "k",
                 client=make_client(bad)).rerank("q", ["a"], top_k=1)
