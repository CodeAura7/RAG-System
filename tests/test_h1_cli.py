"""`hrag h1 probe|discover|fetch` end to end against a fake HackerOne transport (shapes ASSUMED - see test_h1_api.py)."""
import json

import httpx
import pytest

from hrag import cli, h1_api, secrets

CONFIG = """
[paths]
db = "work/hrag.db"
cache_dir = "work/cache"
eval_dir = "work/eval"
[defaults]
chunk_set = "c150"
embedding = "fake"
answer = "fake_answer"
judge = "fake_judge"
rerank = "none"
[ingest]
n_reports = 10
seed = 1
min_body_chars = 100
allowed_substates = ["resolved"]
near_duplicate_jaccard = 0.9
dataset_base_url = "https://unused.example/"
dataset_files = ["a.parquet"]
[retrieval]
mode = "vector"
top_k = 3
candidates = 10
max_chunks_per_report = 2
dedupe_cosine = 0.0
[chunk_sets.c150]
max_tokens = 150
overlap_tokens = 20
min_tokens = 10
[embeddings.fake]
provider = "openai_compat"
base_url = "http://127.0.0.1:1/v1"
model = "fake-embed"
api_key_name = "HRAG_FAKE_KEY"
[llm.fake_answer]
provider = "openai_compat"
base_url = "http://127.0.0.1:1/v1"
model = "fake-chat"
api_key_name = "HRAG_FAKE_KEY"
[llm.fake_judge]
provider = "openai_compat"
base_url = "http://127.0.0.1:1/v1"
model = "fake-judge"
api_key_name = "HRAG_FAKE_KEY"
"""


def fake_api(with_full_text: bool):
    def handler(request):
        path = request.url.path
        if path.endswith("/hacktivity"):
            data = [{"id": str(i), "attributes": {"report_id": i, "title": f"bug {i}"},
                     "relationships": {"program": {"data": {"attributes": {"handle": f"prog{i % 4}"}}}}} for i in range(1, 13)]
            return httpx.Response(200, json={"data": data, "links": {}})
        rid = path.rsplit("/", 1)[1]
        if not with_full_text:
            return httpx.Response(403, json={})
        return httpx.Response(200, json={"data": {"id": rid, "attributes": {"vulnerability_information": f"full text of {rid}"}}})
    return handler


@pytest.fixture()
def project(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(secrets, "_dotenv_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    monkeypatch.setenv("H1_API_USERNAME", "alice")
    monkeypatch.setenv("H1_API_TOKEN", "tok-123")
    state = {"full": True}

    def factory(config):
        http = httpx.Client(transport=httpx.MockTransport(fake_api(state["full"])))
        return h1_api.H1Client("alice", "tok-123", client=http, cache_dir=config.cache_dir / "h1", min_interval=0, sleep=lambda s: None)

    monkeypatch.setattr(cli, "make_h1_client", factory)
    return {"cfg": str(cfg), "root": tmp_path, "state": state}


def run(project, *argv, capsys):
    code = cli.main(["--config", project["cfg"], *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_bulk_commands_are_blocked_until_a_probe_has_passed(project, capsys):
    code, _, err = run(project, "h1", "discover", capsys=capsys)
    assert code == 1 and "probe" in err
    code, _, err = run(project, "h1", "fetch", capsys=capsys)
    assert code == 1 and "probe" in err
    assert not (project["root"] / "work" / "cache" / "h1").exists()          # nothing was collected


def test_failed_probe_keeps_bulk_blocked_and_explains_why(project, capsys):
    project["state"]["full"] = False
    code, out, _ = run(project, "h1", "probe", capsys=capsys)
    assert code == 1 and "NOT usable" in out
    code, _, err = run(project, "h1", "discover", capsys=capsys)
    assert code == 1 and "blocked" in err


def test_probe_discover_fetch_happy_path(project, capsys):
    code, out, _ = run(project, "h1", "probe", capsys=capsys)
    assert code == 0 and "bulk collection is now allowed" in out
    assert "tok-123" not in out and "full text of" not in out                # no secret, no report text on screen

    code, out, _ = run(project, "h1", "discover", "--n", "6", "--per-program", "2", capsys=capsys)
    assert code == 0 and "from 4 programs" in out
    lines = (project["root"] / "work" / "h1_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 6
    assert max(sum(1 for r in rows if r["program"] == p) for p in {r["program"] for r in rows}) <= 2

    code, out, _ = run(project, "h1", "fetch", capsys=capsys)
    assert code == 0 and "fetched 6 of 6" in out
    cached = sorted((project["root"] / "work" / "cache" / "h1").glob("*.json"))
    assert len(cached) == 6


def test_doctor_lists_the_collector_keys_and_the_env_file(project, capsys, monkeypatch):
    monkeypatch.delenv("H1_API_TOKEN")
    (project["root"] / ".env").write_text("H1_API_TOKEN=from-file\n", encoding="utf-8")
    code, out, _ = run(project, "doctor", capsys=capsys)
    assert code == 0
    assert "H1_API_USERNAME" in out and "H1_API_TOKEN" in out
    token_line = next(line for line in out.splitlines() if line.strip().startswith("H1_API_TOKEN"))
    assert ".env" in token_line and "from-file" not in out
