"""Drive the real CLI end to end against a tiny local fake of the embeddings and chat APIs.
No network, no real keys: this proves the commands are wired together correctly."""
import json
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hrag.cli import main

TOPICS = {
    "ssrf": "server side request forgery fetches internal metadata service credentials through the url parameter",
    "xss": "stored cross site scripting payload in the comment field executes script for every visitor",
    "idor": "insecure direct object reference lets a user read other customers invoices by changing the numeric id",
    "sqli": "sql injection in the search parameter allows dumping the users table with a union select",
    "csrf": "cross site request forgery on the change email form lacks any anti forgery token",
    "rce": "remote code execution via unsafe deserialization of the uploaded session cookie object",
    "redirect": "open redirect in the login return url parameter sends victims to an attacker controlled domain",
    "xxe": "xml external entity processing in the import feature discloses local files from the server",
    "path": "path traversal in the download endpoint reads arbitrary files using dot dot slash sequences",
    "authz": "missing authorization check on the admin api allows any member to delete other organisations",
    "race": "race condition when redeeming a coupon lets the same code be applied many times concurrently",
    "jwt": "json web token signature is not verified so the none algorithm grants administrator access",
}


def vec(text: str, dim: int = 2048) -> list[float]:
    v = np.zeros(dim)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(tok.encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeApi(BaseHTTPRequestHandler):
    def log_message(self, *a):          # keep pytest output quiet
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer test-key"
        if self.path.endswith("/embeddings"):
            return self._send({"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(body["input"])]})
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if "retrieval aid" in system:
            reply = {"when_to_try": "Try this when the application exposes the behaviour described in the report on a similar feature."}
        elif "grader" in system:
            n_facts = len(re.findall(r"^\d+\. ", user.split("KEY FACTS:")[1].split("ANSWER:")[0], flags=re.M))
            reply = {"facts_present": [True] * n_facts, "unsupported_claims": [], "faithful": True}
        elif "FIXED set" in system:
            reply = {"questions": ["How does CVE-2099-0001 in AcmeCorp Router firmware work?"]}
        elif "evaluation questions" in system:
            topic = user.split("\n")[1]
            reply = {"question": "How is the issue described as '" + " ".join(topic.split()[:6]) + "' exploited in practice?",
                     "key_facts": ["fact one", "fact two"]}
        else:
            ids = re.findall(r"\[C(\d+)\]", user)
            reply = {"answer": f"Based on the sources [C{ids[0]}].", "citations": [f"C{ids[0]}"], "abstained": False}
        self._send({"choices": [{"message": {"content": json.dumps(reply)}}]})


@pytest.fixture()
def project(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"

    rows = []
    for i, (name, text) in enumerate(TOPICS.items()):
        body = f"## Summary\nThe {name} issue. " + text + ". " + (f"Detail sentence {i} about {name} behaviour. " * 6) + \
               f"\n\n## Steps To Reproduce\n1. Open the page for {name}.\n2. Send:\n```\nGET /{name}?x={i} HTTP/1.1\n```\n\n## Impact\n{text}."
        rows.append({"id": 1000 + i, "title": f"{name.upper()} bug", "created_at": "2023-01-01", "substate": "resolved",
                     "vulnerability_information": body, "team": {"handle": "acme"}, "disclosed_at": "2023-02-01",
                     "weakness": {"id": 1, "name": name}, "original_report_id": None, "vote_count": 1,
                     "structured_scope": {"max_severity": "high"}})
    data = tmp_path / "parquet"
    data.mkdir()
    for f in ("a.parquet", "b.parquet", "c.parquet"):
        pq.write_table(pa.Table.from_pylist(rows if f == "a.parquet" else rows[:1] and []) if f == "a.parquet"
                       else pa.Table.from_pylist([dict(rows[0], id=9000 + len(f))]), data / f)

    cfg = tmp_path / "config.toml"
    cfg.write_text(f"""
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
enrich = "fake_answer"
[ingest]
n_reports = 10
seed = 1
min_body_chars = 100
allowed_substates = ["resolved"]
near_duplicate_jaccard = 0.9
dataset_base_url = "https://unused.example/"
dataset_files = ["a.parquet", "b.parquet", "c.parquet"]
[retrieval]
mode = "hybrid"
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
base_url = "{base}"
model = "fake-embed"
api_key_name = "HRAG_FAKE_KEY"
batch_size = 5
max_retries = 0
[llm.fake_answer]
provider = "openai_compat"
base_url = "{base}"
model = "fake-chat"
api_key_name = "HRAG_FAKE_KEY"
max_retries = 0
[llm.fake_judge]
provider = "openai_compat"
base_url = "{base}"
model = "fake-judge"
api_key_name = "HRAG_FAKE_KEY"
max_retries = 0
""", encoding="utf-8")
    monkeypatch.setenv("HRAG_FAKE_KEY", "test-key")
    yield {"cfg": str(cfg), "data": str(data), "root": tmp_path}
    server.shutdown()


def run(project, *argv, capsys):
    code = main(["--config", project["cfg"], *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_full_workflow_through_the_cli(project, capsys):
    code, out, err = run(project, "ingest-dataset", "--parquet-dir", project["data"], capsys=capsys)
    assert code == 0, err
    assert "selected 10" in out or "selected 11" in out or "selected" in out
    code, out, _ = run(project, "ingest-dataset", "--parquet-dir", project["data"], capsys=capsys)       # refuses to clobber
    assert code == 1 and "REPLACES" in out

    assert run(project, "chunk", capsys=capsys)[0] == 0
    code, out, err = run(project, "embed", capsys=capsys)
    assert code == 0 and "embedded" in out, err
    assert '"reports": 10' in run(project, "stats", capsys=capsys)[1]

    code, out, _ = run(project, "search", "stored cross site scripting comment", "--mode", "hybrid", "--k", "2", capsys=capsys)
    assert code == 0 and "XSS bug" in out
    code, out, _ = run(project, "search", "unsafe deserialization", "--mode", "vector", "--k", "1", capsys=capsys)
    assert "RCE bug" in out

    code, out, err = run(project, "ask", "How does the sql injection work?", capsys=capsys)
    assert code == 0, err
    assert "Based on the sources" in out and "https://hackerone.com/reports/" in out

    # drafting -> review gate -> validate -> retrieval/answers evaluation
    qpath = project["root"] / "work" / "eval" / "questions.draft.jsonl"
    code, out, err = run(project, "eval", "draft", "--n", "6", "--unanswerable", "1", capsys=capsys)
    assert code == 0 and qpath.is_file(), err
    items = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines()]
    assert len(items) >= 4 and all(i["needs_review"] for i in items)

    final = project["root"] / "work" / "eval" / "questions.jsonl"
    final.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    code, _, err = run(project, "eval", "retrieval", capsys=capsys)
    assert code == 1 and "needs_review" in err                                                    # human review is enforced

    for i in items:
        i["needs_review"] = False
        i["key_facts"] = i["key_facts"] or ["x"]
    final.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    assert run(project, "eval", "validate", capsys=capsys)[0] == 0

    code, out, err = run(project, "eval", "retrieval", "--arms", "vector,bm25,hybrid", capsys=capsys)
    assert code == 0, err
    assert "recall@1" in out and "Paired comparison" in out
    results = sorted((project["root"] / "work" / "eval" / "results").glob("*retrieval-hybrid.json"))
    assert results and json.loads(results[0].read_text())["summary"]["n_answerable"] >= 3

    code, _, err = run(project, "eval", "retrieval", "--arms", "fuzzy", capsys=capsys)
    assert code == 1 and "hybrid" in err                                                  # unknown arms are refused with a clear message

    code, out, err = run(project, "eval", "answers", capsys=capsys)
    assert code == 0, err
    summary = json.loads(out[out.index("{"):out.rindex("}") + 1])
    assert summary["parse_errors"] == 0 and summary["judge_errors"] == 0 and summary["citation_valid_rate"] == 1.0

    files = sorted((project["root"] / "work" / "eval" / "results").glob("*.json"))
    a = next(f for f in files if "retrieval-vector" in f.name)
    b = next(f for f in files if "retrieval-bm25" in f.name)
    code, out, _ = run(project, "eval", "compare", str(a), str(b), capsys=capsys)
    assert code == 0 and "A - B" in out


def test_helpful_errors_not_tracebacks(project, capsys):
    code, _, err = run(project, "chunk", capsys=capsys)
    assert code == 1 and "no reports" in err
    code, _, err = run(project, "search", "x", capsys=capsys)
    assert code == 1 and err.startswith("error:")
    code, _, err = run(project, "ingest-dataset", "--parquet-dir", str(project["root"]), capsys=capsys)
    assert code == 1 and "missing file" in err


def test_doctor_reports_key_sources_without_leaking_values(project, capsys):
    code, out, _ = run(project, "doctor", capsys=capsys)
    assert code == 0 and "HRAG_FAKE_KEY" in out and "env" in out and "test-key" not in out


def test_build_from_collected_sources_then_search_and_ask(project, capsys):
    """The new pipeline end to end: cached GitLab copies -> collect -> enrich -> chunk -> embed -> vector search -> cited answer."""
    cache = project["root"] / "work" / "cache" / "gitlab"
    cache.mkdir(parents=True)
    for i, (name, text) in enumerate(TOPICS.items()):
        desc = (f"**[HackerOne report #{7000 + i}](https://hackerone.com/reports/{7000 + i})** by `r` on 2022-01-01:\n\n## Report\n\n##### Summary\n\n"
                f"The {name} issue: {text}. " + (f"Detail sentence {i} about {name} behaviour. " * 8) +
                f"\n\n##### Steps to reproduce\n\n1. Open the page for {name}.\n2. Send this request:\n\n```http\nGET /{name}?x={i} HTTP/1.1\n```\n\n"
                f"##### Impact\n\n{text}.\n")
        issue = {"iid": 100 + i, "title": f"{name.upper()} bug", "description": desc, "labels": ["HackerOne", f"Weakness::CWE-{900 + i}"],     # distinct CWEs: the per-weakness cap must not hide a topic
                 "web_url": f"https://gitlab.com/gitlab-org/gitlab/-/issues/{100 + i}", "created_at": "2022-01-01T00:00:00Z", "user_notes_count": 0}
        (cache / f"gitlab-org__gitlab__{100 + i}.json").write_text(json.dumps({"issue": issue, "notes": None}), encoding="utf-8")

    code, out, err = run(project, "build", capsys=capsys)
    assert code == 0, err
    assert "selected" in out and "llm=" in out and "embedded" in out
    stats = json.loads(run(project, "stats", capsys=capsys)[1])
    assert stats["sources"] == {"gitlab": stats["reports"]} and stats["reports"] >= 6 and stats["embeddings"][0]["n"] >= stats["reports"]

    code, out, _ = run(project, "search", "stored cross site scripting comment", "--k", "2", "--show-text", capsys=capsys)
    assert code == 0 and "XSS bug" in out and "When to try this" in out
    code, out, err = run(project, "ask", "How does the sql injection work?", capsys=capsys)
    assert code == 0, err
    assert "https://gitlab.com/gitlab-org/gitlab/-/issues/" in out and "https://hackerone.com/reports/" not in out.split("Based on")[-1]
    # all three search modes work from the command line (keyword mode needs no embedding call at all)
    for mode in ("vector", "bm25", "hybrid"):
        code, out, err = run(project, "search", "stored cross site scripting comment", "--mode", mode, "--k", "2", capsys=capsys)
        assert code == 0 and "XSS bug" in out, (mode, err)
    assert "'bm25'" in run(project, "search", "stored cross site scripting", "--mode", "bm25", capsys=capsys)[1]
    with pytest.raises(SystemExit):
        run(project, "search", "x", "--mode", "fuzzy", capsys=capsys)
