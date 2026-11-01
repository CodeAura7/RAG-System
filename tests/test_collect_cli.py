"""gitlab discover/fetch -> collect -> enrich -> chunk through the real CLI, against fake GitLab / LLM transports."""
import json

import httpx
import pytest

from hrag import cli, gitlab_src, secrets
from tests.test_h1_cli import CONFIG as BASE_CONFIG

CONFIG = BASE_CONFIG + """
[collect]
n_reports = 6
min_body_chars = 200
per_program = 3
per_weakness = 3
max_gitlab = 4
gitlab_urls_file = "sources/gitlab_urls.txt"
[llm.fake_enrich]
provider = "openai_compat"
base_url = "http://127.0.0.1:1/v1"
model = "fake-enrich"
api_key_name = "HRAG_FAKE_KEY"
"""


def desc(i):
    return (f"**[HackerOne report #{i}](https://hackerone.com/reports/{i})** by `r` on 2020-01-01:\n\n## Report\n\n##### Summary\n\n"
            + " ".join(f"issue{i}word{j}" for j in range(60)) + f"\n\n##### Steps to reproduce\n\n1. Send:\n\n```http\nGET /x{i} HTTP/1.1\n```\n2. See it.\n\n"
            f"##### Impact\n\nAn attacker can read data of other users.\n")


def gl_issue(iid, cwe):
    return {"id": iid * 10, "iid": iid, "title": f"Bug {iid}", "description": desc(5000 + iid), "state": "closed", "confidential": False,
            "created_at": "2021-01-01T00:00:00Z", "closed_at": "2021-02-01T00:00:00Z", "web_url": f"https://gitlab.com/gitlab-org/gitlab/-/issues/{iid}",
            "labels": ["HackerOne", f"Weakness::CWE-{cwe}"], "user_notes_count": 0}


def handler(request):
    path = request.url.path
    if path.endswith("/issues"):
        return httpx.Response(200, json=[gl_issue(i, [79, 918, 89, 22, 79, 79][i % 6]) for i in range(1, 9)])
    iid = int(path.rsplit("/", 1)[1])
    return httpx.Response(200, json=gl_issue(iid, 79))


@pytest.fixture()
def project(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(secrets, "_dotenv_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    monkeypatch.delenv("HRAG_FAKE_KEY", raising=False)

    def factory(config):
        http = httpx.Client(transport=httpx.MockTransport(handler))
        return gitlab_src.GitLabClient(client=http, cache_dir=config.cache_dir / "gitlab", min_interval=0, sleep=lambda s: None)
    monkeypatch.setattr(cli, "make_gitlab_client", factory)
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "gitlab_urls.txt").write_text("# samples\nhttps://gitlab.com/gitlab-org/gitlab/-/issues/7\n", encoding="utf-8")
    return {"cfg": str(cfg), "root": tmp_path}


def run(project, *argv, capsys):
    code = cli.main(["--config", project["cfg"], *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_collect_with_nothing_downloaded_says_what_to_do(project, capsys):
    code, _, err = run(project, "collect", capsys=capsys)
    assert code == 1 and "gitlab fetch" in err and "h1 fetch" in err


def test_discover_fetch_collect_enrich_chunk(project, capsys, monkeypatch):
    code, out, _ = run(project, "gitlab", "discover", "--n", "4", "--per-weakness", "2", capsys=capsys)
    assert code == 0 and "picked 4" in out
    code, out, _ = run(project, "gitlab", "fetch", capsys=capsys)
    assert code == 0 and "fetched 1 of 1" in out
    cached = list((project["root"] / "work" / "cache" / "gitlab").glob("*.json"))
    assert len(cached) >= 5                                                    # 4 discovered + the pinned sample link (#7)

    code, out, _ = run(project, "collect", capsys=capsys)
    assert code == 0 and "selected" in out and "pinned" in out
    summary = json.loads((project["root"] / "work" / "collect_summary.json").read_text(encoding="utf-8"))
    assert summary["pinned_selected"] == 1 and summary["by_source"]["gitlab"] == summary["selected"] <= 6
    assert any("comments skipped" in n for n in summary["notes"]) is False     # these fixtures have no comments

    code, out, _ = run(project, "collect", capsys=capsys)                       # a second run must not wipe the database
    assert code == 1 and "REPLACES" in out
    assert run(project, "collect", "--force", capsys=capsys)[0] == 0

    class FakeEnrichLLM:
        def complete(self, system, user):
            return json.dumps({"when_to_try": "Try this when an endpoint reflects or stores user-controlled input in pages."})
    monkeypatch.setattr(cli, "make_enrich_llm", lambda cfg, name, require: FakeEnrichLLM())
    code, out, _ = run(project, "enrich", capsys=capsys)
    assert code == 0 and "llm=" in out
    assert run(project, "chunk", capsys=capsys)[0] == 0
    stats = json.loads(run(project, "stats", capsys=capsys)[1])
    assert stats["reports"] == summary["selected"] and stats["sources"] == {"gitlab": summary["selected"]}
    store = cli.open_store(cli.load_config(project["cfg"]))
    chunk = store.chunks("c150")[0]
    assert "When to try this: Try this when an endpoint" in chunk.header and chunk.meta["source"] == "gitlab"


def test_enrich_without_a_key_warns_and_uses_templates_but_can_be_made_strict(project, capsys):
    run(project, "gitlab", "fetch", capsys=capsys)
    run(project, "collect", capsys=capsys)
    code, out, err = run(project, "enrich", "--llm", "fake_enrich", capsys=capsys)
    assert code == 0 and "template=" in out and "warning" in (out + err).lower()
    code, _, err = run(project, "enrich", "--llm", "fake_enrich", "--force", "--require-llm", capsys=capsys)
    assert code == 1 and "HRAG_FAKE_KEY" in err


def test_gitlab_fetch_without_a_links_file_explains(project, capsys):
    (project["root"] / "sources" / "gitlab_urls.txt").unlink()
    code, _, err = run(project, "gitlab", "fetch", capsys=capsys)
    assert code == 1 and "gitlab_urls.txt" in err
