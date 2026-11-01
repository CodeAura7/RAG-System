"""GitLab copies of HackerOne reports: URL parsing, polite fetching, listing/search, mapping to SourceReport. No network."""
import json

import httpx
import pytest

from hrag import gitlab_src
from hrag.gitlab_src import GitLabClient, GitLabError

DESC = ("**[HackerOne report #1039435](https://hackerone.com/reports/1039435)** by `ledz1996` on 2020-11-20, assigned to @x:\n\n"
        "## Report\n\n##### Summary\n\nTokens leak.\n")


def issue(iid=288484, desc=DESC, labels=None, **kw):
    base = {"id": iid * 10, "iid": iid, "title": "Geo OAUTH2 flow access_token stealing", "description": desc, "state": "closed",
            "confidential": False, "created_at": "2020-11-20T10:00:00Z", "closed_at": "2021-03-01T00:00:00Z",
            "labels": labels or ["HackerOne", "Weakness::CWE-322", "severity::2", "group::geo"],
            "web_url": f"https://gitlab.com/gitlab-org/gitlab/-/issues/{iid}", "user_notes_count": 2}
    base.update(kw)
    return base


def client(handler, tmp_path=None, token=None):
    sleeps = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return GitLabClient(client=http, cache_dir=tmp_path, token=token, min_interval=0.0, sleep=sleeps.append, clock=lambda: 0.0), sleeps


@pytest.mark.parametrize("url,expected", [
    ("https://gitlab.com/gitlab-org/gitlab/-/issues/288484", ("gitlab-org/gitlab", 288484)),
    ("https://gitlab.com/gitlab-org/gitlab/-/issues/288484#note_1", ("gitlab-org/gitlab", 288484)),
    ("https://gitlab.com/gitlab-org/gitlab/-/work_items/288484?x=1", ("gitlab-org/gitlab", 288484)),
    ("https://gitlab.com/a/b/c/-/issues/7/", ("a/b/c", 7)),
])
def test_issue_urls_are_parsed(url, expected):
    assert gitlab_src.parse_issue_url(url) == expected


@pytest.mark.parametrize("bad", ["https://evil.example/gitlab-org/gitlab/-/issues/1", "https://gitlab.com/gitlab-org/gitlab",
                                 "https://gitlab.com/gitlab-org/gitlab/-/merge_requests/3", "not a url"])
def test_other_hosts_and_non_issue_urls_are_refused(bad):
    with pytest.raises(GitLabError):
        gitlab_src.parse_issue_url(bad)


def test_fetch_maps_a_copy_to_a_source_report_and_caches_it(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert "authorization" not in request.headers and "private-token" not in request.headers     # public: no credentials sent
        if request.url.path.endswith("/notes"):
            return httpx.Response(401, json={"message": "401 Unauthorized"})
        return httpx.Response(200, json=issue())

    c, _ = client(handler, tmp_path)
    r = c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/288484")
    assert (r.source, r.report_id, r.program) == ("gitlab", 1039435, "gitlab")
    assert r.source_url == "https://gitlab.com/gitlab-org/gitlab/-/issues/288484"
    assert (r.cwe, r.weakness, r.severity) == ("CWE-322", "CWE-322 (Key Exchange without Entity Authentication)", "high")
    assert r.title.startswith("Geo OAUTH2") and r.created_at.startswith("2020-11-20") and "Tokens leak." in r.body
    assert "HackerOne" in r.labels
    assert any("comments" in n for n in r.notes)                       # honest note: comments need a token
    n_before = len(calls)
    c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/288484")
    assert len(calls) == n_before                                       # second call served from the disk cache


def test_comments_are_fetched_only_when_a_token_is_given_and_sent_only_to_gitlab(tmp_path):
    seen = {}

    def handler(request):
        if request.url.path.endswith("/notes"):
            seen["token"] = request.headers.get("private-token")
            return httpx.Response(200, json=[{"body": "Fixed in 13.9", "system": False, "author": {"username": "carol", "bot": False}}])
        return httpx.Response(200, json=issue())

    c, _ = client(handler, tmp_path, token="glpat-secret-value")
    r = c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/288484")
    assert seen["token"] == "glpat-secret-value" and r.comments[0]["body"] == "Fixed in 13.9"
    assert "glpat-secret-value" not in json.dumps([p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")])   # never cached


def test_an_issue_that_is_not_a_hackerone_copy_is_rejected():
    c, _ = client(lambda r: httpx.Response(200, json=issue(desc="Just a normal bug report.", labels=["bug"])))
    with pytest.raises(GitLabError, match="not a copy of a HackerOne report"):
        c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/5")


def test_confidential_or_missing_issues_give_clear_errors():
    c, _ = client(lambda r: httpx.Response(404, json={"message": "404 Not found"}))
    with pytest.raises(GitLabError, match="not found or not public"):
        c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/5")


def test_listing_follows_pages_and_find_copy_searches_by_report_id():
    def handler(request):
        p = request.url.params
        if p.get("search"):
            assert "hackerone.com/reports/1039435" in p["search"]
            return httpx.Response(200, json=[issue(1, desc="mentions hackerone.com/reports/1039435 only"), issue(288484)])
        page = int(p.get("page", "1"))
        headers = {"x-next-page": "2"} if page == 1 else {}
        return httpx.Response(200, json=[issue(page * 10), issue(page * 10 + 1)], headers=headers)

    c, _ = client(handler)
    assert [i["iid"] for i in c.iter_hackerone_issues(max_items=3)] == [10, 11, 20]
    assert c.find_copy(1039435)["iid"] == 288484                        # skips the issue that merely mentions the id


def test_rate_limit_is_waited_out():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "3"}) if state["n"] == 1 else httpx.Response(200, json=[])

    c, sleeps = client(handler)
    assert list(c.iter_hackerone_issues()) == [] and 3.0 in sleeps


def test_weakness_names_for_common_cwes():
    assert "Server-Side Request Forgery" in gitlab_src.weakness_from_labels(["Weakness::CWE-918"])[1]
    assert gitlab_src.weakness_from_labels(["Weakness::CWE-99999"]) == ("CWE-99999", "CWE-99999")
    assert gitlab_src.weakness_from_labels(["bug"]) == ("", "")


# ---------- choosing candidates from a listing (cheap quality prefilter, diverse weaknesses) ----------
def _good_desc(i):
    return (f"**[HackerOne report #{i}](https://hackerone.com/reports/{i})** by `r` on 2020-01-01:\n\n## Report\n\n##### Summary\n\n"
            + f"Bug number {i} " * 30 + f"\n\n##### Steps to reproduce\n\n1. Send:\n\n```http\nGET /x{i} HTTP/1.1\n```\n2. See it.\n\n"
            f"##### Impact\n\nAn attacker can read data of other users.\n")


def _gl(i, cwe, desc=None, **kw):
    return issue(i, desc=desc or _good_desc(1000 + i), labels=["HackerOne", f"Weakness::CWE-{cwe}"], **kw)


def test_pick_candidates_prefers_complete_reports_and_spreads_weaknesses():
    pool = [_gl(1, 79), _gl(2, 79), _gl(3, 79), _gl(4, 79), _gl(5, 918), _gl(6, 89),
            _gl(7, 22, desc=DESC),                                                # too thin
            issue(8, desc="No header at all, just a plain bug. " * 40, labels=["bug"])]     # not a HackerOne copy
    picked = gitlab_src.pick_candidates(pool, n=5, per_weakness=2, min_chars=200)
    assert [i["iid"] for i in picked] == [1, 2, 5, 6]                            # 2 x CWE-79, then 918 and 89; thin/non-copy skipped
    assert len(gitlab_src.pick_candidates(pool, n=2, per_weakness=2, min_chars=200)) == 2


def test_cache_issue_writes_a_record_the_fetcher_can_reuse_without_a_request(tmp_path):
    def handler(request):
        raise AssertionError("no request expected: served from the cache")

    c, _ = client(handler, tmp_path)
    c.cache_issue("gitlab-org/gitlab", _gl(9, 918))
    r = c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/9")
    assert r.report_id == 1009 and r.cwe == "CWE-918"
    c.cache_issue("gitlab-org/gitlab", _gl(9, 79))                              # an existing record is never overwritten (it may hold notes)
    assert c.fetch_report("https://gitlab.com/gitlab-org/gitlab/-/issues/9").cwe == "CWE-918"
