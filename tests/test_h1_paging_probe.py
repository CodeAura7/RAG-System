"""Tests for tools/h1_paging_probe.py: how does the HackerOne listing REALLY paginate? Fake API, no network, no token.
The probe prints structure, counts and dates only - never titles or report text."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from hrag.h1_api import H1Client
from tests.test_h1_discover import FakeListing, days_items, make_item
from tools import h1_paging_probe as pp

TITLE_SENTINEL = "ZZ-SECRET-TITLE"


def run(tmp_path, fake, argv=None, **kw):
    printed: list[str] = []
    out = tmp_path / "out"
    code = pp.main(["--since", "2025-01-01", "--out-dir", str(out)] + (argv or []), client_factory=fake.client, out=printed.append, **kw)
    return code, "\n".join(printed), out


def test_probe_finds_a_page_cap_of_50_with_no_next_link_and_working_page_numbers(tmp_path):
    fake = FakeListing(days_items(130), page_cap=50, send_next=False)
    code, text, out = run(tmp_path, fake)
    verdict = json.loads((out / "paging.json").read_text(encoding="utf-8"))["verdict"]
    assert code == 0
    assert verdict["page_size_cap"] == 50 and verdict["next_link_sent"] is False and verdict["page_number_works"] is True
    assert "50" in text and "page[number]" in text


def test_probe_notices_when_page_numbers_are_ignored(tmp_path):
    fake = FakeListing(days_items(130), page_cap=50, send_next=False, honor_page_number=False)
    _, _, out = run(tmp_path, fake)
    verdict = json.loads((out / "paging.json").read_text(encoding="utf-8"))["verdict"]
    assert verdict["page_number_works"] is False


def test_probe_records_the_next_link_parameters_when_the_api_sends_one(tmp_path):
    fake = FakeListing(days_items(130), page_cap=50, send_next=True)
    _, _, out = run(tmp_path, fake)
    data = json.loads((out / "paging.json").read_text(encoding="utf-8"))
    assert data["verdict"]["next_link_sent"] is True
    assert "page[number]" in data["verdict"]["next_link_parameters"] and "page[size]" in data["verdict"]["next_link_parameters"]


def test_probe_records_what_a_far_out_of_range_page_returns(tmp_path):
    fake = FakeListing(days_items(60), page_cap=50, send_next=False)
    _, _, out = run(tmp_path, fake)
    far = [r for r in json.loads((out / "paging.json").read_text(encoding="utf-8"))["requests"] if r["label"] == "page_far"][0]
    assert far["status"] == 200 and far["items"] == 0
    fake2 = FakeListing(days_items(60), page_cap=50, send_next=False, reject_pages_after=3)
    _, _, out2 = run(tmp_path / "again", fake2)
    far2 = [r for r in json.loads((out2 / "paging.json").read_text(encoding="utf-8"))["requests"] if r["label"] == "page_far"][0]
    assert far2["status"] == 400


def test_probe_captures_pagination_related_headers_and_only_names_of_the_rest(tmp_path):
    class WithHeaders(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            return httpx.Response(resp.status_code, content=resp.content, headers={
                "Content-Type": "application/json", "Link": '<https://api.hackerone.com/v1/hackers/hacktivity?page=2>; rel="next"',
                "X-Total-Count": "977", "Set-Cookie": "session=ZZ-COOKIE-VALUE", "X-RateLimit-Remaining": "590"})

    _, _, out = run(tmp_path, WithHeaders(days_items(130), page_cap=50, send_next=False))
    data = json.loads((out / "paging.json").read_text(encoding="utf-8"))
    blob = json.dumps(data)
    assert data["verdict"]["link_header"] is True and data["verdict"]["total_count_header"] == "977"
    assert "ZZ-COOKIE-VALUE" not in blob and "set-cookie" in data["header_names_seen"]


def test_probe_uses_only_the_listing_endpoint_and_at_most_ten_requests(tmp_path):
    fake = FakeListing(days_items(130), page_cap=50, send_next=True)
    run(tmp_path, fake)
    assert len(fake.requests) <= 10


def test_probe_saves_no_titles_or_text(tmp_path):
    fake = FakeListing([make_item(i, f"2025-02-{i % 28 + 1:02d}", title=TITLE_SENTINEL) for i in range(1, 70)], page_cap=50)
    code, text, out = run(tmp_path, fake)
    assert TITLE_SENTINEL not in text and TITLE_SENTINEL not in (out / "paging.json").read_text(encoding="utf-8")


def test_probe_never_overwrites_and_reports_missing_keys(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "paging.json").write_text("KEEP", encoding="utf-8")
    fake = FakeListing(days_items(10))
    code, _, _ = run(tmp_path, fake)
    assert code == 2 and fake.requests == [] and (out / "paging.json").read_text(encoding="utf-8") == "KEEP"
    from hrag.secrets import MissingKeyError

    def no_keys():
        raise MissingKeyError("No API key found for H1_API_TOKEN")

    printed: list[str] = []
    assert pp.main(["--out-dir", str(tmp_path / "o2")], client_factory=no_keys, out=printed.append) == 2 and "H1_API_TOKEN" in "\n".join(printed)


def test_probe_stops_on_rejected_credentials_without_the_token(tmp_path):
    def handler(request):
        return httpx.Response(401, json={})

    client = lambda: H1Client("alice", "tok-ZZ", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0.0,
                              sleep=lambda s: None, clock=lambda: 0.0)
    printed: list[str] = []
    code = pp.main(["--out-dir", str(tmp_path / "o")], client_factory=client, out=printed.append)
    assert code == 1 and "tok-ZZ" not in "\n".join(printed) and "credentials" in "\n".join(printed).lower()


def test_probe_has_no_database_report_or_download_code():
    src = Path(pp.__file__).read_text(encoding="utf-8")
    for banned in ("sqlite3", "hrag.store", "get_report", "vulnerability_information", "import requests", "urlopen"):
        assert banned not in src


def test_probe_really_asks_for_a_page_far_past_the_end(tmp_path):
    fake = FakeListing(days_items(60), page_cap=50, send_next=False)
    run(tmp_path, fake)
    assert any(q.get("page[number]") == "500" for q in fake.requests)
