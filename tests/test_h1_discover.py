"""Tests for tools/h1_discover.py: list disclosed reports since a date through the official Hacker API (metadata only) and count them.

A fake HackerOne listing: no network, no real token. The query syntax and sort parameter are UNVERIFIED against the real API, so the
tool must validate what comes back (date filter honoured? order honoured?) and refuse to continue, rather than trust them."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import httpx
import pytest

from hrag.h1_api import H1Client
from tools import h1_discover as dis

TOKEN = "tok-secret-ZZ"
BASE = "https://api.hackerone.com/v1/hackers"


def make_item(n: int, day: str, severity="high", cwe="CWE-79 - XSS", title=None, program="acme", summary_len=0):
    item = {"id": n, "type": "hacktivity-item",
            "attributes": {"title": title or f"Report {n}", "substate": "resolved", "url": f"https://hackerone.com/reports/{n}",
                           "disclosed_at": f"{day}T10:00:00.000Z", "severity_rating": severity, "cwe": cwe, "votes": 3,
                           "total_awarded_amount": None, "cve_ids": [], "submitted_at": f"{day}T08:00:00.000Z", "disclosed": True},
            "relationships": {"program": {"data": {"type": "program", "attributes": {"handle": program}}},
                              "reporter": {"data": {"type": "user", "attributes": {"username": "ZZ-reporter-name"}}}}}
    if summary_len:
        item["relationships"]["report_generated_content"] = {"data": {"type": "report-generated-content",
                                                                       "attributes": {"hacktivity_summary": "ZZ-SUMMARY-TEXT " + "x" * (summary_len - 16)}}}
    return item


class FakeListing:
    """Emulates GET /hacktivity. supports_filter / supports_sort switch the two features the real API may or may not honour."""

    def __init__(self, items, supports_filter=True, supports_sort=True, reject_filter_with=None, page_cap=None, send_next=True,
                 honor_page_number=True, reject_pages_after=None, empty_after_first=False, skip_offset_bug=False, hide_after=None, hole_mod=None):
        self.skip_offset_bug, self.hide_after, self.hole_mod = skip_offset_bug, hide_after, hole_mod
        self.empty_after_first = empty_after_first
        self.page_cap, self.send_next, self.honor_page_number, self.reject_pages_after = page_cap, send_next, honor_page_number, reject_pages_after
        self.items = list(items)
        self.supports_filter = supports_filter
        self.supports_sort = supports_sort
        self.reject_filter_with = reject_filter_with
        self.requests: list[dict] = []

    def handler(self, request):
        assert request.url.path.endswith("/hacktivity"), request.url.path
        q = dict(request.url.params)
        self.requests.append(q)
        query = q.get("queryString", "")
        items = list(self.items)
        m = re.search(r"disclosed_at:(>=|>)\"?(\d{4}-\d{2}-\d{2}(?:T[0-9:.]+Z)?)", query)
        if m and self.reject_filter_with:
            return httpx.Response(self.reject_filter_with, json={"errors": [{"detail": "bad query"}]})
        if m and self.supports_filter:
            op, val = m.group(1), m.group(2)
            if "T" in val:                                                     # an exact timestamp
                items = [i for i in items if (i["attributes"]["disclosed_at"] >= val if op == ">=" else i["attributes"]["disclosed_at"] > val)]
            else:
                items = [i for i in items if i["attributes"]["disclosed_at"][:10] >= val]
        sort = q.get("sort") or ""
        if sort and self.supports_sort:
            items.sort(key=lambda i: i["attributes"]["disclosed_at"], reverse=sort.startswith("-"))
        size = int(q.get("page[size]", 25))
        if self.page_cap:
            size = min(size, self.page_cap)                      # the real API returned 50 when 100 were asked for
        number = int(q.get("page[number]", 1)) if self.honor_page_number else 1
        if self.reject_pages_after and number > self.reject_pages_after:
            return httpx.Response(400, json={"errors": [{"detail": "page too deep"}]})
        chunk = items[(number - 1) * size: number * size]
        if self.skip_offset_bug and number > 1:
            asked = int(q.get("page[size]", 25))
            chunk = items[(number - 1) * asked: (number - 1) * asked + 10]          # offset from the size ASKED, but only 10 returned
        if self.hole_mod and not sort.startswith("-"):
            chunk = [i for i in chunk if i["id"] % self.hole_mod != 0]            # the oldest-first listing silently drops some reports
        if self.hide_after and not sort.startswith("-") and number == 1:
            chunk = [i for i in chunk if i["attributes"]["disclosed_at"][:10] <= self.hide_after]   # the oldest-first listing stops early
        if self.hide_after and not sort.startswith("-") and number > 1:
            chunk = []
        if self.empty_after_first and int(q.get("page[number]", 1)) > 1:
            chunk = []                                              # a server whose page 2 is simply empty
        links = {}
        if self.send_next and number * size < len(items):
            nq = dict(q)
            nq["page[number]"] = str(number + 1)
            links["next"] = str(httpx.URL(f"{BASE}/hacktivity", params=nq))
        return httpx.Response(200, json={"data": chunk, "links": links})

    def client(self, **kw):
        return H1Client("alice", TOKEN, client=httpx.Client(transport=httpx.MockTransport(self.handler)), min_interval=0.0,
                        sleep=lambda s: None, clock=lambda: 0.0, **kw)


def months_items():
    days = ["2024-12-30", "2025-01-02", "2025-01-15", "2025-02-03", "2025-02-20", "2025-02-21", "2025-03-09"]
    sev = ["high", "low", "high", "critical", "medium", "high", "low"]
    cwes = ["CWE-79 - XSS", "CWE-89 - SQLi", "CWE-79 - XSS", "CWE-79 - XSS", "", "CWE-89 - SQLi", "CWE-22 - Path"]
    return [make_item(100 + i, d, s, c) for i, (d, s, c) in enumerate(zip(days, sev, cwes))]


def collect(fake, since="2025-01-01", **kw):
    logs: list[str] = []
    res = dis.collect(fake.client(), since=since, log=logs.append, **kw)
    return res, logs


# ---------------------------------------------------------------- one row of the manifest

def test_row_has_the_requested_fields_and_no_reporter_name():
    row = dis.row_of(make_item(7, "2025-03-04", "critical", "CWE-79 - XSS", title="Stored XSS", program="acme"))
    assert row["report_id"] == 7 and row["title"] == "Stored XSS" and row["url"] == "https://hackerone.com/reports/7"
    assert row["disclosed_at"].startswith("2025-03-04") and row["severity"] == "critical" and row["cwe"] == "CWE-79 - XSS"
    assert row["cwe_id"] == "CWE-79" and row["program"] == "acme"
    assert "ZZ-reporter-name" not in json.dumps(row)


def test_row_builds_the_url_when_the_item_lacks_one_and_tolerates_missing_fields():
    item = {"id": 55, "attributes": {"disclosed_at": "2025-05-05T00:00:00Z"}}
    row = dis.row_of(item)
    assert row["url"] == "https://hackerone.com/reports/55" and row["severity"] == "" and row["cwe"] == "" and row["program"] == ""


def test_an_item_without_a_usable_id_gives_no_row():
    assert dis.row_of({"attributes": {"title": "x"}}) is None


# ---------------------------------------------------------------- date handling

def test_since_must_be_a_calendar_day():
    assert dis.parse_day("2025-01-01") == "2025-01-01"
    for bad in ("2025-1-1", "01/01/2025", "2025-13-01", "tomorrow", ""):
        with pytest.raises(ValueError):
            dis.parse_day(bad)


# ---------------------------------------------------------------- collection modes

def test_server_filter_mode_returns_only_reports_since_the_date_each_once():
    fake = FakeListing(months_items())
    res, _ = collect(fake, page_size=2)
    assert res.complete and res.mode == "server_filter_sorted"
    ids = [r["report_id"] for r in res.rows]
    assert ids == [101, 102, 103, 104, 105, 106] and len(set(ids)) == len(ids)
    assert all(r["disclosed_at"][:10] >= "2025-01-01" for r in res.rows)


def test_pages_are_followed_with_the_requested_page_size_and_never_exceed_100():
    fake = FakeListing(months_items())
    collect(fake, page_size=2)
    assert any(q.get("page[size]") == "2" for q in fake.requests)
    with pytest.raises(ValueError):
        dis.collect(fake.client(), since="2025-01-01", page_size=101, log=lambda s: None)


def test_falls_back_to_newest_first_when_the_date_filter_is_rejected():
    fake = FakeListing(months_items(), reject_filter_with=400)
    res, logs = collect(fake, page_size=3)
    assert res.complete and res.mode == "newest_first_client_stop"
    assert [r["report_id"] for r in res.rows] == [106, 105, 104, 103, 102, 101]
    assert any("filter" in line.lower() for line in logs)


def test_falls_back_to_newest_first_when_the_filter_is_ignored_but_sorting_works():
    fake = FakeListing(months_items(), supports_filter=False)
    res, _ = collect(fake, page_size=3)
    assert res.complete and res.mode == "newest_first_client_stop"
    assert len(res.rows) == 6


def test_filter_without_sort_is_used_when_only_sorting_is_ignored():
    fake = FakeListing(list(reversed(months_items())), supports_sort=False)          # unsorted (newest first by accident)
    res, _ = collect(fake, page_size=3)
    assert res.mode == "server_filter_unsorted" and sorted(r["report_id"] for r in res.rows) == [101, 102, 103, 104, 105, 106]


def test_stops_with_a_clear_message_when_neither_filter_nor_order_can_be_trusted():
    shuffled = [make_item(1, "2025-02-01"), make_item(2, "2023-01-01"), make_item(3, "2025-06-01"), make_item(4, "2022-01-01")]
    fake = FakeListing(shuffled, supports_filter=False, supports_sort=False)
    res, logs = collect(fake, page_size=2)
    assert not res.complete and res.rows == [] and "could not" in res.stopped_because.lower()


def test_an_order_violation_midway_stops_the_run_as_incomplete():
    class Lying(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            if request.url.params.get("page[number]") == "2" and body["data"]:
                body["data"].reverse()
                body["data"][0]["attributes"]["disclosed_at"] = "2025-01-01T00:00:00Z"        # older than what page 1 ended with
                return httpx.Response(200, json=body)
            return resp

    items = [make_item(200 + i, f"2025-02-{i + 1:02d}") for i in range(8)]
    res, _ = collect(Lying(items), page_size=3, paging="pages")
    assert not res.complete and "order" in res.stopped_because.lower()


def test_until_limits_the_range_in_every_mode():
    res, _ = collect(FakeListing(months_items()), until="2025-02-20", page_size=2)
    assert [r["report_id"] for r in res.rows] == [101, 102, 103, 104] and res.complete
    res2, _ = collect(FakeListing(months_items(), reject_filter_with=400), until="2025-02-20", page_size=2)
    assert sorted(r["report_id"] for r in res2.rows) == [101, 102, 103, 104]


def test_max_items_stops_early_and_marks_the_run_incomplete():
    res, _ = collect(FakeListing(months_items()), max_items=3, page_size=2)
    assert len(res.rows) == 3 and not res.complete and "max" in res.stopped_because.lower()


def test_max_requests_is_a_hard_stop():
    fake = FakeListing([make_item(300 + i, f"2025-03-{i % 28 + 1:02d}") for i in range(40)])
    res, _ = collect(fake, page_size=1, max_requests=6)
    assert not res.complete and res.requests <= 6 and "requests" in res.stopped_because.lower()


def test_a_long_listing_is_resumed_from_a_date_when_the_page_limit_per_segment_is_reached():
    items = [make_item(400 + i, f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}") for i in range(84)]        # 3 months, one per day
    fake = FakeListing(items)
    res, _ = collect(fake, page_size=10, segment_pages=3)
    assert res.complete and len(res.rows) == 84 and len({r["report_id"] for r in res.rows}) == 84
    assert res.segments >= 2


def test_a_segment_that_makes_no_progress_does_not_loop_forever():
    same_day = [make_item(500 + i, "2025-04-01") for i in range(30)]
    res, _ = collect(FakeListing(same_day), page_size=5, segment_pages=2, paging="pages")
    assert not res.complete and "progress" in res.stopped_because.lower() and res.requests < 40


def test_a_next_link_to_another_host_is_refused_by_the_client_and_ends_the_run():
    class Evil(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            if body.get("links", {}).get("next"):
                body["links"]["next"] = "https://evil.example.com/steal"
            return httpx.Response(200, json=body)

    res, _ = collect(Evil(months_items()), page_size=2, paging="pages")
    assert not res.complete and TOKEN not in res.stopped_because


def test_a_next_link_that_repeats_while_pages_keep_coming_is_detected():
    class Loop(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            q = dict(request.url.params)
            q["page[number]"] = "2"
            body["links"] = {"next": str(httpx.URL(f"{BASE}/hacktivity", params=q))}          # always the same page 2
            return httpx.Response(200, json=body)

    res, _ = collect(Loop(days_items(60)), page_size=10, max_requests=30, paging="pages")
    assert not res.complete and "twice" in res.stopped_because and res.requests < 30


def test_authentication_failure_is_reported_without_the_token():
    def handler(request):
        return httpx.Response(401, json={})

    client = H1Client("alice", TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0.0,
                      sleep=lambda s: None, clock=lambda: 0.0)
    res = dis.collect(client, since="2025-01-01", log=lambda s: None)
    assert not res.complete and TOKEN not in res.stopped_because and "credentials" in res.stopped_because.lower()


# ---------------------------------------------------------------- counts

def test_counts_by_month_severity_and_cwe():
    res, _ = collect(FakeListing(months_items()))
    counts = dis.count_rows(res.rows)
    assert counts["total"] == 6
    assert counts["by_month"] == {"2025-01": 2, "2025-02": 3, "2025-03": 1}
    assert counts["by_severity"] == {"high": 2, "low": 2, "critical": 1, "medium": 1}
    assert counts["by_cwe"]["CWE-79 - XSS"] == 2 and counts["by_cwe"]["(none)"] == 1
    assert counts["by_month_severity"]["2025-02"] == {"critical": 1, "medium": 1, "high": 1}


def test_a_missing_severity_or_date_is_counted_under_a_visible_label():
    rows = [dis.row_of({"id": 1, "attributes": {"disclosed_at": "2025-01-05T00:00:00Z"}}), dis.row_of({"id": 2, "attributes": {}})]
    counts = dis.count_rows(rows)
    assert counts["by_severity"] == {"(none)": 2} and counts["by_month"] == {"2025-01": 1, "(unknown)": 1}


# ---------------------------------------------------------------- files

def run_main(tmp_path, fake, argv, **kw):
    printed: list[str] = []
    out = tmp_path / "out"
    code = dis.main(["--out-dir", str(out)] + argv, client_factory=fake.client, out=printed.append, **kw)
    return code, "\n".join(printed), out


def test_main_writes_manifest_json_and_csv_with_the_same_rows(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(months_items()), ["--since", "2025-01-01"])
    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    with (out / "manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert manifest["complete"] is True and manifest["count"] == 6 and manifest["since"] == "2025-01-01"
    assert [int(r["report_id"]) for r in rows] == [r["report_id"] for r in manifest["reports"]]
    assert {"report_id", "title", "url", "disclosed_at", "severity", "cwe"} <= set(rows[0])


def test_main_writes_counts_files_and_prints_the_tables(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(months_items()), ["--since", "2025-01-01"])
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    assert counts["by_month"]["2025-02"] == 3
    with (out / "counts.csv").open(encoding="utf-8-sig", newline="") as fh:
        flat = list(csv.DictReader(fh))
    assert {"dimension": "month", "key": "2025-02", "count": "3"} in flat
    assert "2025-02" in text and "critical" in text and "CWE-79" in text


def test_csv_cells_cannot_run_as_spreadsheet_formulas(tmp_path):
    items = [make_item(1, "2025-02-01", title="=HYPERLINK(\"http://evil\",\"x\")"), make_item(2, "2025-02-02", title="+cmd|' /C calc'!A0"),
             make_item(3, "2025-02-03", title="@SUM(1)"), make_item(4, "2025-02-04", title="-2+3")]
    code, _, out = run_main(tmp_path, FakeListing(items), ["--since", "2025-01-01"])
    with (out / "manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        titles = [r["title"] for r in csv.DictReader(fh)]
    assert all(t.startswith("'") for t in titles)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reports"][0]["title"].startswith("=")                 # JSON keeps the true value


def test_an_incomplete_run_still_saves_what_it_found_and_says_so(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(months_items()), ["--since", "2025-01-01", "--max-items", "2", "--page-size", "2"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 1 and manifest["complete"] is False and manifest["count"] == 2 and "max" in manifest["stopped_because"].lower()
    assert "INCOMPLETE" in text


def test_main_never_overwrites_an_existing_manifest(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text("KEEP", encoding="utf-8")
    fake = FakeListing(months_items())
    code, text, _ = run_main(tmp_path, fake, ["--since", "2025-01-01"])
    assert code == 2 and (out / "manifest.json").read_text(encoding="utf-8") == "KEEP" and fake.requests == []


def test_main_validates_arguments_before_any_request(tmp_path):
    fake = FakeListing(months_items())
    for argv in (["--since", "banana"], ["--since", "2025-01-01", "--page-size", "0"], ["--since", "2025-01-01", "--page-size", "500"],
                 ["--since", "2025-01-01", "--min-interval", "0.01"], ["--since", "2025-03-01", "--until", "2025-01-01"],
                 ["--since", "2025-01-01", "--max-items", "0"]):
        code, text, _ = run_main(tmp_path, fake, argv)
        assert code == 2, argv
    assert fake.requests == []


def test_a_future_since_date_is_refused(tmp_path):
    code, text, _ = run_main(tmp_path, FakeListing(months_items()), ["--since", "2999-01-01"])
    assert code == 2 and "future" in text.lower()


def test_missing_credentials_give_a_plain_message(tmp_path):
    from hrag.secrets import MissingKeyError

    def no_keys():
        raise MissingKeyError("No API key found for H1_API_TOKEN")

    printed: list[str] = []
    code = dis.main(["--since", "2025-01-01", "--out-dir", str(tmp_path / "o")], client_factory=no_keys, out=printed.append)
    assert code == 2 and "H1_API_TOKEN" in "\n".join(printed)


def test_the_token_never_appears_in_any_output_or_file(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(months_items()), ["--since", "2025-01-01"])
    blob = text + "".join(p.read_text(encoding="utf-8-sig") for p in out.iterdir())
    assert TOKEN not in blob and "ZZ-reporter-name" not in blob


def test_default_output_folder_is_new_each_run_so_nothing_is_overwritten(tmp_path):
    cfg = tmp_path / "config.toml"
    src = Path(__file__).resolve().parent.parent / "config.toml"
    cfg.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    printed: list[str] = []
    code = dis.main(["--since", "2025-01-01", "--config", str(cfg)], client_factory=FakeListing(months_items()).client, out=printed.append,
                    now=lambda: dis.datetime(2026, 11, 6, 3, 0, 0, tzinfo=dis.timezone.utc))
    assert code == 0 and (tmp_path / "data" / "h1_discovery" / "20261106T030000Z" / "manifest.json").is_file()


def test_the_tool_has_no_database_and_no_report_content_code():
    src = Path(dis.__file__).read_text(encoding="utf-8")
    for banned in ("sqlite3", "hrag.store", "Store(", "get_report", "vulnerability_information", "import requests", "urlopen"):
        assert banned not in src
    assert not re.search(r"get_json\(\s*f?[\"']/reports", src)            # the detail endpoint is never called


# ---------------------------------------------------------------- the real API capped pages at 50 and sent no usable next link:
# a missing next link must never be taken as "the list is finished"

def days_items(n, start=0):
    """n reports, one per distinct day starting 2025-01-01 (so the watermark moves on every page)."""
    from datetime import date, timedelta
    return [make_item(1000 + i, (date(2025, 1, 1) + timedelta(days=start + i)).isoformat()) for i in range(n)]


def test_capped_pages_without_next_links_are_walked_to_the_end_with_explicit_page_numbers():
    fake = FakeListing(days_items(130), page_cap=50, send_next=False)
    res, logs = collect(fake, page_size=100, paging="pages")
    assert res.complete and len(res.rows) == 130 and len({r["report_id"] for r in res.rows}) == 130
    assert res.largest_page == 50 and res.requests <= 12
    assert any(q.get("page[number]") == "2" for q in fake.requests)


def test_when_page_numbers_are_ignored_too_the_date_watermark_still_reaches_everything():
    fake = FakeListing(days_items(130), page_cap=50, send_next=False, honor_page_number=False)
    res, _ = collect(fake, page_size=100)
    assert res.complete and len(res.rows) == 130 and res.segments >= 3


def test_a_rejected_deeper_page_falls_back_to_the_date_watermark():
    fake = FakeListing(days_items(130), page_cap=50, send_next=False, reject_pages_after=1)
    res, _ = collect(fake, page_size=100)
    assert res.complete and len(res.rows) == 130


def test_a_short_non_empty_last_page_without_a_next_link_is_not_the_end():
    """Exactly the failure seen on the real API: 50 of 130 reports, no next link."""
    fake = FakeListing(days_items(130), page_cap=50, send_next=False, honor_page_number=False)
    res, _ = collect(fake, page_size=100, max_items=300)
    assert len(res.rows) == 130


def test_more_reports_on_one_day_than_a_page_holds_is_reported_not_silently_cut():
    same_day = [make_item(2000 + i, "2025-04-01") for i in range(120)]
    fake = FakeListing(same_day, page_cap=50, send_next=False, honor_page_number=False)
    res, _ = collect(fake, page_size=100)
    assert not res.complete and len(res.rows) == 50
    assert [g["day"] for g in res.gap_days] == ["2025-04-01"] or "tail check" in res.stopped_because.lower()


def test_the_normal_case_with_next_links_still_finishes_with_few_extra_requests():
    fake = FakeListing(days_items(60))
    res, _ = collect(fake, page_size=20)
    assert res.complete and len(res.rows) == 60 and res.requests <= 9


def test_an_empty_listing_since_the_date_is_complete_with_no_rows():
    fake = FakeListing([make_item(1, "2024-01-01")])
    res, _ = collect(fake)
    assert res.complete and res.rows == []


def test_main_reports_the_page_size_the_api_really_returned_and_saves_it(tmp_path):
    fake = FakeListing(days_items(130), page_cap=50, send_next=False)
    code, text, out = run_main(tmp_path, fake, ["--since", "2025-01-01", "--page-size", "100"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0 and manifest["count"] == 130 and manifest["complete"] is True
    assert manifest["pagination"]["requested_page_size"] == 100 and manifest["pagination"]["largest_page_returned"] == 50
    assert "asked for 100" in text and "50" in text


def test_an_empty_page_right_after_fresh_reports_is_not_taken_as_the_end():
    """If page 2 is empty only because page numbers are unsupported, the date watermark must still carry on."""
    fake = FakeListing(days_items(130), page_cap=50, send_next=False, empty_after_first=True)
    res, _ = collect(fake, page_size=100)
    assert res.complete and len(res.rows) == 130


def test_a_repeated_page_of_reports_outside_until_is_noticed_not_looped():
    items = days_items(30)
    fake = FakeListing(items, reject_filter_with=400, send_next=False, honor_page_number=False, page_cap=10)       # newest-first mode
    res, _ = collect(fake, until="2025-01-05", page_size=10, max_requests=40)
    assert not res.complete and "repeated" in res.stopped_because.lower() and res.requests < 12


# ---------------------------------------------------------------- v3: date-watermark ("keyset") paging is the default; page numbers are a fallback

def test_keyset_paging_survives_a_server_whose_later_pages_skip_reports():
    """The real API gave 50 on page 1 and about 10 on later pages: page numbers cannot be trusted, dates can."""
    items = days_items(300)
    walked, _ = collect(FakeListing(items, page_cap=100, send_next=False, skip_offset_bug=True), page_size=100, paging="pages")
    assert len(walked.rows) < 300                                              # the page-number walk really loses reports on this server
    res, _ = collect(FakeListing(items, page_cap=100, send_next=False, skip_offset_bug=True), page_size=100)
    assert res.complete and len(res.rows) == 300 and len({r["report_id"] for r in res.rows}) == 300


def test_keyset_paging_uses_only_first_pages_and_never_page_numbers():
    fake = FakeListing(days_items(130), page_cap=50, send_next=False)
    res, _ = collect(fake, page_size=100)
    assert res.complete and len(res.rows) == 130
    assert all("page[number]" not in q for q in fake.requests) and res.pages_by_number == 0


def test_keyset_paging_works_when_every_page_is_capped_at_ten():
    res, _ = collect(FakeListing(days_items(95), page_cap=10, send_next=False), page_size=100)
    assert res.complete and len(res.rows) == 95


def test_an_exact_multiple_of_the_page_cap_does_not_stall():
    res, _ = collect(FakeListing(days_items(100), page_cap=50, send_next=False), page_size=100)
    assert res.complete and len(res.rows) == 100


def test_one_busy_day_filling_a_whole_page_is_collected_with_page_numbers_when_they_work_but_flagged():
    busy = [make_item(3000 + i, "2025-03-01") for i in range(120)] + [make_item(4000 + i, f"2025-03-{i + 2:02d}") for i in range(5)]
    res, _ = collect(FakeListing(busy, page_cap=50, send_next=False), page_size=100)
    assert len(res.rows) == 125 and not res.complete and [g["day"] for g in res.gap_days] == ["2025-03-01"]


def test_the_newest_first_tail_check_catches_a_listing_that_stops_early():
    fake = FakeListing(days_items(60), hide_after="2025-01-30")
    res, _ = collect(fake, page_size=100)
    assert not res.complete and "tail check" in res.stopped_because.lower() and res.tail_missing > 0
    assert res.requests < 40                                                        # it gives up after a few retries, it does not hammer the API


def test_a_listing_that_silently_drops_reports_is_caught_by_the_tail_check():
    fake = FakeListing(days_items(120), hole_mod=5, page_cap=50, send_next=False)
    res, _ = collect(fake, page_size=100)
    assert not res.complete and "tail check" in res.stopped_because.lower() and res.tail_missing > 0 and res.requests < 60


def test_a_clean_run_records_a_passed_tail_check():
    res, _ = collect(FakeListing(days_items(60), page_cap=50, send_next=False), page_size=100)
    assert res.complete and res.tail_checked > 0 and res.tail_missing == 0


def test_manifest_has_a_page_log_that_explains_every_request(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(days_items(130), page_cap=50, send_next=False), ["--since", "2025-01-01", "--page-size", "100"])
    pag = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["pagination"]
    assert pag["paging"] == "keyset" and pag["tail_check"]["checked"] > 0 and pag["tail_check"]["missing"] == 0
    log = pag["page_log"]
    assert log and all({"n", "segment", "via", "items", "new", "first_day", "last_day"} <= set(e) for e in log)
    assert sum(e["new"] for e in log) == 130 and {e["via"] for e in log} == {"first"}


def test_paging_pages_mode_is_still_available_and_logs_page_numbers(tmp_path):
    code, text, out = run_main(tmp_path, FakeListing(days_items(130), page_cap=50, send_next=False), ["--since", "2025-01-01", "--paging", "pages"])
    pag = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["pagination"]
    assert code == 0 and "page_number" in {e["via"] for e in pag["page_log"]}


def test_generated_summary_length_is_recorded_but_never_the_text(tmp_path):
    items = [make_item(1, "2025-02-01", summary_len=700), make_item(2, "2025-02-02")]
    code, text, out = run_main(tmp_path, FakeListing(items), ["--since", "2025-01-01"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [r["summary_chars"] for r in manifest["reports"]] == [700, 0]
    blob = text + "".join(p.read_text(encoding="utf-8-sig") for p in out.iterdir())
    assert "ZZ-SUMMARY-TEXT" not in blob
    assert "summary_chars" in (out / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert json.loads((out / "counts.json").read_text(encoding="utf-8"))["with_generated_summary"] == 1


def test_the_default_keyset_paging_never_follows_a_next_link_even_a_hostile_one():
    class Evil(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            if body.get("links", {}).get("next"):
                body["links"]["next"] = "https://evil.example.com/steal"
            return httpx.Response(200, json=body)

    res, _ = collect(Evil(months_items()), page_size=2)
    assert res.complete and len(res.rows) == 6 and res.pages_by_next_link == 0


def test_after_a_tail_check_failure_the_walk_moves_over_known_reports_by_date_even_without_page_numbers():
    fake = FakeListing(days_items(120), hole_mod=5, page_cap=50, send_next=False, honor_page_number=False)
    res, _ = collect(fake, page_size=100)
    assert not res.complete and "tail check" in res.stopped_because.lower() and "no progress" not in res.stopped_because.lower()


# ---------------------------------------------------------------- v4: a busy day must never make the date marker LEAP over reports

def busy_timeline():
    """30 quiet days, one BUSY day (120 reports), then 60 quiet days - the shape that made the real run skip most of 2026."""
    from datetime import date, timedelta
    quiet_before = [make_item(100 + i, (date(2025, 1, 1) + timedelta(days=i)).isoformat()) for i in range(30)]
    busy = [make_item(1000 + i, "2025-02-01") for i in range(120)]
    quiet_after = [make_item(5000 + i, (date(2025, 2, 2) + timedelta(days=i)).isoformat()) for i in range(60)]
    return quiet_before, busy, quiet_after


def test_a_busy_day_does_not_make_the_walk_skip_the_quiet_days_after_it():
    before, busy, after = busy_timeline()
    fake = FakeListing(before + busy + after, page_cap=50, send_next=False, skip_offset_bug=True)
    res, _ = collect(fake, page_size=100)
    got = {r["report_id"] for r in res.rows}
    quiet_ids = {i["id"] for i in before + after}
    assert quiet_ids <= got                                                   # every quiet-day report is there - no leap
    assert not res.complete and [g["day"] for g in res.gap_days] == ["2025-02-01"]
    assert "busy" in res.stopped_because.lower() and "2025-02-01" in res.stopped_because
    day_rows = sum(1 for r in res.rows if r["disclosed_at"].startswith("2025-02-01"))
    assert 50 <= day_rows <= 120 and res.gap_days[0]["collected_on_day"] == day_rows


def test_the_old_page_walk_does_leap_on_the_same_timeline():
    """Documents why page numbers are not trusted: this is the failure the real run showed."""
    before, busy, after = busy_timeline()
    fake = FakeListing(before + busy + after, page_cap=50, send_next=False, skip_offset_bug=True)
    res, _ = collect(fake, page_size=100, paging="pages")
    got = {r["report_id"] for r in res.rows}
    assert not ({i["id"] for i in before + after} <= got)


def test_a_busy_day_where_page_numbers_work_is_still_flagged_as_unverified():
    before, busy, after = busy_timeline()
    res, _ = collect(FakeListing(before + busy + after, page_cap=50, send_next=False), page_size=100)
    assert len(res.rows) == 210 and not res.complete and res.gap_days and "may be partial" in res.stopped_because.lower()


def test_main_saves_the_gap_days_and_exits_with_one(tmp_path):
    before, busy, after = busy_timeline()
    fake = FakeListing(before + busy + after, page_cap=50, send_next=False, skip_offset_bug=True)
    code, text, out = run_main(tmp_path, fake, ["--since", "2025-01-01", "--no-audit"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 1 and manifest["complete"] is False and manifest["pagination"]["gap_days"][0]["day"] == "2025-02-01"
    assert "BUSY" in text and "2025-02-01" in text


# ---------------------------------------------------------------- v4: "complete" is only said after an independent month-by-month audit

def early_busy_timeline(n_after=200):
    from datetime import date, timedelta
    before = [make_item(100 + i, (date(2025, 1, 1) + timedelta(days=i)).isoformat()) for i in range(4)]
    busy = [make_item(1000 + i, "2025-01-05") for i in range(120)]
    after = [make_item(5000 + i, (date(2025, 1, 6) + timedelta(days=i)).isoformat()) for i in range(n_after)]
    return before + busy + after


def test_the_final_audit_catches_a_walk_that_leapt_and_marks_the_run_incomplete(tmp_path):
    fake = FakeListing(early_busy_timeline(), page_cap=50, send_next=False, skip_offset_bug=True)
    code, text, out = run_main(tmp_path, fake, ["--since", "2025-01-01", "--paging", "pages"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 1 and manifest["complete"] is False and manifest["audit"]["verdict"]["missing_total"] > 0
    assert "audit" in manifest["stopped_because"].lower()


def test_a_clean_run_passes_the_final_audit_and_records_it(tmp_path):
    from datetime import date, timedelta
    items = [make_item(100 + i, (date(2025, 1, 1) + timedelta(days=i)).isoformat()) for i in range(70)]
    code, text, out = run_main(tmp_path, FakeListing(items, page_cap=50, send_next=False), ["--since", "2025-01-01"], )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0 and manifest["complete"] is True and manifest["audit"]["verdict"]["problem_months"] == []


def test_the_audit_is_skipped_for_a_bounded_or_trial_run(tmp_path):
    from datetime import date, timedelta
    items = [make_item(100 + i, (date(2025, 1, 1) + timedelta(days=i)).isoformat()) for i in range(70)]
    fake = FakeListing(items)
    _, text, out = run_main(tmp_path, fake, ["--since", "2025-01-01", "--until", "2025-01-20"])
    assert "audit" not in json.loads((out / "manifest.json").read_text(encoding="utf-8")) and "audit skipped" in text.lower()


def test_the_busy_day_walk_stops_at_once_when_the_api_repeats_a_page():
    before, busy, after = busy_timeline()
    fake = FakeListing(before + busy + after, page_cap=50, send_next=False, honor_page_number=False)
    res, _ = collect(fake, page_size=100)
    gather = [e for e in res.page_log if str(e["segment"]).startswith("busy:")]
    assert 1 <= len(gather) <= 2 and any(g["day"] == "2025-02-01" for g in res.gap_days)


def test_the_busy_day_walk_gives_up_after_three_pages_with_nothing_new():
    class Cycling(FakeListing):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            n = int(request.url.params.get("page[number]", 1))
            if n > 1 and body["data"]:
                body["data"] = known[(n % 3):(n % 3) + 10]                                # different ids every page, all already known
            return httpx.Response(200, json=body)

    busy = [make_item(1000 + i, "2025-02-01") for i in range(500)]                          # a day so busy there are ~10 pages of it
    quiet = [make_item(100 + i, f"2025-01-{i + 1:02d}") for i in range(20)]
    known = sorted(quiet + busy, key=lambda i: i["attributes"]["disclosed_at"])[:50]
    res, _ = collect(Cycling(quiet + busy, page_cap=50, send_next=False), page_size=100)
    gather = [e for e in res.page_log if str(e["segment"]).startswith("busy:")]
    assert 1 <= len(gather) <= 4 and res.gap_days
