"""Tests for tools/h1_syntax_probe.py: which listing queries does the API really obey? A configurable fake API decides which spellings,
bounds, filters, sorts and page semantics it honours; the probe must judge each by the returned items, not by the HTTP status.
No network, no token. Output holds statuses, counts, dates and ids only."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import httpx

from hrag.h1_api import H1Client
from tools import h1_syntax_probe as sp

SECRET_TITLE = "ZZ-SECRET-TITLE"
SUBSTATES = ["resolved", "informative", "duplicate"]
SEVERITIES = ["low", "medium", "high"]
PROGRAMS = ["acme", "globex", "initech"]


def make_items(n=260):
    out = []
    for i in range(n):
        day = (date(2025, 1, 1) + timedelta(days=i // 3)).isoformat()                    # three reports per day
        out.append({"id": 100 + i, "type": "hacktivity-item",
                    "attributes": {"title": SECRET_TITLE, "substate": SUBSTATES[i % 3], "url": f"https://hackerone.com/reports/{100 + i}",
                                   "disclosed_at": f"{day}T{8 + i % 3:02d}:{i % 60:02d}:{(i * 7) % 60:02d}.{i % 1000:03d}Z",
                                   "severity_rating": SEVERITIES[(i // 3) % 3], "votes": (i * 37) % 101,
                                   "submitted_at": f"2024-{(i * 5) % 12 + 1:02d}-10T00:00:00.000Z",
                                   "latest_disclosable_activity_at": f"2025-06-{(i * 11) % 28 + 1:02d}T00:00:00.000Z"},
                    "relationships": {"program": {"data": {"attributes": {"handle": PROGRAMS[(i // 2) % 3]}}}}})
    return out


ITEMS = make_items()
ROWS = [{"report_id": i["id"], "disclosed_at": i["attributes"]["disclosed_at"]} for i in ITEMS]


class SyntaxFake:
    """support: the feature names this API obeys. Anything else is rejected (HTTP 400) or, with unsupported='ignore', silently ignored."""

    def __init__(self, support=(), unsupported="reject", page_semantics="exact", cap=50, items=None):
        self.support, self.unsupported, self.page_semantics, self.cap = set(support), unsupported, page_semantics, cap
        self.items = items if items is not None else ITEMS
        self.requests: list[dict] = []

    def _feature(self, name, query_has):
        if not query_has:
            return None
        return "ok" if name in self.support else ("reject" if self.unsupported == "reject" else "ignore")

    def handler(self, request):
        q = dict(request.url.params)
        self.requests.append(q)
        query, sort = q.get("queryString", ""), q.get("sort") or ""
        items = list(self.items)
        m = re.search(r"disclosed_at:>=(\d{4}-\d\d-\d\d)(?: |$)", query)
        if m:
            items = [i for i in items if i["attributes"]["disclosed_at"][:10] >= m.group(1)]
        ts_forms = {"ts_ms": r'disclosed_at:>=(\d{4}-\d\d-\d\dT[\d:]+\.\d+Z)(?: |$)', "ts_no_ms": r'disclosed_at:>=(\d{4}-\d\d-\d\dT[\d:]+Z)(?: |$)',
                    "ts_quoted": r'disclosed_at:>="(\d{4}-\d\d-\d\dT[^"]+)"', "ts_escaped": r'disclosed_at:>=(\d{4}-\d\d-\d\dT[\d\\:]+\.\d+Z)',
                    "ts_range": r'disclosed_at:\[(\S+) TO \*\]', "ts_space": r'disclosed_at:>="(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"'}
        for name, rx in ts_forms.items():
            mm = re.search(rx, query)
            if mm:
                state = self._feature(name, True)
                if state == "reject":
                    return httpx.Response(400, json={})
                value = mm.group(1).replace("\\", "").replace(" ", "T")
                if state == "ok":
                    items = [i for i in items if i["attributes"]["disclosed_at"][:19] >= value[:19]]
                else:
                    items = [i for i in items if i["attributes"]["disclosed_at"][:10] >= value[:10]]
                break
        for name, rx, inclusive in (("upper_range", r"disclosed_at:\[(\d{4}-\d\d-\d\d) TO (\d{4}-\d\d-\d\d)\]", True),
                                    ("upper_le", r"disclosed_at:<=(\d{4}-\d\d-\d\d)", True), ("upper_ge_lt", r">=\d{4}-\d\d-\d\d AND disclosed_at:<(\d{4}-\d\d-\d\d)", False),
                                    ("upper_lt", r"disclosed_at:<(\d{4}-\d\d-\d\d)", False)):
            mm = re.search(rx, query)
            if mm:
                state = self._feature(name, True)
                if state == "reject":
                    return httpx.Response(400, json={})
                if state == "ok":
                    hi = mm.group(mm.lastindex)
                    items = [i for i in items if (i["attributes"]["disclosed_at"][:10] <= hi if inclusive else i["attributes"]["disclosed_at"][:10] < hi)]
                break
        for name, field, getter in (("filter_substate", "substate", lambda i: i["attributes"]["substate"]),
                                    ("filter_severity", "severity_rating", lambda i: i["attributes"]["severity_rating"]),
                                    ("filter_team", "team", lambda i: i["relationships"]["program"]["data"]["attributes"]["handle"]),
                                    ("filter_program", "program", lambda i: i["relationships"]["program"]["data"]["attributes"]["handle"])):
            mm = re.search(rf" AND {field}:(\S+)", query)
            if mm:
                state = self._feature(name, True)
                if state == "reject":
                    return httpx.Response(400, json={})
                if state == "ok":
                    items = [i for i in items if getter(i) == mm.group(1)]
        key = sort.lstrip("-")
        if key in ("submitted_at", "votes", "latest_disclosable_activity_at", "disclosed_at"):
            if key != "disclosed_at" and f"sort_{key}" not in self.support:
                if self.unsupported == "reject":
                    return httpx.Response(400, json={})
                key = "disclosed_at"
            items.sort(key=lambda i: i["attributes"][key], reverse=sort.startswith("-"))
        size = min(int(q.get("page[size]", 25)), self.cap)
        number = int(q.get("page[number]", 1))
        asked = int(q.get("page[size]", 25))
        if self.page_semantics == "wrong_offset" and number > 1:
            chunk = items[(number - 1) * asked:(number - 1) * asked + 10]
        else:
            chunk = items[(number - 1) * size:number * size]
        return httpx.Response(200, json={"data": chunk, "links": {}})

    def client(self):
        return H1Client("alice", "tok-ZZ", client=httpx.Client(transport=httpx.MockTransport(self.handler)), min_interval=0.0,
                        sleep=lambda s: None, clock=lambda: 0.0)


def run(tmp_path, fake, rows=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"reports": rows if rows is not None else ROWS}), encoding="utf-8")
    printed: list[str] = []
    out = tmp_path / "syn"
    code = sp.main(["--manifest", str(man), "--out-dir", str(out)], client_factory=fake.client, out=printed.append)
    data = json.loads((out / "syntax.json").read_text(encoding="utf-8")) if (out / "syntax.json").exists() else None
    return code, "\n".join(printed), data


def verdicts(data, group):
    return {r["name"]: r["verdict"] for r in data["results"] if r["group"] == group}


# ---------------------------------------------------------------- the reference report

def test_the_reference_is_the_last_report_of_a_busy_day_with_later_days_after_it():
    ref = sp.pick_reference(ROWS)
    day_rows = sorted((r for r in ROWS if r["disclosed_at"][:10] == ref["day"]), key=lambda r: r["disclosed_at"])
    assert ref["report_id"] == day_rows[-1]["report_id"] and ref["reports_that_day"] >= 2
    last_day = max(r["disclosed_at"][:10] for r in ROWS)
    assert (date.fromisoformat(last_day) - date.fromisoformat(ref["day"])).days >= 5


def test_no_reference_when_no_day_has_two_reports():
    rows = [{"report_id": i, "disclosed_at": f"2025-01-{i:02d}T00:00:00.000Z"} for i in range(1, 20)]
    assert sp.pick_reference(rows) is None


# ---------------------------------------------------------------- timestamps: judged by the items, not by the status

def test_only_the_spelling_the_api_obeys_is_reported_as_working(tmp_path):
    code, text, data = run(tmp_path, SyntaxFake(support={"ts_quoted"}))
    v = verdicts(data, "timestamp")
    assert v["ts_quoted"] == "obeyed" and v["ts_ms"] == "rejected" and v["ts_range"] == "rejected"
    assert data["works"]["timestamp"] == ["ts_quoted"] and "ts_quoted" in text


def test_a_time_that_is_silently_ignored_is_not_reported_as_working(tmp_path):
    _, _, data = run(tmp_path, SyntaxFake(support=(), unsupported="ignore"))
    v = verdicts(data, "timestamp")
    assert set(v.values()) <= {"day level only", "ignored", "cannot tell"} and "timestamp" not in data["works"]


def test_when_nothing_works_the_summary_says_so(tmp_path):
    class Shuffled(SyntaxFake):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            if int(request.url.params.get("page[number]", 1)) > 1:
                body["data"] = ITEMS[200:210]
            return httpx.Response(200, json=body)

    _, text, data = run(tmp_path, Shuffled())
    assert "work: NONE" in text and "nothing found that gets past a busy day" in text
    _, text2, _ = run(tmp_path / "again", SyntaxFake())
    assert "page numbers are exact at size 100" in text2


# ---------------------------------------------------------------- upper bounds, filters, sorts

def test_upper_bounds_are_judged_newest_first_so_an_ignored_bound_is_obvious(tmp_path):
    _, _, data = run(tmp_path, SyntaxFake(support={"upper_le", "upper_range"}))
    v = verdicts(data, "upper")
    assert v["upper_le"] == "obeyed" and v["upper_range"] == "obeyed" and v["upper_lt"] == "rejected"
    _, _, data2 = run(tmp_path / "b", SyntaxFake(support=(), unsupported="ignore"))
    assert set(verdicts(data2, "upper").values()) == {"ignored"}


def test_a_filter_counts_as_working_only_if_every_item_matches_and_fewer_come_back(tmp_path):
    _, _, data = run(tmp_path, SyntaxFake(support={"filter_substate", "filter_severity"}))
    v = verdicts(data, "filter")
    assert v["substate"] == "obeyed" and v["severity_rating"] == "obeyed" and v["team"] == "rejected" and v["program"] == "rejected"
    _, _, data2 = run(tmp_path / "b", SyntaxFake(support=(), unsupported="ignore"))
    assert set(verdicts(data2, "filter").values()) == {"ignored"}


def test_a_sort_key_works_only_if_the_order_is_really_by_that_key(tmp_path):
    _, _, data = run(tmp_path, SyntaxFake(support={"sort_submitted_at", "sort_votes"}))
    v = verdicts(data, "sort")
    assert v["submitted_at"] == "obeyed" and v["-votes"] == "obeyed" and v["latest_disclosable_activity_at"] == "rejected"
    _, _, data2 = run(tmp_path / "b", SyntaxFake(support=(), unsupported="ignore"))
    assert "obeyed" not in verdicts(data2, "sort").values()


# ---------------------------------------------------------------- page numbers against the ground truth in the manifest

def test_exact_page_numbers_are_recognised(tmp_path):
    _, text, data = run(tmp_path, SyntaxFake())
    v = verdicts(data, "pages")
    assert v["size 10 page 2"] == "exact" and v["size 25 page 2"] == "exact" and v["size 50 page 2"] == "exact"
    assert data["page_numbers"]["safe_page_size"] == 100 and data["page_numbers"]["returned_on_page_1"]["100"] == 50


def test_page_numbers_that_follow_the_size_asked_instead_of_the_size_returned_are_named_as_skipping(tmp_path):
    """The suspected real behaviour: asked for 100, page 1 returns 50, page 2 starts at record 100 - so records 50-99 are never seen."""
    _, text, data = run(tmp_path, SyntaxFake(page_semantics="wrong_offset"))
    v = verdicts(data, "pages")
    assert v["size 100 page 2"] == "SKIPS" and v["size 100 page 3"] == "SKIPS"
    assert v["size 10 page 2"] == "exact" and v["size 50 page 2"] == "exact"            # fine whenever asked size == returned size
    assert data["page_numbers"]["safe_page_size"] == 50
    skip = [r for r in data["results"] if r["name"] == "size 100 page 2"][0]
    assert "skipped" in skip["note"]


def test_totally_wrong_positions_are_flagged_too(tmp_path):
    class Shuffled(SyntaxFake):
        def handler(self, request):
            resp = super().handler(request)
            body = json.loads(resp.content)
            if int(request.url.params.get("page[number]", 1)) > 1:
                body["data"] = ITEMS[200:210]
            return httpx.Response(200, json=body)

    _, _, data = run(tmp_path, Shuffled())
    assert verdicts(data, "pages")["size 10 page 2"] == "WRONG POSITION" and data["page_numbers"]["safe_page_size"] is None


def test_page_numbers_are_not_judged_when_the_manifest_is_too_small(tmp_path):
    rows = ROWS[:60]
    _, _, data = run(tmp_path, SyntaxFake(), rows=rows)
    assert verdicts(data, "pages")["all"] == "cannot tell"


# ---------------------------------------------------------------- hygiene

def test_request_count_is_bounded(tmp_path):
    fake = SyntaxFake()
    run(tmp_path, fake)
    assert len(fake.requests) <= 30


def test_nothing_but_statuses_counts_dates_and_ids_is_printed_or_saved(tmp_path):
    code, text, data = run(tmp_path, SyntaxFake(support={"ts_ms"}))
    blob = text + (tmp_path / "syn" / "syntax.json").read_text(encoding="utf-8")
    assert SECRET_TITLE not in blob and "tok-ZZ" not in blob and "http" not in json.dumps(data["results"])


def test_unreadable_manifest_overwrite_and_missing_keys_are_handled(tmp_path):
    fake = SyntaxFake()
    printed: list[str] = []
    assert sp.main(["--manifest", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")], client_factory=fake.client, out=printed.append) == 2
    assert "not found" in "\n".join(printed) and fake.requests == []
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"reports": ROWS}), encoding="utf-8")
    out = tmp_path / "keep"
    out.mkdir()
    (out / "syntax.json").write_text("KEEP", encoding="utf-8")
    assert sp.main(["--manifest", str(man), "--out-dir", str(out)], client_factory=fake.client, out=printed.append) == 2
    assert (out / "syntax.json").read_text(encoding="utf-8") == "KEEP" and fake.requests == []
    from hrag.secrets import MissingKeyError

    def no_keys():
        raise MissingKeyError("No API key found for H1_API_TOKEN")

    assert sp.main(["--manifest", str(man), "--out-dir", str(tmp_path / "o3")], client_factory=no_keys, out=printed.append) == 2


def test_rejected_credentials_stop_the_probe_without_the_token(tmp_path):
    def handler(request):
        return httpx.Response(401, json={})

    client = lambda: H1Client("alice", "tok-ZZ", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0.0,
                              sleep=lambda s: None, clock=lambda: 0.0)
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"reports": ROWS}), encoding="utf-8")
    printed: list[str] = []
    code = sp.main(["--manifest", str(man), "--out-dir", str(tmp_path / "o")], client_factory=client, out=printed.append)
    assert code == 1 and "tok-ZZ" not in "\n".join(printed)


def test_the_tool_has_no_database_report_body_or_download_code():
    src = Path(sp.__file__).read_text(encoding="utf-8")
    for banned in ("sqlite3", "hrag.store", "get_report", "vulnerability_information", "import requests", "urlopen"):
        assert banned not in src


def test_a_sort_that_is_ignored_but_happens_to_match_is_not_reported_as_obeyed(tmp_path):
    """If submitted_at grows with disclosed_at, an ignored sort looks sorted. That must read 'cannot tell', not 'obeyed'."""
    mono = make_items()
    for n, item in enumerate(mono):
        item["attributes"]["submitted_at"] = f"2024-01-01T00:{n // 60:02d}:{n % 60:02d}.000Z"
        item["attributes"]["latest_disclosable_activity_at"] = f"2025-02-01T00:{n // 60:02d}:{n % 60:02d}.000Z"
    rows = [{"report_id": i["id"], "disclosed_at": i["attributes"]["disclosed_at"]} for i in mono]
    _, _, data = run(tmp_path, SyntaxFake(support=(), unsupported="ignore", items=mono), rows=rows)
    v = verdicts(data, "sort")
    assert v["submitted_at"] == "cannot tell" and v["latest_disclosable_activity_at"] == "cannot tell"


def test_the_reference_day_is_not_one_of_the_last_days_of_the_manifest():
    rows = [{"report_id": i, "disclosed_at": f"2025-01-{i:02d}T10:00:00.000Z"} for i in range(1, 21)]
    rows += [{"report_id": 100 + k, "disclosed_at": f"2025-01-20T1{k}:00:00.000Z"} for k in range(1, 6)]            # the busiest day is the LAST day
    rows += [{"report_id": 200 + k, "disclosed_at": f"2025-01-05T1{k}:30:00.000Z"} for k in range(1, 3)]
    ref = sp.pick_reference(rows)
    assert ref["day"] == "2025-01-05"


def test_upper_bounds_say_cannot_tell_when_the_manifest_has_nothing_after_the_bound(tmp_path):
    rows = [{"report_id": 1, "disclosed_at": "2025-01-01T01:00:00.000Z"}, {"report_id": 2, "disclosed_at": "2025-01-01T02:00:00.000Z"},
            {"report_id": 3, "disclosed_at": "2025-01-02T01:00:00.000Z"}, {"report_id": 4, "disclosed_at": "2025-01-02T02:00:00.000Z"}]
    items = [i for i in ITEMS if i["id"] in (101, 102, 103, 104)]
    _, _, data = run(tmp_path, SyntaxFake(support=(), unsupported="ignore", items=items), rows=rows)
    assert set(verdicts(data, "upper").values()) == {"cannot tell"}
