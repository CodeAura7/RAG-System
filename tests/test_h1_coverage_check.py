"""Tests for tools/h1_coverage_check.py: an independent audit of a discovery manifest against the API, month by month.
Fake API, no network. Prints counts, dates and report ids only - never titles."""
from __future__ import annotations

import json
from pathlib import Path

from tests.test_h1_discover import FakeListing, make_item
from tools import h1_coverage_check as cc

NOW = lambda: cc.datetime(2025, 6, 15, tzinfo=cc.timezone.utc)


def items_for(months: dict[str, int]) -> list[dict]:
    out, n = [], 100
    for month, count in months.items():
        for d in range(count):
            out.append(make_item(n, f"{month}-{d % 27 + 1:02d}", title="ZZ-TITLE-SECRET"))
            n += 1
    return out


def manifest_of(items, drop=lambda i: False, tmp_path=None, since="2025-01-01"):
    rows = [{"report_id": i["id"], "disclosed_at": i["attributes"]["disclosed_at"]} for i in items if not drop(i)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"since": since, "complete": True, "reports": rows}), encoding="utf-8")
    return path


def run(tmp_path, fake, manifest, extra=None):
    printed: list[str] = []
    out = tmp_path / "cov"
    code = cc.main(["--manifest", str(manifest), "--out-dir", str(out)] + (extra or []), client_factory=fake.client, out=printed.append, now=NOW)
    return code, "\n".join(printed), out


def test_a_complete_manifest_is_reported_as_fully_covered(tmp_path):
    items = items_for({"2025-01": 12, "2025-02": 9, "2025-03": 15})
    code, text, out = run(tmp_path, FakeListing(items), manifest_of(items, tmp_path=tmp_path))
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert code == 0 and data["verdict"]["problem_months"] == [] and data["verdict"]["recall_sampled"] == 1.0


def test_a_month_missing_from_the_manifest_is_flagged_when_the_api_has_reports_in_it(tmp_path):
    items = items_for({"2025-01": 12, "2025-02": 9, "2025-03": 15})
    man = manifest_of(items, drop=lambda i: i["attributes"]["disclosed_at"].startswith("2025-02"), tmp_path=tmp_path)
    code, text, out = run(tmp_path, FakeListing(items), man)
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert code == 1 and "2025-02" in data["verdict"]["problem_months"] and "MISSING" in text


def test_a_partly_missing_month_is_flagged_as_incomplete(tmp_path):
    items = items_for({"2025-01": 12, "2025-02": 9})
    man = manifest_of(items, drop=lambda i: i["id"] % 2 == 0 and i["attributes"]["disclosed_at"].startswith("2025-01"), tmp_path=tmp_path)
    code, text, out = run(tmp_path, FakeListing(items), man)
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert "2025-01" in data["verdict"]["problem_months"] and data["verdict"]["recall_sampled"] < 1.0 and data["verdict"]["missing_report_ids"]


def test_a_month_with_no_disclosures_is_recognised_as_genuinely_empty(tmp_path):
    items = items_for({"2025-01": 10, "2025-03": 10})                      # February has nothing at all
    code, text, out = run(tmp_path, FakeListing(items), manifest_of(items, tmp_path=tmp_path))
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    feb = [m for m in data["months"] if m["month"] == "2025-02"][0]
    assert feb["verdict"] == "empty_per_api" and "2025-02" not in data["verdict"]["problem_months"] and code == 0


def test_the_newest_reports_are_checked_against_the_manifest(tmp_path):
    items = items_for({"2025-05": 30})
    man = manifest_of(items, drop=lambda i: i["id"] >= 120, tmp_path=tmp_path)           # the newest ten are missing
    code, text, out = run(tmp_path, FakeListing(items), man)
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert data["tail"]["checked"] > 0 and data["tail"]["missing"] > 0 and code == 1


def test_the_exact_timestamp_syntax_is_tested(tmp_path):
    items = items_for({"2025-01": 20})
    _, text, out = run(tmp_path, FakeListing(items), manifest_of(items, tmp_path=tmp_path))
    syn = json.loads((out / "coverage.json").read_text(encoding="utf-8"))["timestamp_filter"]
    assert syn["ge_works"] is True and syn["gt_works"] is True
    _, _, out2 = run(tmp_path / "again", FakeListing(items, reject_filter_with=400), manifest_of(items, tmp_path=tmp_path))


def test_months_after_now_are_not_probed_and_requests_are_bounded(tmp_path):
    items = items_for({"2025-01": 12})
    fake = FakeListing(items)
    run(tmp_path, fake, manifest_of(items, tmp_path=tmp_path))
    assert len(fake.requests) <= 2 + 12 + 1                                    # syntax tests + Jan..Jun x 2 probes + tail


def test_nothing_is_saved_or_printed_but_counts_dates_and_ids(tmp_path):
    items = items_for({"2025-01": 12})
    code, text, out = run(tmp_path, FakeListing(items), manifest_of(items, tmp_path=tmp_path))
    assert "ZZ-TITLE-SECRET" not in text + (out / "coverage.json").read_text(encoding="utf-8")


def test_unreadable_manifest_gives_the_specific_message_and_makes_no_request(tmp_path):
    fake = FakeListing([make_item(1, "2025-01-01")])
    printed: list[str] = []
    code = cc.main(["--manifest", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "o")], client_factory=fake.client, out=printed.append, now=NOW)
    assert code == 2 and "not found" in "\n".join(printed) and fake.requests == []


def test_never_overwrites_and_reports_missing_keys(tmp_path):
    items = items_for({"2025-01": 5})
    man = manifest_of(items, tmp_path=tmp_path)
    out = tmp_path / "cov"
    out.mkdir()
    (out / "coverage.json").write_text("KEEP", encoding="utf-8")
    fake = FakeListing(items)
    code, text, _ = run(tmp_path, fake, man)
    assert code == 2 and fake.requests == [] and (out / "coverage.json").read_text(encoding="utf-8") == "KEEP"
    from hrag.secrets import MissingKeyError

    def no_keys():
        raise MissingKeyError("No API key found for H1_API_TOKEN")

    printed: list[str] = []
    assert cc.main(["--manifest", str(man), "--out-dir", str(tmp_path / "o2")], client_factory=no_keys, out=printed.append, now=NOW) == 2


def test_the_tool_has_no_database_report_body_or_download_code():
    src = Path(cc.__file__).read_text(encoding="utf-8")
    for banned in ("sqlite3", "hrag.store", "get_report", "vulnerability_information", "import requests", "urlopen"):
        assert banned not in src


def test_a_gap_only_among_the_newest_reports_still_fails_the_audit(tmp_path):
    items = items_for({"2025-05": 40})
    man = manifest_of(items, drop=lambda i: i["attributes"]["disclosed_at"][8:10] in ("26", "27"), tmp_path=tmp_path)    # only the newest two days
    code, text, out = run(tmp_path, FakeListing(items), man)
    data = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert data["verdict"]["problem_months"] == [] and data["tail"]["missing"] > 0 and code == 1


def test_timestamp_filter_is_reported_false_when_rejected_or_ignored(tmp_path):
    items = items_for({"2025-01": 20})
    man = manifest_of(items, tmp_path=tmp_path)
    _, _, out = run(tmp_path / "a", FakeListing(items, reject_filter_with=400), man)
    assert json.loads((out / "coverage.json").read_text(encoding="utf-8"))["timestamp_filter"]["ge_works"] is False
    _, _, out2 = run(tmp_path / "b", FakeListing(items, supports_filter=False), man)
    syn = json.loads((out2 / "coverage.json").read_text(encoding="utf-8"))["timestamp_filter"]
    assert syn["ge_works"] is False


def test_probes_before_the_since_date_are_not_made(tmp_path):
    items = items_for({"2025-01": 20})
    fake = FakeListing(items)
    run(tmp_path, fake, manifest_of(items, tmp_path=tmp_path), ["--since", "2025-01-16"])
    assert not any("disclosed_at:>=2025-01-01" in q.get("queryString", "") for q in fake.requests)
