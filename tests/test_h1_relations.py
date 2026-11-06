"""Tests for tools/h1_relations.py: what do a report's relationships (generated content, activities, attachments, summaries) hold?

No network and no real token: a fake HackerOne transport. The JSON shapes are ASSUMED from HackerOne's public documentation (JSON:API
envelopes) - the real run prints which of them exist. The tool must print/save NAMES, COUNTS and LENGTHS only."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from hrag.h1_api import H1Client
from tools import h1_relations as rel

SECRET_BODY = "ZZ-SECRET-COMMENT-TEXT the reporter sent a request to http://10.0.0.5/admin and got the config back"
INTERNAL_BODY = "ZZ-INTERNAL-ONLY-NOTE do not share this"
ATTACH_URL = "https://hackerone-us-west-2-production.s3.amazonaws.com/ZZ-EXPIRING-LINK?X-Amz-Signature=abc"
FILE_NAME = "ZZ-secret-screenshot-name.png"
TOKEN = "tok-123-ZZ"


def activity(kind, text, internal=False, rid=1):
    attrs = {"message": text, "created_at": "2026-01-01T00:00:00Z", "internal": internal}
    return {"id": str(rid), "type": kind, "attributes": attrs}


def detail(rid, activities=None, attachments=None, summaries=None, bare_activities=False):
    acts = activities if activities is not None else [
        activity("activity-comment", SECRET_BODY, rid=1),
        activity("activity-comment", INTERNAL_BODY, internal=True, rid=2),
        activity("activity-bug-triaged", "", rid=3),
    ]
    atts = attachments if attachments is not None else [
        {"id": "9", "type": "attachment", "attributes": {"file_name": FILE_NAME, "content_type": "image/png", "file_size": 2048,
                                                         "expiring_url": ATTACH_URL}}]
    sums = summaries if summaries is not None else [
        {"id": "5", "type": "summary", "attributes": {"category": "team", "content": "A short team summary that exists.", "created_at": "x"}}]
    rels = {"reporter": {"data": {"id": "1", "type": "user"}},
            "activities": {"data": acts}, "attachments": {"data": atts}, "summaries": {"data": sums}, "bounties": {"data": []}}
    return {"data": {"id": str(rid), "type": "report", "attributes": {"title": "t", "state": "resolved"}, "relationships": rels}}


def hacktivity_item(rid, summary="A generated one-line summary"):
    return {"id": rid, "type": "hacktivity-item", "attributes": {"title": "t", "disclosed_at": "2026-01-01T00:00:00Z"},
            "relationships": {"report_generated_content": {"data": {"type": "report-generated-content",
                                                                    "attributes": {"hacktivity_summary": summary}}}}}


class Fake:
    def __init__(self, details=None, items=None):
        self.details = details or {}
        self.items = items if items is not None else [hacktivity_item(11), hacktivity_item(12, summary="")]
        self.calls: list[str] = []

    def handler(self, request):
        path = request.url.path
        self.calls.append(path)
        if path.endswith("/hacktivity"):
            return httpx.Response(200, json={"data": self.items, "links": {}})
        rid = int(path.rsplit("/", 1)[1])
        if rid in self.details:
            return httpx.Response(200, json=self.details[rid])
        return httpx.Response(404, json={"errors": []})

    def client(self):
        return H1Client("alice", TOKEN, client=httpx.Client(transport=httpx.MockTransport(self.handler)), min_interval=0.0,
                        sleep=lambda s: None, clock=lambda: 0.0)


# ---------------------------------------------------------------- profiling one relationship

def test_a_relationship_is_profiled_by_count_type_and_text_length_not_by_text():
    entries = [activity("activity-comment", SECRET_BODY), activity("activity-comment", "short note here")]
    p = rel.profile_resources(entries)
    assert p["count"] == 2 and p["types"] == {"activity-comment": 2}
    msg = p["text_fields"]["message"]
    assert msg["entries"] == 2 and msg["max_chars"] == len(SECRET_BODY) and msg["min_chars"] == len("short note here")
    assert SECRET_BODY not in json.dumps(p)


def test_internal_entries_are_counted_but_their_text_is_never_read():
    entries = [activity("activity-comment", SECRET_BODY), activity("activity-comment", INTERNAL_BODY, internal=True)]
    p = rel.profile_resources(entries, preview=200)
    assert p["internal"] == 1 and p["public"] == 1
    assert p["text_fields"]["message"]["entries"] == 1                       # only the public one
    assert INTERNAL_BODY not in json.dumps(p) and all(INTERNAL_BODY not in s for s in p["previews"])


def test_an_entry_with_no_internal_flag_is_reported_as_unknown_visibility_and_not_previewed():
    p = rel.profile_resources([{"id": "1", "type": "x", "attributes": {"message": INTERNAL_BODY}}], preview=200)
    assert p["visibility_unknown"] == 1 and p["previews"] == []


def test_url_and_file_name_values_are_never_kept_but_sizes_and_content_types_are():
    atts = [{"id": "9", "type": "attachment", "attributes": {"file_name": FILE_NAME, "content_type": "image/png", "file_size": 2048,
                                                             "expiring_url": ATTACH_URL}}]
    p = rel.profile_resources(atts, preview=200)
    blob = json.dumps(p)
    assert ATTACH_URL not in blob and FILE_NAME not in blob and "X-Amz" not in blob
    assert p["value_counts"]["content_type"] == {"image/png": 1}
    assert p["total_file_size"] == 2048
    assert p["previews"] == []


def test_technical_signals_are_counted_per_entry():
    texts = ["see ```\ncode\n``` here", "POST /import HTTP/1.1\nHost: x", "curl -X POST http://a.b/c", "![shot](x.png)", "1. open\n2. click",
             "plain words only here"]
    p = rel.profile_resources([activity("activity-comment", t, rid=i) for i, t in enumerate(texts)])
    sig = p["signals"]
    assert sig["code_fence"] == 1 and sig["http_request"] == 1 and sig["curl"] == 1 and sig["image_markdown"] == 1
    assert sig["numbered_steps"] == 1 and sig["url"] >= 1


def test_preview_is_off_by_default_capped_and_only_for_public_entries():
    entries = [activity("activity-comment", SECRET_BODY)]
    assert rel.profile_resources(entries)["previews"] == []
    p = rel.profile_resources(entries, preview=10)
    assert p["previews"] == [SECRET_BODY[:10]]
    assert len(rel.profile_resources(entries, preview=100_000)["previews"][0]) <= rel.MAX_PREVIEW


def test_an_empty_relationship_is_reported_as_empty():
    assert rel.profile_resources([])["count"] == 0


def test_identifier_only_entries_are_resolved_from_the_included_section():
    body = {"data": {"id": "1", "type": "report", "relationships": {"summaries": {"data": [{"id": "5", "type": "summary"}]}}},
            "included": [{"id": "5", "type": "summary", "attributes": {"category": "team", "content": "resolved from included"}}]}
    entries = rel.relationship_entries(body, "summaries")
    assert entries[0]["attributes"]["content"] == "resolved from included"


def test_identifier_only_entries_without_included_are_flagged_as_ids_only():
    body = {"data": {"id": "1", "relationships": {"summaries": {"data": [{"id": "5", "type": "summary"}]}}}}
    p = rel.profile_resources(rel.relationship_entries(body, "summaries"))
    assert p["ids_only"] == 1


def test_a_single_object_relationship_is_treated_as_one_entry_and_a_missing_one_as_none():
    body = {"data": {"id": "1", "relationships": {"severity": {"data": {"id": "1", "type": "severity", "attributes": {"rating": "high"}}}}}}
    assert len(rel.relationship_entries(body, "severity")) == 1
    assert rel.relationship_entries(body, "nothing") is None


# ---------------------------------------------------------------- the whole run

def run(tmp_path: Path, fake: Fake, argv: list[str], **kw):
    printed: list[str] = []
    code = rel.main(argv + ["--out-dir", str(tmp_path / "out")], client_factory=fake.client, out=printed.append, **kw)
    return code, "\n".join(printed)


def test_main_profiles_each_report_and_calls_only_the_two_read_endpoints(tmp_path):
    fake = Fake(details={11: detail(11), 12: detail(12)})
    code, text = run(tmp_path, fake, ["11", "12"])
    assert code == 0
    assert set(fake.calls) <= {"/v1/hackers/hacktivity", "/v1/hackers/reports/11", "/v1/hackers/reports/12"}
    assert fake.calls.count("/v1/hackers/hacktivity") == 1
    assert "report 11" in text and "report 12" in text and "activities" in text and "attachments" in text and "summaries" in text


def test_main_output_and_saved_file_hold_no_report_text_urls_or_names(tmp_path):
    fake = Fake(details={11: detail(11)})
    code, text = run(tmp_path, fake, ["11"])
    saved = (tmp_path / "out" / "relations.json").read_text(encoding="utf-8")
    for secret in (SECRET_BODY, INTERNAL_BODY, ATTACH_URL, FILE_NAME, TOKEN, "A short team summary", "A generated one-line summary"):
        assert secret not in text and secret not in saved
    assert json.loads(saved)["reports"][0]["report_id"] == 11


def test_main_reports_internal_entries_as_counted_and_skipped(tmp_path):
    fake = Fake(details={11: detail(11)})
    _, text = run(tmp_path, fake, ["11"])
    assert "internal 1" in text


def test_main_profiles_the_generated_content_of_the_listing(tmp_path):
    fake = Fake(details={11: detail(11)})
    _, text = run(tmp_path, fake, ["11"])
    saved = json.loads((tmp_path / "out" / "relations.json").read_text(encoding="utf-8"))
    g = saved["hacktivity_generated_content"]
    assert g["items"] == 2 and g["with_text"] == 1 and "hacktivity_summary" in g["text_fields"]


def test_main_marks_a_report_that_cannot_be_read_and_continues(tmp_path):
    fake = Fake(details={12: detail(12)})
    code, text = run(tmp_path, fake, ["11", "12"])
    assert "report 11" in text and "not available" in text and "report 12" in text
    assert code == 1


def test_main_takes_at_most_five_reports(tmp_path):
    code, text = run(tmp_path, Fake(), [str(i) for i in range(1, 7)])
    assert code == 2 and "at most 5" in text


def test_main_refuses_non_numeric_ids(tmp_path):
    fake = Fake()
    code, _ = run(tmp_path, fake, ["abc"])
    assert code == 2 and fake.calls == []


def test_without_ids_main_uses_the_probes_sample_ids(tmp_path):
    probe = tmp_path / "h1_probe.json"
    probe.write_text(json.dumps({"sample_report_ids": [11, 12, 13, 14]}), encoding="utf-8")
    fake = Fake(details={11: detail(11), 12: detail(12), 13: detail(13)})
    code, text = run(tmp_path, fake, ["--probe-file", str(probe)])
    assert "report 11" in text and "report 13" in text and "report 14" not in text      # first three only


def test_without_ids_and_without_a_probe_file_main_says_what_to_do(tmp_path):
    fake = Fake()
    code, text = run(tmp_path, fake, ["--probe-file", str(tmp_path / "missing.json")])
    assert code == 2 and "h1 probe" in text and fake.calls == []


def test_main_never_writes_over_an_existing_output_folder_file(tmp_path):
    fake = Fake(details={11: detail(11)})
    out = tmp_path / "out"
    out.mkdir()
    (out / "relations.json").write_text("KEEP", encoding="utf-8")
    code, text = run(tmp_path, fake, ["11"])
    assert code == 2 and (out / "relations.json").read_text(encoding="utf-8") == "KEEP" and fake.calls == []


def test_missing_credentials_give_a_plain_message_and_no_request(tmp_path):
    from hrag.secrets import MissingKeyError

    def no_keys():
        raise MissingKeyError("No API key found for H1_API_TOKEN")

    printed: list[str] = []
    code = rel.main(["11", "--out-dir", str(tmp_path / "o")], client_factory=no_keys, out=printed.append)
    assert code == 2 and "H1_API_TOKEN" in "\n".join(printed)


def test_the_tool_has_no_database_and_no_download_code():
    src = Path(rel.__file__).read_text(encoding="utf-8")
    for banned in ("sqlite3", "hrag.store", "Store(", "urlretrieve", "httpx.get", "httpx.stream", "import requests", "urlopen", ".stream("):
        assert banned not in src


def test_preview_text_is_shown_on_screen_only_and_never_saved(tmp_path):
    fake = Fake(details={11: detail(11)})
    code, text = run(tmp_path, fake, ["11", "--preview", "40"])
    saved = (tmp_path / "out" / "relations.json").read_text(encoding="utf-8")
    assert SECRET_BODY[:40] in text                                    # the user asked to see it
    assert SECRET_BODY[:40] not in saved and "ZZ-SECRET" not in saved  # but it is not kept
    assert INTERNAL_BODY[:20] not in text                               # and internal entries are never shown


def test_link_and_file_name_fields_are_ignored_not_counted_as_text():
    atts = [{"id": "9", "type": "attachment", "attributes": {"file_name": FILE_NAME, "content_type": "image/png", "file_size": 10,
                                                             "expiring_url": ATTACH_URL}}]
    p = rel.profile_resources(atts)
    assert "expiring_url" not in p["text_fields"] and "file_name" not in p["text_fields"]
    assert set(p["ignored_fields"]) == {"expiring_url", "file_name"}
    assert p["signals"].get("url", 0) == 0                                  # an expiring download link is not a technical finding


def test_a_very_long_preview_request_is_cut_to_the_cap():
    long_text = "x" * 5000
    p = rel.profile_resources([activity("activity-comment", long_text)], preview=5000)
    assert len(p["previews"][0]) == rel.MAX_PREVIEW


# ---------------------------------------------------------------- visibility: which entries are EXPLICITLY public?

def test_every_attribute_name_is_listed_and_visibility_like_names_are_called_out():
    entries = [{"id": "1", "type": "activity-comment", "attributes": {"message": "hello there", "created_at": "x", "visibility": "public"}}]
    p = rel.profile_resources(entries)
    assert {"message", "created_at", "visibility"} <= set(p["attribute_names"])
    assert p["visibility_fields"] == ["visibility"]
    assert "hello there" not in json.dumps(p)


def test_entries_without_any_visibility_marker_are_not_eligible_for_ingestion():
    entries = [activity("activity-comment", "a note", internal=False, rid=1), {"id": "2", "type": "activity-comment", "attributes": {"message": "no flag"}},
               activity("activity-comment", "secret", internal=True, rid=3)]
    p = rel.profile_resources(entries)
    assert (p["public"], p["visibility_unknown"], p["internal"]) == (1, 1, 1)
    assert p["eligible_for_ingestion"] == 1                                   # only the explicitly public one


def test_unstated_entries_are_previewed_only_on_request_and_only_text_fields():
    entry = {"id": "2", "type": "activity-comment", "attributes": {"message": "unstated visibility note text", "name": "Jane Reporter"}}
    assert rel.profile_resources([entry], preview=20)["previews"] == []                              # default: not shown
    shown = rel.profile_resources([entry], preview=20, preview_unstated=True)["previews"]
    assert shown == ["unstated visibility note text"[:20]]
    assert all("Jane" not in s for s in shown)                                                       # 'name' is not a preview field


def test_internal_entries_are_never_previewed_even_when_unstated_ones_are_requested():
    p = rel.profile_resources([activity("activity-comment", INTERNAL_BODY, internal=True)], preview=100, preview_unstated=True)
    assert p["previews"] == [] and INTERNAL_BODY not in json.dumps(p)


def test_person_and_attachment_relationships_are_never_previewed():
    body = {"data": {"id": "1", "relationships": {
        "reporter": {"data": {"id": "1", "type": "user", "attributes": {"bio": "ZZ-private-bio text", "content": "ZZ-person-content", "internal": False}}},
        "attachments": {"data": [{"id": "2", "type": "attachment", "attributes": {"description": "ZZ-desc", "message": "ZZ-att-message", "internal": False}}]},
        "summaries": {"data": [{"id": "3", "type": "report-summary", "attributes": {"content": "a public summary text here"}}]}}}}
    prof = rel.profile_report(1, body, preview=30, preview_unstated=True)
    assert prof["relationships"]["reporter"]["previews"] == [] and prof["relationships"]["attachments"]["previews"] == []
    assert prof["relationships"]["summaries"]["previews"] == ["a public summary text here"]


def test_main_shows_unstated_previews_on_screen_only_when_asked(tmp_path):
    fake = Fake(details={11: detail(11, activities=[{"id": "1", "type": "activity-comment", "attributes": {"message": SECRET_BODY}}])})
    _, text = run(tmp_path, fake, ["11", "--preview", "30"])
    assert SECRET_BODY[:30] not in text                                                      # off by default, even with --preview
    out2 = tmp_path / "second"
    printed: list[str] = []
    rel.main(["11", "--preview", "30", "--preview-unstated", "--out-dir", str(out2)], client_factory=fake.client, out=printed.append)
    assert SECRET_BODY[:30] in "\n".join(printed)
    assert SECRET_BODY[:30] not in (out2 / "relations.json").read_text(encoding="utf-8")


def test_main_states_plainly_what_may_be_ingested(tmp_path):
    fake = Fake(details={11: detail(11)})
    _, text = run(tmp_path, fake, ["11"])
    assert "explicitly public" in text.lower() and "not stated" in text.lower() and "held back" in text.lower()


# ---------------------------------------------------------------- surveying many reports

def manifest_file(tmp_path, n=100):
    path = tmp_path / "manifest.json"
    rows = [{"report_id": 5000 + i, "disclosed_at": f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}T00:00:00Z"} for i in range(n)]
    path.write_text(json.dumps({"reports": rows[::-1]}), encoding="utf-8")                      # stored newest first on purpose
    return path


def test_choose_sample_is_spread_over_time_and_includes_both_ends():
    rows = [{"report_id": i, "disclosed_at": f"2025-01-{i % 28 + 1:02d}T00:00:00Z" if i < 28 else f"2025-02-{i % 28 + 1:02d}T00:00:00Z"} for i in range(56)]
    ids = rel.choose_sample(rows, 5)
    assert len(ids) == 5 and len(set(ids)) == 5
    ordered = sorted(rows, key=lambda r: r["disclosed_at"])
    assert ids[0] == ordered[0]["report_id"] and ids[-1] == ordered[-1]["report_id"]
    assert rel.choose_sample(rows[:3], 10) == [r["report_id"] for r in sorted(rows[:3], key=lambda r: r["disclosed_at"])]


def test_survey_reads_sampled_reports_and_aggregates_without_text(tmp_path):
    man = manifest_file(tmp_path)
    ids = rel.choose_sample(json.loads(man.read_text(encoding="utf-8"))["reports"], 4)
    fake = Fake(details={i: detail(i) for i in ids})
    code, text = run(tmp_path, fake, ["--manifest", str(man), "--sample", "4"])
    saved = json.loads((tmp_path / "out" / "relations.json").read_text(encoding="utf-8"))
    assert code == 0 and len(saved["reports"]) == 4 and len(fake.calls) == 1 + 4
    s = saved["survey"]
    assert s["reports"] == 4 and s["relationships"]["activities"]["reports_with_entries"] == 4
    assert s["relationships"]["activities"]["entries_total"] == 12 and s["relationships"]["activities"]["internal"] == 4
    assert s["relationships"]["attachments"]["content_types"] == {"image/png": 4}
    assert "survey" in text.lower() and SECRET_BODY not in text and SECRET_BODY not in json.dumps(saved)


def test_survey_arguments_are_checked_before_any_request(tmp_path):
    man = manifest_file(tmp_path)
    fake = Fake()
    for argv in (["--sample", "5"], ["--manifest", str(man)], ["--manifest", str(man), "--sample", "41"], ["--manifest", str(man), "--sample", "0"],
                 ["--manifest", str(man), "--sample", "3", "11"], ["--manifest", str(tmp_path / "nope.json"), "--sample", "3"]):
        code, text = run(tmp_path / f"r{len(argv)}{argv[-1][:3]}", fake, argv)
        assert code == 2, argv
    assert fake.calls == []


def test_aggregate_sums_counts_and_unions_names():
    a = rel.profile_report(1, detail(1))
    b = rel.profile_report(2, detail(2, summaries=[]))
    s = rel.aggregate([a, b])
    summaries = s["relationships"]["summaries"]
    assert s["reports"] == 2 and summaries["reports_with_entries"] == 1 and summaries["entries_total"] == 1
    assert s["relationships"]["activities"]["types"]["activity-comment"] == 4
    assert "message" in s["relationships"]["activities"]["attribute_names"]


# ---------------------------------------------------------------- reading the manifest: harmless variants work, real problems are named

def _good_manifest_text(n=40):
    rows = [{"report_id": 7000 + i, "disclosed_at": f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}T00:00:00Z", "title": "T █ ünï"} for i in range(n)]
    return json.dumps({"complete": True, "reports": rows}, ensure_ascii=False)


@pytest.mark.parametrize("label,encode", [("plain", lambda t: t.encode("utf-8")), ("bom", lambda t: b"\xef\xbb\xbf" + t.encode("utf-8")),
                                           ("utf16", lambda t: t.encode("utf-16")), ("crlf", lambda t: t.replace("\n", "\r\n").encode("utf-8"))])
def test_manifest_loader_accepts_harmless_encoding_variants(tmp_path, label, encode):
    path = tmp_path / "manifest.json"
    path.write_bytes(encode(_good_manifest_text()))
    assert len(rel.load_manifest_rows(str(path))) == 40


def test_manifest_loader_accepts_the_folder_that_holds_manifest_json(tmp_path):
    (tmp_path / "manifest.json").write_text(_good_manifest_text(), encoding="utf-8")
    assert len(rel.load_manifest_rows(str(tmp_path))) == 40


@pytest.mark.parametrize("setup,needle", [
    (lambda d: None, "not found"),
    (lambda d: (d / "manifest.json").write_text("{ this is not json", encoding="utf-8"), "not valid JSON"),
    (lambda d: (d / "manifest.json").write_text(json.dumps({"count": 3, "rows": []}), encoding="utf-8"), "'reports'"),
    (lambda d: (d / "manifest.json").write_text(json.dumps({"complete": False, "stopped_because": "x", "reports": []}), encoding="utf-8"), "0 reports"),
    (lambda d: (d / "manifest.json").write_text(json.dumps({"reports": [{"title": "t"}]}), encoding="utf-8"), "report_id"),
    (lambda d: (d / "manifest.json").write_bytes(b'{"reports": "\x81\x80\x81"}'), "decode"),
])
def test_manifest_loader_names_the_real_problem(tmp_path, setup, needle):
    d = tmp_path / "run"
    d.mkdir()
    setup(d)
    with pytest.raises(rel.ManifestError) as err:
        rel.load_manifest_rows(str(d / "manifest.json"))
    assert needle in str(err.value)


def test_a_missing_manifest_message_shows_the_full_path_and_how_to_find_the_file(tmp_path):
    with pytest.raises(rel.ManifestError) as err:
        rel.load_manifest_rows("data\\h1_discovery\\nope\\manifest.json")
    assert "Get-ChildItem" in str(err.value) and "current folder" in str(err.value)


def test_main_prints_the_specific_manifest_problem(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"reports": []}), encoding="utf-8")
    code, text = run(tmp_path / "x", Fake(), ["--manifest", str(bad), "--sample", "3"])
    assert code == 2 and "0 reports" in text and "could not read any reports" not in text


def test_main_samples_from_a_bom_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xef\xbb\xbf" + _good_manifest_text().encode("utf-8"))
    ids = rel.choose_sample(rel.load_manifest_rows(str(path)), 4)
    fake = Fake(details={i: detail(i) for i in ids})
    code, text = run(tmp_path / "y", fake, ["--manifest", str(path), "--sample", "4"])
    assert code == 0 and len(fake.calls) == 1 + 4
