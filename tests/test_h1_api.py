"""HackerOne API collector, tested against a fake transport - no network, no real token.

IMPORTANT: the JSON shapes below are ASSUMED from HackerOne's public Hacker API documentation
(a JSON:API-style envelope: {"data": [...], "links": {"next": ...}}). They have NOT been checked against
a real response. `hrag h1 probe` run with a real token is what verifies them, and these fixtures must be
corrected to match whatever it reports.
"""
import json

import httpx
import pytest

from hrag import h1_api
from hrag.h1_api import H1Client, H1Error

TOKEN = "super-secret-token-123"
BASE = "https://api.hackerone.com/v1/hackers"


def item(rid, handle="acme", title="SSRF"):
    return {"id": str(rid), "type": "hacktivity-item",
            "attributes": {"report_id": rid, "title": title, "disclosed_at": "2023-01-01T00:00:00Z"},
            "relationships": {"program": {"data": {"attributes": {"handle": handle}}}}}


def make_client(handler, tmp_path=None, **kw):
    sleeps: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = H1Client("alice", TOKEN, client=http, cache_dir=tmp_path, min_interval=kw.pop("min_interval", 0.0),
                      sleep=sleeps.append, clock=kw.pop("clock", lambda: 0.0), **kw)
    return client, sleeps


def test_sends_basic_auth_and_never_leaks_the_token_in_errors():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(401, json={"errors": [{"detail": "nope"}]})   # body must not contain the hint words itself

    client, _ = make_client(handler)
    with pytest.raises(H1Error) as err:
        client.get_json("/hacktivity")
    assert seen["auth"].startswith("Basic ")
    assert err.value.status == 401
    assert TOKEN not in str(err.value) and "alice" not in str(err.value)
    assert "credentials" in str(err.value).lower()                    # tells the user what to check


def test_pagination_follows_next_links_and_respects_max_items():
    pages = {
        None: {"data": [item(1), item(2)], "links": {"next": f"{BASE}/hacktivity?page%5Bnumber%5D=2"}},
        "2": {"data": [item(3), item(4)], "links": {}},
    }

    def handler(request):
        return httpx.Response(200, json=pages[request.url.params.get("page[number]")])

    client, _ = make_client(handler)
    assert [h1_api.report_id_of(i) for i in client.iter_hacktivity()] == [1, 2, 3, 4]
    client, _ = make_client(handler)
    assert [h1_api.report_id_of(i) for i in client.iter_hacktivity(max_items=3)] == [1, 2, 3]


def test_never_follows_a_next_link_to_another_host_with_the_credentials():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        # the foreign link is offered ONCE, so a broken guard fails the assertions instead of looping forever
        links = {"next": "https://evil.example/steal"} if len(calls) == 1 else {}
        return httpx.Response(200, json={"data": [item(1)], "links": links})

    client, _ = make_client(handler)
    with pytest.raises(H1Error, match="another host"):
        list(client.iter_hacktivity())
    assert all("evil.example" not in c for c in calls)


def test_retries_rate_limits_using_retry_after_then_gives_up():
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "2"}) if state["n"] == 1 else httpx.Response(200, json={"data": []})

    client, sleeps = make_client(flaky)
    assert client.get_json("/hacktivity") == {"data": []}
    assert 2.0 in sleeps

    client, sleeps = make_client(lambda r: httpx.Response(429), max_retries=2)
    with pytest.raises(H1Error) as err:
        client.get_json("/hacktivity")
    assert err.value.status == 429 and TOKEN not in str(err.value)


def test_does_not_retry_auth_failures_or_missing_reports():
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        return httpx.Response(403, json={})

    client, _ = make_client(handler)
    with pytest.raises(H1Error):
        client.get_json("/reports/5")
    assert n["calls"] == 1


def test_politeness_spacing_between_requests():
    client, sleeps = make_client(lambda r: httpx.Response(200, json={"data": []}), min_interval=1.0)
    client.get_json("/hacktivity")
    client.get_json("/hacktivity")
    assert sleeps == [1.0]                                             # first call free, second waits (fake clock never advances)


def test_get_report_is_cached_on_disk(tmp_path):
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        return httpx.Response(200, json={"data": {"id": "7", "attributes": {"vulnerability_information": "details"}}})

    client, _ = make_client(handler, tmp_path)
    first = client.get_report(7)
    second = client.get_report(7)
    assert first == second and n["calls"] == 1
    assert json.loads((tmp_path / "7.json").read_text(encoding="utf-8")) == first


def test_report_id_of_handles_the_plausible_shapes():
    assert h1_api.report_id_of({"id": "12", "attributes": {}}) == 12
    assert h1_api.report_id_of({"id": "x", "attributes": {"report_id": 34}}) == 34
    assert h1_api.report_id_of({"attributes": {}, "relationships": {"report": {"data": {"id": "56"}}}}) == 56
    assert h1_api.report_id_of({"attributes": {"title": "no id anywhere"}}) is None


def probe_handler(detail_body, detail_status=200, list_extra=None):
    def handler(request):
        if request.url.path.endswith("/hacktivity"):
            it = item(11)
            it["attributes"].update(list_extra or {})
            return httpx.Response(200, json={"data": [it, item(12)], "links": {}})
        return httpx.Response(detail_status, json=detail_body)
    return handler


def test_probe_reports_usable_when_detail_has_full_text():
    body = {"data": {"id": "11", "attributes": {"title": "t", "vulnerability_information": "STEP ONE: do the secret thing"}}}
    client, _ = make_client(probe_handler(body))
    verdict = h1_api.run_probe(client)
    assert verdict["auth_ok"] and verdict["hacktivity_ok"] and verdict["report_detail_ok"]
    assert verdict["usable_for_bulk"] is True
    assert any("vulnerability_information" in p for p, _ in verdict["detail_full_text_fields"])
    dumped = json.dumps(verdict)
    assert "secret thing" not in dumped and TOKEN not in dumped       # lengths and names only, never report text


def test_probe_reports_not_usable_when_only_summaries_come_back():
    client, _ = make_client(probe_handler({"errors": [{"detail": "forbidden"}]}, detail_status=403))
    verdict = h1_api.run_probe(client)
    assert verdict["hacktivity_ok"] is True and verdict["report_detail_ok"] is False
    assert verdict["usable_for_bulk"] is False
    assert verdict["detail_status"]["11"] == 403
    assert any("summar" in n.lower() or "full text" in n.lower() for n in verdict["notes"])


def test_probe_with_bad_credentials_is_a_clean_verdict_not_a_crash():
    client, _ = make_client(lambda r: httpx.Response(401, json={}))
    verdict = h1_api.run_probe(client)
    assert verdict["auth_ok"] is False and verdict["usable_for_bulk"] is False


def test_bulk_collection_is_gated_on_a_passing_probe(tmp_path):
    path = tmp_path / "h1_probe.json"
    with pytest.raises(H1Error, match="probe"):
        h1_api.require_probe_ok(path)                                  # never probed
    h1_api.write_probe(path, {"usable_for_bulk": False, "notes": ["summaries only"]})
    with pytest.raises(H1Error, match="summaries only"):
        h1_api.require_probe_ok(path)                                  # probed, failed: message explains why
    h1_api.write_probe(path, {"usable_for_bulk": True, "notes": []})
    h1_api.require_probe_ok(path)                                      # passes silently


def test_pick_diverse_caps_reports_per_program_and_deduplicates():
    items = [item(i, handle="big") for i in range(1, 8)] + [item(20, "small"), item(21, "other"), item(1, "big")]
    picked = h1_api.pick_diverse(items, n=6, per_program=2)
    ids = [h1_api.report_id_of(i) for i in picked]
    assert len(ids) == len(set(ids)) == 4                             # 2 from big + small + other: all the caps allow, no repeats
    assert sum(1 for i in picked if h1_api.program_of(i) == "big") == 2


# ---------- mapping an API report to the common SourceReport (field names ASSUMED, see module docstring) ----------
def api_report(**attrs):
    base = {"title": "SSRF in webhook", "vulnerability_information": "## Summary\nThe webhook fetches internal URLs.",
            "state": "resolved", "created_at": "2022-01-01T00:00:00Z", "disclosed_at": "2022-03-01T00:00:00Z",
            "weakness": {"id": 68, "name": "Server-Side Request Forgery (SSRF)", "external_id": "cwe-918"},
            "severity": {"rating": "high"}}
    base.update(attrs)
    return {"data": {"id": "1234", "type": "report", "attributes": base,
                     "relationships": {"program": {"data": {"id": "9", "attributes": {"handle": "acme"}}}}}}


def test_api_report_maps_to_a_source_report():
    r = h1_api.to_source_report(api_report())
    assert (r.source, r.report_id, r.title, r.program) == ("hackerone_api", 1234, "SSRF in webhook", "acme")
    assert r.source_url == "https://hackerone.com/reports/1234" and "internal URLs" in r.body
    assert (r.weakness, r.cwe, r.severity, r.substate) == ("Server-Side Request Forgery (SSRF)", "CWE-918", "high", "resolved")
    assert r.disclosed_at.startswith("2022-03-01")


def test_api_report_without_body_text_is_not_usable_and_separate_fields_are_joined():
    assert h1_api.to_source_report(api_report(vulnerability_information="")) is None
    r = h1_api.to_source_report(api_report(vulnerability_information="", steps_to_reproduce="1. Send POST /hook", impact="Reads metadata."))
    assert "1. Send POST /hook" in r.body and "Reads metadata." in r.body
    assert h1_api.to_source_report({"data": {"attributes": {"title": "x", "vulnerability_information": "y"}}}) is None   # no id
