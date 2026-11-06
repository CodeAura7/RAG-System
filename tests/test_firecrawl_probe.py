"""Tests for tools/firecrawl_probe.py - a tiny, polite feasibility test of Firecrawl on 3-5 PUBLIC HackerOne report pages.

No test here touches the network: the Firecrawl SDK is replaced by a fake that records how it was called.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import firecrawl_probe as fp

KEY = "fc-SECRETKEYVALUE-1234"
SENTINEL = "UNIQUE-SENTINEL-PAGE-TEXT"

# a report body long enough to be assessed, with steps, a PoC and an impact statement
HF_BODY = (
    "## Summary\nThe image importer fetches any URL it is given without checking the host.\n\n"
    "## Steps to reproduce\n1. Log in as any user.\n2. Send the request below.\n3. Read the response from the internal host.\n\n"
    "```\nPOST /import HTTP/1.1\nHost: app.example.com\n\nurl=http://169.254.169.254/latest/meta-data/\n```\n\n"
    "## Impact\nAn attacker can read cloud credentials from the metadata service and move into the internal network. "
    + " ".join(f"detail{i}" for i in range(80))
)
PAGE_CHROME_TOP = "# HackerOne\n\n[Hacktivity](/hacktivity) [Programs](/directory) [Log in](/users/sign_in)\n\n"
PAGE_COMMENTS = f"\n\n## Activity\n\nteam-member posted a comment. {SENTINEL}\n\nteam-member changed the status to Resolved.\n"


def doc(markdown: str | None, status: int | None = 200, credits: int | None = 1, error: str | None = None):
    meta = SimpleNamespace(status_code=status, credits_used=credits, proxy_used="basic", error=error)
    return SimpleNamespace(markdown=markdown, metadata=meta)


class FakeSDK:
    """Stands in for firecrawl.Firecrawl: returns queued documents (or raises queued exceptions) and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def scrape(self, url, **kwargs):
        self.calls.append((url, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(argv, sdk=None, tmp_path: Path | None = None, hf=None, key=KEY):
    """Run main() with everything injected; returns (exit_code, printed_text, sdk)."""
    printed: list[str] = []
    sleeps: list[float] = []
    factory_calls: list[str] = []

    def factory(k):
        factory_calls.append(k)
        return fp.SdkFetcher(sdk)

    code = fp.main(
        argv + ["--cache-dir", str(tmp_path / "probe")], client_factory=factory, key_getter=lambda name: key,
        hf_loader=lambda ids: (hf or {}), sleep=sleeps.append, out=printed.append)
    return code, "\n".join(printed), SimpleNamespace(sdk=sdk, sleeps=sleeps, factory_calls=factory_calls)


# ------------------------------------------------------------------ which URLs are allowed

@pytest.mark.parametrize("arg", ["783392", "https://hackerone.com/reports/783392", "https://hackerone.com/reports/783392/"])
def test_accepts_a_report_id_or_the_canonical_public_report_url(arg):
    assert fp.report_url(arg) == (783392, "https://hackerone.com/reports/783392")


@pytest.mark.parametrize("arg", [
    "http://hackerone.com/reports/1",                    # not https
    "https://example.com/reports/1",                     # another site
    "https://hackerone.com.evil.example/reports/1",      # look-alike host
    "https://hackerone.com@evil.example/reports/1",      # user-info trick: the real host is evil.example
    "https://www.hackerone.com/reports/1",               # only the exact host we were told about
    "https://hackerone.com/reports/1?x=1",               # no query string
    "https://hackerone.com/reports/1#comments",          # no fragment
    "https://hackerone.com/reports/abc",                 # not a number
    "https://hackerone.com/reports/1/edit",              # not the report page
    "https://hackerone.com/hacktivity",                  # a listing, not a report
    "", "  ", "0", "-5", "12 34",
])
def test_rejects_anything_that_is_not_one_public_report_page(arg):
    with pytest.raises(fp.ProbeError):
        fp.report_url(arg)


def test_refuses_more_than_five_reports_and_makes_no_request(tmp_path):
    sdk = FakeSDK([doc(HF_BODY)] * 6)
    code, text, ctx = run(["1", "2", "3", "4", "5", "6", "--run"], sdk, tmp_path)
    assert code == 2 and "at most 5" in text
    assert sdk.calls == [] and ctx.factory_calls == []


def test_duplicate_ids_are_fetched_once(tmp_path):
    sdk = FakeSDK([doc(HF_BODY)])
    code, _, _ = run(["77", "https://hackerone.com/reports/77", "--run"], sdk, tmp_path)
    assert code == 0 and len(sdk.calls) == 1


# ------------------------------------------------------------------ dry run is the default

def test_without_run_nothing_is_fetched_and_no_key_is_read(tmp_path):
    sdk = FakeSDK([])
    code, text, ctx = run(["783392", "135946"], sdk, tmp_path)
    assert code == 0
    assert sdk.calls == [] and ctx.factory_calls == []
    assert "dry run" in text.lower() and "--run" in text
    assert "https://hackerone.com/reports/783392" in text
    assert KEY not in text


def test_with_run_but_no_key_it_stops_with_a_clear_message(tmp_path):
    sdk = FakeSDK([doc(HF_BODY)])
    code, text, ctx = run(["1", "--run"], sdk, tmp_path, key=None)
    assert code == 2 and "FIRECRAWL_API_KEY" in text
    assert sdk.calls == []


def test_interval_below_the_free_plan_limit_is_refused(tmp_path):
    sdk = FakeSDK([doc(HF_BODY)])
    code, text, _ = run(["1", "--run", "--interval", "1"], sdk, tmp_path)
    assert code == 2 and "interval" in text.lower()
    assert sdk.calls == []


# ------------------------------------------------------------------ how Firecrawl is called (the no-bypass rules)

def test_the_sdk_is_called_with_the_basic_proxy_and_markdown_only_and_never_with_actions(tmp_path):
    sdk = FakeSDK([doc(HF_BODY)])
    run(["1", "--run"], sdk, tmp_path)
    (url, kwargs), = sdk.calls
    assert url == "https://hackerone.com/reports/1"
    assert kwargs["proxy"] == "basic"                      # 'auto' would silently escalate to enhanced proxies after a block
    assert kwargs["formats"] == ["markdown"]
    assert kwargs["store_in_cache"] is False               # the SDK default (True) would keep a copy of the page on Firecrawl's side
    allowed = {"formats", "proxy", "only_main_content", "timeout", "wait_for", "store_in_cache"}
    assert set(kwargs) <= allowed                          # no actions, no headers, no location tricks, no mobile
    assert "actions" not in kwargs and "headers" not in kwargs


def test_requests_are_spaced_by_the_interval(tmp_path):
    sdk = FakeSDK([doc(HF_BODY), doc(HF_BODY), doc(HF_BODY)])
    _, _, ctx = run(["1", "2", "3", "--run", "--interval", "9"], sdk, tmp_path)
    assert ctx.sleeps == [9.0, 9.0]                        # between requests, not before the first


# ------------------------------------------------------------------ stop at the first problem, never retry

@pytest.mark.parametrize("status", [401, 403, 429, 451, 503])
def test_a_blocking_status_stops_the_whole_run(status, tmp_path):
    sdk = FakeSDK([doc(HF_BODY), doc("Forbidden", status=status), doc(HF_BODY)])
    code, text, _ = run(["1", "2", "3", "--run"], sdk, tmp_path)
    assert code == 1
    assert len(sdk.calls) == 2                             # the third report was never requested
    assert "blocked" in text.lower() and "stopped" in text.lower()


@pytest.mark.parametrize("page", [
    "Just a moment...\nChecking your browser before accessing hackerone.com",
    "Please verify you are human to continue. CAPTCHA",
    "Access denied. Your request was blocked.",
])
def test_a_challenge_page_with_status_200_is_treated_as_blocked(page, tmp_path):
    sdk = FakeSDK([doc(page, status=200), doc(HF_BODY)])
    code, _, _ = run(["1", "2", "--run"], sdk, tmp_path)
    assert code == 1 and len(sdk.calls) == 1


def test_a_long_report_that_merely_mentions_captcha_is_not_a_block(tmp_path):
    long_report = HF_BODY + "\n\nThe login form has a CAPTCHA that can be skipped by omitting the parameter. " * 20
    assert fp.classify(doc(long_report))[0] == "ok"


def test_an_empty_page_is_a_failure_not_a_success(tmp_path):
    assert fp.classify(doc(""))[0] == "empty"
    assert fp.classify(doc(None))[0] == "empty"


def test_a_404_is_reported_as_failed_and_stops_the_run(tmp_path):
    sdk = FakeSDK([doc("Not found", status=404), doc(HF_BODY)])
    code, _, _ = run(["1", "2", "--run"], sdk, tmp_path)
    assert code == 1 and len(sdk.calls) == 1


def test_an_sdk_exception_is_recorded_not_retried_and_stops_the_run(tmp_path):
    sdk = FakeSDK([RuntimeError("rate limited " + KEY), doc(HF_BODY)])
    code, text, _ = run(["1", "2", "--run"], sdk, tmp_path)
    assert code == 1 and len(sdk.calls) == 1               # one attempt, no retry
    assert KEY not in text                                 # an error message is never allowed to leak the key


# ------------------------------------------------------------------ cache: free re-runs, never overwrite

def test_a_cached_report_costs_no_request_and_is_not_overwritten(tmp_path):
    cache = tmp_path / "probe"
    cache.mkdir()
    original = {"url": "https://hackerone.com/reports/1", "status_code": 200, "markdown": HF_BODY, "credits_used": 1,
                "proxy_used": "basic", "error": None}
    (cache / "1.json").write_text(json.dumps(original), encoding="utf-8")
    sdk = FakeSDK([doc("something different")])
    code, text, ctx = run(["1", "--run"], sdk, tmp_path)
    assert code == 0 and sdk.calls == [] and ctx.factory_calls == []
    assert json.loads((cache / "1.json").read_text(encoding="utf-8")) == original
    assert "cached" in text.lower()


def test_only_successful_pages_are_cached(tmp_path):
    sdk = FakeSDK([doc("Forbidden", status=403)])
    run(["1", "--run"], sdk, tmp_path)
    assert not (tmp_path / "probe" / "1.json").exists()


def test_a_successful_page_is_cached_without_the_key(tmp_path):
    sdk = FakeSDK([doc(PAGE_CHROME_TOP + HF_BODY)])
    run(["1", "--run"], sdk, tmp_path)
    saved = (tmp_path / "probe" / "1.json").read_text(encoding="utf-8")
    assert "Steps to reproduce" in saved and KEY not in saved


# ------------------------------------------------------------------ comparison with the Hugging Face text

def test_page_containing_the_whole_dataset_text_is_reported_as_contained():
    m = fp.compare(HF_BODY, PAGE_CHROME_TOP + HF_BODY + PAGE_COMMENTS)
    assert m["verdict"] == "page_contains_dataset_text"
    assert m["containment"] >= 0.95
    assert m["page_chars"] > m["dataset_chars"]


def test_a_page_with_different_text_is_reported_as_different():
    other = " ".join(f"unrelated{i}" for i in range(200))
    m = fp.compare(HF_BODY, other)
    assert m["verdict"] == "different_text" and m["containment"] < 0.1


def test_a_partly_matching_page_is_reported_as_partial():
    half = HF_BODY[: len(HF_BODY) // 2]
    m = fp.compare(HF_BODY, PAGE_CHROME_TOP + half)
    assert m["verdict"] == "partial" and 0.3 < m["containment"] < 0.9


def test_a_report_with_no_body_in_the_dataset_says_so_and_still_describes_the_page():
    m = fp.compare("", PAGE_CHROME_TOP + HF_BODY)
    assert m["verdict"] == "dataset_has_no_body"
    assert m["page_chars"] > 0 and m["page_flags"]["has_steps"] is True


def test_a_report_missing_from_the_dataset_is_reported_as_such():
    assert fp.compare(None, HF_BODY)["verdict"] == "not_in_dataset"


def test_comparison_lists_what_the_page_adds_over_the_dataset():
    thin = "## Summary\nThe importer fetches any URL.\n" + " ".join(f"w{i}" for i in range(100))
    m = fp.compare(thin, PAGE_CHROME_TOP + thin + "\n\n```\nGET /import?url=http://127.0.0.1\n```\n\n## Impact\nAn attacker can read secrets.\n")
    assert "has_poc" in m["page_adds"] and "has_impact" in m["page_adds"]


def test_activity_markers_count_comments_and_status_changes():
    m = fp.compare(HF_BODY, HF_BODY + PAGE_COMMENTS)
    assert m["activity_markers"] >= 2
    assert fp.compare(HF_BODY, HF_BODY)["activity_markers"] == 0


def test_outline_lists_headings_with_short_text_only():
    long_heading = "## " + "x" * 300
    m = fp.compare(HF_BODY, long_heading + "\n" + HF_BODY)
    assert all(len(h["text"]) <= 60 for h in m["outline"]) and len(m["outline"]) <= 40


# ------------------------------------------------------------------ what gets printed and saved (privacy + key safety)

def test_summary_and_output_contain_metrics_but_no_report_text_and_no_key(tmp_path):
    sdk = FakeSDK([doc(PAGE_CHROME_TOP + HF_BODY + PAGE_COMMENTS)])
    code, text, _ = run(["1", "--run"], sdk, tmp_path, hf={1: HF_BODY})
    assert code == 0
    summary = (tmp_path / "probe" / "summary.json").read_text(encoding="utf-8")
    for blob in (summary, text):
        assert SENTINEL not in blob and "169.254.169.254" not in blob and KEY not in blob
    data = json.loads(summary)
    assert data["reports"][0]["report_id"] == 1 and data["reports"][0]["verdict"] == "page_contains_dataset_text"


def test_credits_are_totalled_and_shown(tmp_path):
    sdk = FakeSDK([doc(HF_BODY, credits=1), doc(HF_BODY, credits=1)])
    _, text, _ = run(["1", "2", "--run"], sdk, tmp_path)
    assert "credits spent this run: 2" in text.lower()


def test_exit_code_is_zero_only_when_every_report_was_ok(tmp_path):
    sdk = FakeSDK([doc(HF_BODY), doc(HF_BODY)])
    assert run(["1", "2", "--run"], sdk, tmp_path)[0] == 0


def test_dict_shaped_replies_from_the_sdk_work_too():
    reply = {"markdown": HF_BODY, "metadata": {"status_code": 200, "credits_used": 1, "proxy_used": "basic"}}
    rec = fp.SdkFetcher(FakeSDK([reply])).fetch("https://hackerone.com/reports/1")
    assert rec["status_code"] == 200 and rec["markdown"] == HF_BODY and rec["credits_used"] == 1


# ------------------------------------------------------------------ two robustness gaps found on review

@pytest.mark.parametrize("bad", ["nan", "inf", "1000"])
def test_an_interval_that_is_not_a_sensible_number_is_refused(bad, tmp_path):
    sdk = FakeSDK([doc(HF_BODY)])
    code, text, _ = run(["1", "--run", "--interval", bad], sdk, tmp_path)
    assert code == 2 and "interval" in text.lower() and sdk.calls == []


def test_an_unusual_value_type_in_the_reply_never_loses_a_paid_page(tmp_path):
    reply = doc(HF_BODY)
    reply.metadata.proxy_used = Path("not-json-serialisable")       # a real SDK could return an enum or a model here
    code, _, _ = run(["1", "--run"], FakeSDK([reply]), tmp_path)
    assert code == 0
    assert "Steps to reproduce" in (tmp_path / "probe" / "1.json").read_text(encoding="utf-8")
