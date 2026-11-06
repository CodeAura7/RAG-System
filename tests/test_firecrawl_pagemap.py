"""Tests for tools/firecrawl_pagemap.py: an OFFLINE, read-only profile of pages already cached by firecrawl_probe.
Nothing here touches the network, a key, a database or any cache except reading the probe's own folder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import firecrawl_pagemap as pm

PROSE = ("The image importer fetches any URL it is given without checking the host and returns the response body to the caller "
         "so a logged in user can read internal services that should never be reachable from the internet at all.")
CODE = "POST /import HTTP/1.1\nHost: app.example.com\nContent-Type: application/x-www-form-urlencoded\n\nurl=http://169.254.169.254/latest/meta-data/"
SENTINEL = "ZZ-UNIQUE-COMMENT-SENTENCE"
DATASET = f"## Summary\n{PROSE}\n\n## Steps to reproduce\n1. Log in as any user.\n2. Send this request.\n\n```\n{CODE}\n```\n"

CHROME_TOP = "# HackerOne\n\n[Hacktivity](/h) [Programs](/p) [Log in](/l)\n"
COMMENT = (f"[alice](/alice) posted a comment 3 days ago\n\nThanks for the report we confirmed the behaviour and a fix is being deployed {SENTINEL} "
           "to every region this week and we will update this thread when it is done.")
FOOTER = "[Privacy](/a) [Terms](/b) [Security](/c)"
PAGE_FULL = f"{CHROME_TOP}\n{DATASET}\n## Activity\n\n{COMMENT}\n\n{FOOTER}\n"


def test_blocks_are_split_on_blank_lines_but_never_inside_a_code_fence():
    blocks = pm.blocks("intro words here\n\n```\nline one\n\nline after a blank\n```\n\nlast paragraph")
    assert len(blocks) == 3 and "line after a blank" in blocks[1]["text"]


def test_headings_are_remembered_for_the_blocks_under_them():
    blocks = pm.blocks("## Summary\n\nfirst paragraph here\n\n## Activity\n\nsecond paragraph here")
    assert [b["heading"] for b in blocks if not b["text"].startswith("#")] == ["Summary", "Activity"]


def test_a_page_holding_the_whole_dataset_text_has_full_coverage_and_no_missing_groups():
    r = pm.analyse(DATASET, PAGE_FULL)
    assert r["coverage"] >= 0.95 and sum(r["missing_by_category"].values()) <= 2


def test_missing_code_is_attributed_to_code_not_to_prose():
    page_without_code = PAGE_FULL.replace(CODE, "(code block not rendered)")
    r = pm.analyse(DATASET, page_without_code)
    assert r["coverage"] < 0.95
    assert r["missing_by_category"]["code"] > 0 and r["missing_by_category"]["prose"] == 0


def test_missing_prose_is_attributed_to_prose():
    r = pm.analyse(DATASET, PAGE_FULL.replace(PROSE, "see the report"))
    assert r["missing_by_category"]["prose"] > 0


def test_blocks_are_classified_as_dataset_text_chrome_or_other():
    r = pm.analyse(DATASET, PAGE_FULL)
    words = r["words_by_kind"]
    assert words["dataset_text"] > 20            # the report body found on the page
    assert r["blocks_by_kind"]["chrome_like"] >= 2   # the nav links and the footer links
    assert words["other_text"] > 0               # the comment is neither the dataset text nor chrome


def test_a_dated_entry_with_real_prose_counts_as_a_discussion_entry_but_a_bare_label_does_not():
    r = pm.analyse(DATASET, PAGE_FULL)
    assert r["discussion_entries"] >= 1
    bare = pm.analyse(DATASET, PAGE_FULL.replace(COMMENT, "triaged"))
    assert bare["discussion_entries"] == 0


def test_a_page_without_dated_entries_reports_none():
    assert pm.analyse(DATASET, f"{CHROME_TOP}\n{DATASET}\n{FOOTER}\n")["discussion_entries"] == 0


def test_sections_list_each_heading_with_its_word_count():
    r = pm.analyse(DATASET, PAGE_FULL)
    names = [s["heading"] for s in r["sections"]]
    assert "Summary" in names and "Activity" in names
    assert all(s["words"] >= 0 for s in r["sections"])


def test_the_result_has_numbers_and_headings_but_no_body_or_comment_text():
    blob = json.dumps(pm.analyse(DATASET, PAGE_FULL))
    assert SENTINEL not in blob and "169.254.169.254" not in blob and "importer" not in blob


def test_an_empty_dataset_body_is_handled():
    r = pm.analyse("", PAGE_FULL)
    assert r["coverage"] is None and r["words_by_kind"]["other_text"] > 0


def _cache_with(tmp_path: Path, rid: int, markdown: str) -> Path:
    cache = tmp_path / "probe"
    cache.mkdir()
    (cache / f"{rid}.json").write_text(json.dumps({"url": f"https://hackerone.com/reports/{rid}", "status_code": 200, "markdown": markdown}),
                                       encoding="utf-8")
    return cache


def test_main_profiles_a_cached_page_prints_no_text_and_writes_a_summary(tmp_path):
    cache = _cache_with(tmp_path, 7, PAGE_FULL)
    printed: list[str] = []
    code = pm.main(["7", "--cache-dir", str(cache)], hf_loader=lambda ids: {7: DATASET}, out=printed.append)
    text = "\n".join(printed)
    assert code == 0 and "report 7" in text
    assert SENTINEL not in text
    saved = json.loads((cache / "pagemap.json").read_text(encoding="utf-8"))
    assert saved["reports"][0]["report_id"] == 7 and SENTINEL not in json.dumps(saved)


def test_main_says_so_when_a_page_was_never_fetched_and_changes_nothing(tmp_path):
    cache = tmp_path / "probe"
    cache.mkdir()
    printed: list[str] = []
    code = pm.main(["9", "--cache-dir", str(cache)], hf_loader=lambda ids: {}, out=printed.append)
    assert code == 1 and "no cached page" in "\n".join(printed).lower()
    assert [p.name for p in cache.iterdir()] == []


def test_main_refuses_a_non_numeric_report_id(tmp_path):
    printed: list[str] = []
    assert pm.main(["abc", "--cache-dir", str(tmp_path)], hf_loader=lambda ids: {}, out=printed.append) == 2


def test_the_tool_never_imports_a_network_client():
    src = Path(pm.__file__).read_text(encoding="utf-8")
    assert "httpx" not in src and "requests" not in src and "firecrawl import" not in src and "urlopen" not in src


# ------------------------------------------------------------------ which section do the "steps" and "activity" signals come from?

def test_numbered_lines_are_counted_per_section():
    page = (f"## Summary\n\n{PROSE}\n\n## Linear Agent Reasoning\n\n1. first thing the agent says here\n2. second thing the agent says here\n")
    secs = {s["heading"]: s for s in pm.analyse(DATASET, page)["sections"]}
    assert secs["Linear Agent Reasoning"]["numbered_lines"] == 2 and secs["Summary"]["numbered_lines"] == 0


def test_activity_phrases_are_counted_per_section():
    page = f"## Summary\n\n{PROSE}\n\n## Activity\n\nteam changed the status to Resolved and triaged the report.\n"
    secs = {s["heading"]: s for s in pm.analyse(DATASET, page)["sections"]}
    assert secs["Activity"]["activity_markers"] >= 2 and secs["Summary"]["activity_markers"] == 0


def test_main_prints_the_per_section_signal_counts(tmp_path):
    cache = _cache_with(tmp_path, 7, "## Notes\n\n1. one thing here now\n2. two things here now\n")
    printed: list[str] = []
    pm.main(["7", "--cache-dir", str(cache)], hf_loader=lambda ids: {7: DATASET}, out=printed.append)
    assert any("Notes" in line and "numbered 2" in line for line in printed)


def test_a_long_undated_paragraph_is_not_a_discussion_entry():
    """Machine-written or extra text with no date must not be mistaken for a comment thread."""
    undated = ("The exploit path looks valid because the importer accepts any URL and the response is returned in full "
               "which means an internal service can be read by a normal user without any special privileges at all.")
    r = pm.analyse(DATASET, f"{CHROME_TOP}\n{DATASET}\n## Exploit Agent Reasoning\n\n{undated}\n")
    assert r["words_by_kind"]["other_text"] >= 12            # the paragraph really is new text ...
    assert r["dated_blocks"] == 0 and r["discussion_entries"] == 0   # ... but nothing marks it as a dated entry
