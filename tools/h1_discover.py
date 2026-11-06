"""Diagnostic: discover disclosed HackerOne reports since a date through the OFFICIAL Hacker API and count them. METADATA ONLY.

What it does: pages through GET /hacktivity (title, url, disclosure date, severity, CWE, program, ...) and writes a manifest (JSON + CSV)
plus counts by month, severity and CWE. It never reads a report body, never opens either SQLite database, and never touches hackerone.com
pages - only api.hackerone.com, through the project's own H1Client (spacing between requests, 429 waits, token never in errors, refusal to
send the token to any other host).

The date-filter and sort syntax of the API are NOT verified, so the tool checks what comes back before trusting it:
  1. server_filter_sorted      query `disclosed:true AND disclosed_at:>=DATE`, sort=disclosed_at (oldest first); restarts from the last date
                                seen when a segment hits its page limit, so deep page numbers are never needed;
  2. server_filter_unsorted    the same filter without a sort (only if sorting is ignored);
  3. newest_first_client_stop  no date filter, sort=-disclosed_at, stops at the first report older than DATE (only if the filter is rejected).
A missing "next page" link is NEVER taken as the end of the list (the real API caps a page at 50 and sent no usable link): the list ends
only when the API returns an EMPTY page, or when a restart from the last date seen finds nothing new. Pages are followed by the link when
there is one, else by explicit page[number]; if neither moves forward, the date watermark does.
A mode is accepted only if a small test page obeys it; while running, every item is re-checked and the run stops as INCOMPLETE if the
API stops obeying. Hard limits: --max-items, --max-requests, --segment-pages.

Usage (from the project folder, via h1_discover.bat):   h1_discover.bat --max-items 300      (small trial first)
                                                         h1_discover.bat                      (everything since 2025-01-01)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from hrag.h1_api import H1Client, H1Error, report_id_of
from tools.h1_relations import default_client

DEFAULT_SINCE = "2025-01-01"
REPORT_URL = "https://hackerone.com/reports/"
MIN_INTERVAL_FLOOR = 0.25            # the documented read limit is 600 per minute; this keeps us at a fraction of it
PREFLIGHT_SIZE = 10
GAP_WALK_PAGES = 30                  # pages walked, for COLLECTION only, on a busy day; the date marker never follows them
CSV_COLUMNS = ("report_id", "title", "url", "disclosed_at", "severity", "cwe", "cwe_id", "program", "substate", "votes", "bounty_awarded",
               "cve_ids", "submitted_at", "summary_chars")
_CWE_ID = re.compile(r"\bCWE-\d+\b", re.I)
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REJECTED_STATUSES = (400, 404, 422)           # the API understood us but not the query: try the next mode


class _Stop(Exception):
    """Ends the run on purpose; the text becomes `stopped_because`."""


@dataclass
class Result:
    rows: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    stopped_because: str = ""
    mode: str = ""
    requests: int = 0
    segments: int = 0
    pages: int = 0
    requested_page_size: int = 0
    largest_page: int = 0                  # the biggest page the API really returned (the real API gave 50 when 100 were asked for)
    pages_by_next_link: int = 0
    pages_by_number: int = 0
    paging: str = "keyset"
    tail_checked: int = 0                   # newest reports compared with the list at the end ...
    tail_missing: int = 0                   # ... and how many of them the oldest-first walk had NOT returned
    page_log: list[dict[str, Any]] = field(default_factory=list)
    gap_days: list[dict[str, Any]] = field(default_factory=list)       # busy days that hold more reports than one page can show


# ---------------------------------------------------------------- small helpers

def parse_day(value: str) -> str:
    if not _DAY.match(value or ""):
        raise ValueError(f"'{value}' is not a date; use YYYY-MM-DD, for example 2025-01-01")
    datetime.strptime(value, "%Y-%m-%d")
    return value


def _stamp(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def row_of(item: dict) -> dict[str, Any] | None:
    """One manifest row from one listing item. No reporter name, no report text."""
    rid = report_id_of(item)
    if rid is None:
        return None
    a = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
    prog = ((((item.get("relationships") or {}).get("program") or {}).get("data") or {}).get("attributes") or {}).get("handle")
    cwe = str(a.get("cwe") or "")
    m = _CWE_ID.search(cwe)
    gen = ((((item.get("relationships") or {}).get("report_generated_content") or {}).get("data") or {}).get("attributes") or {})
    summary_chars = len(str(gen.get("hacktivity_summary") or ""))                  # the LENGTH only - the text is never kept
    return {"report_id": rid, "title": str(a.get("title") or ""), "url": str(a.get("url") or f"{REPORT_URL}{rid}"),
            "disclosed_at": str(a.get("disclosed_at") or ""), "severity": str(a.get("severity_rating") or ""), "cwe": cwe,
            "cwe_id": m.group(0).upper() if m else "", "program": str(prog or ""), "substate": str(a.get("substate") or ""),
            "votes": a.get("votes") if isinstance(a.get("votes"), int) else "", "bounty_awarded": a.get("total_awarded_amount") or "",
            "cve_ids": ";".join(str(c) for c in a.get("cve_ids") or []), "submitted_at": str(a.get("submitted_at") or ""), "summary_chars": summary_chars}


def _next_day(day: str) -> str:
    from datetime import timedelta  # noqa: PLC0415
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _add_item(item: dict, rows: dict[int, dict[str, Any]], since: str, until: str | None) -> bool:
    day = _day(item)
    if day and (day < since or (until and day > until)):
        return False
    row = row_of(item)
    if row is None or row["report_id"] in rows:
        return False
    rows[row["report_id"]] = row
    return True


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_month: Counter[str] = Counter()
    by_sev: Counter[str] = Counter()
    by_cwe: Counter[str] = Counter()
    by_ms: dict[str, Counter[str]] = {}
    for r in rows:
        month = r["disclosed_at"][:7] if re.match(r"^\d{4}-\d{2}", r["disclosed_at"]) else "(unknown)"
        sev = r["severity"] or "(none)"
        by_month[month] += 1
        by_sev[sev] += 1
        by_cwe[r["cwe"] or "(none)"] += 1
        by_ms.setdefault(month, Counter())[sev] += 1
    return {"total": len(rows), "with_generated_summary": sum(1 for r in rows if r.get("summary_chars")), "by_month": dict(sorted(by_month.items())), "by_severity": dict(by_sev.most_common()),
            "by_cwe": dict(by_cwe.most_common()), "by_month_severity": {m: dict(c.most_common()) for m, c in sorted(by_ms.items())}}


# ---------------------------------------------------------------- collection

class _Runner:
    def __init__(self, client: H1Client, page_size: int, max_requests: int, log: Callable[[str], None]):
        self.client, self.page_size, self.max_requests, self.log = client, page_size, max_requests, log
        self.requests = 0
        self.largest_page = 0
        self.by_next = 0
        self.by_number = 0
        self.via = "first"

    def get(self, url: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if self.requests >= self.max_requests:
            raise _Stop(f"reached the --max-requests limit ({self.max_requests} requests)")
        self.requests += 1
        return self.client.get_json(url, params)

    def pages(self, query: str, sort: str | None, size: int):
        """Yields page after page, forever: by the `links.next` URL when the API sends one, else by explicit page[number].
        The caller decides when a segment is over (empty page, nothing new, page limit). A repeated next link is refused. A deeper
        page the API rejects raises _PageRejected so the caller can fall back to the date watermark."""
        base: dict[str, Any] = {"queryString": query, "page[size]": size}
        if sort:
            base["sort"] = sort
        url, params, number, seen = "/hacktivity", dict(base), 1, set()
        self.via = "first"
        while True:
            try:
                body = self.get(url, params)
            except H1Error as exc:
                if number > 1 and exc.status in REJECTED_STATUSES:
                    raise _PageRejected(f"the API rejected page {number} (HTTP {exc.status})") from exc
                raise
            size_got = len(body.get("data") or [])
            self.largest_page = max(self.largest_page, size_got)
            yield body
            nxt = (body.get("links") or {}).get("next")
            number += 1
            if nxt:
                if nxt in seen:
                    raise _Stop("the API sent the same next-page link twice")
                seen.add(nxt)
                url, params = nxt, None
                self.by_next += 1
                self.via = "next_link"
            else:
                url, params = "/hacktivity", {**base, "page[number]": number}
                self.by_number += 1
                self.via = "page_number"


class _PageRejected(Exception):
    pass


MODES = (("server_filter_sorted", True, "disclosed_at"), ("server_filter_unsorted", True, None), ("newest_first_client_stop", False, "-disclosed_at"))


def _query(use_filter: bool, since: str) -> str:
    return f"disclosed:true AND disclosed_at:>={since}" if use_filter else "disclosed:true"


def _day(item: dict) -> str:
    return str((item.get("attributes") or {}).get("disclosed_at") or "")[:10]


def _obeys(items: list[dict], use_filter: bool, sort: str | None, since: str) -> bool:
    if not items:
        return False
    if use_filter and any(_day(i) and _day(i) < since for i in items):
        return False
    stamps = [s for s in (_stamp((i.get("attributes") or {}).get("disclosed_at")) for i in items) if s]
    if sort and len(stamps) > 1:
        pairs = list(zip(stamps, stamps[1:]))
        return all(a <= b for a, b in pairs) if not sort.startswith("-") else all(a >= b for a, b in pairs)
    return True


def _choose_mode(run: _Runner, since: str) -> tuple[str, bool, str | None] | None:
    for name, use_filter, sort in MODES:
        try:
            body = run.get("/hacktivity", {"queryString": _query(use_filter, since), "page[size]": PREFLIGHT_SIZE,
                                           **({"sort": sort} if sort else {})})
        except H1Error as exc:
            if exc.status in REJECTED_STATUSES:
                run.log(f"mode {name}: the API rejected this query (HTTP {exc.status}); trying the next way")
                continue
            raise
        items = body.get("data") or []
        if _obeys(items, use_filter, sort, since):
            run.log(f"mode {name}: accepted ({len(items)} test items obey the date filter and order)")
            return name, use_filter, sort
        if name == "newest_first_client_stop" and items and all(_day(i) and _day(i) < since for i in items):
            return name, use_filter, sort                    # valid and simply nothing newer than the date
        run.log(f"mode {name}: not accepted (the test page ignored the date filter or the sort order)")
    return None


def _gather_busy_day(run: "_Runner", res: Result, use_filter: bool, sort: str | None, since: str, until: str | None,
                     rows: dict[int, dict[str, Any]], day: str, max_items: int) -> None:
    """A day holds more reports than one page shows. Walk page numbers FROM that day purely to COLLECT what they return (page numbers have
    been seen to return the wrong records, so nothing found here moves the date marker). Stops after a repeat, an empty page, three pages
    with nothing new, or GAP_WALK_PAGES pages."""
    prev_ids, idle = None, 0
    try:
        for n, body in enumerate(run.pages(_query(use_filter, day), sort, run.page_size), start=1):
            items = body.get("data") or []
            res.pages += 1
            ids = [report_id_of(i) for i in items]
            added = sum(1 for item in items if _add_item(item, rows, since, until))
            days = sorted(d for d in (_day(i) for i in items) if d)
            res.page_log.append({"n": res.pages, "segment": f"busy:{day}", "via": run.via, "requested_size": run.page_size, "items": len(items),
                                 "new": added, "first_day": days[0] if days else "", "last_day": days[-1] if days else ""})
            if not items or (prev_ids is not None and ids == prev_ids) or n >= GAP_WALK_PAGES:
                break
            prev_ids = ids
            idle = 0 if added else idle + 1
            if idle >= 3 or len(rows) >= max_items:
                break
    except _PageRejected:
        pass


def _busy_message(res: Result) -> str:
    days = ", ".join(g["day"] for g in res.gap_days[:10]) + (" ..." if len(res.gap_days) > 10 else "")
    return (f"{len(res.gap_days)} busy day(s) hold more reports than one page of this API shows, so those days may be partial: {days} "
            f"(see gap_days in the manifest)")


def _tail_check(run: "_Runner", since: str, until: str | None, rows: dict[int, dict[str, Any]]) -> tuple[int, list[str]] | None:
    """An independent look at the END of the list from the other side: the newest reports, newest first. Returns (how many were checked,
    the disclosure days of those the oldest-first walk had not returned), or None if the API refuses newest-first sorting."""
    try:
        body = run.get("/hacktivity", {"queryString": _query(True, since), "page[size]": min(run.page_size, 50), "sort": "-disclosed_at"})
    except H1Error as exc:
        if exc.status in REJECTED_STATUSES:
            return None
        raise
    checked, missing = 0, []
    for item in body.get("data") or []:
        day, rid = _day(item), report_id_of(item)
        if rid is None or (day and (day < since or (until and day > until))):
            continue
        checked += 1
        if rid not in rows:
            missing.append(day)
    return checked, missing


def collect(client: H1Client, *, since: str, until: str | None = None, page_size: int = 100, segment_pages: int = 100,
            max_items: int = 50_000, max_requests: int = 500, paging: str = "keyset", log: Callable[[str], None] = print) -> Result:
    """paging="keyset" (default): every request is the FIRST page of "reports since DAY, oldest first"; the day then moves to the newest day
    seen. It never relies on page numbers or next links. paging="pages": walks page by page (page numbers / next links) up to
    `segment_pages`, then resumes from the last day. Either way the list ends only on an empty page or a restart that finds nothing new,
    and is then cross-checked newest-first."""
    since = parse_day(since)
    until = parse_day(until) if until else None
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if paging not in ("keyset", "pages"):
        raise ValueError("paging must be 'keyset' or 'pages'")
    res = Result(paging=paging)
    run = _Runner(client, page_size, max_requests, log)
    rows: dict[int, dict[str, Any]] = {}
    try:
        try:
            chosen = _choose_mode(run, since)
        except H1Error as exc:
            raise _Stop(str(exc)) from exc
        if chosen is None:
            raise _Stop("could not find a way to ask the API for reports since that date that it obeys (filter and sort were both ignored "
                        "or rejected); nothing was collected")
        res.mode, use_filter, sort = chosen
        start, newest_first = since, bool(sort and sort.startswith("-"))
        resumable = sort == "disclosed_at"                                   # only an oldest-first listing can be resumed from a date
        walk, tail_retries = paging == "pages", 0
        while True:
            res.segments += 1
            limit = segment_pages if (walk or not resumable) else 1
            query, new_in_segment, pages, last_day, prev = _query(use_filter, start), 0, 0, None, None
            ended, max_day, last_len, prev_ids = "", "", 0, None
            try:
                for body in run.pages(query, sort, page_size):
                    pages += 1
                    res.pages += 1
                    items = body.get("data") or []
                    last_len = len(items)
                    ids = [report_id_of(i) for i in items]
                    days = sorted(d for d in (_day(i) for i in items) if d)
                    entry = {"n": res.pages, "segment": res.segments, "via": run.via, "requested_size": page_size, "items": len(items), "new": 0,
                             "first_day": days[0] if days else "", "last_day": days[-1] if days else ""}
                    res.page_log.append(entry)
                    if not items:
                        log(f"  page {res.pages} ({run.via}): empty")
                        ended = "empty"                                        # the one reliable end-of-list signal
                        break
                    if prev_ids is not None and ids == prev_ids:
                        entry["repeat"] = True
                        log(f"  page {res.pages} ({run.via}): {len(items)} items, exactly the previous page again (the API repeated a page)")
                        ended = "repeat"
                        break
                    prev_ids = ids
                    added = 0
                    for item in items:
                        day, stamp = _day(item), _stamp((item.get("attributes") or {}).get("disclosed_at"))
                        if stamp and prev and sort and ((newest_first and stamp > prev) or (not newest_first and stamp < prev)):
                            raise _Stop("the API stopped returning reports in the requested order, so the list cannot be trusted")
                        if stamp:
                            prev = stamp
                        if day and day < since:
                            if use_filter:
                                raise _Stop("the API returned a report older than the requested date although a date filter was sent")
                            res.complete = True                               # newest-first: everything after this is older
                            raise _Stop("")
                        if day and until and day > until:
                            if sort == "disclosed_at":
                                res.complete = True                           # oldest-first: everything after this is newer
                                raise _Stop("")
                            continue
                        if day:
                            max_day = max(max_day, day)
                        row = row_of(item)
                        if row is None or row["report_id"] in rows:
                            continue
                        rows[row["report_id"]] = row
                        added += 1
                        new_in_segment += 1
                        last_day = day or last_day
                        if len(rows) >= max_items:
                            entry["new"] = added
                            raise _Stop(f"reached the --max-items limit ({max_items} reports)")
                    entry["new"] = added
                    log(f"  page {res.pages} ({run.via}): {len(items)} items, +{added} new, {len(rows)} reports so far, newest date seen {max_day or 'n/a'}")
                    if pages >= limit:
                        ended = "limit"
                        break
            except _PageRejected as rejected:
                ended = "rejected"
                log(f"  {rejected}; continuing from the last date seen instead")
            if not resumable:                                                  # newest-first / unsorted: no date to resume from
                if ended == "empty":
                    res.complete = True
                    break
                raise _Stop({"limit": "the page limit per segment was reached and this mode cannot be resumed",
                             "repeat": "the API repeated a page instead of moving on (the same page was repeated), and this mode cannot be resumed from a date",
                             "rejected": "the API rejected a deeper page and this mode cannot be resumed from a date"}.get(ended, "stopped early"))
            if new_in_segment:
                if not max_day:
                    raise _Stop("no progress: the reports found carry no disclosure date to resume from")
                start, walk = max_day, paging == "pages"
                continue
            # nothing new from this start day. A SHORT page of known reports (or an empty page) is the end; a FULL page of known reports
            # means more may lie beyond it, so move on to the newest day it shows.
            expected_full = max(1, min(page_size, run.largest_page))
            if not (ended == "empty" or last_len < expected_full):
                if max_day and max_day > start:
                    start, walk = max_day, paging == "pages"
                    log(f"  a full page of known reports; moving on from {start}")
                    continue
                if paging == "keyset":
                    before = len(rows)
                    _gather_busy_day(run, res, use_filter, sort, since, until, rows, start, max_items)
                    if len(rows) >= max_items:
                        raise _Stop(f"reached the --max-items limit ({max_items} reports)")
                    on_day = sum(1 for r in rows.values() if r["disclosed_at"][:10] == start)
                    res.gap_days.append({"day": start, "collected_on_day": on_day, "page_size_seen": expected_full,
                                         "found_by_page_numbers": len(rows) - before})
                    log(f"  BUSY DAY {start}: a whole page holds only reports of that day; {on_day} collected, more may exist; "
                        f"continuing from the next day")
                    nxt = _next_day(start)
                    if until and nxt > until:
                        res.stopped_because = _busy_message(res)
                        break
                    start, walk = nxt, False
                    continue
                if not walk:
                    walk = True
                    log(f"  a whole page holds only known reports of {start}; walking page numbers for that day")
                    continue
                raise _Stop(f"no progress: at least {last_len} reports share the day {start}, which is more than one page of this API holds, "
                            f"and page numbers do not get past them")
            # the oldest-first walk says it is finished: ask from the other side (newest first) before believing it
            tail = _tail_check(run, since, until, rows)
            if tail is None:
                log("  tail check skipped: the API refused newest-first sorting")
                res.complete = not res.gap_days
                res.stopped_because = _busy_message(res) if res.gap_days else ""
                break
            res.tail_checked += tail[0]
            res.tail_missing = len(tail[1])
            if not tail[1]:
                log(f"  tail check passed: the {tail[0]} newest reports are all in the list")
                res.complete = not res.gap_days
                res.stopped_because = _busy_message(res) if res.gap_days else ""
                break
            tail_retries += 1
            if tail_retries > 2:
                raise _Stop(f"tail check: {len(tail[1])} of the {tail[0]} newest reports are missing from the oldest-first list, which the "
                            f"listing did not return even when asked again from their date")
            start = min(d for d in tail[1] if d) if any(tail[1]) else start
            walk = paging == "pages"
            log(f"  tail check found {len(tail[1])} newest reports not in the list; asking again from {start}")
    except _Stop as stop:
        res.stopped_because = str(stop)
        if stop.args and not stop.args[0]:
            res.complete = True
    except H1Error as exc:
        res.stopped_because = str(exc)
    res.rows = list(rows.values())
    res.requests = run.requests
    res.requested_page_size, res.largest_page = page_size, run.largest_page
    res.pages_by_next_link, res.pages_by_number = run.by_next, run.by_number
    return res


# ---------------------------------------------------------------- files

def _safe_cell(value: Any) -> Any:
    """A spreadsheet runs a cell that starts with = + - @ as a formula; a leading apostrophe keeps it as text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _write_text(path: Path, text: str, newline: str | None = None, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding, newline=newline) as fh:
        fh.write(text)
    os.replace(tmp, path)


def _csv_text(header: Sequence[str], rows: list[dict[str, Any]]) -> str:
    import io  # noqa: PLC0415
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(header), extrasaction="ignore", lineterminator="\r\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _safe_cell(r.get(k, "")) for k in header})
    return buf.getvalue()


def write_outputs(out_dir: Path, res: Result, since: str, until: str | None, counts: dict[str, Any], audit: dict[str, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "since": since, "until": until, "mode": res.mode,
                "complete": res.complete, "stopped_because": res.stopped_because, "requests": res.requests, "count": len(res.rows),
                "pagination": {"requested_page_size": res.requested_page_size, "largest_page_returned": res.largest_page, "pages": res.pages,
                               "segments": res.segments, "pages_by_next_link": res.pages_by_next_link, "pages_by_page_number": res.pages_by_number,
                               "paging": res.paging, "tail_check": {"checked": res.tail_checked, "missing": res.tail_missing},
                               "page_log": res.page_log, "gap_days": res.gap_days},
                "note": "metadata only (no report text, no reporter names)", **({"audit": audit} if audit else {}), "reports": res.rows}
    _write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    _write_text(out_dir / "manifest.csv", "﻿" + _csv_text(CSV_COLUMNS, res.rows), newline="")
    _write_text(out_dir / "counts.json", json.dumps(counts, indent=2, ensure_ascii=False))
    flat = [{"dimension": dim, "key": key, "count": n} for dim, src in (("month", counts["by_month"]), ("severity", counts["by_severity"]),
                                                                       ("cwe", counts["by_cwe"])) for key, n in src.items()]
    flat += [{"dimension": "month_severity", "key": f"{m}/{s}", "count": n} for m, d in counts["by_month_severity"].items() for s, n in d.items()]
    _write_text(out_dir / "counts.csv", "﻿" + _csv_text(("dimension", "key", "count"), flat), newline="")


def _print_counts(counts: dict[str, Any], out: Callable[[str], None], top_cwe: int = 25) -> None:
    out(f"\nreports found: {counts['total']}")
    out("by month:")
    for m, n in counts["by_month"].items():
        out(f"  {m}  {n:>6}   " + ", ".join(f"{s} {c}" for s, c in counts["by_month_severity"][m].items()))
    out("by severity:")
    for s, n in counts["by_severity"].items():
        out(f"  {s:<12}{n:>6}")
    out(f"by CWE (top {top_cwe} of {len(counts['by_cwe'])}):")
    for c, n in list(counts["by_cwe"].items())[:top_cwe]:
        out(f"  {n:>6}  {c}")


# ---------------------------------------------------------------- main

def _data_dir(config: str | None) -> Path:
    from hrag.config import load_config  # noqa: PLC0415
    return load_config(config).db_path.parent


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[], H1Client] | None = None, out: Callable[[str], None] = print,
         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    ap = argparse.ArgumentParser(prog="h1_discover", description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=DEFAULT_SINCE, help="first disclosure day, YYYY-MM-DD (default 2025-01-01)")
    ap.add_argument("--until", help="last disclosure day, YYYY-MM-DD (default: no upper limit)")
    ap.add_argument("--page-size", type=int, default=100, help="items per request, 1-100 (default 100)")
    ap.add_argument("--segment-pages", type=int, default=100, help="pages per segment before resuming from the last date seen (default 100)")
    ap.add_argument("--max-items", type=int, default=50_000, help="stop after this many reports (default 50000); use 300 for a first trial")
    ap.add_argument("--max-requests", type=int, default=500, help="hard limit on API requests in this run (default 500)")
    ap.add_argument("--paging", choices=("keyset", "pages"), default="keyset",
                    help="keyset (default): every request is the first page 'since DAY', no page numbers; pages: walk page numbers - a fallback that can silently skip reports if later pages are inconsistent, so audit its result with h1_coverage_check")
    ap.add_argument("--no-audit", action="store_true", help="skip the final month-by-month audit (about 50 extra requests); not recommended")
    ap.add_argument("--min-interval", type=float, default=0.6, help=f"seconds between requests (default 0.6, minimum {MIN_INTERVAL_FLOOR})")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--out-dir", help="output folder (default: <data folder>/h1_discovery/<UTC time>)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)
    try:
        since = parse_day(args.since)
        until = parse_day(args.until) if args.until else None
    except ValueError as exc:
        out(f"error: {exc}")
        return 2
    problems = []
    if since > now().strftime("%Y-%m-%d"):
        problems.append("--since is in the future")
    if until and until < since:
        problems.append("--until is before --since")
    if not 1 <= args.page_size <= 100:
        problems.append("--page-size must be between 1 and 100")
    if args.min_interval < MIN_INTERVAL_FLOOR:
        problems.append(f"--min-interval must be at least {MIN_INTERVAL_FLOOR} seconds (the API allows 600 reads a minute)")
    for name in ("segment_pages", "max_items", "max_requests"):
        if getattr(args, name) < 1:
            problems.append(f"--{name.replace('_', '-')} must be at least 1")
    if problems:
        out("error: " + "; ".join(problems))
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else _data_dir(args.config) / "h1_discovery" / now().strftime("%Y%m%dT%H%M%SZ")
    existing = [p for p in ("manifest.json", "manifest.csv", "counts.json", "counts.csv") if (out_dir / p).exists()]
    if existing:
        out(f"error: {out_dir} already holds {', '.join(existing)} - this tool never overwrites. Choose another --out-dir.")
        return 2
    try:
        client = (client_factory or default_client)()
    except Exception as exc:  # noqa: BLE001 - MissingKeyError and friends: print the plain message, which never holds a key
        out(f"error: {exc}")
        return 2
    client.min_interval = args.min_interval                # already checked against the floor above

    out(f"discovering disclosed reports since {since}" + (f" until {until}" if until else "") + f"  (metadata only; output folder {out_dir})")
    res = collect(client, since=since, until=until, page_size=args.page_size, segment_pages=args.segment_pages, max_items=args.max_items,
                  max_requests=args.max_requests, paging=args.paging, log=out)
    audit_result: dict[str, Any] | None = None
    if args.no_audit:
        out("final audit skipped (--no-audit): completeness is NOT independently checked")
    elif until or not (res.complete or res.gap_days) or not res.rows:
        out("final audit skipped: it only runs after a full, unbounded walk (no --until, not stopped early)")
    else:
        from tools.h1_coverage_check import audit  # noqa: PLC0415 - imported here: that module imports this one
        out("final audit: asking the API month by month whether anything is missing ...")
        try:
            audit_result = audit(client, res.rows, since, now(), 2, out)
        except H1Error as exc:
            res.complete = False
            res.stopped_because = (res.stopped_because + "; " if res.stopped_because else "") + f"the final audit could not run ({exc})"
        else:
            bad_months = audit_result["verdict"]["problem_months"]
            if bad_months or audit_result["tail"]["missing"]:
                res.complete = False
                res.stopped_because = (res.stopped_because + "; " if res.stopped_because else "") + (
                    f"the coverage audit found gaps in: {', '.join(bad_months) or 'the newest reports'} "
                    f"({audit_result['verdict']['missing_total']} reports the API listed are missing)")
    counts = count_rows(res.rows)
    write_outputs(out_dir, res, since, until, counts, audit_result)
    for g in res.gap_days:
        out(f"BUSY DAY {g['day']}: {g['collected_on_day']} reports collected on that day; more may exist (one page of this API shows {g['page_size_seen']})")
    _print_counts(counts, out)
    if res.largest_page and res.largest_page < res.requested_page_size:
        out(f"note: asked for {res.requested_page_size} reports per page, the API never returned more than {res.largest_page}; "
            f"{res.pages_by_next_link} pages followed a next link, {res.pages_by_number} used page numbers.")
    out(f"tail check (newest reports compared with the list): checked {res.tail_checked}, missing {res.tail_missing}")
    out(f"\nmode: {res.mode or 'none'}; paging: {res.paging}; requests used: {res.requests}; pages: {res.pages}; rate limit remaining reported by the API: "
        f"{client.rate_remaining or 'n/a'}")
    out(f"saved: {out_dir} (manifest.json, manifest.csv, counts.json, counts.csv)")
    if not res.complete:
        out(f"INCOMPLETE: {res.stopped_because or 'stopped early'}. What was found is saved, but the counts are partial.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
