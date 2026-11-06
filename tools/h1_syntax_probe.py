"""Diagnostic: which queries does the official HackerOne listing REALLY obey? (about 25 requests, listing endpoint only, structure only)

The coverage check showed that a day holding more reports than one page (about 50) cannot be listed by dates alone, and that the exact-
timestamp filter was "not accepted" without saying why. This probe tries, one request each, the ways that could get past a busy day, and
judges each by what comes back - not by whether the API answered:
  * exact timestamps in six spellings (with / without milliseconds, quoted, escaped colons, range, date + space + time),
  * upper date bounds (`<`, `<=`, `[A TO B]`, `>= AND <`) - checked newest-first, where an ignored bound is obvious,
  * extra filters that could split a busy day (substate, severity_rating, team / program),
  * other sort keys (submitted_at, votes, latest_disclosable_activity_at),
  * what page[number] really does at page sizes 10 / 25 / 50 / 100, checked against the ORDER of the early part of your manifest
    (2025 passed the coverage check, so it is the ground truth).
For every query it records the HTTP status, the number of items and a verdict: obeyed / ignored / rejected / cannot tell. It prints and saves
counts, dates, parameter names and report ids only - no titles, no report text.

Usage (from the project folder, via h1_syntax_probe.bat):
    h1_syntax_probe.bat --manifest data\\h1_discovery\\<time>\\manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from hrag.h1_api import H1Client, H1Error, report_id_of
from tools.h1_discover import REJECTED_STATUSES
from tools.h1_relations import ManifestError, default_client, load_manifest_rows

BASE = "disclosed:true"
PAGE_TESTS = ((10, 1), (10, 2), (10, 5), (25, 1), (25, 2), (50, 1), (50, 2), (100, 1), (100, 2), (100, 3))


def _attr(item: dict, name: str) -> Any:
    return (item.get("attributes") or {}).get(name)


def _stamp(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _handle(item: dict) -> str:
    return str(((((item.get("relationships") or {}).get("program") or {}).get("data") or {}).get("attributes") or {}).get("handle") or "")


class Prober:
    def __init__(self, client: H1Client, log: Callable[[str], None]):
        self.client, self.log, self.results, self.requests = client, log, [], 0

    def ask(self, name: str, query: str, size: int, sort: str | None, number: int | None = None) -> tuple[str, list[dict]]:
        """(status, items): status is 'ok' or 'rejected (HTTP n)'. Credential problems are raised."""
        params: dict[str, Any] = {"queryString": query, "page[size]": size}
        if sort:
            params["sort"] = sort
        if number:
            params["page[number]"] = number
        self.requests += 1
        try:
            body = self.client.get_json("/hacktivity", params)
        except H1Error as exc:
            if exc.status in REJECTED_STATUSES:
                return f"rejected (HTTP {exc.status})", []
            raise
        return "ok", [i for i in body.get("data") or [] if isinstance(i, dict)]

    def record(self, group: str, name: str, query: str, status: str, items: list, verdict: str, note: str = "", **extra: Any) -> None:
        rec = {"group": group, "name": name, "query": query, "status": status, "items": len(items), "verdict": verdict, "note": note, **extra}
        self.results.append(rec)
        self.log(f"  {group:<10} {name:<22} {status:<20} {len(items):>3} items  -> {verdict}{(' (' + note + ')') if note else ''}")


def pick_reference(rows: list[dict]) -> dict[str, Any] | None:
    """The LAST report of a busy day in the middle of the manifest (a day-level filter would return earlier same-day reports first, so a
    time that is honoured can be told from one that is ignored; later days must exist so an ignored upper bound shows). Needs a day with at
    least two reports."""
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        ts = str(r.get("disclosed_at") or "")
        if "T" in ts and r.get("report_id") is not None:
            by_day.setdefault(ts[:10], []).append(r)
    all_days = sorted(by_day)
    days = [d for d in all_days if len(by_day[d]) >= 2]
    if not days:
        return None
    cutoff = (datetime.strptime(all_days[-1], "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    early = [d for d in days if d <= cutoff] or days
    middle = all_days[len(all_days) // 2]
    day = max(early, key=lambda d: (len(by_day[d]), -abs((datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(middle, "%Y-%m-%d")).days)))
    last = sorted(by_day[day], key=lambda r: str(r["disclosed_at"]))[-1]
    return {"day": day, "timestamp": str(last["disclosed_at"]), "report_id": int(last["report_id"]), "reports_that_day": len(by_day[day])}


def ts_variants(ts: str) -> list[tuple[str, str]]:
    plain = ts.split(".")[0].rstrip("Z") + "Z" if "." in ts else ts
    return [("ts_ms", f"disclosed_at:>={ts}"), ("ts_no_ms", f"disclosed_at:>={plain}"), ("ts_quoted", f'disclosed_at:>="{ts}"'),
            ("ts_escaped", "disclosed_at:>=" + ts.replace(":", "\\:")), ("ts_range", f"disclosed_at:[{ts} TO *]"),
            ("ts_space", 'disclosed_at:>="' + ts.replace("T", " ").split(".")[0].rstrip("Z") + '"')]


def test_timestamps(p: Prober, ref: dict[str, Any]) -> None:
    T = _stamp(ref["timestamp"])
    for name, clause in ts_variants(ref["timestamp"]):
        query = f"{BASE} AND {clause}"
        status, items = p.ask(name, query, 5, "disclosed_at")
        if status != "ok":
            p.record("timestamp", name, query, status, items, "rejected")
        elif not items:
            p.record("timestamp", name, query, status, items, "cannot tell", "no items returned")
        else:
            first = _stamp(_attr(items[0], "disclosed_at"))
            if first is not None and T is not None and first >= T - timedelta(seconds=1):
                p.record("timestamp", name, query, status, items, "obeyed", f"first item {first.isoformat()} is at or after the time asked")
            elif first is not None and T is not None and first.date() == T.date():
                p.record("timestamp", name, query, status, items, "day level only", "same day but earlier: the time of day is ignored")
            else:
                p.record("timestamp", name, query, status, items, "ignored", "earlier days returned")


def test_upper_bounds(p: Prober, ref: dict[str, Any], rows: list[dict]) -> None:
    lo = ref["day"]
    hi = (datetime.strptime(lo, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")
    later_exists = any(str(r.get("disclosed_at") or "")[:10] > hi for r in rows)
    forms = [("upper_lt", f"{BASE} AND disclosed_at:<{hi}", False), ("upper_le", f"{BASE} AND disclosed_at:<={hi}", True),
             ("upper_range", f"{BASE} AND disclosed_at:[{lo} TO {hi}]", True), ("upper_ge_lt", f"{BASE} AND disclosed_at:>={lo} AND disclosed_at:<{hi}", False)]
    for name, query, inclusive in forms:
        status, items = p.ask(name, query, 5, "-disclosed_at")
        if status != "ok":
            p.record("upper", name, query, status, items, "rejected")
        elif not items:
            p.record("upper", name, query, status, items, "cannot tell", "no items returned")
        elif not later_exists:
            p.record("upper", name, query, status, items, "cannot tell", "the manifest has nothing after the bound")
        else:
            newest = str(_attr(items[0], "disclosed_at") or "")[:10]
            ok = newest <= hi if inclusive else newest < hi
            p.record("upper", name, query, status, items, "obeyed" if ok else "ignored", f"newest returned {newest}, bound {hi}")


def test_field_filters(p: Prober, ref: dict[str, Any]) -> None:
    status, base = p.ask("base", f"{BASE} AND disclosed_at:>={ref['day']}", 50, "disclosed_at")
    if status != "ok" or not base:
        p.record("filter", "base page", "", status, base, "cannot tell", "no base page to derive values from")
        return

    def choose(values: list[str]) -> str | None:
        counts = Counter(v for v in values if v)
        return next((v for v, n in counts.most_common() if n < len(values)), None)

    plans = [("substate", "substate", lambda i: str(_attr(i, "substate") or "")), ("severity_rating", "severity_rating", lambda i: str(_attr(i, "severity_rating") or "")),
             ("team", "team", _handle), ("program", "program", _handle)]
    for name, field, getter in plans:
        value = choose([getter(i) for i in base])
        if value is None:
            p.record("filter", name, "", "not run", [], "cannot tell", "no value that splits the base page")
            continue
        query = f"{BASE} AND disclosed_at:>={ref['day']} AND {field}:{value}"
        status, items = p.ask(name, query, 50, "disclosed_at")
        if status != "ok":
            p.record("filter", name, query, status, items, "rejected")
        elif items and all(getter(i) == value for i in items) and len(items) < len(base):
            p.record("filter", name, query, status, items, "obeyed", f"{len(items)} of {len(base)} base items match")
        elif items and len(items) == len(base):
            p.record("filter", name, query, status, items, "ignored", "same number of items as without the filter")
        else:
            p.record("filter", name, query, status, items, "ignored" if items else "cannot tell")


def test_sorts(p: Prober, ref: dict[str, Any]) -> None:
    query = f"{BASE} AND disclosed_at:>={ref['day']}"
    status0, baseline = p.ask("sort_base", query, 20, "disclosed_at")
    base_order = [report_id_of(i) for i in baseline]
    for key, attr, desc in (("submitted_at", "submitted_at", False), ("-votes", "votes", True), ("latest_disclosable_activity_at", "latest_disclosable_activity_at", False)):
        status, items = p.ask(f"sort {key}", query, 20, key)
        if status != "ok":
            p.record("sort", key, query, status, items, "rejected")
            continue
        vals = [_attr(i, attr) for i in items]
        vals = [v for v in vals if v is not None]
        mono = all(a >= b for a, b in zip(vals, vals[1:])) if desc else all(a <= b for a, b in zip(vals, vals[1:]))
        same_as_default = [report_id_of(i) for i in items] == base_order[:len(items)]
        p.record("sort", key, query, status, items, ("obeyed" if mono and not same_as_default else "cannot tell" if mono else "ignored"),
                 "same order as the disclosed_at sort" if mono and same_as_default else "")


def test_page_numbers(p: Prober, rows: list[dict]) -> dict[str, Any]:
    """Compares each page with the ORDER of the manifest's oldest part (ground truth). Page n of size S can start at
    (n-1) x (size the server RETURNED on page 1)   = consistent paging, or at
    (n-1) x S (the size ASKED)                     = it silently skips records whenever the server returns fewer than it was asked for."""
    ordered = sorted((r for r in rows if r.get("report_id") is not None), key=lambda r: (str(r.get("disclosed_at") or ""), int(r["report_id"])))
    if len(ordered) < 120:
        p.record("pages", "all", "", "not run", [], "cannot tell", "the manifest has fewer than 120 reports to compare with")
        return {"safe_page_size": None, "exact": {}, "returned_on_page_1": {}}
    truth = [int(r["report_id"]) for r in ordered]
    query = f"{BASE} AND disclosed_at:>={str(ordered[0]['disclosed_at'])[:10]}"

    def overlap(ids: list[int], start: int) -> float:
        return len(set(ids) & set(truth[start:start + len(ids)])) / len(ids) if ids else 0.0

    returned_on_page_1: dict[int, int] = {}
    exact: dict[str, str] = {}
    for size, number in PAGE_TESTS:
        status, items = p.ask(f"page {size}/{number}", query, size, "disclosed_at", number)
        ids = [rid for rid in (report_id_of(i) for i in items) if rid is not None]
        if number == 1:
            returned_on_page_1[size] = len(ids)
        l1 = returned_on_page_1.get(size) or size
        by_returned, by_asked = (number - 1) * l1, (number - 1) * size
        if not ids:
            verdict, note = "empty", "no items"
        elif overlap(ids, by_returned) >= 0.9:
            verdict, note = "exact", f"starts where {number - 1} page(s) of {l1} would end"
        elif by_asked != by_returned and overlap(ids, by_asked) >= 0.9:
            verdict, note = "SKIPS", f"starts {by_asked} records in (the size asked), not {by_returned} (the size returned): the records between are skipped"
        else:
            first_pos = next((i for i, t in enumerate(truth) if t == ids[0]), None)
            verdict, note = "WRONG POSITION", f"first item is #{first_pos + 1 if first_pos is not None else '?'} of the oldest-first order"
        exact[f"{size}/{number}"] = verdict
        p.record("pages", f"size {size} page {number}", query, status, items, verdict, note, returned=len(ids))
    safe = [s for s in sorted({s for s, _ in PAGE_TESTS})
            if all(exact.get(f"{s}/{n}") == "exact" for (ss, n) in PAGE_TESTS if ss == s and n > 1)]
    return {"safe_page_size": max(safe) if safe else None, "exact": exact, "returned_on_page_1": returned_on_page_1}


def run_probe(client: H1Client, rows: list[dict], log: Callable[[str], None] = print) -> dict[str, Any]:
    p = Prober(client, log)
    ref = pick_reference(rows)
    log("query syntax tests (one request each; verdict judged on the returned items):")
    paging: dict[str, Any] = {}
    if ref is None:
        p.record("setup", "reference report", "", "not run", [], "cannot tell", "no day with two reports to use as a reference")
    else:
        log(f"  reference: report {ref['report_id']} at {ref['timestamp']} (the last of {ref['reports_that_day']} reports that day)")
        test_timestamps(p, ref)
        test_upper_bounds(p, ref, rows)
        test_field_filters(p, ref)
        test_sorts(p, ref)
    paging = test_page_numbers(p, rows)
    by_group: dict[str, list[str]] = {}
    for r in p.results:
        if r["verdict"] in ("obeyed", "exact") and not (r["group"] == "pages" and r["name"].endswith("page 1")):
            by_group.setdefault(r["group"], []).append(r["name"])
    return {"reference": ref, "requests": p.requests, "results": p.results, "works": by_group, "page_numbers": paging}


def _explain(res: dict[str, Any], out: Callable[[str], None]) -> None:
    w = res["works"]
    out("\nwhat this means for busy days:")
    out(f"  exact-time filter spellings that work: {', '.join(w.get('timestamp', [])) or 'NONE'}")
    out(f"  upper date bounds that work:           {', '.join(w.get('upper', [])) or 'NONE'}")
    out(f"  extra filters that split a day:        {', '.join(w.get('filter', [])) or 'NONE'}")
    out(f"  other sort keys that work:             {', '.join(w.get('sort', [])) or 'NONE'}")
    safe = res["page_numbers"].get("safe_page_size")
    out(f"  page numbers are exact at page size:   {safe if safe else 'NO tested size (page numbers are not trustworthy)'}")
    if w.get("timestamp"):
        out("  -> exact time works: the date marker can move to the last TIMESTAMP seen, so a busy day can be walked without page numbers.")
    elif w.get("filter"):
        out("  -> extra filters work: a busy day can be split (for example by substate x severity) until every part fits one page.")
    elif safe:
        out(f"  -> page numbers are exact at size {safe}: busy days can be walked with that size only.")
    else:
        out("  -> nothing found that gets past a busy day: such days can only be reported as partial, or sourced from elsewhere.")


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[], H1Client] | None = None, out: Callable[[str], None] = print,
         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    ap = argparse.ArgumentParser(prog="h1_syntax_probe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="a manifest.json (or its folder) written by h1_discover")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--out-dir", help="output folder (default: <data folder>/h1_syntax/<UTC time>)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)
    try:
        rows = load_manifest_rows(args.manifest)
    except ManifestError as exc:
        out(f"error: {exc}")
        return 2
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from hrag.config import load_config  # noqa: PLC0415
        out_dir = load_config(args.config).db_path.parent / "h1_syntax" / now().strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / "syntax.json"
    if target.exists():
        out(f"error: {target} already exists - this tool never overwrites. Choose another --out-dir.")
        return 2
    try:
        client = (client_factory or default_client)()
    except Exception as exc:  # noqa: BLE001 - plain message, never holds a key
        out(f"error: {exc}")
        return 2
    if client_factory is None:
        client.min_interval = max(client.min_interval, 0.6)
    try:
        result = run_probe(client, rows, out)
    except H1Error as exc:
        out(f"stopped: {exc}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checked_at": now().isoformat(timespec="seconds"), **result,
                                  "note": "statuses, counts, dates, parameter names and report ids only"}, indent=2), encoding="utf-8")
    _explain(result, out)
    out(f"saved: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
