"""Audit: is a discovery manifest COMPLETE? Asks the API directly, month by month, and compares with the manifest. Listing endpoint only.

For every month since --since it asks the API "what are the first reports on or after day 1 (and day 16) of this month, oldest first?" and checks
whether those reports are in the manifest. From that it can tell
  * a month that is genuinely empty (the API's first report on or after the 1st lies in a LATER month),
  * a month the walk MISSED entirely (the API has reports in it, the manifest has none),
  * a month that is INCOMPLETE (some of the API's first reports of the month are not in the manifest),
and it compares the 50 newest reports (newest first) with the manifest. It also tests whether the API accepts an EXACT TIMESTAMP in the date
filter (`>=` and `>`), which would allow exact, page-number-free paging.
It samples: it can prove a gap, it cannot prove the absence of every gap. The report prints counts, dates and report ids only.

Usage (from the project folder, via h1_coverage_check.bat):
    h1_coverage_check.bat --manifest data\\h1_discovery\\<time>\\manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from hrag.h1_api import H1Client, H1Error, report_id_of
from tools.h1_discover import DEFAULT_SINCE, REJECTED_STATUSES, parse_day
from tools.h1_relations import ManifestError, default_client, load_manifest_rows

MAX_MONTHS = 36
PROBE_SIZE = 10
TAIL_SIZE = 50
MAX_LISTED_MISSING = 50


def _day(item: dict) -> str:
    return str((item.get("attributes") or {}).get("disclosed_at") or "")[:10]


def months_since(since: str, now: datetime) -> list[str]:
    y, m = int(since[:4]), int(since[5:7])
    out = []
    while (y, m) <= (now.year, now.month) and len(out) < MAX_MONTHS:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _ask(client: H1Client, query: str, size: int, sort: str) -> list[dict] | None:
    """The items of one listing page, or None when the API rejects the query (HTTP 400/404/422). Credentials trouble is raised."""
    try:
        body = client.get_json("/hacktivity", {"queryString": query, "page[size]": size, "sort": sort})
    except H1Error as exc:
        if exc.status in REJECTED_STATUSES:
            return None
        raise
    return [i for i in body.get("data") or [] if isinstance(i, dict)]


def test_timestamp_syntax(client: H1Client, rows: list[dict]) -> dict[str, Any]:
    """Does the filter understand an exact timestamp? `>=T` must return the report with disclosure time T first; `>T` must skip it."""
    stamped = sorted((r for r in rows if "T" in str(r.get("disclosed_at") or "")), key=lambda r: str(r["disclosed_at"]))
    if not stamped:
        return {"ge_works": None, "gt_works": None, "reference": None}
    ref = stamped[len(stamped) // 2]
    ts, rid = str(ref["disclosed_at"]), int(ref["report_id"])
    ge = _ask(client, f"disclosed:true AND disclosed_at:>={ts}", 3, "disclosed_at")
    gt = _ask(client, f"disclosed:true AND disclosed_at:>{ts}", 3, "disclosed_at")
    return {"reference": ts, "reference_report": rid,
            "ge_works": bool(ge) and report_id_of(ge[0]) == rid,
            "gt_works": bool(gt) and report_id_of(gt[0]) != rid and str((gt[0].get("attributes") or {}).get("disclosed_at") or "") > ts}


def audit(client: H1Client, rows: list[dict], since: str, now: datetime, probes_per_month: int = 2,
          log: Callable[[str], None] = print) -> dict[str, Any]:
    known = {int(r["report_id"]) for r in rows if r.get("report_id") is not None}
    in_manifest = Counter(str(r.get("disclosed_at") or "")[:7] for r in rows)
    syntax = test_timestamp_syntax(client, rows)
    log(f"timestamp filter: '>=' {syntax['ge_works']}, '>' {syntax['gt_works']}")
    months, missing_ids = [], []
    sampled_total = present_total = 0
    for month in months_since(since, now):
        days = ["01", "16"][:probes_per_month]
        probes, sampled, present, first_in_month = [], 0, 0, None
        for d in days:
            start = f"{month}-{d}"
            if start < since:
                continue
            items = _ask(client, f"disclosed:true AND disclosed_at:>={start}", PROBE_SIZE, "disclosed_at")
            if items is None:
                probes.append({"from": start, "rejected": True})
                continue
            mine = [i for i in items if _day(i).startswith(month)]
            earliest = _day(items[0]) if items else ""
            if d == "01" or first_in_month is None:
                first_in_month = earliest
            p_present = 0
            for i in mine:
                rid = report_id_of(i)
                if rid is not None:
                    sampled += 1
                    if rid in known:
                        present += 1
                        p_present += 1
                    else:
                        missing_ids.append(rid)
            probes.append({"from": start, "items": len(items), "earliest_day": earliest, "in_month": len(mine), "present": p_present})
        sampled_total += sampled
        present_total += present
        count = in_manifest.get(month, 0)
        api_has = any(p.get("in_month") for p in probes)
        verdict = ("empty_per_api" if not api_has else "missing" if count == 0 else "incomplete" if present < sampled else "ok")
        months.append({"month": month, "manifest_count": count, "api_first_report_day": first_in_month or "", "sampled": sampled,
                       "present": present, "verdict": verdict, "probes": probes})
        log(f"  {month}: manifest {count:>4}, API first report {first_in_month or '-':<10}, sampled present {present}/{sampled}, {verdict.upper() if verdict in ('missing', 'incomplete') else verdict}")
    newest = _ask(client, f"disclosed:true AND disclosed_at:>={since}", TAIL_SIZE, "-disclosed_at") or []
    tail_ids = [r for r in (report_id_of(i) for i in newest) if r is not None]
    tail_missing = [r for r in tail_ids if r not in known]
    missing_ids.extend(tail_missing)
    problem = [m["month"] for m in months if m["verdict"] in ("missing", "incomplete")]
    unique_missing = sorted(set(missing_ids))
    return {"since": since, "timestamp_filter": syntax, "months": months,
            "tail": {"checked": len(tail_ids), "missing": len(tail_missing), "newest_day": _day(newest[0]) if newest else ""},
            "verdict": {"problem_months": problem, "empty_months": [m["month"] for m in months if m["verdict"] == "empty_per_api"],
                        "recall_sampled": round(present_total / sampled_total, 3) if sampled_total else None,
                        "sampled": sampled_total, "present": present_total, "manifest_reports": len(known),
                        "missing_report_ids": unique_missing[:MAX_LISTED_MISSING], "missing_total": len(unique_missing)}}


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[], H1Client] | None = None, out: Callable[[str], None] = print,
         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    ap = argparse.ArgumentParser(prog="h1_coverage_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="a manifest.json (or its folder) written by h1_discover")
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD (default 2025-01-01)")
    ap.add_argument("--probes-per-month", type=int, choices=(1, 2), default=2, help="sample points per month: day 1 and day 16 (default 2)")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--out-dir", help="output folder (default: <data folder>/h1_coverage/<UTC time>)")
    try:
        args = ap.parse_args(argv)
        since = parse_day(args.since)
    except SystemExit as exc:
        return int(exc.code or 2)
    except ValueError as exc:
        out(f"error: {exc}")
        return 2
    try:
        rows = load_manifest_rows(args.manifest)
    except ManifestError as exc:
        out(f"error: {exc}")
        return 2
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from hrag.config import load_config  # noqa: PLC0415
        out_dir = load_config(args.config).db_path.parent / "h1_coverage" / now().strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / "coverage.json"
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
    out(f"auditing {len(rows)} manifest reports against the API, month by month since {since} (listing endpoint only):")
    try:
        result = audit(client, rows, since, now(), args.probes_per_month, out)
    except H1Error as exc:
        out(f"stopped: {exc}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checked_at": now().isoformat(timespec="seconds"), **result,
                                  "note": "counts, dates and report ids only"}, indent=2), encoding="utf-8")
    v, t = result["verdict"], result["tail"]
    out("\nsummary:")
    out(f"  months with a problem: {', '.join(v['problem_months']) or 'none'}")
    out(f"  months the API says are genuinely empty (no report on or after the 1st until a later month): {', '.join(v['empty_months']) or 'none'}")
    out(f"  sampled reports present in the manifest: {v['present']} of {v['sampled']}"
        + (f" ({100 * v['recall_sampled']:.0f}%)" if v["recall_sampled"] is not None else ""))
    out(f"  newest reports (newest first): {t['checked']} checked, {t['missing']} missing from the manifest; newest day {t['newest_day'] or 'n/a'}")
    if v["missing_total"]:
        out(f"  reports the API listed that the manifest lacks ({v['missing_total']}): {v['missing_report_ids'][:20]}")
    out(f"  exact-timestamp filter: '>=' {result['timestamp_filter']['ge_works']}, '>' {result['timestamp_filter']['gt_works']}")
    out(f"saved: {target}")
    bad = bool(v["problem_months"]) or t["missing"] > 0
    out("RESULT: " + ("GAPS FOUND - the manifest is NOT complete." if bad else
                      "no gap found in the sampled points (a sample cannot prove there are none)."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
