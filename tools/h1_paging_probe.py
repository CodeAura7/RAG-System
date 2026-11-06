"""Diagnostic: how does the official HackerOne listing REALLY paginate? (at most 10 requests, listing endpoint only, structure only)

The first discovery run returned exactly 50 reports from ONE page although 100 were requested and no limit was reached: the API seems
to cap a page at 50 and sent no next-page link the tool could use. This probe asks the same question in several ways and records:
  * how many items come back for page[size] 100 / 50 / 51 / 25 (the real page-size cap),
  * which keys the response has (`links`, `meta`, ...) and what a `next` link looks like, if any,
  * whether an explicit page[number]=2 / 3 returns NEW reports or repeats page 1 (do page numbers work?),
  * what a page far past the end returns (empty page, or an error),
  * pagination-related HTTP headers (`Link`, `X-Total-Count`, ...); the names - not the values - of all other headers.
It prints and saves counts, dates, parameter names and report ids only: no titles, no report text, no reporter names.

Usage (from the project folder, via h1_paging_probe.bat):   h1_paging_probe.bat
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, urlparse

import httpx

from hrag.h1_api import H1Client, H1Error, report_id_of
from tools.h1_discover import DEFAULT_SINCE, parse_day
from tools.h1_relations import default_client

MAX_REQUESTS = 10
KEEP_HEADER_VALUES = ("link", "x-total-count", "x-total-pages", "x-per-page", "x-page", "x-next-page", "x-ratelimit-limit",
                      "x-ratelimit-remaining", "retry-after")
# (label, page[size], page[number] or None)
PLAN = (("size100", 100, None), ("size50", 50, None), ("size51", 51, None), ("size25", 25, None),
        ("page2", 50, 2), ("page3", 50, 3), ("page_far", 50, 500))


def _day(item: dict) -> str:
    return str((item.get("attributes") or {}).get("disclosed_at") or "")[:10]


def probe(client: H1Client, since: str, log: Callable[[str], None] = print) -> dict[str, Any]:
    query = f"disclosed:true AND disclosed_at:>={since}"
    headers_seen: dict[str, str] = {}
    names_seen: set[str] = set()

    def on_response(response: httpx.Response) -> None:
        names_seen.update(k.lower() for k in response.headers.keys())
        for k in KEEP_HEADER_VALUES:
            if k in response.headers:
                headers_seen[k] = response.headers[k]

    client.http.event_hooks.setdefault("response", []).append(on_response)
    records: list[dict[str, Any]] = []
    ids_by_label: dict[str, set[int]] = {}
    next_url: str | None = None

    def ask(label: str, params: dict[str, Any] | None, url: str = "/hacktivity") -> dict[str, Any]:
        rec: dict[str, Any] = {"label": label, "status": 200, "items": 0}
        try:
            body = client.get_json(url, params)
        except H1Error as exc:
            if exc.status in (401, 403) or exc.status is None or exc.status >= 500:
                raise
            rec["status"] = exc.status
            records.append(rec)
            log(f"  {label}: HTTP {exc.status}")
            return rec
        items = [i for i in body.get("data") or [] if isinstance(i, dict)]
        ids = {rid for rid in (report_id_of(i) for i in items) if rid is not None}
        days = sorted(d for d in (_day(i) for i in items) if d)
        links = body.get("links") if isinstance(body.get("links"), dict) else {}
        rec.update({"items": len(items), "first_day": days[0] if days else "", "last_day": days[-1] if days else "",
                    "top_level_keys": sorted(body.keys()), "links_keys": sorted(links.keys()),
                    "meta_keys": sorted(body["meta"].keys()) if isinstance(body.get("meta"), dict) else [],
                    "next": links.get("next") or "", "report_ids": sorted(ids)})
        ids_by_label[label] = ids
        records.append(rec)
        log(f"  {label}: {len(items)} items, {rec['first_day'] or '-'} to {rec['last_day'] or '-'}, keys {rec['top_level_keys']}, "
            f"links {rec['links_keys']}")
        return rec

    log("asking the listing in several ways (structure only):")
    for label, size, number in PLAN:
        params: dict[str, Any] = {"queryString": query, "page[size]": size, "sort": "disclosed_at"}
        if number:
            params["page[number]"] = number
        rec = ask(label, params)
        if label == "size100" and rec.get("next"):
            next_url = rec["next"]
    if next_url:
        ask("follow_next_link", None, next_url)

    def rec_of(label: str) -> dict[str, Any]:
        return next((r for r in records if r["label"] == label), {})

    first, second, third = ids_by_label.get("size50", set()), ids_by_label.get("page2", set()), ids_by_label.get("page3", set())
    sizes = {s: rec_of(l).get("items", 0) for l, s, _ in PLAN if l.startswith("size")}
    parsed = parse_qs(urlparse(next_url).query) if next_url else {}
    verdict = {
        "page_size_cap": max(sizes.values()) if sizes and max(sizes.values()) < 100 else None,
        "items_by_requested_size": {str(k): v for k, v in sizes.items()},
        "next_link_sent": any(r.get("next") for r in records if r["label"].startswith("size")),
        "next_link_parameters": sorted(parsed.keys()),
        "page_number_works": bool(second) and not (second & first) if first else None,
        "page_number_overlap_with_page1": len(second & first),
        "page3_new_vs_page2": len(third - second - first),
        "page_far_status": rec_of("page_far").get("status"), "page_far_items": rec_of("page_far").get("items"),
        "link_header": "link" in headers_seen, "total_count_header": headers_seen.get("x-total-count"),
        "pagination_headers": {k: v for k, v in headers_seen.items() if k != "link"},
    }
    verdict["page_size_cap"] = verdict["page_size_cap"] if verdict["page_size_cap"] else (100 if sizes.get(100) == 100 else None)
    return {"since": since, "verdict": verdict, "requests": records, "header_names_seen": sorted(names_seen)}


def _explain(v: dict[str, Any], out: Callable[[str], None]) -> None:
    out("\nwhat this means:")
    out(f"  page size: asked for {v['items_by_requested_size']} -> the API's page cap looks like {v['page_size_cap'] or 'not reached (100 or more)'}")
    out(f"  next link in the response: {'yes, parameters ' + str(v['next_link_parameters']) if v['next_link_sent'] else 'NO - the response has no usable next link'}")
    out(f"  Link / X-Total-Count headers: {'Link header present' if v['link_header'] else 'no Link header'}; X-Total-Count {v['total_count_header'] or 'absent'}")
    works = v["page_number_works"]
    out("  page[number]: " + ("WORKS - page 2 holds different reports than page 1" if works else
                              "IGNORED - page 2 repeated page 1 (use the date watermark)" if works is False else "could not be judged"))
    out(f"  a page far past the end (page 500): HTTP {v['page_far_status']}, {v['page_far_items']} items")


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[], H1Client] | None = None, out: Callable[[str], None] = print,
         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    ap = argparse.ArgumentParser(prog="h1_paging_probe", description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=DEFAULT_SINCE, help="YYYY-MM-DD (default 2025-01-01)")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--out-dir", help="output folder (default: <data folder>/h1_paging/<UTC time>)")
    try:
        args = ap.parse_args(argv)
        since = parse_day(args.since)
    except SystemExit as exc:
        return int(exc.code or 2)
    except ValueError as exc:
        out(f"error: {exc}")
        return 2
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from hrag.config import load_config  # noqa: PLC0415
        out_dir = load_config(args.config).db_path.parent / "h1_paging" / now().strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / "paging.json"
    if target.exists():
        out(f"error: {target} already exists - this tool never overwrites. Choose another --out-dir.")
        return 2
    try:
        client = (client_factory or default_client)()
    except Exception as exc:  # noqa: BLE001 - plain message, never holds a key
        out(f"error: {exc}")
        return 2
    client.min_interval = max(client.min_interval, 0.6) if client_factory is None else client.min_interval
    try:
        result = probe(client, since, out)
    except H1Error as exc:
        out(f"stopped: {exc}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checked_at": now().isoformat(timespec="seconds"), **result,
                                  "note": "counts, dates, parameter names and report ids only"}, indent=2), encoding="utf-8")
    _explain(result["verdict"], out)
    out(f"saved: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
