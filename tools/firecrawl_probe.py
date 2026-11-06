"""Feasibility test: can Firecrawl read a PUBLIC HackerOne report page completely, and is the text better than the dataset's?

This is a STAGE-1 TOOL, not a pipeline stage. It writes only into its own folder (data/cache/firecrawl_probe/) and never touches
a database, config.toml, or any existing cache. It reuses the project's cleaning (`clean_report`), quality check (`assess`),
dataset reader (`read_rows`) and key lookup (`get_secret`), so it does not duplicate any of them.

The rules it enforces in code (not just in this text):
  * only https://hackerone.com/reports/<number> pages, at most 5 per run (there is no flag to raise this);
  * the BASIC proxy only (Firecrawl's default 'auto' escalates to 'enhanced' proxies after a block - that would be bypassing one);
  * no browser actions, custom headers, location or mobile tricks; the page is not stored in Firecrawl's cache (store_in_cache=False);
  * one request per page, spaced out (>= 6 s: the free plan allows 10 scrapes a minute), NO retries;
  * it STOPS at the first blocked, empty, challenge, error or not-found page - it never tries a way around it;
  * a dry run is the default; real requests need --run and the key FIRECRAWL_API_KEY (environment / .env / OS store, never code);
  * pages already cached are not requested again, and a cached file is never overwritten;
  * what is printed and saved in summary.json is numbers and short headings only - no report text and never the key.

Usage (from the project folder):
    python -m tools.firecrawl_probe 783392 135946 174470            # dry run: shows the plan, sends nothing
    python -m tools.firecrawl_probe 783392 135946 174470 --run      # real: about 1 Firecrawl credit per page
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from hrag.clean import clean_report
from hrag.quality import assess

KEY_NAME = "FIRECRAWL_API_KEY"
HOST = "hackerone.com"
MAX_REPORTS = 5
MIN_INTERVAL_S = 6.0           # the free plan allows 10 scrapes/minute
MAX_INTERVAL_S = 300.0
DEFAULT_INTERVAL_S = 7.0
CACHE_FOLDER = "firecrawl_probe"
BLOCKING_STATUSES = {401, 403, 429, 451, 503}
SHORT_PAGE_CHARS = 2000        # challenge / login text is only trusted as a block signal on SHORT pages (a long report may mention CAPTCHA)

_REPORT_PATH = re.compile(r"^/reports/(\d+)/?$")
_CHALLENGE = re.compile(
    r"just a moment|verify you are (?:a )?human|captcha|access denied|request (?:was )?blocked|attention required|"
    r"enable javascript and cookies|unusual traffic|are you (?:a )?robot|checking your browser", re.I)
_LOGIN_WALL = re.compile(r"(?:log|sign) ?in to (?:continue|view)|you (?:need|must) (?:to )?(?:log|sign) ?in", re.I)
_ACTIVITY = re.compile(r"posted a comment|added a comment|changed the (?:status|state)|triaged|bounty (?:was )?(?:awarded|paid)|"
                       r"closed this report|disclosed this report", re.I)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
_WORDS = re.compile(r"[a-z0-9]+")
_SHINGLE = 5


class ProbeError(ValueError):
    pass


# ------------------------------------------------------------------ which pages are allowed

def report_url(arg: str) -> tuple[int, str]:
    """('783392' or 'https://hackerone.com/reports/783392[/]') -> (783392, canonical url). Anything else is refused."""
    text = (arg or "").strip()
    if text.isdigit():
        rid = int(text)
    else:
        parsed = urlparse(text)
        if (parsed.scheme != "https" or parsed.hostname != HOST or parsed.username or parsed.password or parsed.port
                or parsed.query or parsed.fragment or parsed.params or "@" in parsed.netloc):
            raise ProbeError(f"not a public report page on https://{HOST}/reports/<number>: '{text[:80]}'")
        m = _REPORT_PATH.match(parsed.path)
        if not m:
            raise ProbeError(f"not a report page (expected https://{HOST}/reports/<number>): '{text[:80]}'")
        rid = int(m.group(1))
    if rid <= 0:
        raise ProbeError(f"not a valid report id: '{text[:80]}'")
    return rid, f"https://{HOST}/reports/{rid}"


# ------------------------------------------------------------------ talking to Firecrawl (kept thin so tests can fake it)

def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _record(obj: Any, url: str = "") -> dict[str, Any]:
    """One plain dict from an SDK `Document` object or from a dict (already-normalised records pass through)."""
    if isinstance(obj, dict) and "status_code" in obj and "markdown" in obj:
        return dict(obj)
    meta = _get(obj, "metadata")
    status = _get(meta, "status_code") if meta is not None else _get(obj, "status_code")
    return {"url": url or _get(meta, "source_url") or "", "status_code": status, "markdown": _get(obj, "markdown"),
            "credits_used": _get(meta, "credits_used"), "proxy_used": _get(meta, "proxy_used"), "error": _get(meta, "error")}


class SdkFetcher:
    """Calls the Firecrawl SDK's `scrape` once, with the only options this tool allows."""

    def __init__(self, sdk: Any):
        self.sdk = sdk

    def fetch(self, url: str) -> dict[str, Any]:
        doc = self.sdk.scrape(url, formats=["markdown"], proxy="basic", only_main_content=False, timeout=60_000, wait_for=2_000,
                               store_in_cache=False)
        return _record(doc, url)


def _default_factory(key: str) -> SdkFetcher:
    try:
        from firecrawl import Firecrawl  # noqa: PLC0415 - optional dependency, only needed for --run
    except ImportError as exc:
        raise ProbeError("the 'firecrawl' package is not installed. Install it with:  .venv\\Scripts\\pip install firecrawl") from exc
    return SdkFetcher(Firecrawl(api_key=key))


# ------------------------------------------------------------------ is the page usable?

def classify(doc: Any) -> tuple[str, str]:
    """('ok' | 'blocked' | 'failed' | 'empty', why). Anything but 'ok' stops the run."""
    rec = _record(doc)
    status = rec.get("status_code")
    text = rec.get("markdown")
    if isinstance(status, int) and not 200 <= status < 300:
        return ("blocked" if status in BLOCKING_STATUSES else "failed"), f"HTTP {status}"
    if not isinstance(text, str) or not text.strip():
        return "empty", "the page came back with no text"
    if len(text) < SHORT_PAGE_CHARS:
        if _CHALLENGE.search(text):
            return "blocked", "the page looks like a bot-check / challenge page"
        if _LOGIN_WALL.search(text):
            return "blocked", "the page asks for a login"
    return "ok", ""


# ------------------------------------------------------------------ compare with the Hugging Face text (numbers only)

def _shingles(text: str) -> set[tuple[str, ...]]:
    words = _WORDS.findall(text.lower())
    if len(words) < _SHINGLE:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def _flags(text: str) -> dict[str, bool]:
    a = assess(text)
    return {"has_steps": a.has_steps, "has_poc": a.has_poc, "has_impact": a.has_impact, "sparse": "sparse" in a.flags}


def compare(dataset_body: str | None, page_markdown: str) -> dict[str, Any]:
    """Describe the page against the dataset's text for the same report. Never returns any report text."""
    page = clean_report(page_markdown or "").text
    out: dict[str, Any] = {
        "page_chars": len(page), "page_flags": _flags(page),
        "activity_markers": len(_ACTIVITY.findall(page_markdown or "")),
        "outline": [{"level": len(m.group(1)), "text": m.group(2)[:60]} for m in _HEADING.finditer(page_markdown or "")][:40],
    }
    if dataset_body is None:
        return {**out, "verdict": "not_in_dataset", "dataset_chars": None, "containment": None, "page_adds": []}
    data = clean_report(dataset_body).text
    if not data.strip():
        return {**out, "verdict": "dataset_has_no_body", "dataset_chars": 0, "containment": None, "page_adds": []}
    a, b = _shingles(data), _shingles(page)
    containment = len(a & b) / len(a) if a else 0.0
    verdict = "page_contains_dataset_text" if containment >= 0.9 else "partial" if containment >= 0.3 else "different_text"
    data_flags = _flags(data)
    return {**out, "verdict": verdict, "dataset_chars": len(data), "dataset_flags": data_flags, "containment": round(containment, 3),
            "page_minus_dataset_chars": len(page) - len(data),
            "page_adds": [k for k in ("has_steps", "has_poc", "has_impact") if out["page_flags"][k] and not data_flags[k]]}


# ------------------------------------------------------------------ the run

def _redact(text: str, secrets: Sequence[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text[:200]


def _default_hf_loader(config_path: str | None) -> Callable[[list[int]], dict[int, str]]:
    def load(ids: list[int]) -> dict[int, str]:
        from hrag.config import load_config  # noqa: PLC0415
        from hrag.ingest import read_rows  # noqa: PLC0415

        files = sorted(load_config(config_path).cache_dir.glob("*.parquet"))
        wanted = set(ids)
        return {int(r["id"]): str(r.get("vulnerability_information") or "")
                for r in read_rows(files) if int(r["id"]) in wanted} if files else {}
    return load


def _default_cache_dir(config_path: str | None) -> Path:
    from hrag.config import load_config  # noqa: PLC0415
    return load_config(config_path).cache_dir / CACHE_FOLDER


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[str], Any] | None = None,
         key_getter: Callable[[str], str | None] | None = None, hf_loader: Callable[[list[int]], dict[int, str]] | None = None,
         sleep: Callable[[float], None] = time.sleep, out: Callable[[str], None] = print) -> int:
    p = argparse.ArgumentParser(prog="firecrawl_probe", description=__doc__.split("\n\n")[0])
    p.add_argument("reports", nargs="+", help="report ids or https://hackerone.com/reports/<id> links (at most 5)")
    p.add_argument("--run", action="store_true", help="really send the requests (default: dry run)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help=f"seconds between requests (minimum {MIN_INTERVAL_S:g})")
    p.add_argument("--config", help="path to config.toml (used to find the cache folder and the dataset)")
    p.add_argument("--cache-dir", help="where to cache pages and write summary.json (default: <cache_dir>/firecrawl_probe)")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)

    if len(args.reports) > MAX_REPORTS:
        out(f"error: at most {MAX_REPORTS} reports per run (you gave {len(args.reports)}). This is a feasibility test, not a crawl.")
        return 2
    if not MIN_INTERVAL_S <= args.interval <= MAX_INTERVAL_S:        # also false for nan
        out(f"error: the interval must be between {MIN_INTERVAL_S:g} and {MAX_INTERVAL_S:g} seconds "
            f"(the free plan allows 10 scrapes a minute); got {args.interval:g}")
        return 2
    targets: list[tuple[int, str]] = []
    try:
        for arg in args.reports:
            item = report_url(arg)
            if item not in targets:
                targets.append(item)
    except ProbeError as exc:
        out(f"error: {exc}")
        return 2

    out("Firecrawl feasibility test - public HackerOne report pages only, basic proxy only, one request per page, no retries,")
    out("and it stops at the first block, challenge page or error. Nothing here goes into any database.")
    for rid, url in targets:
        out(f"  report {rid}: {url}")
    if not args.run:
        out(f"DRY RUN: nothing was sent and no key was read. About {len(targets)} Firecrawl credit(s) would be used. Add --run to do it.")
        return 0

    try:
        cache = Path(args.cache_dir) if args.cache_dir else _default_cache_dir(args.config)
    except Exception as exc:  # noqa: BLE001 - a config problem should be a message, not a traceback
        out(f"error: could not read the config ({type(exc).__name__}); pass --config or --cache-dir")
        return 2
    cache.mkdir(parents=True, exist_ok=True)
    factory = client_factory or _default_factory
    get_key = key_getter
    if get_key is None:
        from hrag.secrets import get_secret as get_key  # noqa: PLC0415
    fetcher: Any = None
    key = ""

    rows: list[dict[str, Any]] = []
    credits = 0
    stopped: str | None = None
    fetched = 0
    records: dict[int, dict[str, Any]] = {}
    for rid, url in targets:
        path = cache / f"{rid}.json"
        if path.is_file():
            records[rid] = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"report_id": rid, "status": "cached", "note": "cached, no credit used"})
            continue
        if fetcher is None:
            key = get_key(KEY_NAME) or ""
            if not key:
                out(f"error: no {KEY_NAME} found. Put a line  {KEY_NAME}=...  in your .env file (or set the environment variable). "
                    "Keys are never put in code.")
                return 2
            try:
                fetcher = factory(key)
            except ProbeError as exc:
                out(f"error: {exc}")
                return 2
        if fetched:
            sleep(args.interval)
        fetched += 1
        try:
            rec = fetcher.fetch(url)
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure is recorded, never retried
            msg = f"{type(exc).__name__}: {_redact(str(exc), [key])}"
            rows.append({"report_id": rid, "status": "failed", "note": msg})
            stopped = f"Stopped at report {rid}: {msg}. No further reports were requested."
            break
        credits += rec.get("credits_used") if isinstance(rec.get("credits_used"), int) else 1
        kind, why = classify(rec)
        if kind != "ok":
            label = f"{kind} ({why})"
            rows.append({"report_id": rid, "status": kind, "note": why})
            stopped = f"Stopped at report {rid}: {label}. No further reports were requested, and nothing was retried."
            break
        text = json.dumps(rec, ensure_ascii=False, default=str)       # default=str: an odd value type must never lose a PAID page
        with path.open("x", encoding="utf-8") as fh:                  # 'x': refuse to overwrite, by construction
            fh.write(text)
        records[rid] = rec
        rows.append({"report_id": rid, "status": "ok", "note": ""})

    load_hf = hf_loader or _default_hf_loader(args.config)
    try:
        dataset = load_hf([rid for rid in records])
    except Exception as exc:  # noqa: BLE001
        out(f"note: the Hugging Face dataset could not be read ({type(exc).__name__}); comparison skipped")
        dataset = {}
    for row in rows:
        rec = records.get(row["report_id"])
        if rec:
            row.update(compare(dataset.get(row["report_id"]), rec["markdown"] or ""))

    summary = {"reports": rows, "credits_spent_this_run": credits, "stopped": stopped, "proxy": "basic", "interval_s": args.interval}
    (cache / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out("")
    for row in rows:
        if "verdict" in row:
            contained = "n/a" if row["containment"] is None else f"{row['containment']:.2f}"
            adds = ",".join(row["page_adds"]) or "nothing"
            out(f"  report {row['report_id']}: {row['status']}; page {row['page_chars']} chars; dataset "
                f"{row['dataset_chars'] if row['dataset_chars'] is not None else 'n/a'} chars; dataset text found on page: {contained}; "
                f"{row['verdict']}; page adds: {adds}; activity markers: {row['activity_markers']}")
        else:
            out(f"  report {row['report_id']}: {row['status']} - {row['note']}")
    out(f"credits spent this run: {credits}")
    if stopped:
        out(stopped)
    out(f"saved: {cache / 'summary.json'} (numbers and short headings only - no report text, no key)")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
