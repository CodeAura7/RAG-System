"""Diagnostic: what do a few disclosed reports' RELATIONSHIPS hold in the official HackerOne API?

For up to 5 report ids (default: the first 3 ids `h1 probe` saved) it reads the report once (GET /reports/{id}) and the newest page of the
listing once (GET /hacktivity), then reports - for report_generated_content, activities, attachments, summaries and every other relationship -
how many entries there are, what kinds, which fields hold text, how long the text is, and whether it carries technical signals (code fence,
HTTP request, curl, URL, image, numbered steps). It prints and saves NAMES, COUNTS and LENGTHS only.

Safety rules built in:
  * only two read endpoints are called, through the project's own H1Client (rate limit, retries, token never in errors);
  * entries the API marks `internal: true` are counted and then skipped - their text is never read or kept;
  * attachments are NEVER downloaded; URL fields and file names are ignored, only content types and sizes are counted;
  * no database is opened and no report text is saved. `--preview N` (off by default, at most 300) prints the first N characters of
    PUBLIC entries on screen only, never into the saved file.

Usage (from the project folder, via h1_relations.bat):   h1_relations.bat            (uses the probe's sample ids)
                                                          h1_relations.bat 4152553 4029867
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from hrag.h1_api import H1Client, H1Error

MAX_IDS = 5
MAX_PREVIEW = 300
PREVIEWS_PER_RELATIONSHIP = 2
LISTING_PAGE_SIZE = 25
PREVIEW_FIELDS = {"message", "content", "hacktivity_summary"}          # the only fields that may ever be shown on screen
PREVIEW_RELATIONSHIPS = {"activities", "summaries"}                      # never people, attachments or anything else
MAX_SAMPLE = 40
VISIBILITY_WORDS = ("internal", "visib", "public", "private", "hidden", "confidential")
ENUM_KEYS = {"type", "category", "content_type", "state", "status", "substate", "kind"}
NEVER_KEEP_KEYS = {"file_name", "filename"}                  # names of uploaded files can be sensitive; URLs are dropped by "url" in the key
SIGNALS = {
    "code_fence": re.compile(r"```|~~~"),
    "http_request": re.compile(r"^\s*(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+HTTP/\d|^\s*HTTP/\d(?:\.\d)?\s+\d{3}", re.M),
    "curl": re.compile(r"\bcurl\s+(?:-|https?://)"),
    "url": re.compile(r"https?://\S+"),
    "image_markdown": re.compile(r"!\[[^\]]*\]\("),
    "numbered_steps": re.compile(r"(?:^\s*\d+[.)]\s+\S.*(?:\n|$)){2,}", re.M),
}


# ---------------------------------------------------------------- reading the JSON:API envelope

def relationship_entries(body: dict, name: str) -> list[dict] | None:
    """The entries of one relationship, or None if the report has no such relationship. Identifier-only entries are filled in from
    the response's `included` section when it holds them."""
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    rels = data.get("relationships")
    if not isinstance(rels, dict) or name not in rels or not isinstance(rels[name], dict):
        return None
    inner = rels[name].get("data")
    if inner is None:
        return []
    entries = [e for e in (inner if isinstance(inner, list) else [inner]) if isinstance(e, dict)]
    included = {(str(i.get("type")), str(i.get("id"))): i for i in body.get("included") or [] if isinstance(i, dict)}
    return [e if e.get("attributes") else included.get((str(e.get("type")), str(e.get("id"))), e) for e in entries]


# ---------------------------------------------------------------- profiling

def _signals_of(text: str) -> set[str]:
    return {name for name, rx in SIGNALS.items() if rx.search(text)}


def profile_resources(entries: list[dict], preview: int = 0, preview_unstated: bool = False) -> dict[str, Any]:
    """Counts, field names and text LENGTHS of a list of entries - never their text (except the optional on-screen `previews`)."""
    preview = max(0, min(int(preview), MAX_PREVIEW))
    p: dict[str, Any] = {"count": len(entries), "types": {}, "internal": 0, "public": 0, "visibility_unknown": 0, "ids_only": 0,
                         "text_fields": {}, "value_counts": {}, "signals": {}, "total_file_size": 0, "ignored_fields": [],
                         "nested_fields": [], "previews": [], "attribute_names": [], "visibility_fields": [], "eligible_for_ingestion": 0}
    types: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {}
    values: dict[str, Counter[str]] = {}
    signals: Counter[str] = Counter()
    ignored: set[str] = set()
    nested: set[str] = set()
    names: set[str] = set()
    for entry in entries:
        types[str(entry.get("type") or "?")] += 1
        attrs = entry.get("attributes")
        if not isinstance(attrs, dict) or not attrs:
            p["ids_only"] += 1
            continue
        flag = attrs.get("internal")
        if flag is True:
            p["internal"] += 1                                  # counted, then skipped: nothing else of this entry is read
            continue
        if flag is False:
            p["public"] += 1
        else:
            p["visibility_unknown"] += 1
        found: set[str] = set()
        previews_allowed = (flag is False or (flag is None and preview_unstated)) and len(p["previews"]) < PREVIEWS_PER_RELATIONSHIP
        names.update(attrs.keys())
        for key, value in attrs.items():
            lk = key.lower()
            if "url" in lk or lk in NEVER_KEEP_KEYS:
                ignored.add(key)
            elif lk.endswith("_at") or key == "internal":
                continue
            elif isinstance(value, str):
                if key in ENUM_KEYS and 0 < len(value) <= 60:
                    values.setdefault(key, Counter())[value] += 1
                elif value.strip():
                    lengths.setdefault(key, []).append(len(value))
                    found |= _signals_of(value)
                    if preview and previews_allowed and key in PREVIEW_FIELDS:
                        p["previews"].append(value[:preview])
                        previews_allowed = len(p["previews"]) < PREVIEWS_PER_RELATIONSHIP
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if lk in {"file_size", "size", "byte_size"}:
                    p["total_file_size"] += int(value)
            elif isinstance(value, (dict, list)):
                nested.add(key)
        signals.update(found)
    p["types"] = dict(types)
    p["text_fields"] = {k: {"entries": len(v), "min_chars": min(v), "median_chars": int(statistics.median(v)), "max_chars": max(v),
                            "total_chars": sum(v)} for k, v in lengths.items()}
    p["value_counts"] = {k: dict(v) for k, v in values.items()}
    p["signals"] = dict(signals)
    p["ignored_fields"] = sorted(ignored)
    p["nested_fields"] = sorted(nested)
    p["attribute_names"] = sorted(names)
    p["visibility_fields"] = sorted(n for n in names if any(w in n.lower() for w in VISIBILITY_WORDS))
    p["eligible_for_ingestion"] = p["public"]                 # only entries the API itself marks internal=false; "not stated" is held back
    return p


def profile_generated_content(items: list[dict]) -> dict[str, Any]:
    """The listing's `report_generated_content` relationship across all items of the page."""
    entries: list[dict] = []
    for item in items:
        data = ((item.get("relationships") or {}).get("report_generated_content") or {}).get("data")
        if isinstance(data, dict) and isinstance(data.get("attributes"), dict):
            entries.append(data)
    p = profile_resources(entries)
    with_text = max((f["entries"] for f in p["text_fields"].values()), default=0)
    return {"items": len(items), "with_generated_content": len(entries), "with_text": with_text, "text_fields": p["text_fields"],
            "signals": p["signals"], "nested_fields": p["nested_fields"]}


def profile_report(report_id: int, body: dict, preview: int = 0, preview_unstated: bool = False) -> dict[str, Any]:
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    attrs = data.get("attributes") if isinstance(data.get("attributes"), dict) else {}
    out: dict[str, Any] = {
        "report_id": report_id,
        "attributes_with_value": sorted(k for k, v in attrs.items() if v not in (None, "", [], {})),
        "attributes_empty": sorted(k for k, v in attrs.items() if v in (None, "", [], {})),
        "relationships": {}}
    for name in sorted(data.get("relationships") or {}):
        entries = relationship_entries(body, name)
        out["relationships"][name] = profile_resources(entries or [], preview if name in PREVIEW_RELATIONSHIPS else 0, preview_unstated)
    out["has_included_section"] = bool(body.get("included"))
    return out


# ---------------------------------------------------------------- surveying many reports

class ManifestError(ValueError):
    """The manifest could not be used; the message says exactly why."""


def load_manifest_rows(path_text: str) -> list[dict]:
    """The `reports` list of a manifest.json written by h1_discover. Accepts the file or the folder that holds it, UTF-8 with or without a
    BOM, and UTF-16; every other problem is named."""
    path = Path(path_text)
    if path.is_dir():
        if not (path / "manifest.json").is_file():
            raise ManifestError(f"{path.resolve()} is a folder without a manifest.json inside")
        path = path / "manifest.json"
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path.resolve()} (current folder: {Path.cwd()}). To list the manifests that exist, run:  "
                            f"Get-ChildItem data\\h1_discovery -Recurse -Filter manifest.json | Select-Object FullName,Length,LastWriteTime")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{path.resolve()} could not be read ({type(exc).__name__})") from exc
    try:
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{path.resolve()} could not be decoded as UTF-8 or UTF-16 (byte {exc.start}); was it edited or re-saved?") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path.resolve()} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}") from exc
    reports = data.get("reports") if isinstance(data, dict) else None
    if not isinstance(reports, list):
        keys = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise ManifestError(f"{path.resolve()} has no 'reports' list; its top-level content is: {keys}")
    if not reports:
        extra = f" (complete: {data.get('complete')}, stopped_because: '{data.get('stopped_because', '')}')" if isinstance(data, dict) else ""
        raise ManifestError(f"{path.resolve()} holds 0 reports{extra}")
    if not any(isinstance(r, dict) and r.get("report_id") is not None for r in reports):
        first = sorted(reports[0]) if isinstance(reports[0], dict) else type(reports[0]).__name__
        raise ManifestError(f"{path.resolve()}: no row has a report_id; the first row has these fields: {first}")
    return reports


def choose_sample(rows: list[dict], n: int) -> list[int]:
    """n report ids spread evenly over the disclosure dates (oldest and newest included)."""
    ordered = sorted((r for r in rows if isinstance(r, dict) and r.get("report_id") is not None), key=lambda r: str(r.get("disclosed_at") or ""))
    if n >= len(ordered):
        return [int(r["report_id"]) for r in ordered]
    picks = [0] if n == 1 else [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
    return [int(ordered[i]["report_id"]) for i in dict.fromkeys(picks)]


def aggregate(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Adds up the per-report profiles: how often does each relationship hold something, and how reliably public is it?"""
    rels: dict[str, dict[str, Any]] = {}
    for prof in profiles:
        for name, p in prof["relationships"].items():
            a = rels.setdefault(name, {"reports_with_entries": 0, "entries_total": 0, "types": Counter(), "explicit_public": 0, "internal": 0,
                                       "not_stated": 0, "signals": Counter(), "content_types": Counter(), "text_fields": {},
                                       "attribute_names": set(), "visibility_fields": set(), "total_file_size": 0})
            a["reports_with_entries"] += 1 if p["count"] else 0
            a["entries_total"] += p["count"]
            a["types"].update(p["types"])
            a["explicit_public"] += p["public"]
            a["internal"] += p["internal"]
            a["not_stated"] += p["visibility_unknown"]
            a["signals"].update(p["signals"])
            a["content_types"].update(p["value_counts"].get("content_type", {}))
            a["attribute_names"].update(p["attribute_names"])
            a["visibility_fields"].update(p["visibility_fields"])
            a["total_file_size"] += p["total_file_size"]
            for field, s in p["text_fields"].items():
                t = a["text_fields"].setdefault(field, {"entries": 0, "total_chars": 0, "max_chars": 0})
                t["entries"] += s["entries"]
                t["total_chars"] += s["total_chars"]
                t["max_chars"] = max(t["max_chars"], s["max_chars"])
    out: dict[str, Any] = {}
    for name, a in sorted(rels.items()):
        out[name] = {"reports_with_entries": a["reports_with_entries"], "entries_total": a["entries_total"], "types": dict(a["types"]),
                     "explicit_public": a["explicit_public"], "internal": a["internal"], "not_stated": a["not_stated"],
                     "signals": dict(a["signals"]), "content_types": dict(a["content_types"]), "total_file_size": a["total_file_size"],
                     "attribute_names": sorted(a["attribute_names"]), "visibility_fields": sorted(a["visibility_fields"]),
                     "text_fields": {f: {**t, "mean_chars": round(t["total_chars"] / t["entries"])} for f, t in a["text_fields"].items()}}
    return {"reports": len(profiles), "relationships": out}


# ---------------------------------------------------------------- printing

def _describe(name: str, p: dict[str, Any], say: Callable[[str], None]) -> None:
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(p["types"].items())) or "none"
    say(f"  {name}: {p['count']} entries ({kinds}); internal {p['internal']} (skipped), public {p['public']}, "
        f"visibility not stated {p['visibility_unknown']}, identifiers only {p['ids_only']}")
    for field, s in p["text_fields"].items():
        say(f"      text field '{field}': {s['entries']} entries, {s['min_chars']} to {s['max_chars']} characters (median {s['median_chars']})")
    for field, counts in p["value_counts"].items():
        say(f"      {field}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if p["signals"]:
        say("      technical signals (entries containing): " + ", ".join(f"{k} {v}" for k, v in sorted(p["signals"].items())))
    if p["total_file_size"]:
        say(f"      total attachment size listed: {p['total_file_size']} bytes (not downloaded)")
    if p["ignored_fields"]:
        say(f"      ignored on purpose (links / file names): {', '.join(p['ignored_fields'])}")
    for text in p["previews"]:
        say(f"      preview (screen only): {text!r}")


def _print_survey(survey: dict[str, Any], out: Callable[[str], None], show: bool) -> None:
    if not show:
        return
    out(f"\nsurvey of {survey['reports']} reports (counts and lengths only):")
    for name in ("activities", "summaries", "attachments"):
        a = survey["relationships"].get(name)
        if not a:
            out(f"  {name}: never present")
            continue
        out(f"  {name}: present in {a['reports_with_entries']} of {survey['reports']} reports, {a['entries_total']} entries; "
            f"types {a['types'] or '-'}")
        for field, t in a["text_fields"].items():
            out(f"      text '{field}': {t['entries']} entries, mean {t['mean_chars']} characters, longest {t['max_chars']}")
        if a["signals"]:
            out("      technical signals: " + ", ".join(f"{k} {v}" for k, v in sorted(a["signals"].items())))
        if a["content_types"]:
            out(f"      attachment types: {a['content_types']}, {a['total_file_size']} bytes listed (not downloaded)")


def _print_eligibility(survey: dict[str, Any], out: Callable[[str], None]) -> None:
    out("\nvisibility (what could safely be ingested):")
    for name in ("activities", "summaries", "attachments"):
        a = survey["relationships"].get(name)
        if a and a["entries_total"]:
            marks = ", ".join(a["visibility_fields"]) or "none found among " + ", ".join(a["attribute_names"][:12])
            out(f"  {name}: explicitly public {a['explicit_public']}, internal {a['internal']} (excluded), "
                f"not stated {a['not_stated']} (held back); visibility fields: {marks}")
    out("  Only entries the API itself marks internal=false count as explicitly public. Entries with no visibility field are held back "
        "until their status is confirmed.")
    out("  report_generated_content comes from the public Hacktivity feed itself and has no per-item visibility field.")


# ---------------------------------------------------------------- main

def default_client() -> H1Client:
    from hrag.secrets import require_secret  # noqa: PLC0415
    return H1Client(require_secret("H1_API_USERNAME"), require_secret("H1_API_TOKEN"), cache_dir=None)


def _data_dir(config: str | None) -> Path:
    from hrag.config import load_config  # noqa: PLC0415
    return load_config(config).db_path.parent


def main(argv: Sequence[str] | None = None, *, client_factory: Callable[[], H1Client] | None = None,
         out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(prog="h1_relations", description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="*", help=f"up to {MAX_IDS} report ids (default: the first 3 sample ids saved by `h1 probe`)")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--probe-file", help="the probe's h1_probe.json (default: <data folder>/h1_probe.json)")
    ap.add_argument("--out-dir", help="where relations.json is written (default: <data folder>/h1_relations/<UTC time>)")
    ap.add_argument("--preview", type=int, default=0, help=f"print the first N characters (max {MAX_PREVIEW}) of PUBLIC entries on screen; default 0 = off")
    ap.add_argument("--preview-unstated", action="store_true",
                    help="with --preview: also show entries whose visibility the API does not state (to compare them with the public page); "
                         "internal entries, people and attachments are never shown")
    ap.add_argument("--manifest", help="a manifest.json written by h1_discover; with --sample, survey reports drawn from it")
    ap.add_argument("--sample", type=int, default=0, help=f"survey this many reports spread over the manifest's dates (1-{MAX_SAMPLE})")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)

    if not all(a.strip().isdigit() for a in args.reports):
        out("error: give report ids (numbers) only")
        return 2
    ids = list(dict.fromkeys(int(a) for a in args.reports))
    if len(ids) > MAX_IDS:
        out(f"error: at most {MAX_IDS} reports per run (you gave {len(ids)})")
        return 2
    sample_mode = bool(args.manifest or args.sample)
    if sample_mode:
        if not (args.manifest and args.sample):
            out("error: --manifest and --sample go together")
            return 2
        if ids or not 1 <= args.sample <= MAX_SAMPLE:
            out(f"error: with --sample give no report ids, and a sample size from 1 to {MAX_SAMPLE}")
            return 2
        try:
            ids = choose_sample(load_manifest_rows(args.manifest), args.sample)
        except ManifestError as exc:
            out(f"error: {exc}")
            return 2
    if not ids:
        probe = Path(args.probe_file) if args.probe_file else _data_dir(args.config) / "h1_probe.json"
        try:
            ids = [int(i) for i in json.loads(probe.read_text(encoding="utf-8")).get("sample_report_ids", [])][:3]
        except (OSError, ValueError, TypeError):
            ids = []
        if not ids:
            out(f"error: no report ids given and none found in {probe}. Run  hrag.bat h1 probe  first, or pass ids on the command line.")
            return 2
    out_dir = Path(args.out_dir) if args.out_dir else _data_dir(args.config) / "h1_relations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / "relations.json"
    if target.exists():
        out(f"error: {target} already exists - this tool never overwrites. Choose another --out-dir.")
        return 2

    try:
        client = (client_factory or default_client)()
    except Exception as exc:  # noqa: BLE001 - MissingKeyError and friends: print the plain message, which never holds a key
        out(f"error: {exc}")
        return 2

    rc = 0
    generated: dict[str, Any] = {}
    try:
        page = client.get_json("/hacktivity", {"queryString": "disclosed:true", "page[size]": LISTING_PAGE_SIZE})
        generated = profile_generated_content(page.get("data") or [])
        out(f"listing (newest {generated['items']} disclosed items): report_generated_content present on {generated['with_generated_content']}, "
            f"with text on {generated['with_text']}")
        for field, s in generated["text_fields"].items():
            out(f"    text field '{field}': {s['entries']} items, {s['min_chars']} to {s['max_chars']} characters (median {s['median_chars']})")
    except H1Error as exc:
        out(f"listing: not available ({exc})")
        rc = 1

    reports: list[dict[str, Any]] = []
    for rid in ids:
        try:
            body = client.get_json(f"/reports/{rid}")
        except H1Error as exc:
            out(f"report {rid}: not available (HTTP {exc.status})")
            rc = 1
            continue
        prof = profile_report(rid, body, args.preview, args.preview_unstated)
        if sample_mode:
            acts, sums = prof["relationships"].get("activities", {}), prof["relationships"].get("summaries", {})
            out(f"report {rid}: activities {acts.get('count', 0)}, summaries {sums.get('count', 0)}, "
                f"attachments {prof['relationships'].get('attachments', {}).get('count', 0)}")
        else:
            out(f"report {rid}:")
            out(f"  attributes with a value: {', '.join(prof['attributes_with_value']) or 'none'}")
            out(f"  attributes empty for this token: {', '.join(prof['attributes_empty']) or 'none'}")
            for name, p in prof["relationships"].items():
                _describe(name, p, out)
        for p in prof["relationships"].values():
            p.pop("previews", None)                     # on-screen only; never saved
        reports.append(prof)

    survey = aggregate(reports) if reports else {"reports": 0, "relationships": {}}
    _print_survey(survey, out, show=sample_mode)
    _print_eligibility(survey, out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "hacktivity_generated_content": generated, "survey": survey, "reports": reports,
                                  "note": "names, counts and lengths only; no report text, links or file names"}, indent=2), encoding="utf-8")
    out(f"saved: {target} (names, counts and lengths only)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
