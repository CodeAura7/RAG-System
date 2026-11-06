"""Import a CHOSEN list of report ids from the cached Hugging Face dataset into a SEPARATE database.

Why: `ingest-dataset` can only pick reports by random seed. This takes a list of ids you chose (for example a friend's
ranked shortlist), keeps all of them, and tops up to N from the 300 most-voted other reports (the project's selector then
prefers stronger reports, and among equals the lowest id; it does not rank by votes). It reuses the project's own
`collect_reports` (the same cleaning, quality flags, prompt-injection removal, duplicate removal and diversity caps as
`python -m hrag collect`), so these reports go through exactly the cleaning your other reports do.

Safety: it writes only to its own database (default data/hrag.selection.db) and its own config (config.selection.toml).
It refuses to touch the database named in config.toml, and refuses to overwrite an existing one without --force.
Nothing is sent to any paid API. Next step after it: python -m hrag --config config.selection.toml build
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from hrag.cli import TERMS_NOTICE
from hrag.collect import collect_reports, process
from hrag.config import Config, ConfigError, load_config
from hrag.ingest import IngestError, _first, download_dataset, jaccard, read_rows, shingles
from hrag.sources import SourceReport
from hrag.store import ReportRow, Store

DEFAULT_IDS = "sources/selection_ids_friend90.txt"
DEFAULT_DB = "data/hrag.selection.db"
DEFAULT_CONFIG = "config.selection.toml"
DEFAULT_EVAL_DIR = "eval_selection"


# ------------------------------------------------------------------------------------------------ reading the id list
def read_ids(path: Path) -> list[int]:
    """Ids from a .csv (column 'id') or a text file (numbers separated by whitespace/commas, '#' starts a comment)."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".csv":
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{path}: a .csv needs a column named 'id' (found: {reader.fieldnames})")
        tokens = [str(row["id"]).strip() for row in reader]
    else:
        tokens = [t for line in text.splitlines() for t in re.split(r"[\s,]+", line.split("#", 1)[0]) if t]
    ids: list[int] = []
    for t in tokens:
        if not t.isdigit():
            raise ValueError(f"{path}: '{t}' is not a report id (expected whole numbers)")
        if int(t) not in ids:
            ids.append(int(t))
    if not ids:
        raise ValueError(f"{path}: no report ids found")
    return ids


# ------------------------------------------------------------------------------------------------ selection
def to_source(raw: dict[str, Any]) -> SourceReport:
    """One dataset row in the shape `hrag.collect` expects. Cleaning happens later, in collect, not here."""
    rid = int(raw["id"])
    return SourceReport(
        source="dataset", source_url=f"https://hackerone.com/reports/{rid}", report_id=rid,
        title=str(raw.get("title") or "").strip(), body=raw.get("vulnerability_information") or "",
        program=_first(raw.get("team"), "handle"), weakness=_first(raw.get("weakness"), "name"),
        severity=_first(raw.get("structured_scope"), "max_severity"), substate=str(raw.get("substate") or ""),
        created_at=str(raw.get("created_at") or ""), disclosed_at=str(raw.get("disclosed_at") or ""))


def _usable_for_topup(raw: dict[str, Any], allowed_substates: Sequence[str]) -> bool:
    """The same first filters `ingest-dataset` applies: not a HackerOne-marked duplicate, and an allowed substate."""
    if raw.get("original_report_id") not in (None, 0) or raw.get("substate") == "duplicate":
        return False
    return raw.get("substate") in allowed_substates


def select_with_pinned(rows: list[dict[str, Any]], cfg: Config, pinned_ids: Sequence[int], *, n: int,
                       pool_top_votes: int) -> tuple[list[ReportRow], dict[str, Any]]:
    by_id = {int(r["id"]): r for r in rows}
    pinned_ids = list(dict.fromkeys(pinned_ids))
    not_in_dataset = [i for i in pinned_ids if i not in by_id]
    pinned_raw = {i: by_id[i] for i in pinned_ids if i in by_id}

    pool = [r for i, r in by_id.items() if i not in pinned_raw and _usable_for_topup(r, cfg.ingest.allowed_substates)]
    pool.sort(key=lambda r: (-int(r.get("vote_count") or 0), int(r["id"])))
    pool = pool[:pool_top_votes]

    sources = [to_source(r) for r in pinned_raw.values()] + [to_source(r) for r in pool]
    kept, summary = collect_reports(sources, cfg.collect, pinned=set(pinned_raw), n=n)
    votes = {i: int(r.get("vote_count") or 0) for i, r in by_id.items()}
    kept = [dataclasses.replace(r, vote_count=votes.get(r.report_id, 0)) for r in kept]   # collect leaves votes at 0

    kept_ids = {r.report_id for r in kept}
    dropped = [{"id": i, "reason": _why_dropped(pinned_raw[i], kept, cfg)} for i in pinned_raw if i not in kept_ids]
    topup = [{"id": r.report_id, "program": r.program, "weakness": r.weakness, "votes": r.vote_count, "title": r.title}
             for r in kept if r.report_id not in pinned_raw]
    summary.update({
        "pinned_requested": len(pinned_ids), "pinned_kept": len(pinned_raw) - len(dropped), "pinned_dropped": dropped,
        "pinned_not_in_dataset": not_in_dataset, "topup_selected": topup,
        "topup_pool": f"the {len(pool)} most-voted usable reports not in your list (pool_top_votes={pool_top_votes})",
        "pinned_marked_duplicate_by_hackerone": [i for i, r in pinned_raw.items()
                                                 if r.get("original_report_id") not in (None, 0) or r.get("substate") == "duplicate"],
    })
    return kept, summary


def _why_dropped(raw: dict[str, Any], kept: list[ReportRow], cfg: Config) -> str:
    """collect only counts reasons; for the ids you asked for we say which one and, for duplicates, of which kept report."""
    row, reason, _ = process(to_source(raw), cfg.collect)
    if row is None:
        return reason
    sh = shingles(row.body)
    best = max(kept, key=lambda k: (k.body_hash == row.body_hash, jaccard(sh, shingles(k.body))), default=None)
    if best is not None:
        if best.body_hash == row.body_hash:
            return f"exact_duplicate_of_{best.report_id}"
        sim = jaccard(sh, shingles(best.body))
        if sim >= cfg.collect.near_duplicate_jaccard:
            return f"near_duplicate_of_{best.report_id} (similarity {sim:.2f})"
    return "over_quota (--n is smaller than the number of ids you gave)"


# ------------------------------------------------------------------------------------------------ the derived config
def derive_config(text: str, db: str, eval_dir: str) -> str:
    """config.toml with only [paths] db and eval_dir changed, everything else byte-for-byte (profiles, retrieval, ...)."""
    out: list[str] = []
    section, seen = "", set()
    for line in text.splitlines():
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            if section == "paths":
                out.extend(f'{k:<9} = "{v}"' for k, v in (("db", db), ("eval_dir", eval_dir)) if k not in seen)
            section = header.group(1).strip()
        key = re.match(r"\s*(db|eval_dir)\s*=", line)
        if section == "paths" and key:
            seen.add(key.group(1))
            out.append(f'{key.group(1):<9} = "{db if key.group(1) == "db" else eval_dir}"')
            continue
        out.append(line)
    if section == "paths":
        out.extend(f'{k:<9} = "{v}"' for k, v in (("db", db), ("eval_dir", eval_dir)) if k not in seen)
    head = ("# Generated by tools/import_selection.py from config.toml: only [paths] db and eval_dir differ, so the selected\n"
            "# reports live in their own database and their evaluation files in their own folder. Use it with\n"
            "#   python -m hrag --config config.selection.toml <command>      (and as the config path in the Playground)\n")
    return head + "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ the command
def _posix_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="import_selection", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="path to config.toml (default: next to the project)")
    p.add_argument("--ids", default=DEFAULT_IDS, help=f"ids file: .txt (one id per line) or .csv with an 'id' column (default {DEFAULT_IDS})")
    p.add_argument("--n", type=int, default=100, help="how many reports in total (your ids first, then the top-up)")
    p.add_argument("--pool-top-votes", type=int, default=300, help="top up only from this many most-voted other reports (votes decide who enters the pool, not the order inside it)")
    p.add_argument("--out-db", default=DEFAULT_DB)
    p.add_argument("--out-config", default=DEFAULT_CONFIG)
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--parquet-dir", help="use already-downloaded parquet files from this folder")
    p.add_argument("--force", action="store_true", help="replace an existing selection database / config file")
    p.add_argument("--dry-run", action="store_true", help="show what would be selected; write nothing")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):        # same as hrag/__main__.py: Windows consoles default to a legacy code page
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    try:
        return _run(args)
    except (ValueError, ConfigError, IngestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args) -> int:
    cfg = load_config(args.config)
    base_cfg_path = Path(args.config) if args.config else cfg.root / "config.toml"

    def under_root(value: str) -> Path:
        q = Path(value)
        return q if q.is_absolute() else cfg.root / q

    ids_path, out_db, out_cfg = under_root(args.ids), under_root(args.out_db), under_root(args.out_config)
    if out_db.resolve() == cfg.db_path.resolve():
        print(f"refusing: --out-db is the database named in your config ({cfg.db_path}). Pick a different file.")
        return 1
    if out_db.is_file() and Store(out_db).report_ids() and not args.force and not args.dry_run:
        print(f"{out_db} already has reports. Re-running REPLACES them and deletes their chunks and embeddings (paid API work). "
              "Pass --force if you really want that.")
        return 1

    ids = read_ids(ids_path)
    print(TERMS_NOTICE + "\n")
    if args.parquet_dir:
        paths = [Path(args.parquet_dir) / f for f in cfg.ingest.dataset_files]
        missing = [x for x in paths if not x.is_file()]
        if missing:
            raise IngestError(f"missing file(s) in --parquet-dir: {', '.join(x.name for x in missing)}")
    else:
        paths = download_dataset(cfg.ingest, cfg.cache_dir)
    rows = read_rows(paths)
    kept, summary = select_with_pinned(rows, cfg, ids, n=args.n, pool_top_votes=args.pool_top_votes)

    print(f"your ids: {summary['pinned_requested']}  kept after cleaning: {summary['pinned_kept']}  "
          f"topped up: {len(summary['topup_selected'])}  total: {summary['selected']} of {summary['requested']} requested")
    if summary["pinned_not_in_dataset"]:
        print("NOT IN THE DATASET:", ", ".join(map(str, summary["pinned_not_in_dataset"])))
    for d in summary["pinned_dropped"]:
        print(f"  dropped {d['id']}: {d['reason']}")
    if summary["pinned_marked_duplicate_by_hackerone"]:
        print("note: HackerOne itself marks these as duplicates (kept, because you asked for them):",
              ", ".join(map(str, summary["pinned_marked_duplicate_by_hackerone"])))
    if summary["topup_selected"]:
        print("top-up reports (chosen by this tool, NOT from your list):")
        for t in summary["topup_selected"]:
            print(f"  {t['id']}  {t['votes']:>4} votes  {t['program'] or '-'}  {t['title'][:70]}")
    if summary["shortfall"]:
        print(f"SHORTFALL: {summary['shortfall']} short of {summary['requested']} - raise --pool-top-votes.")
    print("quality flags:", summary["flags"] or "none", "| AI-directed instructions removed:", summary["injection"])
    if args.dry_run:
        print("dry run: nothing written.")
        return 0

    Store(out_db).replace_reports(kept)
    summary_path = out_db.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wrote_cfg = False
    if not out_cfg.exists() or args.force:
        derived = derive_config(base_cfg_path.read_text(encoding="utf-8"), _posix_rel(out_db, cfg.root),
                                _posix_rel(under_root(args.eval_dir), cfg.root))
        out_cfg.write_text(derived, encoding="utf-8", newline="\n")
        wrote_cfg = True
    if load_config(out_cfg).db_path.resolve() != out_db.resolve():
        print(f"warning: {out_cfg} points at a different database than {out_db}. Delete it and re-run, or pass --force.")
    print(f"database -> {out_db}\nsummary  -> {summary_path}\nconfig   -> {out_cfg} ({'written' if wrote_cfg else 'already existed, kept'})")
    print(f"Next: python -m hrag --config {_posix_rel(out_cfg, cfg.root)} build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
