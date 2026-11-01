"""Get the public HackerOne dataset and pick N unique, usable reports.

Source: Hugging Face dataset `Hacker0x01/hackerone_disclosed_reports` (three public parquet files, ~19 MB).
Its card says it is "made available under HackerOne's Terms of Service"; reuse terms beyond that were not
found, so treat the text as for your own private use and do not redistribute it.

Deduplication happens here in layers (and is repeated at retrieval time by the per-report cap):
  1. HackerOne's own flags: substate 'duplicate' or a set original_report_id are dropped;
  2. exact duplicates (same text after whitespace/case normalisation);
  3. near duplicates (word 5-gram Jaccard similarity >= the configured threshold).
Nothing is silently lost: the summary file records how many reports each rule removed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from .config import IngestSettings
from .store import ReportRow

COLUMNS = ["id", "title", "created_at", "substate", "vulnerability_information", "team", "disclosed_at",
           "weakness", "original_report_id", "vote_count", "structured_scope"]
PARQUET_MAGIC = b"PAR1"
_WS = re.compile(r"\s+")


class IngestError(RuntimeError):
    pass


# ---------- download ----------

def _looks_like_parquet(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            return head == PARQUET_MAGIC and f.read(4) == PARQUET_MAGIC
    except OSError:
        return False


def download_dataset(settings: IngestSettings, cache_dir: Path, *, client: httpx.Client | None = None,
                     log: Callable[[str], None] = print) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(timeout=120.0, follow_redirects=True)
    paths: list[Path] = []
    try:
        for name in settings.dataset_files:
            dest = cache_dir / name
            if dest.is_file() and _looks_like_parquet(dest):
                log(f"  have {name}")
                paths.append(dest)
                continue
            url = settings.dataset_base_url.rstrip("/") + "/" + name
            log(f"  downloading {name} ...")
            part = dest.with_suffix(dest.suffix + ".part")
            try:
                with http.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise IngestError(f"download failed for {name}: HTTP {response.status_code}")
                    with part.open("wb") as out:
                        for block in response.iter_bytes(1 << 20):
                            out.write(block)
            except httpx.HTTPError as exc:
                raise IngestError(f"download failed for {name}: {type(exc).__name__}") from exc
            if not _looks_like_parquet(part):
                part.unlink(missing_ok=True)
                raise IngestError(f"{name} did not download as a valid parquet file")
            os.replace(part, dest)
            paths.append(dest)
    finally:
        if client is None:
            http.close()
    return paths


# ---------- read ----------

def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415 - heavy import, only needed here

    rows: list[dict[str, Any]] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        table = pq.read_table(path, columns=[c for c in COLUMNS if c in available])
        rows.extend(table.to_pylist())
    return rows


def _first(value: Any, key: str) -> str:
    """Pull `key` out of a dict, or out of the first dict in a list; '' if absent."""
    if isinstance(value, list):
        for item in value:
            found = _first(item, key)
            if found:
                return found
        return ""
    if isinstance(value, dict):
        v = value.get(key)
        return str(v) if v not in (None, "") else ""
    return ""


def to_report_row(raw: dict[str, Any]) -> ReportRow:
    body = (raw.get("vulnerability_information") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return ReportRow(
        report_id=int(raw["id"]),
        title=str(raw.get("title") or "").strip(),
        program=_first(raw.get("team"), "handle"),
        weakness=_first(raw.get("weakness"), "name"),
        substate=str(raw.get("substate") or ""),
        created_at=str(raw.get("created_at") or ""),
        disclosed_at=str(raw.get("disclosed_at") or ""),
        severity=_first(raw.get("structured_scope"), "max_severity"),
        vote_count=int(raw.get("vote_count") or 0),
        body=body,
        body_hash=hashlib.sha256(_normalise(body).encode("utf-8")).hexdigest(),
    )


# ---------- dedupe helpers ----------

def _normalise(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def shingles(text: str, n: int = 5) -> set[int]:
    words = _normalise(text).split(" ")
    if len(words) < n:
        return {hash(" ".join(words))}
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------- selection ----------

def select_reports(rows: list[dict[str, Any]], settings: IngestSettings, *, n: int | None = None,
                   seed: int | None = None) -> tuple[list[ReportRow], dict[str, Any]]:
    n = settings.n_reports if n is None else n
    seed = settings.seed if seed is None else seed
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    eligible: list[ReportRow] = []
    seen_hashes: set[str] = set()
    for raw in sorted(rows, key=lambda r: int(r["id"])):
        if raw.get("original_report_id") not in (None, 0) or raw.get("substate") == "duplicate":
            reject("marked_duplicate_by_hackerone")
            continue
        if raw.get("substate") not in settings.allowed_substates:
            reject("substate_not_allowed")
            continue
        report = to_report_row(raw)
        if not report.body:
            reject("empty_body")
        elif len(report.body) < settings.min_body_chars:
            reject("body_too_short")
        elif report.body_hash in seen_hashes:
            reject("exact_duplicate")
        else:
            seen_hashes.add(report.body_hash)
            eligible.append(report)

    pool = list(eligible)
    random.Random(seed).shuffle(pool)
    accepted: list[ReportRow] = []
    accepted_shingles: list[set[int]] = []
    for report in pool:
        if len(accepted) == n:
            break
        sh = shingles(report.body)
        if any(jaccard(sh, other) >= settings.near_duplicate_jaccard for other in accepted_shingles):
            reject("near_duplicate")
            continue
        accepted.append(report)
        accepted_shingles.append(sh)

    accepted.sort(key=lambda r: r.report_id)
    summary = {
        "dataset_rows": len(rows), "eligible_after_filters": len(eligible), "requested": n, "selected": len(accepted),
        "seed": seed, "rejected": dict(sorted(rejected.items())), "report_ids": [r.report_id for r in accepted],
    }
    return accepted, summary


def write_summary(summary: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    ids_path = directory / "selected_report_ids.txt"
    json_path = directory / "ingest_summary.json"
    ids_path.write_text("\n".join(str(i) for i in summary["report_ids"]) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ids_path, json_path
