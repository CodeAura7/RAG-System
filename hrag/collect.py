"""collect: turn downloaded source reports into the ~100 reports the database holds.

Offline and deterministic (it only reads the files `h1 fetch` and `gitlab fetch` cached), so it costs nothing and can be re-run.
Steps: clean -> quality flags -> drop trivial -> one report per id -> exact / near duplicates -> select with quality tiers and
diversity caps. The user's pinned sample links always go in. Nothing is dropped silently: the summary counts every reason.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import gitlab_src, h1_api
from .clean import _is_injection, clean_report, filter_comments
from .config import CollectSettings
from .ingest import jaccard, shingles
from .quality import assess
from .sources import SourceReport
from .store import ReportRow

WEAK_FLAGS = {"incomplete", "disputed", "image_dependent", "sparse"}


@dataclass
class Candidate:
    row: ReportRow
    pinned: bool = False

    @property
    def weak(self) -> bool:
        return bool(WEAK_FLAGS & set(self.row.flags))


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def process(src: SourceReport, settings: CollectSettings) -> tuple[ReportRow | None, str, dict]:
    """(row or None, reject reason, stats). Cleaning happens here, once, so every later step sees the final text."""
    cleaned = clean_report(src.body, source=src.source)
    if len(cleaned.text) < settings.min_body_chars:
        return None, "too_short_after_cleaning", {}
    title = " ".join((html.unescape(src.title) if src.source == "gitlab" else src.title).split())
    hits = list(cleaned.injection_hits)
    if not title or _is_injection(title)[0]:
        hits.append("instruction in the title")
        title = f"HackerOne report {src.report_id}"
    human = [str(n.get("body") or "") for n in src.comments
             if not n.get("system") and not (n.get("author") or {}).get("bot")]
    a = assess(cleaned.text, images=cleaned.images, injection_hits=hits, injection_in_code=cleaned.injection_in_code,
               substate=src.substate, labels=src.labels, comments=human, sparse_chars=settings.sparse_chars)
    body = cleaned.text
    kept = filter_comments(src.comments, source=src.source) if src.comments else []
    if kept:
        body += "\n\n## Triage discussion\n\n" + "\n\n---\n\n".join(kept)
    row = ReportRow(
        report_id=src.report_id, title=title, program=src.program, weakness=src.weakness, substate=src.substate, created_at=src.created_at,
        disclosed_at=src.disclosed_at, severity=src.severity, vote_count=0, body=body,
        body_hash=hashlib.sha256(_normalise(body).encode("utf-8")).hexdigest(), source=src.source, source_url=src.source_url, cwe=src.cwe,
        quality_flags=json.dumps(a.flags), quality_score=a.score, images=json.dumps(cleaned.images), injection_hits=len(hits))
    return row, "", {"comments_kept": len(kept), "injection_in_code": cleaned.injection_in_code}


def collect_reports(sources: list[SourceReport], settings: CollectSettings, *, pinned: set[int] | frozenset[int] = frozenset(),
                    n: int | None = None) -> tuple[list[ReportRow], dict]:
    n = settings.n_reports if n is None else n
    rejected: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    stats = Counter()
    notes: list[str] = []

    # 1. clean + flag
    by_id: dict[int, list[Candidate]] = {}
    for src in sorted(sources, key=lambda s: (s.report_id, s.source)):
        row, reason, st = process(src, settings)
        notes.extend(x for x in src.notes if x not in notes)
        if row is None:
            rejected[reason] += 1
            continue
        stats.update(st)
        by_id.setdefault(row.report_id, []).append(Candidate(row, row.report_id in pinned))

    # 2. one report per id: better quality wins, a tie goes to the API
    unique: list[Candidate] = []
    for rid in sorted(by_id):
        group = by_id[rid]
        best = max(group, key=lambda c: (round(c.row.quality_score, 1), c.row.source == "hackerone_api"))
        if len(group) > 1:
            rejected["duplicate_report_id"] += len(group) - 1
        unique.append(best)

    # 3. exact and near duplicates (pinned and better reports are looked at first, so they are the ones kept)
    unique.sort(key=lambda c: (not c.pinned, -c.row.quality_score, c.row.report_id))
    seen_hashes: set[str] = set()
    accepted_shingles: list[set[int]] = []
    deduped: list[Candidate] = []
    for c in unique:
        if c.row.body_hash in seen_hashes:
            rejected["exact_duplicate"] += 1
            continue
        sh = shingles(c.row.body)
        if any(jaccard(sh, other) >= settings.near_duplicate_jaccard for other in accepted_shingles):
            rejected["near_duplicate"] += 1
            continue
        seen_hashes.add(c.row.body_hash)
        accepted_shingles.append(sh)
        deduped.append(c)

    # 4. select: pinned first, then strong reports, then weak ones only if the quota is still open; caps keep it diverse
    order = sorted(deduped, key=lambda c: (not c.pinned, c.weak, -c.row.quality_score, c.row.report_id))
    chosen: list[Candidate] = []
    per_program: Counter[str] = Counter()
    per_weak: Counter[str] = Counter()
    gitlab_count = 0
    for c in order:
        if len(chosen) >= n:
            skipped["over_quota"] += 1
            continue
        r = c.row
        wkey = (r.cwe or r.weakness).lower()
        if not c.pinned:
            if r.source == "gitlab" and gitlab_count >= settings.max_gitlab:
                skipped["cap_gitlab"] += 1
                continue
            if r.source != "gitlab" and r.program and per_program[r.program] >= settings.per_program:
                skipped["cap_program"] += 1
                continue
            if wkey and per_weak[wkey] >= settings.per_weakness:
                skipped["cap_weakness"] += 1
                continue
        chosen.append(c)
        gitlab_count += r.source == "gitlab"
        if r.source != "gitlab" and r.program:
            per_program[r.program] += 1
        if wkey:
            per_weak[wkey] += 1
    chosen.sort(key=lambda c: c.row.report_id)
    rows = [c.row for c in chosen]

    summary = {
        "requested": n, "selected": len(rows), "shortfall": max(0, n - len(rows)),
        "pinned_selected": sum(c.pinned for c in chosen), "sources_in": len(sources),
        "by_source": dict(Counter(r.source for r in rows)), "by_program": dict(Counter(r.program or "(unknown)" for r in rows)),
        "by_weakness": dict(Counter((r.cwe or r.weakness or "(unknown)") for r in rows)),
        "flags": dict(Counter(f for r in rows for f in r.flags)),
        "rejected": dict(rejected), "skipped": dict(skipped),
        "injection": {"reports_with_removals": sum(1 for r in rows if r.injection_hits), "removals": sum(r.injection_hits for r in rows),
                      "left_in_code_blocks": stats["injection_in_code"]},
        "reports_with_images": sum(1 for r in rows if r.image_list), "comments_kept": stats["comments_kept"],
        "notes": notes, "report_ids": [r.report_id for r in rows],
    }
    return rows, summary


def read_urls(path: Path) -> list[str]:
    if not Path(path).is_file():
        return []
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8-sig").splitlines() if ln.strip() and not ln.strip().startswith("#")]


def load_sources(cache_dir: Path, urls_file: Path) -> tuple[list[SourceReport], set[int], list[str]]:
    """Read every cached download: (source reports, ids of the user's pinned sample links, notes about anything unusable)."""
    sources: list[SourceReport] = []
    notes: list[str] = []
    for path in sorted((Path(cache_dir) / "h1").glob("*.json")):
        sr = h1_api.to_source_report(json.loads(path.read_text(encoding="utf-8")))
        if sr is None:
            notes.append(f"HackerOne report {path.stem}: the API returned no report text, so it was skipped")
        else:
            sources.append(sr)
    by_cache_name: dict[str, int] = {}
    for path in sorted((Path(cache_dir) / "gitlab").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        try:
            sr = gitlab_src.to_source_report(record["issue"], record.get("notes") or [], comments_skipped=not record.get("notes"))
        except gitlab_src.GitLabError as exc:
            notes.append(f"GitLab {path.stem}: {exc}")
            continue
        sources.append(sr)
        by_cache_name[path.stem] = sr.report_id
    pinned: set[int] = set()
    for url in read_urls(urls_file):
        try:
            project, iid = gitlab_src.parse_issue_url(url)
        except gitlab_src.GitLabError as exc:
            notes.append(f"pinned link ignored: {exc}")
            continue
        rid = by_cache_name.get(f"{project.replace('/', '__')}__{iid}")
        if rid is None:
            notes.append(f"pinned link not downloaded yet (run `gitlab fetch`): {url}")
        else:
            pinned.add(rid)
    return sources, pinned, notes
