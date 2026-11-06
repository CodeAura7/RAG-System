"""OFFLINE profile of pages that `firecrawl_probe --run` already cached. Costs no credits, makes no request, needs no key.

It answers: where does a page's text sit - report body, page furniture, or something else - and WHY is the share of the dataset's text
found on the page low (code and URLs rendered differently, or prose really missing)? It also shows where "steps" and "activity" signals
come from, section by section. It prints numbers and short section headings only - never report text.

Usage (from the project folder):   python -m tools.firecrawl_pagemap 783392 135946 174470
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from hrag.clean import clean_report
from tools import firecrawl_probe as fp

_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}\s+{_MONTH},?\s+\d{{4}}\b|\b{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}\b|"
                   r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b", re.I)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.M)
_WINDOW = 5
FOUND_AS_DATASET = 0.6          # a block with at least this share of its 5-word groups in the dataset text counts as "dataset text"
MIN_ENTRY_WORDS = 12            # new text of at least this many words after a dated line = a discussion entry
CATEGORIES = ("prose", "code", "url", "image")


def blocks(markdown: str) -> list[dict[str, str]]:
    """Paragraph-like blocks. A blank line ends a block except inside a code fence; a heading line is a block of its own."""
    out: list[dict[str, str]] = []
    cur: list[str] = []
    heading, incode = "", False

    def flush() -> None:
        if cur and "".join(cur).strip():
            out.append({"heading": heading, "text": "\n".join(cur).strip("\n")})
        cur.clear()

    for line in (markdown or "").replace("\r\n", "\n").split("\n"):
        if line.strip().startswith(("```", "~~~")):
            incode = not incode
            cur.append(line)
            continue
        m = None if incode else _HEAD.match(line)
        if m:
            flush()
            heading = m.group(2)[:60]
            out.append({"heading": heading, "text": line.strip()})
        elif not incode and not line.strip():
            flush()
        else:
            cur.append(line)
    flush()
    return out


def _dataset_windows(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Every 5-word group of the dataset text, labelled by the kind of line its words sit in."""
    words: list[tuple[str, str]] = []
    incode = False
    for ln in text.split("\n"):
        if ln.strip().startswith(("```", "~~~")):
            incode = not incode
            continue
        cat = "code" if incode else "image" if "![" in ln else "url" if re.search(r"https?://|www\.", ln) else "prose"
        words += [(w, cat) for w in fp._WORDS.findall(ln.lower())]
    if not words:
        return []
    spans = [words[i:i + _WINDOW] for i in range(len(words) - _WINDOW + 1)] if len(words) >= _WINDOW else [words]
    out = []
    for span in spans:
        cats = [c for _, c in span if c != "prose"]
        out.append((tuple(w for w, _ in span), Counter(cats).most_common(1)[0][0] if cats else "prose"))
    return out


def analyse(dataset_body: str | None, page_markdown: str) -> dict[str, Any]:
    page_clean = clean_report(page_markdown or "").text
    page_sh = fp._shingles(page_clean)
    data_text = clean_report(dataset_body or "").text
    windows = _dataset_windows(data_text)
    data_sh = {w for w, _ in windows}
    coverage = round(len(data_sh & page_sh) / len(data_sh), 3) if data_sh else None
    missing = {c: 0 for c in CATEGORIES}
    for w, cat in windows:
        if w not in page_sh:
            missing[cat] += 1

    words_by_kind: Counter[str] = Counter({k: 0 for k in ("dataset_text", "chrome_like", "other_text", "heading")})
    blocks_by_kind: Counter[str] = Counter({k: 0 for k in ("dataset_text", "chrome_like", "other_text", "heading")})
    sections: list[dict[str, Any]] = []
    entries = dated = 0
    parsed = blocks(page_markdown)
    kinds: list[str] = []
    for b in parsed:
        text = b["text"]
        n = len(fp._WORDS.findall(text.lower()))
        if _HEAD.match(text.split("\n")[0]) and "\n" not in text:
            kind = "heading"
        else:
            sh = fp._shingles(text) if n >= _WINDOW else set()
            found = len(sh & data_sh) / len(sh) if sh else None
            link_share = len("".join(_LINK.findall(text))) / max(1, len(text))
            kind = ("dataset_text" if found is not None and found >= FOUND_AS_DATASET
                    else "chrome_like" if link_share >= 0.5 or n <= 6 else "other_text")
        kinds.append(kind)
        words_by_kind[kind] += n
        blocks_by_kind[kind] += 1
        label = b["heading"] or "(top of page)"
        if not sections or sections[-1]["heading"] != label:
            sections.append({"heading": label, "words": 0, "dataset_text_words": 0, "numbered_lines": 0, "activity_markers": 0})
        sections[-1]["words"] += n
        sections[-1]["numbered_lines"] += len(_NUMBERED.findall(text))
        sections[-1]["activity_markers"] += len(fp._ACTIVITY.findall(text))
        if kind == "dataset_text":
            sections[-1]["dataset_text_words"] += n
    for i, b in enumerate(parsed):
        if kinds[i] == "dataset_text" or kinds[i] == "heading" or not _DATE.search(b["text"]) or len(fp._WORDS.findall(b["text"].lower())) < 4:
            continue
        dated += 1
        for j in (i, i + 1, i + 2):                       # the dated line itself, or the next two blocks, hold the entry's text
            if j < len(parsed) and kinds[j] == "other_text" and len(fp._WORDS.findall(parsed[j]["text"].lower())) >= MIN_ENTRY_WORDS:
                entries += 1
                break
    return {"page_chars": len(page_clean), "page_words": sum(words_by_kind.values()), "dataset_words": len(fp._WORDS.findall(data_text.lower())),
            "coverage": coverage, "missing_by_category": missing, "words_by_kind": dict(words_by_kind), "blocks_by_kind": dict(blocks_by_kind),
            "dated_blocks": dated, "discussion_entries": entries, "sections": sections[:40]}


def main(argv: Sequence[str] | None = None, *, hf_loader: Callable[[list[int]], dict[int, str]] | None = None,
         out: Callable[[str], None] = print) -> int:
    p = argparse.ArgumentParser(prog="firecrawl_pagemap", description=__doc__.split("\n\n")[0])
    p.add_argument("reports", nargs="+", help="report ids already fetched by firecrawl_probe --run")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--cache-dir", help="the probe's cache folder (default: <cache_dir>/firecrawl_probe)")
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)
    if not all(a.strip().isdigit() for a in args.reports):
        out("error: give report ids (numbers) only")
        return 2
    ids = list(dict.fromkeys(int(a) for a in args.reports))
    cache = Path(args.cache_dir) if args.cache_dir else fp._default_cache_dir(args.config)
    pages: dict[int, str] = {}
    rc = 0
    for rid in ids:
        path = cache / f"{rid}.json"
        if path.is_file():
            pages[rid] = str(json.loads(path.read_text(encoding="utf-8")).get("markdown") or "")
        else:
            out(f"report {rid}: no cached page in {cache} (run firecrawl_probe --run first)")
            rc = 1
    if not pages:
        return rc
    try:
        dataset = (hf_loader or fp._default_hf_loader(args.config))(list(pages))
    except Exception as exc:  # noqa: BLE001
        out(f"note: the Hugging Face dataset could not be read ({type(exc).__name__}); coverage skipped")
        dataset = {}
    results = []
    for rid, md in pages.items():
        r = {"report_id": rid, **analyse(dataset.get(rid), md)}
        results.append(r)
        cov = "n/a" if r["coverage"] is None else f"{100 * r['coverage']:.0f}%"
        wk, mb = r["words_by_kind"], r["missing_by_category"]
        out(f"report {rid}: page {r['page_chars']} chars, {r['page_words']} words; dataset text found on the page: {cov}")
        out(f"  words by kind: dataset text {wk['dataset_text']}, page furniture {wk['chrome_like']}, other text {wk['other_text']}, headings {wk['heading']}")
        out(f"  dataset 5-word groups NOT on the page, by the kind of line they came from: prose {mb['prose']}, code {mb['code']}, "
            f"url {mb['url']}, image {mb['image']}")
        out(f"  dated blocks: {r['dated_blocks']}; discussion entries (a dated line followed by >= {MIN_ENTRY_WORDS} words of new text): {r['discussion_entries']}")
        out("  sections (heading: words / of which dataset text / numbered lines / activity phrases):")
        for s in r["sections"]:
            out(f"    {s['heading']}: words {s['words']}, dataset text {s['dataset_text_words']}, numbered {s['numbered_lines']}, "
                f"activity {s['activity_markers']}")
    (cache / "pagemap.json").write_text(json.dumps({"reports": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    out(f"saved: {cache / 'pagemap.json'} (numbers and short headings only)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
