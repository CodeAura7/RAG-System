"""Draft evaluation questions with an LLM so a person can review and fix them.

"You write half, AI drafts the rest": every drafted question is written with needs_review=true, and the
evaluation commands refuse to run on unreviewed questions. A human removes the flag after reading each one
(and checking the gold report ids and key facts).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .answer import Completer, parse_json_object
from .store import Store

CVE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)
SELF_REFERENCE = re.compile(r"\b(this|the above|the following) (report|excerpt|passage|text)\b|\bin the (report|excerpt)\b", re.I)
KIND_BY_SECTION = {"steps": "procedural", "impact": "conceptual", "fix": "conceptual"}

DRAFT_SYSTEM = ("You write evaluation questions for a retrieval system over public HackerOne vulnerability reports. "
                "Reply with one JSON object and nothing else.")
UNANSWERABLE_SYSTEM = ("You write questions that a retrieval system over a FIXED set of HackerOne reports must refuse to answer, "
                       "because the reports do not cover them. Reply with one JSON object and nothing else.")


def _draft_one(llm: Completer, header: str, text: str, kind: str) -> dict[str, Any] | None:
    user = (f"Excerpt from one report ({header}):\n---\n{text[:3500]}\n---\n"
            f"Write ONE question a security engineer could ask whose answer is stated in this excerpt. Kind: {kind}.\n"
            "It must make sense WITHOUT seeing the excerpt: never say 'this report', 'the excerpt' or 'above' - name the product, "
            "vulnerability class or technique instead. Then list 2-4 short key facts a correct answer must contain.\n"
            'JSON: {"question": string, "key_facts": [string]}')
    data = parse_json_object(llm.complete(DRAFT_SYSTEM, user))
    if not data or not isinstance(data.get("question"), str) or not isinstance(data.get("key_facts"), list):
        return None
    return {"question": data["question"].strip(), "key_facts": [str(f).strip() for f in data["key_facts"] if str(f).strip()]}


def draft_questions(store: Store, chunk_set: str, llm: Completer, *, n_answerable: int, n_unanswerable: int,
                    n_identifier: int = 3, seed: int = 42, model_name: str = "") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chunks = store.chunks(chunk_set)
    if not chunks:
        raise ValueError(f"chunk set '{chunk_set}' is empty")
    reports = store.reports()
    items: list[dict[str, Any]] = []

    def add(kind: str, question: str, gold: list[int], facts: list[str], answerable: bool = True) -> None:
        items.append({"id": f"q{len(items) + 1:03d}", "question": question, "type": kind, "answerable": answerable,
                      "gold_report_ids": gold, "key_facts": facts, "reference_answer": "", "needs_review": True,
                      "drafted_by": model_name or "llm"})

    # exact-identifier questions are built from a template (the identifier is the point, not the wording)
    with_cve = [c for c in chunks if CVE.search(c.text)]
    rng.shuffle(with_cve)
    used_reports: set[int] = set()
    for c in with_cve:
        if len([i for i in items if i["type"] == "exact_identifier"]) >= n_identifier:
            break
        if c.report_id in used_reports:
            continue
        used_reports.add(c.report_id)
        add("exact_identifier", f"What is the vulnerability described in the report that mentions {CVE.search(c.text).group(0).upper()}?",
            [c.report_id], ["<fill in the key facts>"])

    pool = list(chunks)
    rng.shuffle(pool)
    target = n_answerable - len(items)
    for c in pool:
        if target <= 0:
            break
        if c.report_id in used_reports:
            continue
        drafted = _draft_one(llm, c.header, c.text, KIND_BY_SECTION.get(c.section, "lookup"))
        if not drafted or SELF_REFERENCE.search(drafted["question"]) or len(drafted["question"]) < 15:
            continue
        used_reports.add(c.report_id)
        add(KIND_BY_SECTION.get(c.section, "lookup"), drafted["question"], [c.report_id], drafted["key_facts"])
        target -= 1

    if n_unanswerable > 0:
        covered = sorted({r.program for r in reports if r.program} | {r.weakness for r in reports if r.weakness})[:80]
        user = (f"The reports cover these programs and weakness types: {', '.join(covered)}.\n"
                f"Write {n_unanswerable} realistic, SPECIFIC security questions (name a product, CVE, or technique) that these reports "
                'almost certainly do NOT answer. JSON: {"questions": [string]}')
        data = parse_json_object(llm.complete(UNANSWERABLE_SYSTEM, user))
        for text in (data.get("questions", []) if data else [])[:n_unanswerable]:
            if isinstance(text, str) and text.strip():
                add("unanswerable", text.strip(), [], [], answerable=False)
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
