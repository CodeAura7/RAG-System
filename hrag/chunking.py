"""Structure-aware chunking for HackerOne reports.

Goals, in priority order:
  1. never cut a fenced code block (PoC / request / payload) in the middle unless one block
     alone is larger than the budget - and then every piece is re-fenced so it stays valid;
  2. keep a section heading with the text under it, and prefer to break BETWEEN sections;
  3. keep short reports whole (the common case: most reports are a few hundred tokens);
  4. overlap only prose-to-prose breaks (copying half a code block helps nobody).

Token counts are ESTIMATES (characters / 3.6): good enough to size chunks, not to bill.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

CHARS_PER_TOKEN = 3.6

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_ATX = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_SETEXT_UNDER = re.compile(r"^\s{0,3}(=+|-{3,})\s*$")
_BOLD = re.compile(r"^\s*(?:\*\*|__)\s*(.{2,60}?)\s*(?:\*\*|__)\s*:?\s*$")
_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /&'_()-]{1,58}?)\s*:\s*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
# A text line that introduces a request / command / payload shown right below it.
_INTRODUCES_CODE = re.compile(
    r":\s*$|\b(request|payload|command|curl|following|send|sent|run|execute|exploit|poc|script|query|body|response|output|example)\b", re.I)

_SECTION_KEYWORDS = (
    ("steps", ("steps to reproduce", "reproduc", "poc", "proof of concept", "proof-of-concept", "exploit", "how to", "steps")),
    ("impact", ("impact", "risk", "severity", "consequence")),
    ("fix", ("fix", "mitigation", "remediation", "recommend", "solution", "patch", "workaround")),
    ("summary", ("summary", "description", "overview", "vulnerability", "details", "background", "issue", "introduction")),
)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1 if text else 0


def classify_section(label: str) -> str:
    lowered = label.lower()
    for name, words in _SECTION_KEYWORDS:
        if any(w in lowered for w in words):
            return name
    return "other"


@dataclass
class Unit:
    kind: str            # 'heading' | 'text' | 'code'
    text: str
    section: str         # classified type of the section this unit belongs to
    sec_idx: int         # which section (a heading starts a new one)
    tokens: int = 0

    def __post_init__(self) -> None:
        # +1 pays for the blank line chunks are joined with, so a chunk's real size stays inside the budget
        self.tokens = estimate_tokens(self.text) + 1 if self.text else 0


@dataclass
class ChunkDraft:
    ordinal: int
    section: str
    text: str
    has_code: bool
    n_tokens: int


def _glue_steps(units: list[Unit], max_tokens: int) -> list[Unit]:
    """Join a step that introduces a request/command/payload with the code block right below it, so a chunk boundary can
    never separate "send this request:" from the request itself. Skipped when the pair would not fit one chunk."""
    out: list[Unit] = []
    i = 0
    while i < len(units):
        u = units[i]
        nxt = units[i + 1] if i + 1 < len(units) else None
        if (nxt is not None and u.kind == "text" and nxt.kind == "code" and u.sec_idx == nxt.sec_idx
                and _INTRODUCES_CODE.search(u.text.rsplit("\n", 1)[-1]) and u.tokens + nxt.tokens <= max_tokens):
            out.append(Unit("stepcode", f"{u.text}\n{nxt.text}", u.section, u.sec_idx))
            i += 2
            continue
        out.append(u)
        i += 1
    return out


def parse_units(body: str) -> list[Unit]:
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    units: list[Unit] = []
    section, sec_idx = "body", 0
    para: list[str] = []
    n = len(lines)

    def flush_para() -> None:
        text = "\n".join(para).strip()
        if text:
            units.append(Unit("text", text, section, sec_idx))
        para.clear()

    def start_section(label: str) -> None:
        nonlocal section, sec_idx
        flush_para()
        clean = label.strip(" :*_#\t")
        section = classify_section(clean)
        sec_idx += 1
        units.append(Unit("heading", f"## {clean}", section, sec_idx))

    i = 0
    while i < n:
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:                                    # fenced code block: keep verbatim, to its closing fence
            flush_para()
            marker = fence.group(1)[0] * 3
            block = [line]
            i += 1
            while i < n:
                block.append(lines[i])
                closed = lines[i].strip().startswith(marker)
                i += 1
                if closed:
                    break
            units.append(Unit("code", "\n".join(block).rstrip(), section, sec_idx))
            continue
        stripped = line.strip()
        if not stripped:
            flush_para()
            i += 1
            continue
        atx = _ATX.match(line)
        if atx:
            start_section(atx.group(1))
            i += 1
            continue
        if not para and i + 1 < n and _SETEXT_UNDER.match(lines[i + 1]):
            start_section(stripped)
            i += 2
            continue
        if not para:
            bold = _BOLD.match(line)
            label = _LABEL.match(line)
            if bold or (label and len(label.group(1).split()) <= 6):
                start_section((bold or label).group(1))
                i += 1
                continue
        para.append(line)
        i += 1
    flush_para()
    return units


# ---------- splitting units that are bigger than one chunk ----------

def _pack(segments: list[str], budget: int) -> list[str]:
    pieces, cur, cur_tok = [], [], 0
    for seg in segments:
        t = estimate_tokens(seg) + 1
        if cur and cur_tok + t > budget:
            pieces.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(seg)
        cur_tok += t
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _hard_split(text: str, budget: int) -> list[str]:
    size = max(20, int(budget * CHARS_PER_TOKEN))
    return [text[i:i + size] for i in range(0, len(text), size)]


def _segments(text: str, budget: int) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        if estimate_tokens(line) <= budget:
            out.append(line)
            continue
        for sentence in _SENTENCE_BREAK.split(line):
            out.extend([sentence] if estimate_tokens(sentence) <= budget else _hard_split(sentence, budget))
    return out


def _split_code(unit: Unit, max_tokens: int) -> list[Unit]:
    lines = unit.text.split("\n")
    opening = lines[0]
    marker = _FENCE.match(opening).group(1)[0] * 3 if _FENCE.match(opening) else "```"
    closed = len(lines) > 1 and lines[-1].strip().startswith(marker)
    content = lines[1:-1] if closed else lines[1:]
    budget = max(20, max_tokens - estimate_tokens(opening + marker) - 2)
    segments: list[str] = []
    for line in content:
        segments.extend([line] if estimate_tokens(line) <= budget else _hard_split(line, budget))
    return [Unit("code", f"{opening}\n{piece}\n{marker}", unit.section, unit.sec_idx) for piece in _pack(segments, budget)]


def _split_oversize(units: list[Unit], max_tokens: int, overlap_tokens: int = 0) -> list[Unit]:
    # Prose pieces leave room for the overlap that will be prepended to the next piece.
    prose_budget = max_tokens - overlap_tokens if 0 < overlap_tokens < max_tokens // 2 else max_tokens
    out: list[Unit] = []
    for u in units:
        if u.tokens <= max_tokens:
            out.append(u)
        elif u.kind == "code":
            out.extend(_split_code(u, max_tokens))
        else:
            out.extend(Unit("text", piece, u.section, u.sec_idx)
                       for piece in _pack(_segments(u.text, prose_budget), prose_budget))
    return out


def _tail(text: str, overlap_tokens: int) -> str:
    """The last few sentences of `text`, up to about overlap_tokens."""
    if overlap_tokens <= 0:
        return ""
    picked: list[str] = []
    used = 0
    for sentence in reversed(_SENTENCE_BREAK.split(text.strip())):
        t = estimate_tokens(sentence)
        if picked and used + t > overlap_tokens:
            break
        if not picked and t > overlap_tokens:
            return text[-int(overlap_tokens * CHARS_PER_TOKEN):].lstrip()
        picked.insert(0, sentence)
        used += t
    return " ".join(picked)


def _primary_section(group: list[Unit]) -> str:
    weight: dict[str, int] = defaultdict(int)
    for u in group:
        if u.kind != "heading":
            weight[u.section] += u.tokens
    return max(weight, key=weight.get) if weight else (group[0].section if group else "body")


# ---------- the public entry point ----------

def chunk_report(body: str, *, max_tokens: int, overlap_tokens: int = 0, min_tokens: int = 0) -> list[ChunkDraft]:
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        return []
    units = parse_units(body)
    if not units:
        return []
    if estimate_tokens(body) <= max_tokens:          # the common case: keep the report whole
        return [ChunkDraft(0, _primary_section(units), body, any(u.kind == "code" for u in units), estimate_tokens(body))]

    units = _split_oversize(_glue_steps(units, max_tokens), max_tokens, overlap_tokens)
    sec_tokens: dict[int, int] = defaultdict(int)
    for u in units:
        sec_tokens[u.sec_idx] += u.tokens

    groups: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_tok = 0

    def flush(next_unit: Unit) -> None:
        """Close the current chunk. A trailing heading moves to the next chunk (never end on a heading)."""
        nonlocal cur, cur_tok
        carry: list[Unit] = []
        while cur and cur[-1].kind == "heading":
            carry.insert(0, cur.pop())
        if cur:
            groups.append(cur)
            last = cur[-1]
            if (overlap_tokens > 0 and last.kind == "text" and next_unit.kind == "text" and not carry
                    and last.sec_idx == next_unit.sec_idx):
                tail = _tail(last.text, overlap_tokens)
                if tail and estimate_tokens(tail) + next_unit.tokens <= max_tokens:
                    carry = [Unit("text", tail, last.section, last.sec_idx)]
        cur = carry
        cur_tok = sum(u.tokens for u in carry)

    for u in units:
        if cur and cur_tok + u.tokens > max_tokens:
            flush(u)
        elif (cur and u.kind == "heading" and cur_tok >= min_tokens
              and sec_tokens[u.sec_idx] <= max_tokens and cur_tok + sec_tokens[u.sec_idx] > max_tokens):
            flush(u)                                  # prefer breaking between sections
        cur.append(u)
        cur_tok += u.tokens
    if cur:
        groups.append(cur)

    drafts = [(g, "\n\n".join(u.text for u in g)) for g in groups]
    if len(drafts) > 1 and estimate_tokens(drafts[-1][1]) < min_tokens:       # fold a tiny last chunk back
        prev_group, prev_text = drafts[-2]
        last_group, last_text = drafts[-1]
        if estimate_tokens(prev_text + "\n\n" + last_text) <= max_tokens * 1.1:   # soft limit: at most 10% over
            drafts[-2:] = [(prev_group + last_group, prev_text + "\n\n" + last_text)]

    return [ChunkDraft(i, _primary_section(g), text, any(u.kind in ("code", "stepcode") for u in g), estimate_tokens(text))
            for i, (g, text) in enumerate(drafts)]


def make_header(report_id: int, title: str, program: str, weakness: str, substate: str, section: str, *,
                severity: str = "", when_to_try: str = "") -> str:
    """A short label put in front of every chunk so it makes sense on its own (and is embedded with it)."""
    parts = [f"HackerOne report #{report_id}: {title}"]
    if program:
        parts.append(f"program: {program}")
    if weakness:
        parts.append(f"weakness: {weakness}")
    if severity:
        parts.append(f"severity: {severity}")
    if substate:
        parts.append(f"outcome: {substate}")
    parts.append(f"section: {section}")
    header = " | ".join(parts)
    return f"{header}\nWhen to try this: {when_to_try}" if when_to_try else header
