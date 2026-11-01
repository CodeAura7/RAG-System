"""Turn the raw text of a report (a HackerOne API body or a GitLab copy of one) into clean, safe, retrieval-ready text.

What it does, in order:
  * GitLab copies: read the `HackerOne report #N` header, drop GitLab's own boilerplate (attachment list, empty
    "How To Reproduce" template, related-issue links, implementation plan, environment-info dumps, quick actions);
  * decode HTML entities (GitLab copies only - they are an export artefact) and strip hidden/invisible characters;
  * remove instructions that are addressed to AI tools from the PROSE (prompt-injection defence). The same phrases inside
    code blocks are kept - a report about prompt injection needs its PoC payload - but they are counted and flagged;
  * replace images/GIFs/videos with a short note and record them (flag only - nothing is downloaded or described);
  * normalise placeholders and secrets (`Cookie: <COOKIE>`, `YOUR_TOKEN` -> `<TOKEN>`, JWTs, emails ...) while leaving
    endpoints, parameter names and payloads such as `<script>` untouched.

Code blocks are lifted out and processed on their own, so a prose rule can never damage a PoC. Cleaning is idempotent.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

# ----------------------------------------------------------------------------------------------- patterns
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_H1_HEADER = re.compile(r"\*{0,2}\[\*{0,2}HackerOne report #(\d+)\*{0,2}\]\(([^)]+)\)\*{0,2}\s+by\s+`([^`]+)`\s+on\s+(\d{4}-\d{2}-\d{2})")
H1_HEADER = _H1_HEADER                     # public alias: the GitLab-copy header line
_TOC = re.compile(r"^\s*\[Report\]\(#report\).*$")
_QUICK_ACTION = re.compile(r"^/(label|unlabel|assign|unassign|reassign|cc|milestone|confidential|due|relate|title|epic|weight|estimate|spend|subscribe|shrug|tableflip)\b.*$", re.M)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_INLINE_CODE = re.compile(r"(`+)(.+?)\1", re.S)
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")

DROP_H2 = {"attachments", "related issue", "related issues", "implementation plan"}
DROP_ANY_LEVEL = (re.compile(r"results of gitlab environment info"), re.compile(r"^gitlab environment info"))
_TEMPLATE_LINE = (re.compile(r"^\s*please add \[?reproducibility information\]?(\([^)]*\))? to this section:?\s*$", re.I),
                  re.compile(r"^\s*\d+[.)]\s*$"), re.compile(r"^\s*\[reproducibility information\]:.*$", re.I))
HEADING_RENAMES = ((re.compile(r"^what is the current .*behaviou?r\??$", re.I), "Current behavior"),
                   (re.compile(r"^what is the expected .*behaviou?r\??$", re.I), "Expected behavior"),
                   (re.compile(r"^relevant logs and/or screenshots$", re.I), "Logs and screenshots"),
                   (re.compile(r"^output of checks$", re.I), "Environment"))

VIDEO_EXT = {"mp4", "webm", "mov", "avi", "mkv"}
IMAGE_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "svg"}
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)(?:\{[^}]*\})?")
_HTML_IMAGE = re.compile(r"<(?:img|video)\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>(?:\s*</video>)?", re.I)
_MEDIA_LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+?\.(?:%s))(?:\?[^)\s]*)?\s*\)" % "|".join(sorted(VIDEO_EXT | IMAGE_EXT)), re.I)
_ANY_LINK = re.compile(r"\[([^\]]+)\]\(\s*([^)\s]+)\s*\)")

# Instructions aimed at AI tools. Verb lists are kept narrow on purpose: "override all rules in the WAF" is a normal sentence.
_INJECTION = [re.compile(p, re.I) for p in (
    r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|all|any|your|system|the)\b[^.\n]{0,30}\b(instructions?|prompts?|guidelines|directions)\b",
    r"\b(note|message|instructions?|attention|important)\s+(to|for)\s+(the\s+)?(ai|llm|assistant|language model|model|chatbot|agent|bot|automated)\b",
    r"\b(if|when)\s+you\s+are\s+(an?\s+)?(ai|llm|language model|assistant|chatgpt|gpt|claude|bot|automated)\b",
    r"\bas\s+an?\s+(ai|llm|language model)\b",
    r"\byou\s+(must|should|will|are required to|need to)\b[^.\n]{0,60}\b(mark|rate|classify|label|score|approve|accept|triage|treat)\b[^.\n]{0,40}\b(this|the)\s+(report|finding|issue|submission)\b",
    r"\b(mark|rate|classify|label|score|approve|accept|treat)\s+(this|the)\s+(report|finding|issue|submission)\s+(as\s+)?(critical|high|valid|resolved|accepted|severe|p1)\b",
    r"\b(reveal|print|show|output|repeat|leak)\b[^.\n]{0,30}\b(system prompt|hidden prompt|initial instructions|your instructions)\b",
    r"\bnew\s+(system\s+)?instructions?\s*:",
    r"<\|(?:im_start|im_end|system|user|assistant)\|>|\[/?INST\]|<<SYS>>|^\s*(?:system|assistant)\s*:\s*(?:you are|you must|ignore|from now on|your (?:task|role))",
    r"\bdo\s+not\s+(tell|inform|mention|reveal|alert)\b[^.\n]{0,40}\b(the\s+)?(user|human|operator|security team|triager)\b",
)]
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Placeholders / secrets
_HEADER_RULES = (
    (re.compile(r"(?im)^(\s*Cookie\s*:\s*).+$"), r"\1<COOKIE>"),
    (re.compile(r"(?im)^(\s*Set-Cookie\s*:\s*[^=\s;]+=)[^;\r\n]+"), r"\1<COOKIE>"),
    (re.compile(r"(?im)^(\s*Authorization\s*:\s*(?:Bearer|Basic|Token|Digest|Negotiate|OAuth)\s+)\S+"), r"\1<TOKEN>"),
    (re.compile(r"(?im)^(\s*Authorization\s*:\s*)(?!(?:Bearer|Basic|Token|Digest|Negotiate|OAuth)\b)(?!<)\S.*$"), r"\1<TOKEN>"),
    (re.compile(r"(?im)^(\s*(?:PRIVATE-TOKEN|JOB-TOKEN|X-Api-Key|X-Auth-Token|X-Access-Token|X-GitLab-Token)\s*:\s*)\S+"), r"\1<TOKEN>"),
    (re.compile(r"(?im)^(\s*(?:X-CSRF-Token|X-XSRF-TOKEN|CSRF-Token|X-CSRFToken)\s*:\s*)\S+"), r"\1<CSRF_TOKEN>"),
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*")
_TOKEN_PREFIXES = re.compile(r"\b(?:glpat-[\w-]{10,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[\w-]{10,}|sk-[A-Za-z0-9]{20,}|AIza[\w-]{30,})")
_SECRET_PARAM = re.compile(r"\b([A-Za-z_]*(?:token|secret|session|passw(?:or)?d|apikey|api_key|signature|csrf|auth)[A-Za-z_]*)=([A-Za-z0-9%/+_.~=-]{16,})", re.I)
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
_SAFE_EMAIL_DOMAIN = re.compile(r"(^|\.)(example\.(com|org|net)|example|test|localhost|invalid|test\.com|domain\.com|attacker\.com|victim\.com|target\.com|email\.com)$", re.I)
_YOUR = re.compile(r"\bYOUR[_-]([A-Z0-9][A-Z0-9_-]*)\b")
_SECRETISH = r"(?:[\w-]*(?:TOKEN|SESSION|COOKIE|SECRET|PASSWORD|JWT|CSRF|DOMAIN|EMAIL|HOST|URL|USER|API_?KEY)[\w-]*)"
_MUSTACHE = re.compile(r"\{\{\s*(%s)\s*\}\}" % _SECRETISH, re.I)
_SHELLVAR = re.compile(r"\$\{(%s)\}" % _SECRETISH)
_REDACTED = re.compile(r"\[redacted\]|<redacted>|█{2,}", re.I)
_ANGLE = re.compile(r"<([A-Za-z][A-Za-z0-9_ -]{1,38})>")
_HTML_TAGS = frozenset("""a abbr address area article aside audio b base bdi bdo blockquote body br button canvas caption cite code col colgroup
data datalist dd del details dfn dialog div dl dt em embed fieldset figure footer form h1 h2 h3 h4 h5 h6 head header hr html i iframe img input
ins kbd label legend li link main map mark meta meter nav noscript object ol optgroup option output p param path picture pre progress q rp rt
ruby s samp script section select slot small source span strong style sub summary sup svg table tbody td template textarea tfoot th thead time
title tr track u ul var video wbr xml""".split())
_CODE_ANGLE = frozenset({"token", "host", "hostname", "gitlabhost", "csrf_token", "jwt", "cookie", "session_id", "session", "api_key",
                         "access_token", "auth_token", "victim", "attacker", "redacted", "jwt_token", "bearer_token", "email"})
_PROSE_ANGLE_WORDS = frozenset({"token", "host", "hostname", "url", "domain", "username", "user", "password", "cookie", "session", "secret",
                                "key", "email", "jwt", "csrf", "instance", "ip", "id", "project", "group", "victim", "attacker", "target",
                                "payload", "redacted", "uuid", "endpoint", "server", "name", "value", "gitlabhost"})
_PROSE_ANGLE_SUFFIX = ("host", "token", "url", "domain")


@dataclass
class CleanResult:
    text: str
    h1: dict = field(default_factory=dict)              # report_id/url/reporter/date from a GitLab copy's header line
    images: list[dict] = field(default_factory=list)    # [{"kind": image|gif|video, "ref": file name}]
    injection_hits: list[str] = field(default_factory=list)   # removed from prose (short snippets)
    injection_in_code: int = 0                          # AI-directed phrases left in code blocks (quoted PoC payloads)
    dropped_sections: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------- helpers
def _media_kind(name: str) -> str | None:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "gif":
        return "gif"
    if ext in VIDEO_EXT:
        return "video"
    return "image" if ext in IMAGE_EXT else None


def _ref_name(url: str, alt: str = "") -> str:
    name = unquote(urlparse(url).path.rsplit("/", 1)[-1])
    return name or alt or "image"


def _add_media(images: list[dict], kind: str, ref: str) -> None:
    item = {"kind": kind, "ref": ref}
    if item not in images:
        images.append(item)


def _is_injection(sentence: str) -> tuple[bool, bool]:
    """(matches an AI-directed pattern, the match is quoted - i.e. described rather than addressed to a reader)."""
    for pat in _INJECTION:
        m = pat.search(sentence)
        if m:
            before = sentence[:m.start()].rstrip()
            return True, bool(before) and before[-1] in "\"'“‘"
    return False, False


def _extract_code_blocks(text: str) -> tuple[str, list[list[str]]]:
    lines, out, blocks = text.split("\n"), [], []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        marker = m.group(1)
        block = [lines[i]]
        i += 1
        while i < len(lines):
            block.append(lines[i])
            closing = lines[i].strip()
            i += 1
            if closing.startswith(marker[0] * len(marker)) and not closing.strip(marker[0]):
                break
        out.append(f"\x00C{len(blocks)}\x00")
        blocks.append(block)
    return "\n".join(out), blocks


def _heading_parts(line: str) -> tuple[int, str] | None:
    m = _HEADING.match(line)
    return (len(m.group(1)), m.group(2).strip()) if m else None


def _title_key(title: str) -> str:
    return title.strip().strip("*_: ").strip().lower().rstrip("?:").strip()


# ----------------------------------------------------------------------------------------------- placeholders
def normalize_placeholders(text: str, *, broad: bool) -> str:
    """Secrets and placeholder spellings -> one canonical `<UPPER_SNAKE>` form. `broad` is for prose; code uses a strict
    vocabulary so XML/HTML payloads (`<password>x</password>`, `<script>`) are never rewritten."""
    for pattern, repl in _HEADER_RULES:
        text = pattern.sub(repl, text)
    text = _JWT.sub("<JWT>", text)
    text = _TOKEN_PREFIXES.sub("<TOKEN>", text)
    text = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=<{m.group(1).upper()}>", text)

    def email(m: re.Match) -> str:
        return m.group(0) if m.group(1).lower() == "git" or _SAFE_EMAIL_DOMAIN.search(m.group(2)) else "<EMAIL>"
    text = _EMAIL.sub(email, text)
    text = _REDACTED.sub("<REDACTED>", text)
    text = _YOUR.sub(lambda m: "<" + re.sub(r"[_-]HERE$", "", m.group(1)).replace("-", "_") + ">", text)
    text = _MUSTACHE.sub(lambda m: "<" + m.group(1).upper().replace("-", "_") + ">", text)
    text = _SHELLVAR.sub(lambda m: "<" + m.group(1).upper().replace("-", "_") + ">", text)

    def angle(m: re.Match) -> str:
        inner = m.group(1)
        norm = re.sub(r"[\s-]+", "_", inner.strip()).lower()
        if norm in _HTML_TAGS or f"</{inner}>" in text:
            return m.group(0)
        if norm.startswith("your_"):
            norm = norm[5:]
            ok = True
        elif broad:
            ok = any(p in _PROSE_ANGLE_WORDS for p in norm.split("_")) or norm.endswith(_PROSE_ANGLE_SUFFIX)
        else:
            ok = norm in _CODE_ANGLE
        return f"<{norm.upper()}>" if ok and norm else m.group(0)
    return _ANGLE.sub(angle, text)


# ----------------------------------------------------------------------------------------------- sections
def _strip_repro_template(block: list[str]) -> list[str]:
    return [ln for ln in block if ln.strip() and not any(p.match(ln) for p in _TEMPLATE_LINE)]


def _drop_sections(text: str, result: CleanResult) -> str:
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        head = _heading_parts(lines[i])
        if not head:
            out.append(lines[i])
            i += 1
            continue
        level, title = head
        key = _title_key(title)
        end = i + 1
        while end < len(lines):
            nxt = _heading_parts(lines[end])
            if nxt and nxt[0] <= level:
                break
            end += 1
        block = lines[i + 1:end]
        if level == 2 and key == "report":                 # GitLab's wrapper heading: keep the content, lose the heading
            i += 1
            continue
        if level == 2 and key in DROP_H2:
            result.dropped_sections.append(title)
            if key == "attachments":
                for name, url in _ANY_LINK.findall("\n".join(block)):
                    ref = _ref_name(url, name)
                    kind = _media_kind(ref) or _media_kind(name)
                    if kind:
                        _add_media(result.images, kind, ref)
            i = end
            continue
        if level == 2 and key == "how to reproduce":
            real = _strip_repro_template(block)
            if sum(len(ln.strip()) for ln in real) < 20:
                result.dropped_sections.append(title)
            else:
                out.append(lines[i])
                out.extend(real)
            i = end
            continue
        if any(p.search(key) for p in DROP_ANY_LEVEL):
            result.dropped_sections.append(title)
            i = end
            continue
        renamed = next((new for pat, new in HEADING_RENAMES if pat.match(title)), title)
        out.append(f"{'#' * level} {renamed}")
        i += 1
    return "\n".join(out)


def _drop_empty_headings(text: str) -> str:
    lines = text.split("\n")
    changed = True
    while changed:
        changed = False
        for idx, line in enumerate(lines):
            head = _heading_parts(line)
            if not head:
                continue
            k = idx + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            nxt = _heading_parts(lines[k]) if k < len(lines) else None
            if k >= len(lines) or (nxt and nxt[0] <= head[0]):
                del lines[idx]
                changed = True
                break
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- the entry point
def clean_report(text: str, *, source: str = "") -> CleanResult:
    result = CleanResult(text="")
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if source == "gitlab":
        t = html.unescape(t)
    t, hidden = _INVISIBLE.subn("", t)
    if hidden:
        result.injection_hits.append(f"{hidden} hidden/invisible character(s) removed")

    t, codes = _extract_code_blocks(t)
    header = _H1_HEADER.search(t)
    if header:
        result.h1 = {"report_id": int(header.group(1)), "url": header.group(2), "reporter": header.group(3), "date": header.group(4)}
    t = "\n".join(ln for ln in t.split("\n") if not _H1_HEADER.search(ln) and not _TOC.match(ln))

    def comment(m: re.Match) -> str:
        if any(p.search(m.group(0)) for p in _INJECTION):
            result.injection_hits.append("hidden HTML comment: " + " ".join(m.group(0)[4:-3].split())[:80])
        return ""
    t = _HTML_COMMENT.sub(comment, t)

    inline: list[str] = []

    def keep_inline(m: re.Match) -> str:
        inline.append(m.group(0))
        return f"\x00I{len(inline) - 1}\x00"
    t = _INLINE_CODE.sub(keep_inline, t)

    t = _drop_sections(t, result)
    t = _QUICK_ACTION.sub("", t)

    def md_image(m: re.Match) -> str:
        ref = _ref_name(m.group(2), m.group(1))
        kind = _media_kind(ref) or "image"
        _add_media(result.images, kind, ref)
        return f"[{'video' if kind == 'video' else 'image'}: {ref}]"

    def html_image(m: re.Match) -> str:
        ref = _ref_name(m.group(1))
        kind = _media_kind(ref) or "image"
        _add_media(result.images, kind, ref)
        return f"[{'video' if kind == 'video' else 'image'}: {ref}]"

    def media_link(m: re.Match) -> str:
        ref = _ref_name(m.group(2), m.group(1))
        kind = _media_kind(ref) or "image"
        _add_media(result.images, kind, ref)
        return f"[{'video' if kind == 'video' else 'image'}: {ref}]"
    t = _MD_IMAGE.sub(md_image, t)
    t = _HTML_IMAGE.sub(html_image, t)
    t = _MEDIA_LINK.sub(media_link, t)

    kept_lines: list[str] = []
    for line in t.split("\n"):
        if _heading_parts(line) or not line.strip():
            kept_lines.append(line)
            continue
        kept: list[str] = []
        for sentence in _SENTENCE.split(line):
            hit, quoted = _is_injection(sentence)
            if hit and quoted:
                result.injection_in_code += 1
                kept.append(sentence)
            elif hit:
                result.injection_hits.append(" ".join(sentence.split())[:80])
            else:
                kept.append(sentence)
        if kept:
            kept_lines.append(" ".join(kept) if len(kept) > 1 else kept[0])
    t = "\n".join(kept_lines)

    t = _drop_empty_headings(t)
    t = normalize_placeholders(t, broad=True)
    t = re.sub(r"\\\n", "\n", t)                                   # markdown backslash line break
    t = "\n".join(ln.rstrip() for ln in t.split("\n"))              # markdown double-space hard breaks

    for i, code in enumerate(inline):
        t = t.replace(f"\x00I{i}\x00", normalize_placeholders(code, broad=False))
    for i, block in enumerate(codes):
        if f"\x00C{i}\x00" not in t:                                  # its section was dropped: nothing to count or restore
            continue
        body = "\n".join(ln.rstrip() for ln in block)
        result.injection_in_code += sum(1 for ln in block if _is_injection(ln)[0])
        t = t.replace(f"\x00C{i}\x00", normalize_placeholders(body, broad=False))
    t = re.sub(r"\x00[CI]\d+\x00", "", t)                           # tokens of dropped sections
    result.text = re.sub(r"\n{3,}", "\n\n", t).strip()
    return result


# ----------------------------------------------------------------------------------------------- comments
_BOT = re.compile(r"(^|[-_.])bot$|^gitlab-bot$|^project_\d+_bot", re.I)
_TECH_SIGNAL = re.compile(r"`|/\w+|\b(root cause|poc|reproduc\w*|fix(ed)?|patch(ed)?|confirm\w*|impact|vulnerab\w*|cve-\d|exploit\w*|bypass\w*|sanitiz\w*|validat\w*|mitigat\w*)\b", re.I)


def filter_comments(notes: list[dict], *, source: str = "gitlab", min_chars: int = 100) -> list[str]:
    """Keep substantive, technical, human comments (triage verdicts, root causes, fixes). Drops system notes, bots,
    thanks/status chatter and anything short. Each kept comment goes through the same cleaning as the report."""
    kept: list[str] = []
    for note in notes:
        author = note.get("author") or {}
        if note.get("system") or author.get("bot") or _BOT.search(str(author.get("username", ""))):
            continue
        body = clean_report(str(note.get("body") or ""), source=source).text
        if len(body) >= min_chars and _TECH_SIGNAL.search(body):
            kept.append(body)
    return kept
