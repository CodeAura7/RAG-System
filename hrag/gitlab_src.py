"""GitLab's public copies of HackerOne reports (issues carrying the `HackerOne` label in gitlab-org/gitlab).

Public issues can be read through GitLab's REST API without any account. The one thing that needs a token is the COMMENTS
of an issue (GitLab answers 401 without one), so comments are fetched only when GITLAB_TOKEN is configured (a read-only
personal access token). The token is sent only to gitlab.com and is never written to the cache.

Everything here is read-only, spaced out (`min_interval`) and cached on disk, so re-running costs no requests.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote, urlparse

import httpx

from .clean import H1_HEADER
from .sources import SourceReport

API_BASE = "https://gitlab.com/api/v4"
ALLOWED_HOSTS = {"gitlab.com"}
DEFAULT_PROJECT = "gitlab-org/gitlab"
_ISSUE_PATH = re.compile(r"^/(?P<project>.+?)/-/(?:issues|work_items)/(?P<iid>\d+)/?$")
_SEVERITY = {"1": "critical", "2": "high", "3": "medium", "4": "low"}

# Names for the CWE ids that occur most in web-security reports (kept to ones whose official title is certain).
CWE_NAMES = {
    20: "Improper Input Validation", 22: "Path Traversal", 74: "Injection", 77: "Command Injection", 78: "OS Command Injection",
    79: "Cross-site Scripting", 89: "SQL Injection", 94: "Code Injection", 113: "HTTP Response Splitting",
    200: "Exposure of Sensitive Information to an Unauthorized Actor", 209: "Information Exposure Through an Error Message",
    269: "Improper Privilege Management", 284: "Improper Access Control", 285: "Improper Authorization",
    287: "Improper Authentication", 306: "Missing Authentication for Critical Function",
    307: "Improper Restriction of Excessive Authentication Attempts", 319: "Cleartext Transmission of Sensitive Information",
    322: "Key Exchange without Entity Authentication", 346: "Origin Validation Error", 352: "Cross-Site Request Forgery",
    400: "Uncontrolled Resource Consumption", 434: "Unrestricted Upload of File with Dangerous Type",
    444: "HTTP Request Smuggling", 502: "Deserialization of Untrusted Data", 601: "Open Redirect",
    611: "XML External Entity Reference", 639: "Authorization Bypass Through User-Controlled Key",
    732: "Incorrect Permission Assignment for Critical Resource", 770: "Allocation of Resources Without Limits or Throttling",
    798: "Use of Hard-coded Credentials", 862: "Missing Authorization", 863: "Incorrect Authorization",
    918: "Server-Side Request Forgery", 1021: "Improper Restriction of Rendered UI Layers", 1333: "Inefficient Regular Expression Complexity",
}


class GitLabError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def parse_issue_url(url: str) -> tuple[str, int]:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ALLOWED_HOSTS:
        raise GitLabError(f"only issue links on gitlab.com are supported (got '{url[:80]}')")
    m = _ISSUE_PATH.match(parsed.path)
    if not m:
        raise GitLabError(f"not a GitLab issue link: {url[:80]}")
    return m.group("project"), int(m.group("iid"))


def weakness_from_labels(labels: list[str]) -> tuple[str, str]:
    """('CWE-918', 'CWE-918 (Server-Side Request Forgery)') from a `Weakness::CWE-918` label; ('', '') if none."""
    for label in labels:
        m = re.match(r"^Weakness::(CWE-(\d+))$", label)
        if m:
            name = CWE_NAMES.get(int(m.group(2)))
            return m.group(1), f"{m.group(1)} ({name})" if name else m.group(1)
    return "", ""


def severity_from_labels(labels: list[str]) -> str:
    for label in labels:
        m = re.match(r"^severity::([1-4])$", label)
        if m:
            return _SEVERITY[m.group(1)]
    return ""


def _h1_header(issue: dict) -> re.Match | None:
    return H1_HEADER.search(html.unescape(issue.get("description") or ""))


class GitLabClient:
    def __init__(self, *, base_url: str = API_BASE, client: httpx.Client | None = None, cache_dir: Path | None = None,
                 token: str | None = None, min_interval: float = 0.6, max_retries: int = 4, timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._token = token or None
        self.min_interval, self.max_retries = min_interval, max_retries
        self._sleep, self._clock = sleep, clock
        self._last: float | None = None

    # ---- transport ----
    def _throttle(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def _get(self, path: str, params: dict[str, Any] | None = None, *, authed: bool = False) -> tuple[Any, httpx.Headers]:
        headers = {"PRIVATE-TOKEN": self._token} if authed and self._token else {}
        last: GitLabError | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.http.get(self.base_url + path, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = GitLabError(f"network error: {type(exc).__name__}")
                self._sleep(min(30.0, 2.0 ** attempt))
                continue
            code = response.status_code
            if code < 300:
                try:
                    return response.json(), response.headers
                except ValueError as exc:
                    raise GitLabError("response was not valid JSON", code) from exc
            if code == 401 and authed:
                raise GitLabError("HTTP 401: GitLab rejected GITLAB_TOKEN (it needs the read_api scope)", code)
            if code in (401, 403, 404):
                raise GitLabError(f"HTTP {code}: not found or not public ({path.split('?')[0]})", code)
            last = GitLabError(f"HTTP {code}", code)
            if code != 429 and code < 500:
                raise last
            retry_after = response.headers.get("retry-after", "")
            self._sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(30.0, 2.0 ** attempt))
        raise last or GitLabError("request failed")

    # ---- reading ----
    def iter_hackerone_issues(self, project: str = DEFAULT_PROJECT, *, max_items: int | None = None, per_page: int = 100) -> Iterator[dict]:
        page, seen = "1", 0
        while page:
            body, headers = self._get(f"/projects/{quote(project, safe='')}/issues",
                                      {"labels": "HackerOne", "state": "closed", "confidential": "false", "order_by": "updated_at",
                                       "per_page": per_page, "page": page})
            for issue in body:
                yield issue
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            page = headers.get("x-next-page", "")

    def find_copy(self, report_id: int, project: str = DEFAULT_PROJECT) -> dict | None:
        """The GitLab issue that is the copy of HackerOne report `report_id` (not merely an issue that mentions it)."""
        body, _ = self._get(f"/projects/{quote(project, safe='')}/issues",
                            {"search": f"hackerone.com/reports/{report_id}", "in": "description", "per_page": 20})
        for issue in body:
            m = _h1_header(issue)
            if m and int(m.group(1)) == report_id:
                return issue
        return None

    def cache_issue(self, project: str, issue: dict) -> None:
        """Store an issue we already hold (e.g. from a listing) so fetching it later costs no request. Never overwrites."""
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{project.replace('/', '__')}__{issue['iid']}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"issue": issue, "notes": None}, ensure_ascii=False), encoding="utf-8")

    def fetch_report(self, url: str) -> SourceReport:
        project, iid = parse_issue_url(url)
        cache = self.cache_dir / f"{project.replace('/', '__')}__{iid}.json" if self.cache_dir else None
        record: dict[str, Any] | None = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.is_file() else None
        if record is None:
            issue, _ = self._get(f"/projects/{quote(project, safe='')}/issues/{iid}")
            record = {"issue": issue, "notes": None}
        wants_notes = bool(self._token) and (record["issue"].get("user_notes_count") or 0) > 0
        if wants_notes and record["notes"] is None:
            record["notes"], _ = self._get(f"/projects/{quote(project, safe='')}/issues/{iid}/notes",
                                           {"per_page": 100, "sort": "asc"}, authed=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")   # the token is never part of the record
        return to_source_report(record["issue"], record["notes"] or [], comments_skipped=not self._token)


def to_source_report(issue: dict, notes: list[dict], *, comments_skipped: bool = False) -> SourceReport:
    m = _h1_header(issue)
    if not m:
        raise GitLabError("not a copy of a HackerOne report (no 'HackerOne report #N' header line)")
    labels = list(issue.get("labels") or [])
    cwe, weakness = weakness_from_labels(labels)
    return SourceReport(
        source="gitlab", source_url=issue.get("web_url") or "", report_id=int(m.group(1)), title=str(issue.get("title") or "").strip(),
        body=issue.get("description") or "", program="gitlab", weakness=weakness, cwe=cwe, severity=severity_from_labels(labels),
        created_at=str(issue.get("created_at") or ""), disclosed_at=str(issue.get("closed_at") or ""), labels=labels, comments=notes,
        notes=["comments skipped: GitLab only returns comments to a token (set GITLAB_TOKEN)"] if comments_skipped and (issue.get("user_notes_count") or 0) > 0 else [])


def pick_candidates(issues: list[dict], n: int, per_weakness: int = 3, *, min_chars: int = 400) -> list[dict]:
    """From a listing, the n most complete HackerOne copies (steps / PoC / impact), at most `per_weakness` per CWE.
    The listing already contains the full text, so this needs no further requests."""
    from .clean import clean_report  # noqa: PLC0415
    from .quality import assess  # noqa: PLC0415

    scored: list[tuple[float, int, dict, str]] = []
    for issue in issues:
        if not _h1_header(issue):
            continue
        text = clean_report(issue.get("description") or "", source="gitlab").text
        a = assess(text)
        if len(text) < min_chars or not (a.has_steps or a.has_poc) or "disputed" in a.flags:
            continue
        cwe, _ = weakness_from_labels(list(issue.get("labels") or []))
        scored.append((a.score, int(issue.get("iid") or 0), issue, cwe))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked: list[dict] = []
    per: dict[str, int] = {}
    for _, _, issue, cwe in scored:
        if cwe and per.get(cwe, 0) >= per_weakness:
            continue
        per[cwe] = per.get(cwe, 0) + 1
        picked.append(issue)
        if len(picked) == n:
            break
    return picked
