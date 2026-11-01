"""HackerOne Hacker API collector: list public disclosed reports (Hacktivity) and fetch their details.

Safety rules built in:
  * credentials come from secrets.py (environment / .env / credential store) and never appear in any error text;
  * the token is only ever sent to the API's own host - a `links.next` pointing elsewhere is refused;
  * requests are spaced (`min_interval`), rate-limit replies (429) are waited out, auth failures are not retried;
  * bulk collection is GATED: `require_probe_ok` refuses until `hrag h1 probe` has shown that the API returns
    complete report text with the user's own token.

The JSON field names used below are ASSUMED from HackerOne's public documentation and are deliberately
tolerant (several plausible spots are tried). `hrag h1 probe` prints the real field names so they can be fixed.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import httpx

from .sources import SourceReport

BASE_URL = "https://api.hackerone.com/v1/hackers"
# Only fields that hold the report BODY count as full text; short blurbs (summary, impact label) must not make a probe pass.
FULL_TEXT_KEYS = ("vulnerability_information", "vulnerability_information_html", "steps_to_reproduce")


class H1Error(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _snippet(response: httpx.Response, limit: int = 200) -> str:
    return " ".join(response.text.split())[:limit]


class H1Client:
    def __init__(self, username: str, token: str, *, base_url: str = BASE_URL, client: httpx.Client | None = None,
                 cache_dir: Path | None = None, min_interval: float = 0.6, max_retries: int = 4, timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self._auth = (username, token)
        self.base_url = base_url.rstrip("/")
        self.http = client or httpx.Client(timeout=timeout, follow_redirects=False, headers={"Accept": "application/json"})
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.rate_remaining: str | None = None

    # ---- transport ----
    def _url(self, url_or_path: str) -> str:
        if url_or_path.startswith("/"):
            return self.base_url + url_or_path
        if urlparse(url_or_path).hostname != urlparse(self.base_url).hostname:
            raise H1Error("refusing to send credentials to another host than the HackerOne API")
        return url_or_path

    def _throttle(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def get_json(self, url_or_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(url_or_path)
        last: H1Error | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.http.get(url, params=params, auth=self._auth)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = H1Error(f"network error: {type(exc).__name__}")
                self._sleep(min(30.0, 2.0 ** attempt))
                continue
            self.rate_remaining = response.headers.get("x-ratelimit-remaining")
            code = response.status_code
            if code < 300:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise H1Error("response was not valid JSON", code) from exc
                if not isinstance(body, dict):
                    raise H1Error("response JSON was not an object", code)
                return body
            if code in (401, 403):
                raise H1Error(f"HTTP {code}: HackerOne rejected the credentials, or this token cannot access that resource. "
                              f"Check H1_API_USERNAME / H1_API_TOKEN. {_snippet(response)}", code)
            if code == 404:
                raise H1Error(f"HTTP 404: not found (or not visible to this token): {url.split('?')[0]}", code)
            last = H1Error(f"HTTP {code}: {_snippet(response)}", code)
            if code != 429 and code < 500:
                raise last
            retry_after = response.headers.get("retry-after", "")
            self._sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(30.0, 2.0 ** attempt))
        raise last or H1Error("request failed")

    # ---- endpoints ----
    def iter_hacktivity(self, query: str = "disclosed:true", page_size: int = 25, max_items: int | None = None) -> Iterator[dict]:
        url: str | None = "/hacktivity"
        params: dict[str, Any] | None = {"queryString": query, "page[size]": page_size}
        seen = 0
        while url:
            body = self.get_json(url, params)
            params = None                                   # the next link already carries its query
            for item in body.get("data") or []:
                yield item
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            url = (body.get("links") or {}).get("next")

    def get_report(self, report_id: int, use_cache: bool = True) -> dict[str, Any]:
        cached = self.cache_dir / f"{int(report_id)}.json" if self.cache_dir else None
        if use_cache and cached and cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
        body = self.get_json(f"/reports/{int(report_id)}")
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        return body


# ---------- reading a (not yet verified) item shape tolerantly ----------

def _find_first(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for k in keys:
            if obj.get(k) not in (None, "", {}, []):
                return obj[k]
        for v in obj.values():
            hit = _find_first(v, keys)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_first(v, keys)
            if hit is not None:
                return hit
    return None


def report_id_of(item: dict) -> int | None:
    rel = (item.get("relationships") or {}).get("report")
    rel_id = (rel.get("data") or {}).get("id") if isinstance(rel, dict) else None
    for value in ((item.get("attributes") or {}).get("report_id"), rel_id, item.get("id")):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None


def program_of(item: dict) -> str | None:
    value = _find_first(item, ("handle", "program_handle", "team_handle"))
    return str(value) if value else None


def title_of(item: dict) -> str:
    return str(_find_first(item, ("title",)) or "")


def pick_diverse(items: list[dict], n: int, per_program: int = 3) -> list[dict]:
    """Up to n distinct reports, at most `per_program` from one program (unknown programs are not capped)."""
    picked: list[dict] = []
    seen_ids: set[int] = set()
    per: dict[str, int] = {}
    for it in items:
        rid = report_id_of(it)
        if rid is None or rid in seen_ids:
            continue
        prog = program_of(it)
        if prog is not None and per.get(prog, 0) >= per_program:
            continue
        seen_ids.add(rid)
        if prog is not None:
            per[prog] = per.get(prog, 0) + 1
        picked.append(it)
        if len(picked) == n:
            break
    return picked


# ---------- probe: what does the API really return? (names and lengths only, never text) ----------

def field_names(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    if isinstance(obj, dict):
        return {k: (field_names(v, depth + 1, max_depth) if depth < max_depth else "...") for k, v in obj.items()}
    if isinstance(obj, list):
        return [field_names(obj[0], depth + 1, max_depth)] if obj else []
    if isinstance(obj, str):
        return f"str[{len(obj)}]"
    return type(obj).__name__


def find_text_fields(obj: Any, path: str = "") -> list[tuple[str, int]]:
    """Where a full-text field with real content occurs, with its length - never the text itself."""
    found: list[tuple[str, int]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in FULL_TEXT_KEYS and isinstance(v, str) and v.strip():
                found.append((p, len(v)))
            found.extend(find_text_fields(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            found.extend(find_text_fields(v, f"{path}[{i}]"))
    return found


def run_probe(client: H1Client, query: str = "disclosed:true", detail_tries: int = 3) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": query,
        "auth_ok": False, "hacktivity_ok": False, "hacktivity_items_seen": 0, "hacktivity_full_text_fields": [],
        "sample_report_ids": [], "detail_status": {}, "report_detail_ok": False, "detail_full_text_fields": [],
        "usable_for_bulk": False, "item_shape": None, "detail_shape": None, "notes": []}
    notes: list[str] = verdict["notes"]
    try:
        page = client.get_json("/hacktivity", {"queryString": query, "page[size]": 10})
    except H1Error as exc:
        verdict["auth_ok"] = exc.status not in (401, 403)
        notes.append(f"hacktivity request failed (HTTP {exc.status}): check the username/token, then re-run the probe")
        return verdict
    verdict["auth_ok"] = True
    items = page.get("data") or []
    verdict["hacktivity_items_seen"] = len(items)
    verdict["hacktivity_ok"] = bool(items)
    if not items:
        notes.append("hacktivity answered but returned no items for this query")
        return verdict
    verdict["item_shape"] = field_names(items[0])
    verdict["hacktivity_full_text_fields"] = find_text_fields(items)
    ids = [rid for rid in (report_id_of(i) for i in items) if rid is not None]
    verdict["sample_report_ids"] = ids[:detail_tries]
    if not ids:
        notes.append("could not find a report id in the hacktivity items - the item shape differs from what the code expects")
    for rid in ids[:detail_tries]:
        try:
            body = client.get_json(f"/reports/{rid}")
        except H1Error as exc:
            verdict["detail_status"][str(rid)] = exc.status
            continue
        verdict["detail_status"][str(rid)] = 200
        fields = find_text_fields(body)
        if verdict["detail_shape"] is None:
            verdict["detail_shape"] = field_names(body)
        if fields:
            verdict["report_detail_ok"] = True
            verdict["detail_full_text_fields"] = fields
            break
    if verdict["detail_status"] and not verdict["report_detail_ok"]:
        notes.append("report detail was not available with full text for this token (only summaries) - "
                     "other permitted sources for complete public report content are needed")
    verdict["usable_for_bulk"] = bool(verdict["hacktivity_ok"] and (verdict["report_detail_ok"] or verdict["hacktivity_full_text_fields"]))
    if not verdict["usable_for_bulk"] and not notes:
        notes.append("no full-text field (reproduction steps / impact) was found in the list or the detail responses")
    return verdict


def write_probe(path: Path, verdict: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")


def require_probe_ok(path: Path) -> dict[str, Any]:
    """Refuse bulk collection until the probe has shown the API returns complete report text."""
    path = Path(path)
    if not path.is_file():
        raise H1Error("bulk collection is blocked until the API has been probed: run  python -m hrag h1 probe  first")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    if not verdict.get("usable_for_bulk"):
        why = "; ".join(verdict.get("notes") or []) or "the probe did not find complete report text"
        raise H1Error(f"bulk collection is blocked: the last probe said the API is not usable for full reports ({why})")
    return verdict


# ---------- API report -> the common SourceReport ----------

def to_source_report(body: dict) -> SourceReport | None:
    """Map one report response to a SourceReport; None if it has no id or no report text (nothing usable to index).
    Field names are ASSUMED from HackerOne's documentation - the probe output shows the real ones."""
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    attrs = data.get("attributes") if isinstance(data.get("attributes"), dict) else data
    try:
        rid = int(str(data.get("id") or attrs.get("id")))
    except (TypeError, ValueError):
        return None
    text = str(attrs.get("vulnerability_information") or "").strip()
    if not text:
        parts = [(label, str(attrs.get(key) or "").strip()) for label, key in
                 (("Summary", "summary"), ("Steps to reproduce", "steps_to_reproduce"), ("Impact", "impact"))]
        text = "\n\n".join(f"## {label}\n\n{value}" for label, value in parts if value)
    if not text:
        return None
    weak = attrs.get("weakness")
    weakness = str(weak.get("name") or "") if isinstance(weak, dict) else str(weak or "")
    ext = str(weak.get("external_id") or "") if isinstance(weak, dict) else ""
    sev = _find_first(attrs.get("severity") or data, ("rating", "severity_rating", "max_severity"))
    return SourceReport(
        source="hackerone_api", source_url=f"https://hackerone.com/reports/{rid}", report_id=rid, title=str(attrs.get("title") or "").strip(),
        body=text, program=program_of(data) or "", weakness=weakness, cwe=ext.upper() if ext.lower().startswith("cwe-") else "",
        severity=str(sev or ""), substate=str(attrs.get("substate") or attrs.get("state") or ""),
        created_at=str(attrs.get("created_at") or ""), disclosed_at=str(attrs.get("disclosed_at") or ""))
