"""One JSON-POST helper with timeouts and retries, shared by every API client.

Lesson borrowed from Bhedan's Smriti: hosted embedding endpoints can stall for 20+ seconds
on one call and answer instantly on the next, so every call has a bounded timeout and is
retried. Errors never contain request headers, so an API key cannot leak into a log.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import httpx


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retriable: bool = False):
        super().__init__(message)
        self.status = status
        self.retriable = retriable


def _snippet(response: httpx.Response, limit: int = 300) -> str:
    return " ".join(response.text.split())[:limit]


def post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 60.0,
    max_retries: int = 4,
    backoff: float = 1.0,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    own_client = client is None
    http = client or httpx.Client(timeout=timeout)
    last: ApiError | None = None
    try:
        for attempt in range(max_retries + 1):
            delay = min(30.0, backoff * (2 ** attempt))
            try:
                response = http.post(url, headers=headers, json=payload, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ApiError(f"network error: {type(exc).__name__}", None, True)
            else:
                if response.status_code < 300:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise ApiError("response was not valid JSON", response.status_code) from exc
                    if not isinstance(body, dict):
                        raise ApiError("response JSON was not an object", response.status_code)
                    return body
                retriable = response.status_code in (408, 409, 425, 429) or response.status_code >= 500
                last = ApiError(f"HTTP {response.status_code}: {_snippet(response)}", response.status_code, retriable)
                if not retriable:
                    raise last
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = min(60.0, max(delay, float(retry_after)))
                    except ValueError:
                        pass
            if attempt < max_retries:
                sleep(delay)
        assert last is not None
        raise last
    finally:
        if own_client:
            http.close()
