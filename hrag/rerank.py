"""Optional reranker (a cross-encoder behind a hosted API). It can only REORDER candidates
that retrieval already found - it cannot recover a chunk that was never retrieved."""
from __future__ import annotations

from typing import Callable, Sequence

import httpx

from .config import RerankProfile
from .http import ApiError, post_json

VOYAGE_BASE = "https://api.voyageai.com/v1"


class RerankError(RuntimeError):
    pass


class Reranker:
    def __init__(self, profile: RerankProfile, api_key: str, *, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] | None = None):
        if not api_key:
            raise RerankError("empty API key - refusing to send a request")
        self.profile = profile
        self._key = api_key
        self._client = client
        self._sleep = sleep

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        """Return (index into `documents`, relevance_score), best first."""
        if not documents:
            return []
        p = self.profile
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        payload = {"query": query, "documents": list(documents), "model": p.model,
                   "top_k": min(top_k, len(documents)), "truncation": True}
        try:
            body = post_json(f"{p.base_url or VOYAGE_BASE}/rerank",
                             headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                             payload=payload, timeout=p.timeout_s, max_retries=p.max_retries,
                             client=self._client, **kwargs)
        except ApiError as exc:
            raise RerankError(f"{p.name}: {exc}") from exc
        data = body.get("data")
        if not isinstance(data, list):
            raise RerankError(f"{p.name}: unexpected rerank reply shape")
        out: list[tuple[int, float]] = []
        for item in data:
            idx, score = item.get("index"), item.get("relevance_score")
            if not isinstance(idx, int) or not 0 <= idx < len(documents) or not isinstance(score, (int, float)):
                raise RerankError(f"{p.name}: rerank reply had a bad index or score")
            out.append((idx, float(score)))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out
