"""Hosted embedding clients: OpenAI, Voyage, and any OpenAI-compatible endpoint.

Rules this module enforces (all learned the hard way in Bhedan's Smriti):
  * a reply is only accepted if every vector is finite, the right count and the right length;
  * vectors are L2-normalised so cosine similarity is a plain dot product;
  * documents and questions go through the SAME model (the profile) - never mix profiles.
"""
from __future__ import annotations

from typing import Callable, Sequence

import httpx
import numpy as np

from .config import EmbeddingProfile
from .http import ApiError, post_json

OPENAI_BASE = "https://api.openai.com/v1"
VOYAGE_BASE = "https://api.voyageai.com/v1"


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, profile: EmbeddingProfile, api_key: str, *, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] | None = None):
        if not api_key:
            raise EmbeddingError("empty API key - refusing to send a request")
        self.profile = profile
        self._key = api_key
        self._client = client
        self._sleep = sleep

    # ---- public ----
    def embed_documents(self, texts: Sequence[str], progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        out: list[np.ndarray] = []
        size = max(1, self.profile.batch_size)
        for start in range(0, len(texts), size):
            batch = [self.profile.doc_prefix + t for t in texts[start:start + size]]
            out.append(self._embed(batch, "document"))
            if progress:
                progress(min(start + size, len(texts)), len(texts))
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.profile.query_prefix + text], "query")[0]

    # ---- internals ----
    def _url_and_headers(self) -> tuple[str, dict[str, str]]:
        p = self.profile
        if p.provider == "voyage":
            base = p.base_url or VOYAGE_BASE
        elif p.provider == "openai":
            base = p.base_url or OPENAI_BASE
        else:
            base = p.base_url
        return f"{base}/embeddings", {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _payload(self, texts: list[str], kind: str) -> dict:
        p = self.profile
        payload: dict = {"model": p.model, "input": texts}
        if p.provider == "voyage":
            payload["input_type"] = kind            # Voyage prepends its own retrieval prompt
            payload["truncation"] = True
            if p.dimensions:
                payload["output_dimension"] = p.dimensions
        elif p.dimensions:
            payload["dimensions"] = p.dimensions
        return payload

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        url, headers = self._url_and_headers()
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        try:
            body = post_json(url, headers=headers, payload=self._payload(texts, kind),
                             timeout=self.profile.timeout_s, max_retries=self.profile.max_retries,
                             client=self._client, **kwargs)
        except ApiError as exc:
            raise EmbeddingError(f"{self.profile.name}: {exc}") from exc
        return self._parse(body, len(texts))

    def _parse(self, body: dict, expected: int) -> np.ndarray:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(f"{self.profile.name}: expected {expected} embeddings, got "
                                 f"{len(data) if isinstance(data, list) else 'none'}")
        items = sorted(data, key=lambda d: d.get("index", 0) if isinstance(d, dict) else 0)
        rows = []
        for item in items:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or not vec or not all(isinstance(x, (int, float)) for x in vec):
                raise EmbeddingError(f"{self.profile.name}: reply contained a non-numeric or empty embedding")
            rows.append(vec)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise EmbeddingError(f"{self.profile.name}: reply contained non-finite numbers or ragged vectors")
        if self.profile.dimensions and arr.shape[1] != self.profile.dimensions:
            raise EmbeddingError(f"{self.profile.name}: wanted {self.profile.dimensions} dimensions, got {arr.shape[1]}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if (norms == 0).any():
            raise EmbeddingError(f"{self.profile.name}: reply contained an all-zero vector")
        return arr / norms
