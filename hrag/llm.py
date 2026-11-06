"""Chat-model clients used for answering and for judging: Anthropic Messages API and any
OpenAI-compatible /chat/completions endpoint (OpenAI, OpenRouter, DeepInfra, Gemini's compat route)."""
from __future__ import annotations

from typing import Callable

import httpx

from .config import LLMProfile
from .http import ApiError, post_json

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, profile: LLMProfile, api_key: str, *, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] | None = None):
        if not api_key:
            raise LLMError("empty API key - refusing to send a request")
        if profile.model in ("", "SET_ME"):
            raise LLMError(f"llm profile '{profile.name}': set `model` in config.toml before using it")
        self.profile = profile
        self._key = api_key
        self._client = client
        self._sleep = sleep
        self._schema_refused = False

    def _send(self, system: str, user: str, json_schema: dict | None) -> dict:
        p = self.profile
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        if p.provider == "anthropic":
            url = f"{p.base_url or ANTHROPIC_BASE}/messages"
            headers = {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
            payload: dict = {"model": p.model, "max_tokens": p.max_tokens, "system": system,
                             "messages": [{"role": "user", "content": user}]}
            if json_schema is not None:
                payload["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        else:
            url = f"{p.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
            payload = {"model": p.model, p.max_tokens_param: p.max_tokens,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if json_schema is not None:
                payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": json_schema}}
                if "openrouter.ai" in p.base_url:             # route only to providers that can really enforce the schema
                    payload["provider"] = {"require_parameters": True}
        if p.temperature is not None:
            payload["temperature"] = p.temperature
        return post_json(url, headers=headers, payload=payload, timeout=p.timeout_s,
                         max_retries=p.max_retries, client=self._client, **kwargs)

    def complete(self, system: str, user: str, *, json_schema: dict | None = None) -> str:
        """json_schema (optional) asks the provider to guarantee a reply of that shape. A provider or model that cannot do that
        answers 400/404/422 (OpenRouter: no endpoint supports the parameter). The call is then repeated ONCE without the schema.
        Only if that repeat WORKS is the schema blamed and no longer sent by this client; if it fails too, the cause was something
        else (a wrong model id, a rejected setting) and the schema stays on."""
        p = self.profile
        schema = None if self._schema_refused else json_schema
        try:
            body = self._send(system, user, schema)
        except ApiError as exc:
            if schema is None or exc.status not in (400, 404, 422):
                raise LLMError(f"{p.name}: {exc}") from exc
            try:
                body = self._send(system, user, None)
            except ApiError as retry_exc:
                raise LLMError(f"{p.name}: {retry_exc}") from retry_exc
            self._schema_refused = True
        text = self._anthropic_text(body) if p.provider == "anthropic" else self._openai_text(body)
        if not text.strip():
            raise LLMError(f"{p.name}: the model returned no text")
        return text

    @staticmethod
    def _anthropic_text(body: dict) -> str:
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise LLMError("unexpected Anthropic reply shape")
        return "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")

    @staticmethod
    def _openai_text(body: dict) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("unexpected chat-completions reply shape")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""
