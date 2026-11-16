"""OpenAI-compatible chat client with a timeout and bounded retries."""

from __future__ import annotations

import logging
import time

from ..config import get_settings

log = logging.getLogger("scirag.llm")


class LLMUnavailable(RuntimeError):
    """Raised when the LLM is not configured or every attempt failed."""


_CLIENT = None
_CLIENT_KEY: tuple | None = None


def _client():
    global _CLIENT, _CLIENT_KEY
    c = get_settings().llm
    if not c.api_key or not c.chat_model:
        raise LLMUnavailable("LLM not configured: set OPENAI_API_KEY and CHAT_MODEL in .env")
    key = (c.base_url, c.api_key, c.timeout_sec)
    if _CLIENT is None or key != _CLIENT_KEY:
        from openai import OpenAI

        # Retries are ours (below), so the SDK's own retry loop is disabled.
        _CLIENT = OpenAI(base_url=c.base_url or None, api_key=c.api_key,
                         timeout=c.timeout_sec, max_retries=0)
        _CLIENT_KEY = key
    return _CLIENT


def is_configured() -> bool:
    c = get_settings().llm
    return bool(c.api_key and c.chat_model)


def chat(messages: list[dict], *, model: str | None = None, temperature: float | None = None,
         max_tokens: int | None = None, json_mode: bool = False) -> str:
    c = get_settings().llm
    client = _client()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last_exc: Exception | None = None
    for attempt in range(c.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model or c.chat_model,
                messages=messages,
                temperature=c.temperature if temperature is None else temperature,
                max_tokens=max_tokens or c.max_tokens,
                **kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, c.max_retries + 1, exc)
            if attempt < c.max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise LLMUnavailable(f"LLM failed after {c.max_retries + 1} attempts: {last_exc}") from last_exc
