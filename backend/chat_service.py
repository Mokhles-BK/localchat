"""Chat orchestration helpers shared by the /api/chat route and tests.

These three functions are deliberately tiny and pure (no FastAPI, no httpx)
so they can be unit-tested without a running llama-server.
"""

from __future__ import annotations

import json

from context import trim_messages
import storage


def extract_text(data: str) -> str:
    """Pull ``choices[0].delta.content`` out of one SSE JSON chunk.

    Returns "" on any parse error so a malformed chunk never breaks the
    stream.
    """
    try:
        chunk = json.loads(data)
        return chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        return ""


def build_context(conv: dict, history: list[dict], new_user_text: str | None) -> list[dict]:
    """Assemble the model's context from the database and the request.

    Order: stored system prompt (if non-empty), stored history, then the new
    user message (if any).  The result is trimmed to fit the context budget.
    """
    messages: list[dict] = []
    if conv.get("system_prompt"):
        messages.append({"role": "system", "content": conv["system_prompt"]})
    messages.extend(history)
    if new_user_text:
        messages.append({"role": "user", "content": new_user_text})
    return trim_messages(messages)


def effective_settings(conv: dict, req) -> dict:
    """Per-request overrides fall back to the conversation's stored settings.

    ``max_tokens`` of -1 means "unlimited" in the database, but llama-server
    has no unlimited mode — a very large value would overflow the context
    window.  We map -1 to 1024 so a long reply cannot blow past the budget.
    """
    settings = {
        "temperature": req.temperature if req.temperature is not None else conv["temperature"],
        "top_p": req.top_p if req.top_p is not None else conv["top_p"],
        "max_tokens": req.max_tokens if req.max_tokens is not None else conv["max_tokens"],
    }
    if settings["max_tokens"] == -1:
        settings["max_tokens"] = 1024
    return settings