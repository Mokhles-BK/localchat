"""Chat orchestration helpers shared by the /api/chat route and tests.

These three functions are deliberately tiny and pure (no FastAPI, no httpx)
so they can be unit-tested without a running llama-server.
"""

from __future__ import annotations

import json

from context import CHARS_PER_TOKEN, ctx_size, estimate_tokens, trim_messages
import storage
import uploads


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


def message_for_model(m: dict, include_images: bool) -> dict:
    """One stored message as the model should see it.

    History rows also carry id/meta/attachments for the UI.  The model gets
    role + content only, where content is plain text, or (when a vision model
    is loaded and this is the message whose images should be shown) a list of
    OpenAI-style parts with the images as base64 data URLs.  Text files are
    pasted into the message text.
    """
    atts = m.get("attachments") or []
    if not atts:
        return {"role": m["role"], "content": m["content"]}

    text_atts = [a for a in atts if a.get("kind") == "text"]
    # All attached files together may use ~40% of the context window.
    per_file = max(500, int(ctx_size() * CHARS_PER_TOKEN * 0.4) // max(1, len(text_atts)))
    blocks: list[str] = []
    images: list[dict] = []
    for a in atts:
        if a.get("kind") == "text":
            body, cut = uploads.read_text(a["id"], per_file)
            blocks.append(f"[Attached file: {a['name']}]\n```\n{body}\n```" + ("\n(file truncated)" if cut else ""))
        elif a.get("kind") == "image":
            url = uploads.image_data_url(a["id"]) if include_images else None
            if url:
                images.append({"type": "image_url", "image_url": {"url": url}})
            else:
                blocks.append(f"[Attached image: {a['name']} (not shown to the model)]")
    text = "\n\n".join([m["content"], *blocks]) if blocks else m["content"]
    if images:
        return {"role": m["role"], "content": [{"type": "text", "text": text}, *images]}
    return {"role": m["role"], "content": text}


def build_context(
    conv: dict, history: list[dict], new_user_text: str | None, vision: bool = False
) -> list[dict]:
    """Assemble the model's context from the database and the request.

    Order: stored system prompt (if non-empty), stored history, then the new
    user message (if any).  The result is trimmed to fit the context budget.
    Only the most recent message with images shows them to the model (each
    image costs hundreds of tokens); older ones become a short note.
    """
    messages: list[dict] = []
    if conv.get("system_prompt"):
        messages.append({"role": "system", "content": conv["system_prompt"]})
    last_with_images = max(
        (i for i, m in enumerate(history) if any(a.get("kind") == "image" for a in m.get("attachments") or [])),
        default=-1,
    )
    messages.extend(
        message_for_model(m, include_images=vision and i == last_with_images) for i, m in enumerate(history)
    )
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

def reply_meta(
    pieces: list[str],
    stats: dict,
    messages: list[dict],
    t_first: float | None,
    t_last: float | None,
) -> dict:
    """Token counts and speed for one reply, saved with the assistant message.

    Prefers llama-server's own numbers (``usage`` / ``timings``); falls back to
    counting streamed pieces and timing them (marked ``exact: false``).
    """
    usage = stats.get("usage") or {}
    timings = stats.get("timings") or {}
    tokens = usage.get("completion_tokens") or timings.get("predicted_n") or len(pieces)
    prompt = usage.get("prompt_tokens") or timings.get("prompt_n") or estimate_tokens(messages)
    tps = timings.get("predicted_per_second")
    if not tps and t_first is not None and t_last is not None and tokens > 1 and t_last - t_first > 0.05:
        tps = (tokens - 1) / (t_last - t_first)
    meta: dict = {
        "tokens": int(tokens),
        "prompt_tokens": int(prompt),
        "ctx": ctx_size(),
        "exact": bool(usage or timings),
    }
    if tps:
        meta["tps"] = round(float(tps), 1)
    return meta
