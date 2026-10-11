"""Context-window budgeting for the chat endpoint.

llama-server is launched with ``--ctx-size 4096``.  A long conversation will
silently overflow that window, so we trim the oldest messages before sending
them upstream.

Estimate: ~3 characters per token (4 is too optimistic for code and
non-English text; 2-3 is more realistic for Arabic/French).  Budget: 2800
tokens of history, leaving the rest of the 4096 window for the model's reply
and logits.
"""

from __future__ import annotations

import os

# Budget in tokens of *history* we are willing to send llama-server.  The rest
# of the 4096-token window is headroom for the reply and the KV cache.
CONTEXT_BUDGET_TOKENS = 2800
CHARS_PER_TOKEN = 3


def ctx_size() -> int:
    """Context window llama-server is started with (same env var as llama_server)."""
    try:
        return int(os.environ.get("LLAMA_CTX_SIZE", "4096"))
    except ValueError:
        return 4096


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for a list of OpenAI-style messages."""
    chars = sum(len(m.get("content", "")) for m in messages)
    overhead = 4 * len(messages)
    return (chars + overhead) // CHARS_PER_TOKEN


def trim_messages(messages: list[dict], budget: int = CONTEXT_BUDGET_TOKENS) -> list[dict]:
    """Trim oldest messages until the estimate fits the budget.

    Rules:
      * The system prompt (first message) is never dropped.
      * The newest user message is never dropped — it is the one we are
        answering, and dropping it would make the reply incoherent.
      * Messages are dropped oldest-first in user/assistant pairs, so the
        trimmed history never starts with an assistant message.  A single
        odd leftover at the front is dropped too, unless it is the protected
        newest user message.

    Returns a (possibly shorter) list preserving the original order.
    """
    if not messages:
        return messages

    system_prompt = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if system_prompt is not None else messages
    if not rest:
        return messages

    # The newest user message is the last one with role "user" — keep it.
    newest_user_idx = max(
        (i for i, m in enumerate(rest) if m.get("role") == "user"),
        default=-1,
    )
    if newest_user_idx == -1:
        newest_user_idx = len(rest) - 1

    protected = {newest_user_idx}
    kept = list(rest)
    while estimate_tokens(kept) > budget and len(kept) > len(protected):
        # Drop the oldest unprotected pair (user + the assistant reply that
        # follows it).  If the oldest unprotected message is the last one in
        # the list (no assistant follows it), drop just that one message.
        drop_idx = next(i for i in range(len(kept)) if i not in protected)
        if drop_idx + 1 < len(kept) and (drop_idx + 1) not in protected:
            del kept[drop_idx : drop_idx + 2]
            protected = {i - 2 if i > drop_idx + 1 else i for i in protected}
        else:
            kept.pop(drop_idx)
            protected = {i - 1 if i > drop_idx else i for i in protected}
        protected.discard(-1)

    if system_prompt is not None:
        return [system_prompt, *kept]
    return kept