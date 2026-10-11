"""Unit tests for chat_service (no llama-server, no network)."""

from __future__ import annotations

import storage
from chat_service import build_context, effective_settings, extract_text


# --------------------------------------------------------------------------- #
# extract_text
# --------------------------------------------------------------------------- #
def test_extract_text_returns_content():
    assert extract_text(
        '{"choices":[{"delta":{"content":"hello"}}]}'
    ) == "hello"


def test_extract_text_empty_delta():
    assert extract_text('{"choices":[{"delta":{}}]}') == ""


def test_extract_text_malformed_returns_empty():
    assert extract_text("not json") == ""
    assert extract_text('{"choices":[]}') == ""
    assert extract_text("") == ""


# --------------------------------------------------------------------------- #
# build_context
# --------------------------------------------------------------------------- #
def test_build_context_includes_system_prompt_and_history():
    conv = storage.create_conversation(system_prompt="Be concise.")
    storage.add_message(conv["id"], "user", "hi")
    storage.add_message(conv["id"], "assistant", "hello")
    history = storage.list_messages(conv["id"])

    ctx = build_context(conv, history, "second message")

    assert ctx[0] == {"role": "system", "content": "Be concise."}
    assert ctx[-1] == {"role": "user", "content": "second message"}


def test_build_context_no_system_prompt():
    conv = storage.create_conversation(system_prompt="")
    history = storage.list_messages(conv["id"])

    ctx = build_context(conv, history, "hi")

    assert ctx[0]["role"] == "user"
    assert ctx[0]["content"] == "hi"


def test_build_context_trims_oldest_when_over_budget():
    conv = storage.create_conversation(system_prompt="sys")
    for i in range(20):
        storage.add_message(conv["id"], "user", f"old question number {i} " * 40)
        storage.add_message(conv["id"], "assistant", f"old answer number {i} " * 40)
    history = storage.list_messages(conv["id"])

    ctx = build_context(conv, history, "newest question")

    # System prompt and the newest user message are always kept.
    assert ctx[0] == {"role": "system", "content": "sys"}
    assert ctx[-1] == {"role": "user", "content": "newest question"}
    # Trimmed: far fewer than the full 41 messages.
    assert len(ctx) < len(history) + 2


def test_build_context_keeps_newest_user_when_alone_over_budget():
    big = "x" * 40000  # way over the 2800-token budget
    ctx = build_context({"system_prompt": ""}, [], big)

    assert ctx == [{"role": "user", "content": big}]


# --------------------------------------------------------------------------- #
# effective_settings
# --------------------------------------------------------------------------- #
def test_effective_settings_uses_stored_when_request_omits():
    conv = storage.create_conversation(temperature=0.3, top_p=0.5, max_tokens=256)

    class Req:
        temperature = None
        top_p = None
        max_tokens = None

    s = effective_settings(conv, Req())
    assert s == {"temperature": 0.3, "top_p": 0.5, "max_tokens": 256}


def test_effective_settings_request_overrides_stored():
    conv = storage.create_conversation(temperature=0.3, top_p=0.5, max_tokens=256)

    class Req:
        temperature = 0.9
        top_p = None
        max_tokens = None

    s = effective_settings(conv, Req())
    assert s == {"temperature": 0.9, "top_p": 0.5, "max_tokens": 256}


def test_effective_settings_unlimited_maps_to_1024():
    conv = storage.create_conversation(max_tokens=-1)

    class Req:
        temperature = None
        top_p = None
        max_tokens = None

    s = effective_settings(conv, Req())
    assert s["max_tokens"] == 1024