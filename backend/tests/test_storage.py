"""Unit tests for storage.py (temp DB per test, see conftest.py)."""

from __future__ import annotations

import storage


def test_conversation_crud():
    conv = storage.create_conversation(title="a", system_prompt="sys", temperature=0.2)
    assert conv["title"] == "a" and conv["temperature"] == 0.2
    assert storage.get_conversation(conv["id"])["system_prompt"] == "sys"

    updated = storage.update_conversation(conv["id"], title="b", top_p=0.5)
    assert updated["title"] == "b" and updated["top_p"] == 0.5

    assert storage.delete_conversation(conv["id"]) is True
    assert storage.get_conversation(conv["id"]) is None
    assert storage.delete_conversation(conv["id"]) is False


def test_delete_conversation_cascades_messages():
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "user", "hi")
    storage.add_message(conv["id"], "assistant", "hello")
    storage.delete_conversation(conv["id"])
    assert storage.list_messages(conv["id"]) == []


def test_message_order_within_same_second():
    conv = storage.create_conversation()
    for i in range(10):
        storage.add_message(conv["id"], "user" if i % 2 == 0 else "assistant", f"m{i}")
    assert [m["content"] for m in storage.list_messages(conv["id"])] == [f"m{i}" for i in range(10)]


def test_delete_last_assistant_message():
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "user", "q1")
    storage.add_message(conv["id"], "assistant", "a1")
    storage.add_message(conv["id"], "user", "q2")
    storage.add_message(conv["id"], "assistant", "a2")

    assert storage.delete_last_assistant_message(conv["id"]) is True
    assert [m["content"] for m in storage.list_messages(conv["id"])] == ["q1", "a1", "q2"]


def test_delete_last_assistant_message_none():
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "user", "q")
    assert storage.delete_last_assistant_message(conv["id"]) is False


def test_update_conversation_ignores_unknown_keys():
    conv = storage.create_conversation(title="keep")
    updated = storage.update_conversation(conv["id"], hacker="x", id=999)
    assert updated["id"] == conv["id"] and updated["title"] == "keep"


def test_list_conversations_newest_first():
    a = storage.create_conversation(title="a")
    b = storage.create_conversation(title="b")
    ids = [c["id"] for c in storage.list_conversations()]
    assert ids == [b["id"], a["id"]]
