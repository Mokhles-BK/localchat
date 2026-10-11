"""Route tests for /api/chat with a fake llama-server (no GPU, no network)."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

import main
import storage


def _chunk(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class FakeServer:
    """Stands in for LlamaServer: streams canned pieces, optionally fails."""

    is_running = True

    def __init__(self, pieces=("Hel", "lo ", "world"), fail_after: int | None = None):
        self.pieces = list(pieces)
        self.fail_after = fail_after
        self.calls: list[list[dict]] = []

    async def chat_stream_async(self, messages, sink=None, **kwargs):
        self.calls.append(messages)
        sink = sink if sink is not None else []
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("boom")
            sink.append(piece)
            yield _chunk(piece)


@pytest.fixture
def client(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(main, "_server", fake)
    c = TestClient(main.app)  # no `with`: lifespan (llama-server start) is skipped
    c.fake = fake
    return c


def _msgs(conv_id):
    return [(m["role"], m["content"]) for m in storage.list_messages(conv_id)]


def test_user_and_assistant_persisted(client):
    conv = storage.create_conversation(system_prompt="Be brief.")
    r = client.post("/api/chat", json={"conversation_id": conv["id"], "content": "hi"})
    assert r.status_code == 200
    assert "[DONE]" in r.text
    assert _msgs(conv["id"]) == [("user", "hi"), ("assistant", "Hello world")]
    # The model saw system prompt + the new user message.
    sent = client.fake.calls[0]
    assert sent[0] == {"role": "system", "content": "Be brief."}
    assert sent[-1] == {"role": "user", "content": "hi"}


def test_partial_saved_when_stream_fails(client, monkeypatch):
    monkeypatch.setattr(main, "_server", FakeServer(fail_after=2))
    conv = storage.create_conversation()
    r = client.post("/api/chat", json={"conversation_id": conv["id"], "content": "hi"})
    assert '"error"' in r.text
    assert _msgs(conv["id"]) == [("user", "hi"), ("assistant", "Hel" + "lo ")]


def test_nothing_generated_saves_no_assistant(client, monkeypatch):
    monkeypatch.setattr(main, "_server", FakeServer(fail_after=0))
    conv = storage.create_conversation()
    client.post("/api/chat", json={"conversation_id": conv["id"], "content": "hi"})
    assert _msgs(conv["id"]) == [("user", "hi")]


def test_regenerate_replaces_last_assistant(client, monkeypatch):
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "user", "q")
    storage.add_message(conv["id"], "assistant", "old answer")
    monkeypatch.setattr(main, "_server", FakeServer(pieces=["new ", "answer"]))
    r = client.post("/api/chat", json={"conversation_id": conv["id"], "regenerate": True})
    assert r.status_code == 200
    assert _msgs(conv["id"]) == [("user", "q"), ("assistant", "new answer")]


def test_regenerate_without_assistant_is_400(client):
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "user", "q")
    r = client.post("/api/chat", json={"conversation_id": conv["id"], "regenerate": True})
    assert r.status_code == 400
    assert _msgs(conv["id"]) == [("user", "q")]  # nothing deleted


def test_unknown_conversation_is_404(client):
    r = client.post("/api/chat", json={"conversation_id": 9999, "content": "hi"})
    assert r.status_code == 404


def test_empty_content_is_400(client):
    conv = storage.create_conversation()
    r = client.post("/api/chat", json={"conversation_id": conv["id"], "content": "   "})
    assert r.status_code == 400


def test_title_set_once(client):
    conv = storage.create_conversation()
    client.post("/api/chat", json={"conversation_id": conv["id"], "content": "first question"})
    client.post("/api/chat", json={"conversation_id": conv["id"], "content": "second question"})
    assert storage.get_conversation(conv["id"])["title"] == "first question"


def test_503_when_server_down(monkeypatch):
    monkeypatch.setattr(main, "_server", None)
    conv = storage.create_conversation()
    r = TestClient(main.app).post("/api/chat", json={"conversation_id": conv["id"], "content": "hi"})
    assert r.status_code == 503
