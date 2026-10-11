"""Phase 4: message edit, fork, search, export, reply stats."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

import chat_service
import main
import storage


def _chunk(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class FakeServer:
    is_running = True

    def __init__(self, pieces=("Hel", "lo"), stats=None):
        self.pieces = list(pieces)
        self.fake_stats = stats or {}
        self.calls: list[list[dict]] = []

    async def chat_stream_async(self, messages, sink=None, stats=None, **kwargs):
        self.calls.append(messages)
        if stats is not None:
            stats.update(self.fake_stats)
        for p in self.pieces:
            if sink is not None:
                sink.append(p)
            yield _chunk(p)


@pytest.fixture
def client(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(main, "_server", fake)
    monkeypatch.setattr(main, "_loading_model", None)
    c = TestClient(main.app)
    c.fake = fake
    return c


def seed():
    conv = storage.create_conversation(title="Topic", system_prompt="Be brief.", temperature=0.3)
    ids = [
        storage.add_message(conv["id"], "user", "q1")["id"],
        storage.add_message(conv["id"], "assistant", "a1")["id"],
        storage.add_message(conv["id"], "user", "q2")["id"],
        storage.add_message(conv["id"], "assistant", "a2")["id"],
    ]
    return conv, ids


# --- storage ---------------------------------------------------------------- #
def test_messages_have_id_and_meta():
    conv = storage.create_conversation()
    storage.add_message(conv["id"], "assistant", "hi", {"tokens": 3, "tps": 12.5})
    storage.add_message(conv["id"], "user", "yo")
    first, second = storage.list_messages(conv["id"])
    assert first["meta"] == {"tokens": 3, "tps": 12.5} and isinstance(first["id"], int)
    assert second["meta"] is None


def test_migration_adds_meta_column(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT '',
          system_prompt TEXT NOT NULL DEFAULT '', temperature REAL NOT NULL DEFAULT 0.7,
          top_p REAL NOT NULL DEFAULT 0.9, max_tokens INTEGER NOT NULL DEFAULT -1, created_at INTEGER NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL,
          role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL);
        INSERT INTO conversations (title, created_at) VALUES ('old', 1);
        INSERT INTO messages (conversation_id, role, content, created_at) VALUES (1, 'user', 'kept', 1);
        """
    )
    con.commit()
    con.close()
    storage.close()
    monkeypatch.setattr(storage, "DB_PATH", db)
    storage._conn = None
    msgs = storage.list_messages(1)
    assert msgs == [{"id": 1, "role": "user", "content": "kept", "meta": None}]


def test_edit_helpers():
    conv, ids = seed()
    assert storage.update_message_content(ids[0], "edited")
    assert storage.delete_messages_after(conv["id"], ids[0]) == 3
    assert [m["content"] for m in storage.list_messages(conv["id"])] == ["edited"]


def test_fork_copies_up_to_message():
    conv, ids = seed()
    new = storage.fork_conversation(conv["id"], ids[1])
    assert new["id"] != conv["id"] and new["title"] == "Topic (fork)"
    assert new["system_prompt"] == "Be brief." and new["temperature"] == 0.3
    assert [m["content"] for m in storage.list_messages(new["id"])] == ["q1", "a1"]
    assert len(storage.list_messages(conv["id"])) == 4  # original untouched
    other = storage.create_conversation()
    assert storage.fork_conversation(other["id"], ids[0]) is None  # message from another chat
    assert storage.fork_conversation(9999, ids[0]) is None


def test_search():
    conv, _ = seed()
    other = storage.create_conversation(title="Recipes")
    storage.add_message(other["id"], "user", "How to make 100% pure_olive oil pasta?")
    hits = storage.search("topic")
    assert hits[0]["message_id"] is None and hits[0]["conversation_id"] == conv["id"]
    hits = storage.search("A2")  # case-insensitive
    assert [h["snippet"] for h in hits] == ["a2"] and hits[0]["role"] == "assistant"
    assert storage.search("100%")[0]["conversation_id"] == other["id"]  # % is literal
    assert storage.search("pure_olive")[0]["conversation_id"] == other["id"]  # _ is literal
    assert storage.search("pure%olive") == []  # wildcards do not leak
    assert storage.search("   ") == []
    long = storage.add_message(other["id"], "user", "x" * 200 + " needle " + "y" * 200)
    snip = [h for h in storage.search("needle") if h["message_id"] == long["id"]][0]["snippet"]
    assert "needle" in snip and snip.startswith("…") and snip.endswith("…") and len(snip) < 130


# --- chat: edit ------------------------------------------------------------- #
def test_edit_replaces_message_and_drops_the_rest(client):
    conv, ids = seed()
    r = client.post(
        "/api/chat", json={"conversation_id": conv["id"], "edit_message_id": ids[2], "content": "q2 edited"}
    )
    assert r.status_code == 200
    msgs = [(m["role"], m["content"]) for m in storage.list_messages(conv["id"])]
    assert msgs == [("user", "q1"), ("assistant", "a1"), ("user", "q2 edited"), ("assistant", "Hello")]
    sent = client.fake.calls[0]
    assert sent[-1] == {"role": "user", "content": "q2 edited"}
    assert all(set(m) == {"role", "content"} for m in sent)  # no id/meta leaked to the model


def test_edit_validation(client):
    conv, ids = seed()
    other = storage.create_conversation()
    foreign = storage.add_message(other["id"], "user", "x")["id"]
    body = {"conversation_id": conv["id"], "content": "new"}
    assert client.post("/api/chat", json={**body, "edit_message_id": ids[1]}).status_code == 400  # assistant
    assert client.post("/api/chat", json={**body, "edit_message_id": foreign}).status_code == 404
    assert client.post("/api/chat", json={**body, "edit_message_id": 99999}).status_code == 404
    assert client.post("/api/chat", json={**body, "edit_message_id": ids[0], "content": " "}).status_code == 400
    assert len(storage.list_messages(conv["id"])) == 4  # nothing was touched


# --- chat: stats ------------------------------------------------------------ #
def test_reply_saved_with_server_stats(client, monkeypatch):
    fake = FakeServer(
        stats={"usage": {"prompt_tokens": 40, "completion_tokens": 7}, "timings": {"predicted_per_second": 33.3}}
    )
    monkeypatch.setattr(main, "_server", fake)
    conv = storage.create_conversation()
    client.post("/api/chat", json={"conversation_id": conv["id"], "content": "hi"})
    meta = storage.list_messages(conv["id"])[-1]["meta"]
    assert meta["tokens"] == 7 and meta["prompt_tokens"] == 40 and meta["tps"] == 33.3
    assert meta["exact"] is True and meta["ctx"] == 4096


def test_reply_meta_fallbacks():
    meta = chat_service.reply_meta(["a", "b", "c"], {}, [{"role": "user", "content": "x" * 30}], 1.0, 2.0)
    assert meta["tokens"] == 3 and meta["exact"] is False and meta["prompt_tokens"] > 0
    assert meta["tps"] == 2.0  # (3-1) tokens in 1 second
    quick = chat_service.reply_meta(["a"], {}, [], 1.0, 1.0)
    assert "tps" not in quick


def test_health_reports_ctx_size(client, monkeypatch):
    monkeypatch.setenv("LLAMA_CTX_SIZE", "8192")
    monkeypatch.setattr(main, "_server", None)  # health works with no server too
    assert client.get("/api/health").json()["ctx_size"] == 8192


# --- routes ----------------------------------------------------------------- #
def test_messages_endpoint_includes_ids(client):
    conv, ids = seed()
    msgs = client.get(f"/api/conversations/{conv['id']}/messages").json()["messages"]
    assert [m["id"] for m in msgs] == ids


def test_fork_route(client):
    conv, ids = seed()
    r = client.post(f"/api/conversations/{conv['id']}/fork", json={"up_to_message_id": ids[1]})
    assert r.status_code == 201 and r.json()["title"] == "Topic (fork)"
    assert client.post(f"/api/conversations/{conv['id']}/fork", json={"up_to_message_id": 99999}).status_code == 404


def test_search_route(client):
    seed()
    r = client.get("/api/search?q=q2")
    assert r.status_code == 200 and r.json()["results"][0]["snippet"] == "q2"
    assert client.get("/api/search").json() == {"results": []}


def test_export_markdown_and_json(client):
    conv, _ = seed()
    r = client.get(f"/api/conversations/{conv['id']}/export?format=md")
    assert r.status_code == 200
    assert 'filename="Topic.md"' in r.headers["content-disposition"]
    body = r.text
    assert body.startswith("# Topic") and "> Be brief." in body
    assert body.index("## You") < body.index("## Assistant") and "q1" in body and "a2" in body

    r = client.get(f"/api/conversations/{conv['id']}/export?format=json")
    data = r.json()
    assert data["conversation"]["title"] == "Topic"
    assert [m["content"] for m in data["messages"]] == ["q1", "a1", "q2", "a2"]

    assert client.get(f"/api/conversations/{conv['id']}/export?format=pdf").status_code == 400
    assert client.get("/api/conversations/9999/export").status_code == 404


def test_export_filename_is_ascii_safe(client):
    conv = storage.create_conversation(title="مرحبا / héllo")
    r = client.get(f"/api/conversations/{conv['id']}/export?format=md")
    assert 'filename="conversation-' in r.headers["content-disposition"] or "h_llo" in r.headers["content-disposition"]
    r.headers["content-disposition"].encode("ascii")
