"""SQLite conversation storage for LocalChat.

The database lives at ``data/localchat.db`` (created automatically on first
import).  Every conversation carries its own system prompt and generation
settings so a user can have a "code assistant" chat and a "creative writer"
chat side by side without reconfiguring the UI.

Schema
------
conversations
    id          INTEGER PK
    title       TEXT
    system_prompt TEXT
    temperature REAL
    top_p       REAL
    max_tokens  INTEGER
    created_at  INTEGER (unix seconds)

messages
    id              INTEGER PK
    conversation_id  INTEGER FK -> conversations.id
    role            TEXT  ('system' | 'user' | 'assistant')
    content         TEXT
    created_at      INTEGER (unix seconds)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "localchat.db"

# Module-level connection guarded by a lock.  FastAPI runs sync endpoints in a
# threadpool, so concurrent requests can hit storage.py at the same time.
# SQLite is process-local and single-writer, so one connection with a small
# busy-timeout is fine — the lock serialises the writes.
_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
            _conn.row_factory = sqlite3.Row
            _init_schema(_conn)
        return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT NOT NULL DEFAULT '',
            system_prompt TEXT NOT NULL DEFAULT '',
            temperature  REAL NOT NULL DEFAULT 0.7,
            top_p        REAL NOT NULL DEFAULT 0.9,
            max_tokens   INTEGER NOT NULL DEFAULT -1,
            created_at   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id  INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      INTEGER NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, created_at);
        """
    )
    # Migration: per-message metadata (token counts, speed) added in Phase 4.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    if "meta" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
    conn.commit()


# --------------------------------------------------------------------------- #
# Conversation CRUD
# --------------------------------------------------------------------------- #
def list_conversations() -> list[dict]:
    """Most-recently-active conversations first (by last message time).

    Ties in ``last_active`` (same second) break on ``c.id DESC`` so the newest
    conversation stays on top when several were touched in the same second.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT c.*,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count,
               COALESCE(
                   (SELECT MAX(created_at) FROM messages m WHERE m.conversation_id = c.id),
                   c.created_at
               ) AS last_active
        FROM conversations c
        ORDER BY last_active DESC, c.id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def create_conversation(
    *,
    title: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = -1,
) -> dict:
    conn = _get_conn()
    now = int(time.time())
    cur = conn.execute(
        """
        INSERT INTO conversations
            (title, system_prompt, temperature, top_p, max_tokens, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (title, system_prompt, temperature, top_p, max_tokens, now),
    )
    conn.commit()
    return get_conversation(cur.lastrowid)  # type: ignore[return-value]


def get_conversation(conv_id: int) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    return dict(row) if row else None


def update_conversation(conv_id: int, **fields) -> Optional[dict]:
    """Patch any subset of conversation fields.  Unknown keys are ignored."""
    allowed = {
        "title",
        "system_prompt",
        "temperature",
        "top_p",
        "max_tokens",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_conversation(conv_id)
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    conn = _get_conn()
    conn.execute(
        f"UPDATE conversations SET {set_clause} WHERE id = ?",
        list(updates.values()) + [conv_id],
    )
    conn.commit()
    return get_conversation(conv_id)


def delete_conversation(conv_id: int) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def list_messages(conv_id: int) -> list[dict]:
    """Return messages in insertion order.  ``id ASC`` breaks ties when several
    messages share the same second (created_at is only second-resolution)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, role, content, meta FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
        (conv_id,),
    ).fetchall()
    return [_message_dict(r) for r in rows]


def _message_dict(row: sqlite3.Row) -> dict:
    meta = None
    if row["meta"]:
        try:
            meta = json.loads(row["meta"])
        except ValueError:
            meta = None
    return {"id": row["id"], "role": row["role"], "content": row["content"], "meta": meta}


def add_message(conv_id: int, role: str, content: str, meta: Optional[dict] = None) -> dict:
    conn = _get_conn()
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at, meta) VALUES (?, ?, ?, ?, ?)",
        (conv_id, role, content, now, json.dumps(meta) if meta else None),
    )
    conn.commit()
    return {"id": cur.lastrowid, "conversation_id": conv_id, "role": role, "content": content, "meta": meta}


def get_message(msg_id: int) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, conversation_id, role, content, meta FROM messages WHERE id = ?", (msg_id,)
    ).fetchone()
    if row is None:
        return None
    d = _message_dict(row)
    d["conversation_id"] = row["conversation_id"]
    return d


def update_message_content(msg_id: int, content: str) -> bool:
    """Replace a message's text; its old metadata no longer applies, so it is cleared."""
    conn = _get_conn()
    cur = conn.execute("UPDATE messages SET content = ?, meta = NULL WHERE id = ?", (content, msg_id))
    conn.commit()
    return cur.rowcount > 0


def delete_messages_after(conv_id: int, msg_id: int) -> int:
    """Delete every message of the conversation that comes after `msg_id`."""
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM messages WHERE conversation_id = ? AND id > ?", (conv_id, msg_id)
    )
    conn.commit()
    return cur.rowcount


def fork_conversation(conv_id: int, up_to_message_id: int) -> Optional[dict]:
    """Copy a conversation (settings + messages up to and including a message)
    into a new one, so the user can try a different direction from that point."""
    src = get_conversation(conv_id)
    last = get_message(up_to_message_id)
    if src is None or last is None or last["conversation_id"] != conv_id:
        return None
    conn = _get_conn()
    title = f"{src['title']} (fork)" if src["title"] else "Fork"
    new = create_conversation(
        title=title,
        system_prompt=src["system_prompt"],
        temperature=src["temperature"],
        top_p=src["top_p"],
        max_tokens=src["max_tokens"],
    )
    conn.execute(
        """
        INSERT INTO messages (conversation_id, role, content, created_at, meta)
        SELECT ?, role, content, created_at, meta FROM messages
        WHERE conversation_id = ? AND id <= ?
        ORDER BY created_at ASC, id ASC
        """,
        (new["id"], conv_id, up_to_message_id),
    )
    conn.commit()
    return new


def _like_pattern(q: str) -> str:
    esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _snippet(text: str, q: str, radius: int = 50) -> str:
    flat = " ".join(text.split())
    i = flat.lower().find(q.lower())
    if i < 0:
        return flat[: radius * 2]
    start = max(0, i - radius)
    end = min(len(flat), i + len(q) + radius)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


def search(q: str, limit: int = 40) -> list[dict]:
    """Find conversations by title and messages by text (case-insensitive for ASCII)."""
    q = q.strip()
    if not q:
        return []
    conn = _get_conn()
    pat = _like_pattern(q)
    results: list[dict] = []
    for r in conn.execute(
        "SELECT id, title FROM conversations WHERE title LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
        (pat, limit),
    ):
        results.append(
            {"conversation_id": r["id"], "title": r["title"], "message_id": None, "role": None, "snippet": r["title"]}
        )
    rows = conn.execute(
        """
        SELECT m.id, m.conversation_id, m.role, m.content, c.title
        FROM messages m JOIN conversations c ON c.id = m.conversation_id
        WHERE m.content LIKE ? ESCAPE '\\'
        ORDER BY m.id DESC LIMIT ?
        """,
        (pat, limit),
    ).fetchall()
    for r in rows:
        results.append(
            {
                "conversation_id": r["conversation_id"],
                "title": r["title"],
                "message_id": r["id"],
                "role": r["role"],
                "snippet": _snippet(r["content"], q),
            }
        )
    return results[:limit]


def auto_title(conv_id: int, first_user_message: str) -> None:
    """Set a conversation title from its first user message (truncated)."""
    title = first_user_message.strip().replace("\n", " ")[:40]
    if title:
        update_conversation(conv_id, title=title)


def delete_last_assistant_message(conv_id: int) -> bool:
    """Delete the most recent assistant message in a conversation.

    Used by the Regenerate button: the user clicks Regenerate on a bot reply,
    we drop that reply, and the next ``POST /api/chat`` re-answers from the
    remaining history without re-sending the user's message.
    """
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT id FROM messages
        WHERE conversation_id = ? AND role = 'assistant'
        ORDER BY id DESC LIMIT 1
        """,
        (conv_id,),
    ).fetchone()
    if row is None:
        return False
    conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
    conn.commit()
    return True


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None