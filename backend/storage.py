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
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, id ASC",
        (conv_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def add_message(conv_id: int, role: str, content: str) -> dict:
    conn = _get_conn()
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now),
    )
    conn.commit()
    return {"id": cur.lastrowid, "conversation_id": conv_id, "role": role, "content": content}


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