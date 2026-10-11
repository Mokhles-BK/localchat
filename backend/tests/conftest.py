"""Shared pytest fixtures for the LocalChat backend tests.

The backend is NOT a package — it is started with
``uvicorn main:app --app-dir backend`` and pytest runs from the backend
folder.  So every test that does ``import storage`` needs the backend folder
on sys.path.

Every test also gets its own temporary SQLite file, so tests never touch the
real ``data/localchat.db``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point storage at a fresh temp database for every test."""
    import storage

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(storage, "DB_PATH", db_path)
    storage._conn = None
    yield
    storage.close()