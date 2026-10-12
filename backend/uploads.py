"""File attachments: images for vision models, text/code files for any model.

Files live in ``data/uploads/<id>`` with a ``<id>.json`` sidecar (name, mime,
kind, size).  The id is a random 128-bit hex string, which is also the only
thing the API ever uses to refer to a file (never a client-supplied path).
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TEXT_BYTES = 512 * 1024
MAX_PER_MESSAGE = 4

# Text/code files we accept (by extension; the content must also be valid UTF-8).
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".env", ".html", ".htm", ".css", ".scss", ".js", ".jsx", ".ts",
    ".tsx", ".py", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sql",
    ".sh", ".bat", ".ps1", ".kt", ".swift", ".lua", ".r", ".tex", ".vue", ".svelte", ".gradle",
}

ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Uploaded but not yet attached to a saved message: protected from clean-up.
_pending: set[str] = set()
_lock = threading.Lock()


def sniff_image(data: bytes) -> Optional[str]:
    """Image type from the file's own bytes (the client's claim is not trusted).
    SVG is deliberately not accepted: it can carry scripts."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _clean_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:120] or "file"


def save_upload(name: str, data: bytes) -> dict:
    """Validate and store a file; returns its metadata. Raises ValueError."""
    name = _clean_name(name)
    if not data:
        raise ValueError("file is empty")
    mime = sniff_image(data)
    if mime:
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image too large (max 10 MB)")
        kind = "image"
    else:
        if Path(name).suffix.lower() not in TEXT_EXTS:
            raise ValueError("unsupported file type (images, text and code files only)")
        if len(data) > MAX_TEXT_BYTES:
            raise ValueError("text file too large (max 512 KB)")
        if b"\x00" in data:
            raise ValueError("not a text file")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("not a UTF-8 text file") from exc
        kind, mime = "text", "text/plain"

    att_id = uuid.uuid4().hex
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / att_id).write_bytes(data)
    meta = {"id": att_id, "name": name, "mime": mime, "kind": kind, "size": len(data)}
    (UPLOADS_DIR / f"{att_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    with _lock:
        _pending.add(att_id)
    return meta


def get_meta(att_id: str) -> Optional[dict]:
    if not isinstance(att_id, str) or not ID_RE.match(att_id):
        return None
    try:
        return json.loads((UPLOADS_DIR / f"{att_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def file_path(att_id: str) -> Optional[Path]:
    if get_meta(att_id) is None:
        return None
    p = UPLOADS_DIR / att_id
    return p if p.is_file() else None


def mark_attached(ids: list[str]) -> None:
    with _lock:
        _pending.difference_update(ids)


def read_text(att_id: str, max_chars: int) -> tuple[str, bool]:
    """File text, cut to `max_chars`; the bool says whether it was cut."""
    p = file_path(att_id)
    if p is None:
        return "", False
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def image_data_url(att_id: str) -> Optional[str]:
    p = file_path(att_id)
    meta = get_meta(att_id)
    if p is None or meta is None or meta.get("kind") != "image":
        return None
    return f"data:{meta['mime']};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def purge_unreferenced(referenced: set[str], min_age_seconds: float = 0) -> int:
    """Delete stored files that no saved message uses (and that are not a fresh
    upload still waiting to be sent). Called after deletes and at startup so a
    deleted chat does not leave its images on disk."""
    if not UPLOADS_DIR.is_dir():
        return 0
    with _lock:
        keep = set(referenced) | set(_pending)
    removed = 0
    now = time.time()
    for p in UPLOADS_DIR.iterdir():
        att_id = p.name[:-5] if p.name.endswith(".json") else p.name
        if not ID_RE.match(att_id) or att_id in keep:
            continue
        try:
            if now - p.stat().st_mtime < min_age_seconds:
                continue
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed
