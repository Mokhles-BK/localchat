"""Model manager: local GGUF files, Hugging Face search, resumable downloads.

Everything here is plain functions + a small in-memory download-job table, so
it is easy to test (tests swap `_client` for an httpx MockTransport client).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

import app_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
HF_BASE = "https://huggingface.co"

REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")
# Root-level .gguf file names only: no slashes, no backslashes, no "..".
FILE_RE = re.compile(r"^[\w.\-+()\[\] ]+\.gguf$")
QUANT_RE = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_(?:XL|[SML])|Q\d_K|Q\d_\d|BF16|F16|F32)", re.I)
SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}", re.I)

MIN_FREE_BYTES = 512 * 1024 * 1024  # keep this much disk free after a download


# --------------------------------------------------------------------------- #
# Local files
# --------------------------------------------------------------------------- #
def quant_of(filename: str) -> Optional[str]:
    m = QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


_vram_cache: tuple[float, Optional[int]] = (0.0, None)


def vram_mb() -> Optional[int]:
    """Total VRAM of the first NVIDIA GPU in MiB, or None (cached for 60 s)."""
    global _vram_cache
    ts, val = _vram_cache
    if time.monotonic() - ts < 60:
        return val
    val = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            val = int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        val = None
    _vram_cache = (time.monotonic(), val)
    return val


def fit_estimate(size_bytes: int) -> str:
    """Rough guess: 'gpu' (fits in VRAM), 'partial' (needs CPU offload),
    'too_big' (will be very slow) or 'unknown' (no NVIDIA GPU found).
    It is an estimate: context size and the desktop also use VRAM."""
    total = vram_mb()
    if not total:
        return "unknown"
    need_mb = size_bytes / (1024 * 1024) * 1.15 + 600  # weights + KV cache + overhead
    usable = total - 300  # leave a little for the desktop
    if need_mb <= usable:
        return "gpu"
    if need_mb <= usable * 2.5:
        return "partial"
    return "too_big"


def list_local() -> list[dict[str, Any]]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        size = p.stat().st_size
        out.append(
            {"name": p.name, "size_bytes": size, "quant": quant_of(p.name), "fit": fit_estimate(size)}
        )
    return out


def safe_model_path(name: str) -> Path:
    """Resolve `name` to a file inside MODELS_DIR, or raise.

    ValueError  -> the name is not a plain .gguf file name (path tricks)
    FileNotFoundError -> valid name, but no such model
    """
    if not isinstance(name, str) or Path(name).name != name or not FILE_RE.match(name):
        raise ValueError("invalid model name")
    path = (MODELS_DIR / name).resolve()
    if path.parent != MODELS_DIR.resolve():
        raise ValueError("invalid model name")
    if not path.is_file():
        raise FileNotFoundError(name)
    return path


def delete_model(name: str) -> None:
    safe_model_path(name).unlink()


def default_model() -> Path:
    """The model to load at startup: the saved choice, else the best local GGUF."""
    if not MODELS_DIR.exists():
        raise RuntimeError(f"models dir missing: {MODELS_DIR}")
    saved = app_settings.get("active_model")
    if saved:
        try:
            return safe_model_path(saved)
        except (ValueError, FileNotFoundError):
            pass  # saved model was deleted — fall back
    candidates = sorted(MODELS_DIR.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"no .gguf models in {MODELS_DIR}")
    for pref in ("Q4_K_M", "Q4_K_S", "Q4_0"):
        for c in candidates:
            if pref in c.name:
                return c
    return candidates[0]


def gpu_layers_for(name: Optional[str]) -> int:
    """GPU layers for a model: per-model override from settings, else env, else 999 (all)."""
    overrides = app_settings.get("gpu_layers", {})
    if name and isinstance(overrides, dict) and isinstance(overrides.get(name), int):
        return overrides[name]
    return int(os.environ.get("LLAMA_N_GPU_LAYERS", "999"))


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #
def _client() -> httpx.Client:
    headers = {"User-Agent": "LocalChat/0.3"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(
        follow_redirects=True, timeout=httpx.Timeout(30.0, read=60.0), headers=headers
    )


def valid_repo_id(repo_id: str) -> bool:
    return bool(REPO_RE.match(repo_id or ""))


def search_repos(query: str, limit: int = 20) -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get(
            f"{HF_BASE}/api/models",
            params={
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": limit,
            },
        )
        r.raise_for_status()
        return [
            {"repo_id": m["id"], "downloads": m.get("downloads", 0), "likes": m.get("likes", 0)}
            for m in r.json()
            if isinstance(m, dict) and "id" in m
        ]


def list_repo_files(repo_id: str) -> list[dict[str, Any]]:
    """GGUF files at the root of a repo, smallest first."""
    with _client() as c:
        r = c.get(f"{HF_BASE}/api/models/{repo_id}/tree/main")
        r.raise_for_status()
        entries = r.json()
    installed = {m["name"] for m in list_local()}
    out = []
    for e in entries:
        if not isinstance(e, dict) or e.get("type") != "file":
            continue
        path = e.get("path", "")
        if "/" in path or not path.lower().endswith(".gguf"):
            continue
        size = (e.get("lfs") or {}).get("size", e.get("size", 0)) or 0
        split = bool(SPLIT_RE.search(path))
        mmproj = "mmproj" in path.lower()
        note = "split into several files (not supported yet)" if split else (
            "vision projector, not a chat model" if mmproj else None
        )
        out.append(
            {
                "filename": path,
                "size_bytes": size,
                "quant": quant_of(path),
                "fit": fit_estimate(size),
                "installed": path in installed,
                "downloadable": not split and not mmproj and bool(FILE_RE.match(path)),
                "note": note,
            }
        )
    return sorted(out, key=lambda f: f["size_bytes"])


# --------------------------------------------------------------------------- #
# Models already on this PC (Ollama, LM Studio, any folder)
# --------------------------------------------------------------------------- #
MAX_SCAN_FILES = 500
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_gguf(path: Path) -> bool:
    """True if the file starts with the GGUF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def ollama_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


def lmstudio_dirs() -> list[Path]:
    home = Path.home()
    return [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]


def _dest_name(display: str) -> str:
    name = re.sub(r"[^\w.\-+]+", "_", display).strip("._") or "model"
    return name if name.lower().endswith(".gguf") else name + ".gguf"


def _candidate(path: Path, name: str, source: str) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "path": str(path),
        "name": name,
        "size_bytes": size,
        "source": source,
        "quant": quant_of(name),
        "fit": fit_estimate(size),
    }


def _scan_ollama() -> list[dict[str, Any]]:
    root = ollama_dir()
    manifests, blobs = root / "manifests", root / "blobs"
    if not manifests.is_dir():
        return []
    out = []
    for mf in itertools.islice((p for p in manifests.rglob("*") if p.is_file()), MAX_SCAN_FILES):
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for layer in data.get("layers", []) if isinstance(data, dict) else []:
            digest = str(layer.get("digest", ""))
            if not str(layer.get("mediaType", "")).endswith("image.model") or not DIGEST_RE.match(digest):
                continue
            blob = blobs / digest.replace(":", "-")
            if not (blob.is_file() and is_gguf(blob)):
                continue
            parts = list(mf.relative_to(manifests).parts)[1:]  # drop registry host
            if parts and parts[0] == "library":
                parts = parts[1:]
            out.append(_candidate(blob, _dest_name("-".join(parts)), "ollama"))
    return out


def _scan_dir(folder: Path, source: str) -> list[dict[str, Any]]:
    if not folder.is_dir():
        return []
    ours = MODELS_DIR.resolve()
    out = []
    files = (p for p in folder.rglob("*.gguf") if p.is_file())
    for p in itertools.islice(files, MAX_SCAN_FILES):
        low = p.name.lower()
        if "mmproj" in low or SPLIT_RE.search(low) or ours in p.resolve().parents:
            continue
        if is_gguf(p):
            out.append(_candidate(p, p.name, source))
    return out


def scan_local(folder: Optional[str] = None) -> list[dict[str, Any]]:
    """Find GGUF models outside data/models: Ollama + LM Studio, or one folder."""
    if folder:
        path = Path(folder).expanduser()
        if not path.is_dir():
            raise ValueError("folder not found")
        found = _scan_dir(path, "folder")
    else:
        found = _scan_ollama()
        for d in lmstudio_dirs():
            found += _scan_dir(d, "lmstudio")
    have = {m["name"] for m in list_local()}
    for c in found:
        c["installed"] = c["name"] in have
    return sorted(found, key=lambda c: (c["source"], c["name"].lower()))


# --------------------------------------------------------------------------- #
# Downloads (background threads, resumable)
# --------------------------------------------------------------------------- #
@dataclass
class DownloadJob:
    id: str
    repo_id: str
    filename: str
    total: int = 0
    downloaded: int = 0
    status: str = "queued"  # queued | downloading | done | error | cancelled
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "total": self.total,
            "downloaded": self.downloaded,
            "status": self.status,
            "error": self.error,
        }


_jobs: dict[str, DownloadJob] = {}
_jobs_lock = threading.Lock()
ACTIVE = ("queued", "downloading")


def start_download(repo_id: str, filename: str, *, threaded: bool = True) -> DownloadJob:
    if not valid_repo_id(repo_id):
        raise ValueError("invalid repo id")
    if not FILE_RE.match(filename or "") or Path(filename).name != filename:
        raise ValueError("invalid file name")
    if SPLIT_RE.search(filename) or "mmproj" in filename.lower():
        raise ValueError("this file cannot be used as a chat model")
    with _jobs_lock:
        for job in _jobs.values():
            if job.filename == filename and job.status in ACTIVE:
                return job  # already downloading
        if (MODELS_DIR / filename).exists():
            raise FileExistsError(filename)
        job = DownloadJob(id=uuid.uuid4().hex[:12], repo_id=repo_id, filename=filename)
        _jobs[job.id] = job
    if threaded:
        threading.Thread(target=_run_download, args=(job,), daemon=True).start()
    return job


def cancel_download(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job is None or job.status not in ACTIVE:
        return False
    job.cancel.set()
    return True


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.started, reverse=True)[:limit]
    return [j.public() for j in jobs]


def _run_download(job: DownloadJob) -> None:
    dest = MODELS_DIR / job.filename
    part = dest.with_name(dest.name + ".part")
    url = f"{HF_BASE}/{job.repo_id}/resolve/main/{quote(job.filename)}"
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        job.status = "downloading"
        for attempt in (1, 2):
            existing = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with _client() as c, c.stream("GET", url, headers=headers) as r:
                if r.status_code == 416 and attempt == 1:
                    part.unlink(missing_ok=True)  # stale partial file: start over
                    continue
                r.raise_for_status()
                resumed = r.status_code == 206
                if not resumed:
                    existing = 0
                job.total = existing + int(r.headers.get("content-length") or 0)
                job.downloaded = existing
                needed = max(job.total - existing, 0) + MIN_FREE_BYTES
                if job.total and shutil.disk_usage(MODELS_DIR).free < needed:
                    raise OSError("not enough free disk space")
                with open(part, "ab" if resumed else "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        if job.cancel.is_set():
                            job.status = "cancelled"  # keep .part so it can resume
                            return
                        f.write(chunk)
                        job.downloaded += len(chunk)
            break
        if job.total and part.stat().st_size != job.total:
            raise OSError("download incomplete, try again to resume")
        os.replace(part, dest)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__


# --------------------------------------------------------------------------- #
# Import a model from elsewhere on this PC
# --------------------------------------------------------------------------- #
def start_import(path: str, name: Optional[str] = None, *, threaded: bool = True) -> DownloadJob:
    """Bring a local GGUF into data/models: hard link if possible, else copy."""
    source = Path(path).expanduser()
    if not source.is_file() or not is_gguf(source):
        raise ValueError("not a GGUF file")
    filename = _dest_name(name) if name else source.name
    if not FILE_RE.match(filename) or Path(filename).name != filename:
        raise ValueError("invalid file name")
    if SPLIT_RE.search(filename) or "mmproj" in filename.lower():
        raise ValueError("this file cannot be used as a chat model")
    with _jobs_lock:
        for job in _jobs.values():
            if job.filename == filename and job.status in ACTIVE:
                return job
        if (MODELS_DIR / filename).exists():
            raise FileExistsError(filename)
        job = DownloadJob(id=uuid.uuid4().hex[:12], repo_id="local", filename=filename)
        _jobs[job.id] = job
    if threaded:
        threading.Thread(target=_run_import, args=(job, source), daemon=True).start()
    return job


def _run_import(job: DownloadJob, source: Path) -> None:
    dest = MODELS_DIR / job.filename
    part = dest.with_name(dest.name + ".part")
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        job.status = "downloading"
        job.total = source.stat().st_size
        try:
            os.link(source, dest)  # instant, no extra disk space (same drive only)
            job.downloaded = job.total
            job.status = "done"
            return
        except FileExistsError:
            raise
        except OSError:
            pass  # other drive or no hard-link support: copy instead
        if shutil.disk_usage(MODELS_DIR).free < job.total + MIN_FREE_BYTES:
            raise OSError("not enough free disk space")
        with open(source, "rb") as src, open(part, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                if job.cancel.is_set():
                    out.close()
                    part.unlink(missing_ok=True)
                    job.status = "cancelled"
                    return
                out.write(chunk)
                job.downloaded += len(chunk)
        os.replace(part, dest)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        part.unlink(missing_ok=True)
        job.status = "error"
        job.error = str(exc) or exc.__class__.__name__
