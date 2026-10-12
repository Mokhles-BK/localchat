"""Model manager routes: search/download from Hugging Face, delete, switch."""

from __future__ import annotations

import threading
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app_settings
import model_manager
from llama_server import LlamaServer

router = APIRouter(prefix="/api/models", tags=["models"])

_switch_lock = threading.Lock()


class DownloadBody(BaseModel):
    repo_id: str
    filename: str


class ImportBody(BaseModel):
    path: str
    name: Optional[str] = None


class ActivateBody(BaseModel):
    name: str
    n_gpu_layers: Optional[int] = Field(None, ge=0, le=999)


def _hf_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=502, detail=f"Hugging Face request failed: {exc}")


@router.get("/search")
def search(q: str = "", limit: int = 20):
    if not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    try:
        return {"results": model_manager.search_repos(q.strip(), max(1, min(limit, 50)))}
    except (httpx.HTTPError, ValueError) as exc:
        raise _hf_error(exc) from exc


@router.get("/files")
def files(repo_id: str):
    if not model_manager.valid_repo_id(repo_id):
        raise HTTPException(status_code=400, detail="invalid repo id")
    try:
        return {"repo_id": repo_id, "files": model_manager.list_repo_files(repo_id)}
    except (httpx.HTTPError, ValueError) as exc:
        raise _hf_error(exc) from exc


@router.get("/scan")
def scan(folder: Optional[str] = None):
    """Models already on this PC (Ollama + LM Studio, or one folder)."""
    try:
        return {"candidates": model_manager.scan_local(folder)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/import", status_code=201)
def import_model(body: ImportBody):
    try:
        job = model_manager.start_import(body.path, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="a model with that name already exists") from exc
    return job.public()


@router.post("/download", status_code=201)
def download(body: DownloadBody):
    try:
        job = model_manager.start_download(body.repo_id, body.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="model already downloaded") from exc
    return job.public()


@router.get("/downloads")
def downloads():
    return {"downloads": model_manager.list_jobs()}


@router.delete("/downloads/{job_id}", status_code=204)
def cancel_download(job_id: str):
    if not model_manager.cancel_download(job_id):
        raise HTTPException(status_code=404, detail="no active download with that id")


def _start_server(app, path, layers: int, extra_args: list[str]):
    """Start llama-server for `path`; closes it again if it fails to come up."""
    server = LlamaServer(
        binary=app._default_binary(),
        model=path,
        host="127.0.0.1",
        port=8080,
        n_gpu_layers=layers,
        extra_args=extra_args,
    )
    try:
        server.start()
    except Exception:  # noqa: BLE001
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    return server


@router.post("/activate")
def activate(body: ActivateBody):
    """Stop the current llama-server and start another model.

    Sync on purpose: FastAPI runs it in a worker thread, so the event loop
    (and /api/health, which reports `loading_model`) stays responsive while
    the model loads.  If the model has a vision projector and loading with it
    fails, the model is loaded without vision instead (with a warning).
    """
    import main as app  # late import: main imports this module

    try:
        path = model_manager.safe_model_path(body.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if model_manager.is_projector(path.name):
        raise HTTPException(status_code=400, detail="that file is a vision projector, not a chat model")

    if not _switch_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a model switch is already in progress")
    try:
        old = app._server
        old_model = old.model if old is not None else None
        old_layers = old.n_gpu_layers if old is not None else None
        old_args = list(getattr(old, "extra_args", [])) if old is not None else []
        app._loading_model = path.name
        app._server = None  # chat answers 503 "model is loading" meanwhile
        if old is not None:
            old.close()

        layers = body.n_gpu_layers if body.n_gpu_layers is not None else model_manager.gpu_layers_for(path.name)
        extra = model_manager.vision_args(path.name)
        warning = None
        new = None
        failure: Exception | None = None
        try:
            new = _start_server(app, path, layers, extra)
        except Exception as exc:  # noqa: BLE001
            failure = exc
            if extra:
                try:
                    new = _start_server(app, path, layers, [])
                    warning = f"vision projector failed to load ({exc}); loaded without image support"
                    failure = None
                except Exception as exc2:  # noqa: BLE001
                    failure = exc2

        if new is None:
            restored = False
            if old_model is not None and old_model.exists():
                try:
                    app._server = _start_server(
                        app, old_model, old_layers if old_layers is not None else 999, old_args
                    )
                    restored = True
                except Exception:  # noqa: BLE001
                    pass
            suffix = "" if restored else " (the previous model could not be restored either)"
            raise HTTPException(
                status_code=500, detail=f"could not load {path.name}: {failure}{suffix}"
            ) from failure

        app._server = new
        app_settings.set_value("active_model", path.name)
        if body.n_gpu_layers is not None:
            overrides = app_settings.get("gpu_layers", {})
            overrides = overrides if isinstance(overrides, dict) else {}
            overrides[path.name] = body.n_gpu_layers
            app_settings.set_value("gpu_layers", overrides)
        return {
            "model": path.name,
            "n_gpu_layers": layers,
            "vision": app._has_vision(new),
            "warning": warning,
        }
    finally:
        app._loading_model = None
        _switch_lock.release()


class PairBody(BaseModel):
    model: str
    projector: str  # "auto", "none" or a projector file name


@router.post("/pair")
def pair(body: PairBody):
    """Choose which vision projector a model uses ("auto" = detect, "none" = text only)."""
    import main as app

    try:
        path = model_manager.safe_model_path(body.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if model_manager.is_projector(path.name):
        raise HTTPException(status_code=400, detail="that file is a vision projector, not a chat model")
    if body.projector not in ("auto", "none"):
        try:
            proj = model_manager.safe_model_path(body.projector)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="projector not found") from exc
        if not model_manager.is_projector(proj.name):
            raise HTTPException(status_code=400, detail="that file is not a vision projector")
    model_manager.set_pair(path.name, body.projector)
    resolved = model_manager.find_projector(path.name)
    server = app._server
    return {
        "model": path.name,
        "projector": resolved.name if resolved else None,
        # A running model only picks the change up when it is loaded again.
        "restart_needed": server is not None and server.model.name == path.name,
    }


@router.delete("/{name}", status_code=204)
def delete(name: str):
    import main as app

    try:
        path = model_manager.safe_model_path(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    server = app._server
    if server is not None and server.model.name == path.name:
        raise HTTPException(status_code=409, detail="cannot delete the active model; switch first")
    if server is not None and str(path) in [str(a) for a in getattr(server, "extra_args", [])]:
        raise HTTPException(status_code=409, detail="this projector is in use by the active model")
    if app._loading_model == path.name:
        raise HTTPException(status_code=409, detail="model is loading")
    model_manager.delete_model(name)
