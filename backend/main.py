"""LocalChat backend — FastAPI entrypoint.

Exposes:
  GET  /api/health          — backend + llama-server status
  POST /api/chat            — stream a model reply as Server-Sent Events
  GET  /api/models          — list GGUF models available in data/models

llama-server is started once, eagerly, in the FastAPI lifespan handler so the
first chat message never blocks on a 120 s cold start.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from llama_server import LlamaServer, register_exit_handler
from routes_conversations import router
from schemas import ChatRequest
import storage
from chat_service import build_context, effective_settings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
BIN_DIR = DATA_DIR / "bin"

# Single shared llama-server instance for the whole process.
_server: Optional[LlamaServer] = None


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_binary() -> Path:
    return BIN_DIR / "llama-server.exe"


def _default_model() -> Path:
    """Pick the first GGUF in data/models, preferring Q4_K_M."""
    if not MODELS_DIR.exists():
        raise RuntimeError(f"models dir missing: {MODELS_DIR}")
    candidates = sorted(MODELS_DIR.glob("*.gguf"))
    if not candidates:
        raise RuntimeError(f"no .gguf models in {MODELS_DIR}")
    for pref in ("Q4_K_M", "Q4_K_S", "Q4_0"):
        for c in candidates:
            if pref in c.name:
                return c
    return candidates[0]


def _model_status() -> dict[str, Any]:
    if not MODELS_DIR.exists():
        return {"models_dir": str(MODELS_DIR), "exists": False, "models": []}
    models = [
        {"name": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(MODELS_DIR.glob("*.gguf"))
    ]
    return {"models_dir": str(MODELS_DIR), "exists": True, "models": models}


# --------------------------------------------------------------------------- #
# Lifespan: start llama-server eagerly at startup, stop on shutdown.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server
    try:
        _server = LlamaServer(
            binary=_default_binary(),
            model=_default_model(),
            host="127.0.0.1",
            port=8080,
            n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "999")),
        )
        _server.start()
        register_exit_handler(_server)
    except Exception as exc:  # noqa: BLE001 - keep the app up, /api/health reports it
        _server = None
        print(f"[startup] llama-server failed to start: {exc!r}", flush=True)
    yield
    if _server is not None:
        _server.close()
    storage.close()


app = FastAPI(title="LocalChat", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report backend + llama-server health (sync — never blocks on I/O)."""
    server_ok = False
    server_detail: Any = "not started"
    if _server is not None:
        try:
            r = _server._client.get(f"{_server.base_url}/health", timeout=5)  # noqa: SLF001
            server_ok = r.status_code == 200
            server_detail = r.text
        except Exception as exc:  # noqa: BLE001
            server_detail = f"error: {exc!r}"
    return {
        "status": "ok",
        "llama_server": server_ok,
        "llama_server_detail": server_detail,
        "model": _server.model.name if _server else None,
        "log_file": str(_server.log_path) if _server else None,
        **_model_status(),
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return _model_status()


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the model's reply token-by-token as Server-Sent Events.

    The database is the source of truth for history: the client sends only
    the new user message (or ``regenerate: true``), and the server builds the
    model context from the stored system prompt and stored messages.

    The route is *async* so Starlette cancels the response task when the
    client disconnects (Stop).  Cancellation, errors and normal completion all
    end in the ``finally`` block below, which is the ONE place the assistant
    reply (full or partial) is saved.
    """
    server = _server
    if server is None or not server.is_running:
        raise HTTPException(
            status_code=503,
            detail="llama-server is not running — check /api/health for the log file",
        )

    conv = storage.get_conversation(req.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    settings = effective_settings(conv, req)

    if req.regenerate:
        # Re-answer from the stored history; needs an assistant reply to drop.
        if not any(m["role"] == "assistant" for m in storage.list_messages(req.conversation_id)):
            raise HTTPException(status_code=400, detail="nothing to regenerate")
        storage.delete_last_assistant_message(req.conversation_id)
    else:
        if not req.content.strip():
            raise HTTPException(status_code=400, detail="content is required")
        # Save the user message BEFORE streaming so it survives a cancel.
        storage.add_message(req.conversation_id, "user", req.content)
        if not conv["title"]:
            storage.auto_title(req.conversation_id, req.content)

    history = storage.list_messages(req.conversation_id)
    messages = build_context(conv, history, None)

    async def event_generator():
        sink: list[str] = []
        try:
            async for data in server.chat_stream_async(
                messages,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                max_tokens=settings["max_tokens"],
                sink=sink,
            ):
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Runs on completion, error AND client-disconnect cancellation.
            text = "".join(sink)
            if text.strip():
                storage.add_message(req.conversation_id, "assistant", text)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
