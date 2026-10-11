"""Model routes: activate (with rollback), delete, download, HF errors."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

import app_settings
import main
import model_manager as mm
import routes_models


class FakeLlama:
    """Stands in for LlamaServer; class attrs let tests make loads fail."""

    fail_for: set[str] = set()
    instances: list["FakeLlama"] = []

    def __init__(self, binary, model, host="127.0.0.1", port=8080, n_gpu_layers=999, extra_args=None):
        self.model = Path(model)
        self.n_gpu_layers = n_gpu_layers
        self.started = False
        self.closed = False
        FakeLlama.instances.append(self)

    @property
    def is_running(self):
        return self.started and not self.closed

    def start(self):
        if self.model.name in FakeLlama.fail_for:
            raise RuntimeError("out of memory")
        self.started = True

    def close(self):
        self.closed = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for n in ("a-Q4_K_M.gguf", "b-Q4_K_M.gguf"):
        (models / n).write_bytes(b"1")
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(app_settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(mm, "vram_mb", lambda: 4096)
    monkeypatch.setattr(routes_models, "LlamaServer", FakeLlama)
    FakeLlama.fail_for = set()
    FakeLlama.instances = []
    mm._jobs.clear()
    current = FakeLlama(None, models / "a-Q4_K_M.gguf")
    current.start()
    monkeypatch.setattr(main, "_server", current)
    monkeypatch.setattr(main, "_loading_model", None)
    client = TestClient(main.app)
    client.models = models
    client.current = current
    return client


def test_activate_switches_and_persists(env):
    r = env.post("/api/models/activate", json={"name": "b-Q4_K_M.gguf"})
    assert r.status_code == 200, r.text
    assert env.current.closed is True
    assert main._server.model.name == "b-Q4_K_M.gguf" and main._server.is_running
    assert app_settings.get("active_model") == "b-Q4_K_M.gguf"
    assert main._loading_model is None


def test_activate_saves_gpu_layers(env):
    env.post("/api/models/activate", json={"name": "b-Q4_K_M.gguf", "n_gpu_layers": 20})
    assert main._server.n_gpu_layers == 20
    assert mm.gpu_layers_for("b-Q4_K_M.gguf") == 20


def test_activate_failure_restores_previous(env):
    FakeLlama.fail_for = {"b-Q4_K_M.gguf"}
    r = env.post("/api/models/activate", json={"name": "b-Q4_K_M.gguf"})
    assert r.status_code == 500
    assert "out of memory" in r.json()["detail"]
    assert main._server is not None and main._server.model.name == "a-Q4_K_M.gguf"
    assert main._server.is_running
    assert app_settings.get("active_model") is None  # not saved
    assert main._loading_model is None


def test_activate_failure_and_no_restore(env):
    FakeLlama.fail_for = {"b-Q4_K_M.gguf", "a-Q4_K_M.gguf"}
    r = env.post("/api/models/activate", json={"name": "b-Q4_K_M.gguf"})
    assert r.status_code == 500 and "could not be restored" in r.json()["detail"]
    assert main._server is None


def test_activate_unknown_and_invalid(env):
    assert env.post("/api/models/activate", json={"name": "nope.gguf"}).status_code == 404
    assert env.post("/api/models/activate", json={"name": "../a-Q4_K_M.gguf"}).status_code == 400
    assert env.post("/api/models/activate", json={"name": "a.txt"}).status_code == 400


def test_chat_503_while_loading(env, monkeypatch):
    monkeypatch.setattr(main, "_loading_model", "b-Q4_K_M.gguf")
    r = env.post("/api/chat", json={"conversation_id": 1, "content": "hi"})
    assert r.status_code == 503 and "loading" in r.json()["detail"]


def test_delete_model(env):
    assert env.delete("/api/models/a-Q4_K_M.gguf").status_code == 409  # active
    assert (env.models / "a-Q4_K_M.gguf").exists()
    assert env.delete("/api/models/b-Q4_K_M.gguf").status_code == 204
    assert not (env.models / "b-Q4_K_M.gguf").exists()
    assert env.delete("/api/models/b-Q4_K_M.gguf").status_code == 404
    assert env.delete("/api/models/readme.txt").status_code == 400


def test_health_and_models_list(env):
    r = env.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert names == ["a-Q4_K_M.gguf", "b-Q4_K_M.gguf"]
    assert r.json()["models"][0]["quant"] == "Q4_K_M"


def test_download_validation_routes(env):
    assert env.post("/api/models/download", json={"repo_id": "bad", "filename": "x.gguf"}).status_code == 400
    assert env.post("/api/models/download", json={"repo_id": "a/b", "filename": "../x.gguf"}).status_code == 400
    assert env.post("/api/models/download", json={"repo_id": "a/b", "filename": "a-Q4_K_M.gguf"}).status_code == 409
    assert env.delete("/api/models/downloads/doesnotexist").status_code == 404


def test_download_end_to_end(env, monkeypatch):
    payload = b"Z" * 5000
    monkeypatch.setattr(
        mm, "_client", lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=payload)))
    )
    r = env.post("/api/models/download", json={"repo_id": "a/b", "filename": "c-Q4_K_M.gguf"})
    assert r.status_code == 201
    for _ in range(100):
        jobs = env.get("/api/models/downloads").json()["downloads"]
        if jobs[0]["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert jobs[0]["status"] == "done", jobs
    assert (env.models / "c-Q4_K_M.gguf").read_bytes() == payload


def test_search_and_files_errors(env, monkeypatch):
    assert env.get("/api/models/search?q=").status_code == 400
    assert env.get("/api/models/files?repo_id=bad").status_code == 400

    def boom(req):
        raise httpx.ConnectError("no internet")

    monkeypatch.setattr(mm, "_client", lambda: httpx.Client(transport=httpx.MockTransport(boom)))
    r = env.get("/api/models/search?q=llama")
    assert r.status_code == 502 and "Hugging Face" in r.json()["detail"]
    assert env.get("/api/models/files?repo_id=a/b").status_code == 502
