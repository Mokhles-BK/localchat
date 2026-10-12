"""Phase 5: attachments (images / text files), vision projectors, safer CORS."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import app_settings
import chat_service
import context
import main
import model_manager as mm
import routes_models
import storage
import uploads

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64
WEBP = b"RIFF\x10\x00\x00\x00WEBP" + b"0" * 32


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _chunk(text: str) -> str:
    return json.dumps({"choices": [{"delta": {"content": text}}]})


class FakeServer:
    is_running = True

    def __init__(self, vision=False):
        self.extra_args = ["--mmproj", "p.gguf"] if vision else []
        self.model = Path("m.gguf")
        self.n_gpu_layers = 99
        self.log_path = Path("llama.log")
        self.calls: list[list[dict]] = []

    async def chat_stream_async(self, messages, sink=None, **kwargs):
        self.calls.append(messages)
        sink.append("ok")
        yield _chunk("ok")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "UPLOADS_DIR", tmp_path / "uploads")
    uploads._pending.clear()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(app_settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(mm, "vram_mb", lambda: 4096)
    monkeypatch.setattr(main, "_loading_model", None)
    monkeypatch.setattr(main, "_server", FakeServer(vision=False))
    client = TestClient(main.app)
    client.models = models
    client.uploads = tmp_path / "uploads"
    return client


def upload(client, name, data):
    return client.post("/api/uploads", json={"name": name, "data_base64": b64(data)})


# --- uploads module --------------------------------------------------------- #
def test_sniff_image():
    assert uploads.sniff_image(PNG) == "image/png"
    assert uploads.sniff_image(JPEG) == "image/jpeg"
    assert uploads.sniff_image(WEBP) == "image/webp"
    assert uploads.sniff_image(b"GIF89a....") == "image/gif"
    assert uploads.sniff_image(b"<svg xmlns='x'><script/></svg>") is None
    assert uploads.sniff_image(b"") is None


def test_save_upload_validation(env):
    assert uploads.save_upload("a.png", PNG)["kind"] == "image"
    assert uploads.save_upload("../../evil/notes.md", b"# hi")["name"] == "notes.md"
    for name, data, msg in [
        ("x.svg", b"<svg onload=alert(1)>", "unsupported"),
        ("x.exe", b"MZ\x90\x00", "unsupported"),
        ("empty.txt", b"", "empty"),
        ("bin.txt", b"abc\x00def", "not a text"),
        ("latin.txt", "caf\xe9".encode("latin-1"), "UTF-8"),
        ("big.png", PNG + b"0" * (uploads.MAX_IMAGE_BYTES + 1), "too large"),
        ("big.txt", b"a" * (uploads.MAX_TEXT_BYTES + 1), "too large"),
    ]:
        with pytest.raises(ValueError, match=msg):
            uploads.save_upload(name, data)
    # a text file renamed .png is not an image, and .png extension does not make it one
    with pytest.raises(ValueError):
        uploads.save_upload("fake.png", b"just text")


def test_get_meta_rejects_bad_ids(env):
    assert uploads.get_meta("../../etc/passwd") is None
    assert uploads.get_meta("z" * 32) is None
    assert uploads.get_meta("0" * 32) is None
    meta = uploads.save_upload("a.txt", b"hello")
    assert uploads.get_meta(meta["id"])["name"] == "a.txt"
    assert uploads.read_text(meta["id"], 3) == ("hel", True)
    assert uploads.read_text(meta["id"], 100) == ("hello", False)


def test_purge_keeps_referenced_and_pending(env):
    kept = uploads.save_upload("kept.txt", b"x")
    pending = uploads.save_upload("pending.txt", b"x")
    gone = uploads.save_upload("gone.txt", b"x")
    uploads.mark_attached([kept["id"], gone["id"]])
    removed = uploads.purge_unreferenced({kept["id"]})
    assert removed == 2  # gone.txt and its .json
    assert uploads.get_meta(gone["id"]) is None
    assert uploads.get_meta(kept["id"]) and uploads.get_meta(pending["id"])


# --- upload routes ---------------------------------------------------------- #
def test_upload_and_download_routes(env):
    r = upload(env, "photo.png", PNG)
    assert r.status_code == 201
    meta = r.json()
    got = env.get(f"/api/uploads/{meta['id']}")
    assert got.status_code == 200 and got.content == PNG
    assert got.headers["content-type"] == "image/png"
    assert got.headers["x-content-type-options"] == "nosniff"
    assert env.get("/api/uploads/" + "a" * 32).status_code == 404
    assert env.get("/api/uploads/not-an-id").status_code == 404
    assert env.post("/api/uploads", json={"name": "a.txt", "data_base64": "%%%"}).status_code == 400
    assert upload(env, "a.svg", b"<svg/>").status_code == 400


# --- chat with attachments -------------------------------------------------- #
def test_text_file_is_pasted_into_the_prompt(env):
    conv = storage.create_conversation()
    att = upload(env, "notes.py", b"print('hello')").json()
    r = env.post("/api/chat", json={"conversation_id": conv["id"], "content": "explain", "attachments": [att["id"]]})
    assert r.status_code == 200
    sent = main._server.calls[0][-1]["content"]
    assert sent.startswith("explain") and "[Attached file: notes.py]" in sent and "print('hello')" in sent
    saved = storage.list_messages(conv["id"])[0]
    assert saved["content"] == "explain" and saved["attachments"][0]["name"] == "notes.py"  # UI text stays clean


def test_image_rejected_without_vision(env):
    conv = storage.create_conversation()
    att = upload(env, "p.png", PNG).json()
    r = env.post("/api/chat", json={"conversation_id": conv["id"], "content": "what is this", "attachments": [att["id"]]})
    assert r.status_code == 400 and "vision" in r.json()["detail"]
    assert storage.list_messages(conv["id"]) == []  # nothing saved


def test_image_sent_to_vision_model_only_for_latest_image_message(env, monkeypatch):
    server = FakeServer(vision=True)
    monkeypatch.setattr(main, "_server", server)
    conv = storage.create_conversation()
    first = upload(env, "one.png", PNG).json()
    second = upload(env, "two.jpg", JPEG).json()
    env.post("/api/chat", json={"conversation_id": conv["id"], "content": "first", "attachments": [first["id"]]})
    env.post("/api/chat", json={"conversation_id": conv["id"], "content": "second", "attachments": [second["id"]]})
    sent = server.calls[-1]
    user_msgs = [m for m in sent if m["role"] == "user"]
    # older image: just a note; newest image: a real image part
    assert isinstance(user_msgs[0]["content"], str) and "one.png" in user_msgs[0]["content"]
    parts = user_msgs[1]["content"]
    assert parts[0] == {"type": "text", "text": "second"}
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_attachment_only_message_and_validation(env, monkeypatch):
    conv = storage.create_conversation()
    att = upload(env, "todo.txt", b"buy milk").json()
    r = env.post("/api/chat", json={"conversation_id": conv["id"], "content": "", "attachments": [att["id"]]})
    assert r.status_code == 200
    assert storage.get_conversation(conv["id"])["title"] == "todo.txt"
    assert env.post("/api/chat", json={"conversation_id": conv["id"], "content": "x", "attachments": ["f" * 32]}).status_code == 400
    assert env.post("/api/chat", json={"conversation_id": conv["id"], "content": ""}).status_code == 400


def test_edit_keeps_attachments_and_fork_copies_them(env):
    conv = storage.create_conversation()
    att = upload(env, "a.txt", b"data").json()
    env.post("/api/chat", json={"conversation_id": conv["id"], "content": "q", "attachments": [att["id"]]})
    msgs = storage.list_messages(conv["id"])
    env.post("/api/chat", json={"conversation_id": conv["id"], "edit_message_id": msgs[0]["id"], "content": "q2"})
    assert storage.list_messages(conv["id"])[0]["attachments"][0]["id"] == att["id"]
    new = storage.fork_conversation(conv["id"], msgs[0]["id"])
    assert storage.list_messages(new["id"])[0]["attachments"][0]["id"] == att["id"]


def test_deleting_chat_removes_its_files_but_not_shared_ones(env):
    conv = storage.create_conversation()
    att = upload(env, "a.txt", b"data").json()
    env.post("/api/chat", json={"conversation_id": conv["id"], "content": "q", "attachments": [att["id"]]})
    msgs = storage.list_messages(conv["id"])
    fork = storage.fork_conversation(conv["id"], msgs[0]["id"])
    assert env.delete(f"/api/conversations/{conv['id']}").status_code == 204
    assert uploads.get_meta(att["id"]) is not None  # the fork still uses it
    assert env.delete(f"/api/conversations/{fork['id']}").status_code == 204
    assert uploads.get_meta(att["id"]) is None  # now unused -> removed


# --- context ---------------------------------------------------------------- #
def test_image_parts_count_in_token_estimate():
    plain = [{"role": "user", "content": "hi"}]
    withimg = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]}]
    assert context.estimate_tokens(withimg) - context.estimate_tokens(plain) >= context.IMAGE_TOKENS - 2


def test_trim_budget_follows_ctx_size(monkeypatch):
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": "w" * 900} for i in range(20)] + [
        {"role": "user", "content": "last"}
    ]
    small = context.trim_messages(msgs)
    monkeypatch.setenv("LLAMA_CTX_SIZE", "16384")
    big = context.trim_messages(msgs)
    assert len(big) > len(small) and big[-1]["content"] == "last"


# --- vision projectors ------------------------------------------------------ #
def gguf(path: Path, size=64):
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))


def test_projector_detection_and_lists(env):
    for n in ("gemma-3-4b-it-Q4_K_M.gguf", "mmproj-gemma-3-4b-it-f16.gguf", "other-Q4_K_M.gguf"):
        gguf(env.models / n)
    models = {m["name"]: m for m in mm.list_local()}
    assert set(models) == {"gemma-3-4b-it-Q4_K_M.gguf", "other-Q4_K_M.gguf"}  # projector is not a model
    assert models["gemma-3-4b-it-Q4_K_M.gguf"]["projector"] == "mmproj-gemma-3-4b-it-f16.gguf"
    assert models["other-Q4_K_M.gguf"]["projector"] is None
    assert [p["name"] for p in mm.list_projectors()] == ["mmproj-gemma-3-4b-it-f16.gguf"]
    assert mm.default_model().name != "mmproj-gemma-3-4b-it-f16.gguf"


def test_projector_pairing_precedence(env, monkeypatch):
    gguf(env.models / "model-Q4_K_M.gguf")
    gguf(env.models / "mmproj-model-f16.gguf", 80)
    gguf(env.models / "mmproj-model-Q8_0.gguf", 40)
    # no name/source match for "model-Q4_K_M" beyond stem "model": both contain it; f16 preferred
    assert mm.find_projector("model-Q4_K_M.gguf").name == "mmproj-model-f16.gguf"
    mm.set_pair("model-Q4_K_M.gguf", "mmproj-model-Q8_0.gguf")  # explicit choice wins
    assert mm.find_projector("model-Q4_K_M.gguf").name == "mmproj-model-Q8_0.gguf"
    mm.set_pair("model-Q4_K_M.gguf", "none")
    assert mm.find_projector("model-Q4_K_M.gguf") is None and mm.vision_args("model-Q4_K_M.gguf") == []
    mm.set_pair("model-Q4_K_M.gguf", "auto")
    assert mm.find_projector("model-Q4_K_M.gguf") is not None


def test_projector_pairing_by_source_repo(env):
    gguf(env.models / "gemma-3-4b-it-Q4_K_M.gguf")
    gguf(env.models / "mmproj-model-f16.gguf")  # generic name: only the repo links them
    assert mm.find_projector("gemma-3-4b-it-Q4_K_M.gguf") is None
    mm.record_source("gemma-3-4b-it-Q4_K_M.gguf", "ggml-org/gemma-3-4b-it-GGUF")
    mm.record_source("mmproj-model-f16.gguf", "ggml-org/gemma-3-4b-it-GGUF")
    assert mm.find_projector("gemma-3-4b-it-Q4_K_M.gguf").name == "mmproj-model-f16.gguf"
    args = mm.vision_args("gemma-3-4b-it-Q4_K_M.gguf")
    assert args[0] == "--mmproj" and args[1].endswith("mmproj-model-f16.gguf")


def test_mmproj_cpu_env(env, monkeypatch):
    gguf(env.models / "m-Q4_K_M.gguf")
    gguf(env.models / "mmproj-m-f16.gguf")
    monkeypatch.setenv("LLAMA_MMPROJ_CPU", "1")
    assert "--no-mmproj-offload" in mm.vision_args("m-Q4_K_M.gguf")


def test_download_records_source(env, monkeypatch):
    import httpx

    monkeypatch.setattr(mm, "_client", lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"G" * 100))))
    job = mm.start_download("a/b", "mmproj-x-f16.gguf", threaded=False)  # projectors can be downloaded now
    mm._run_download(job)
    assert job.status == "done", job.error
    assert app_settings.get("sources")["mmproj-x-f16.gguf"] == "a/b"


class FakeLlama:
    fail_with_projector = False
    fail_all = False
    instances: list = []

    def __init__(self, binary, model, host="127.0.0.1", port=8080, n_gpu_layers=999, extra_args=None):
        self.model = Path(model)
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = list(extra_args or [])
        self.log_path = Path("llama.log")
        self.started = self.closed = False
        FakeLlama.instances.append(self)

    @property
    def is_running(self):
        return self.started and not self.closed

    def start(self):
        if FakeLlama.fail_all or (FakeLlama.fail_with_projector and self.extra_args):
            raise RuntimeError("projector incompatible")
        self.started = True

    def close(self):
        self.closed = True


@pytest.fixture
def vision_env(env, monkeypatch):
    monkeypatch.setattr(routes_models, "LlamaServer", FakeLlama)
    FakeLlama.fail_with_projector = FakeLlama.fail_all = False
    FakeLlama.instances = []
    gguf(env.models / "a-Q4_K_M.gguf")
    gguf(env.models / "v-Q4_K_M.gguf")
    gguf(env.models / "mmproj-v-f16.gguf")
    current = FakeLlama(None, env.models / "a-Q4_K_M.gguf")
    current.start()
    monkeypatch.setattr(main, "_server", current)
    return env


def test_activate_loads_projector_and_reports_vision(vision_env):
    r = vision_env.post("/api/models/activate", json={"name": "v-Q4_K_M.gguf"})
    assert r.status_code == 200 and r.json()["vision"] is True and r.json()["warning"] is None
    assert main._server.extra_args[0] == "--mmproj"
    assert vision_env.get("/api/health").json()["vision"] is True
    r = vision_env.post("/api/models/activate", json={"name": "a-Q4_K_M.gguf"})
    assert r.json()["vision"] is False and main._server.extra_args == []


def test_activate_falls_back_to_text_only(vision_env):
    FakeLlama.fail_with_projector = True
    r = vision_env.post("/api/models/activate", json={"name": "v-Q4_K_M.gguf"})
    assert r.status_code == 200
    assert r.json()["vision"] is False and "loaded without image support" in r.json()["warning"]
    assert main._server.model.name == "v-Q4_K_M.gguf" and main._server.is_running


def test_activate_projector_file_is_refused(vision_env):
    r = vision_env.post("/api/models/activate", json={"name": "mmproj-v-f16.gguf"})
    assert r.status_code == 400 and "projector" in r.json()["detail"]


def test_pair_endpoint(vision_env):
    gguf(vision_env.models / "mmproj-other-f16.gguf")
    r = vision_env.post("/api/models/pair", json={"model": "a-Q4_K_M.gguf", "projector": "mmproj-other-f16.gguf"})
    assert r.status_code == 200 and r.json()["projector"] == "mmproj-other-f16.gguf"
    assert r.json()["restart_needed"] is True  # a-Q4_K_M is the running model
    r = vision_env.post("/api/models/pair", json={"model": "a-Q4_K_M.gguf", "projector": "none"})
    assert r.json()["projector"] is None
    assert vision_env.post("/api/models/pair", json={"model": "a-Q4_K_M.gguf", "projector": "nope.gguf"}).status_code == 404
    assert vision_env.post("/api/models/pair", json={"model": "a-Q4_K_M.gguf", "projector": "v-Q4_K_M.gguf"}).status_code == 400
    assert vision_env.post("/api/models/pair", json={"model": "mmproj-v-f16.gguf", "projector": "auto"}).status_code == 400


def test_cannot_delete_projector_in_use(vision_env):
    vision_env.post("/api/models/activate", json={"name": "v-Q4_K_M.gguf"})
    assert vision_env.delete("/api/models/mmproj-v-f16.gguf").status_code == 409
    vision_env.post("/api/models/activate", json={"name": "a-Q4_K_M.gguf"})
    assert vision_env.delete("/api/models/mmproj-v-f16.gguf").status_code == 204


def test_models_list_includes_projectors(vision_env):
    data = vision_env.get("/api/models").json()
    assert [p["name"] for p in data["projectors"]] == ["mmproj-v-f16.gguf"]
    assert {m["name"]: m["projector"] for m in data["models"]}["v-Q4_K_M.gguf"] == "mmproj-v-f16.gguf"


# --- CORS ------------------------------------------------------------------- #
def test_cors_only_allows_the_local_ui(env):
    ok = env.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = env.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
