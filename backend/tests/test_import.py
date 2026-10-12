"""Scan + import of GGUF models that already exist on this PC."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import app_settings
import main
import model_manager as mm

HEX = "a" * 64


@pytest.fixture
def env(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(app_settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(mm, "vram_mb", lambda: 4096)
    monkeypatch.setattr(main, "_server", None)
    mm._jobs.clear()

    ollama = tmp_path / "ollama"
    (ollama / "blobs").mkdir(parents=True)
    monkeypatch.setattr(mm, "ollama_dir", lambda: ollama)
    lm = tmp_path / "lmstudio"
    monkeypatch.setattr(mm, "lmstudio_dirs", lambda: [lm])
    return type("Env", (), {"models": models, "ollama": ollama, "lm": lm, "tmp": tmp_path})


def gguf(path: Path, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


def add_ollama(env, rel: str, digest_hex: str = HEX, media="application/vnd.ollama.image.model"):
    manifest = env.ollama / "manifests" / rel
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"layers": [{"mediaType": media, "digest": f"sha256:{digest_hex}", "size": 64}]})
    )
    return env.ollama / "blobs" / f"sha256-{digest_hex}"


def test_is_gguf(tmp_path):
    assert mm.is_gguf(gguf(tmp_path / "a.gguf"))
    (tmp_path / "b.gguf").write_bytes(b"nope")
    assert not mm.is_gguf(tmp_path / "b.gguf")
    assert not mm.is_gguf(tmp_path / "missing.gguf")


def test_scan_ollama_names_and_filters(env):
    gguf(add_ollama(env, "registry.ollama.ai/library/qwen2.5-coder/3b", "b" * 64))
    gguf(add_ollama(env, "registry.ollama.ai/someone/mymodel/latest", "c" * 64))
    # not a model layer
    add_ollama(env, "registry.ollama.ai/library/other/1b", "d" * 64, media="application/vnd.ollama.image.template")
    # model layer but blob missing
    add_ollama(env, "registry.ollama.ai/library/ghost/1b", "e" * 64)
    # manifest with a path-trick digest is ignored
    bad = env.ollama / "manifests" / "registry.ollama.ai" / "library" / "evil" / "1b"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"layers": [{"mediaType": "x.image.model", "digest": "sha256:../../etc"}]}))

    found = mm.scan_local()
    names = [c["name"] for c in found]
    assert names == ["qwen2.5-coder-3b.gguf", "someone-mymodel-latest.gguf"]
    assert all(c["source"] == "ollama" and c["installed"] is False for c in found)


def test_scan_lmstudio_and_folder(env):
    gguf(env.lm / "pub" / "repo" / "m-Q4_K_M.gguf")
    gguf(env.lm / "pub" / "repo" / "mmproj-F16.gguf")  # listed, as a projector
    gguf(env.lm / "pub" / "repo" / "big-00001-of-00002.gguf")  # skipped
    (env.lm / "pub" / "repo" / "fake.gguf").write_bytes(b"text")  # not GGUF
    found = mm.scan_local()
    assert [(c["name"], c["kind"]) for c in found] == [("m-Q4_K_M.gguf", "model"), ("mmproj-F16.gguf", "projector")]
    assert found[0]["quant"] == "Q4_K_M" and found[0]["source"] == "lmstudio"

    other = env.tmp / "Downloads"
    gguf(other / "x-Q8_0.gguf")
    gguf(env.models / "ours.gguf")  # our own folder is never listed
    found = mm.scan_local(str(other))
    assert [c["name"] for c in found] == ["x-Q8_0.gguf"] and found[0]["source"] == "folder"
    with pytest.raises(ValueError):
        mm.scan_local(str(env.tmp / "nope"))


def test_scan_marks_installed(env):
    gguf(env.lm / "m.gguf")
    gguf(env.models / "m.gguf")
    assert mm.scan_local()[0]["installed"] is True


def test_import_hardlink_or_copy(env):
    src = gguf(env.tmp / "src" / "m.gguf", 4096)
    job = mm.start_import(str(src), threaded=False)
    mm._run_import(job, src)
    assert job.status == "done", job.error
    dest = env.models / "m.gguf"
    assert dest.read_bytes() == src.read_bytes()
    assert os.path.samefile(src, dest)  # same drive in tests -> hard link
    # deleting our copy must not touch the original
    mm.delete_model("m.gguf")
    assert src.exists()


def test_import_copy_fallback(env, monkeypatch):
    src = gguf(env.tmp / "src" / "m.gguf", 3 * (1 << 20) + 5)

    def no_link(a, b):
        raise OSError("cross-device")

    monkeypatch.setattr(mm.os, "link", no_link)
    job = mm.start_import(str(src), name="renamed", threaded=False)
    assert job.filename == "renamed.gguf"
    mm._run_import(job, src)
    assert job.status == "done", job.error
    dest = env.models / "renamed.gguf"
    assert dest.read_bytes() == src.read_bytes() and not os.path.samefile(src, dest)
    assert job.downloaded == job.total == src.stat().st_size


def test_import_cancel_copy(env, monkeypatch):
    src = gguf(env.tmp / "src" / "m.gguf", 2 * (1 << 20))
    monkeypatch.setattr(mm.os, "link", lambda a, b: (_ for _ in ()).throw(OSError("x")))
    job = mm.start_import(str(src), threaded=False)
    job.cancel.set()
    mm._run_import(job, src)
    assert job.status == "cancelled"
    assert not (env.models / "m.gguf").exists() and not (env.models / "m.gguf.part").exists()


def test_import_validation(env):
    (env.tmp / "t.gguf").write_bytes(b"text")
    with pytest.raises(ValueError):
        mm.start_import(str(env.tmp / "t.gguf"), threaded=False)
    with pytest.raises(ValueError):
        mm.start_import(str(env.tmp / "missing.gguf"), threaded=False)
    src = gguf(env.tmp / "ok.gguf")
    # path tricks in the requested name are neutralised, never followed
    job = mm.start_import(str(src), name="../evil", threaded=False)
    assert job.filename == "evil.gguf"
    mm._jobs.clear()
    with pytest.raises(ValueError):
        mm.start_import(str(gguf(env.tmp / "m-00001-of-00002.gguf")), threaded=False)
    gguf(env.models / "ok.gguf")
    with pytest.raises(FileExistsError):
        mm.start_import(str(src), threaded=False)


def test_routes_scan_and_import(env):
    client = TestClient(main.app)
    gguf(env.lm / "m-Q4_K_M.gguf")
    r = client.get("/api/models/scan")
    assert r.status_code == 200 and r.json()["candidates"][0]["name"] == "m-Q4_K_M.gguf"
    assert client.get("/api/models/scan?folder=" + str(env.tmp / "nope")).status_code == 400

    path = r.json()["candidates"][0]["path"]
    r = client.post("/api/models/import", json={"path": path})
    assert r.status_code == 201
    import time

    for _ in range(100):
        job = client.get("/api/models/downloads").json()["downloads"][0]
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    assert (env.models / "m-Q4_K_M.gguf").exists()
    assert client.post("/api/models/import", json={"path": path}).status_code == 409
    assert client.post("/api/models/import", json={"path": str(env.tmp / "nope.gguf")}).status_code == 400
