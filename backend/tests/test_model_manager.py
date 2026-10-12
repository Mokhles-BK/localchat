"""model_manager: local files, Hugging Face parsing, resumable downloads."""

from __future__ import annotations

import httpx
import pytest

import app_settings
import model_manager as mm


@pytest.fixture(autouse=True)
def isolated_models(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(mm, "MODELS_DIR", models)
    monkeypatch.setattr(app_settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(mm, "vram_mb", lambda: 4096)
    mm._jobs.clear()
    return models


def mock_hf(monkeypatch, handler):
    monkeypatch.setattr(
        mm, "_client", lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    )


# --- local files ----------------------------------------------------------- #
def test_quant_of():
    assert mm.quant_of("Llama-3.2-3B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert mm.quant_of("model-Q8_0.gguf") == "Q8_0"
    assert mm.quant_of("model-IQ3_XXS.gguf") == "IQ3_XXS"
    assert mm.quant_of("model-f16.gguf") == "F16"
    assert mm.quant_of("model-Q3_K_XL.gguf") == "Q3_K_XL"
    assert mm.quant_of("model-Q4_K_L.gguf") == "Q4_K_L"
    assert mm.quant_of("weird.gguf") is None


def test_fit_estimate(monkeypatch):
    gib = 1024**3
    assert mm.fit_estimate(2 * gib) == "gpu"  # 3B Q4 on a 4 GB card
    assert mm.fit_estimate(int(4.4 * gib)) == "partial"  # 7B Q4
    assert mm.fit_estimate(40 * gib) == "too_big"
    monkeypatch.setattr(mm, "vram_mb", lambda: None)
    assert mm.fit_estimate(2 * gib) == "unknown"


def test_list_local_ignores_part_files(isolated_models):
    (isolated_models / "a-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (isolated_models / "b.gguf.part").write_bytes(b"x")
    names = [m["name"] for m in mm.list_local()]
    assert names == ["a-Q4_K_M.gguf"]


@pytest.mark.parametrize("bad", ["../x.gguf", "..\\x.gguf", "sub/x.gguf", "x.txt", "", "C:\\x.gguf"])
def test_safe_model_path_rejects_tricks(bad):
    with pytest.raises(ValueError):
        mm.safe_model_path(bad)


def test_safe_model_path_missing_and_ok(isolated_models):
    with pytest.raises(FileNotFoundError):
        mm.safe_model_path("nope.gguf")
    (isolated_models / "ok.gguf").write_bytes(b"1")
    assert mm.safe_model_path("ok.gguf").name == "ok.gguf"


def test_default_model_prefers_saved_then_q4km(isolated_models):
    (isolated_models / "a-Q8_0.gguf").write_bytes(b"1")
    (isolated_models / "b-Q4_K_M.gguf").write_bytes(b"1")
    assert mm.default_model().name == "b-Q4_K_M.gguf"
    app_settings.set_value("active_model", "a-Q8_0.gguf")
    assert mm.default_model().name == "a-Q8_0.gguf"
    app_settings.set_value("active_model", "deleted.gguf")  # saved model vanished
    assert mm.default_model().name == "b-Q4_K_M.gguf"


def test_gpu_layers_for(monkeypatch):
    monkeypatch.delenv("LLAMA_N_GPU_LAYERS", raising=False)
    assert mm.gpu_layers_for("m.gguf") == 999
    app_settings.set_value("gpu_layers", {"m.gguf": 20})
    assert mm.gpu_layers_for("m.gguf") == 20
    assert mm.gpu_layers_for("other.gguf") == 999


# --- Hugging Face ----------------------------------------------------------- #
def test_search_repos_parses(monkeypatch):
    def handler(req: httpx.Request):
        assert req.url.path == "/api/models"
        assert req.url.params["search"] == "llama"
        assert req.url.params["filter"] == "gguf"
        return httpx.Response(200, json=[{"id": "bartowski/X-GGUF", "downloads": 5, "likes": 2}, {"nope": 1}])

    mock_hf(monkeypatch, handler)
    assert mm.search_repos("llama") == [{"repo_id": "bartowski/X-GGUF", "downloads": 5, "likes": 2}]


def test_list_repo_files_filters_and_flags(monkeypatch, isolated_models):
    (isolated_models / "m-Q4_K_M.gguf").write_bytes(b"1")
    tree = [
        {"type": "file", "path": "README.md", "size": 10},
        {"type": "file", "path": "m-Q8_0.gguf", "size": 3000, "lfs": {"size": 3_000_000_000}},
        {"type": "file", "path": "m-Q4_K_M.gguf", "size": 100, "lfs": {"size": 2_000_000_000}},
        {"type": "file", "path": "big-00001-of-00003.gguf", "size": 1, "lfs": {"size": 5}},
        {"type": "file", "path": "mmproj-F16.gguf", "size": 1, "lfs": {"size": 6}},
        {"type": "directory", "path": "sub"},
        {"type": "file", "path": "sub/deep.gguf", "size": 1},
    ]
    mock_hf(monkeypatch, lambda req: httpx.Response(200, json=tree))
    files = mm.list_repo_files("a/b")
    by_name = {f["filename"]: f for f in files}
    assert set(by_name) == {"m-Q8_0.gguf", "m-Q4_K_M.gguf", "big-00001-of-00003.gguf", "mmproj-F16.gguf"}
    assert by_name["m-Q4_K_M.gguf"]["installed"] is True
    assert by_name["m-Q4_K_M.gguf"]["size_bytes"] == 2_000_000_000
    assert by_name["m-Q8_0.gguf"]["downloadable"] is True
    assert by_name["big-00001-of-00003.gguf"]["downloadable"] is False
    assert by_name["mmproj-F16.gguf"]["downloadable"] is True  # vision projectors can be downloaded
    assert by_name["mmproj-F16.gguf"]["kind"] == "projector" and by_name["m-Q8_0.gguf"]["kind"] == "model"
    assert [f["size_bytes"] for f in files] == sorted(f["size_bytes"] for f in files)


# --- downloads -------------------------------------------------------------- #
def test_start_download_validation():
    with pytest.raises(ValueError):
        mm.start_download("not a repo", "x.gguf", threaded=False)
    with pytest.raises(ValueError):
        mm.start_download("a/b", "../x.gguf", threaded=False)
    with pytest.raises(ValueError):
        mm.start_download("a/b", "x.bin", threaded=False)
    with pytest.raises(ValueError):
        mm.start_download("a/b", "m-00001-of-00002.gguf", threaded=False)


def test_start_download_existing_and_duplicate(isolated_models):
    (isolated_models / "have.gguf").write_bytes(b"1")
    with pytest.raises(FileExistsError):
        mm.start_download("a/b", "have.gguf", threaded=False)
    j1 = mm.start_download("a/b", "new.gguf", threaded=False)
    j2 = mm.start_download("a/b", "new.gguf", threaded=False)
    assert j1.id == j2.id  # same file already queued


def test_download_success(monkeypatch, isolated_models):
    payload = b"G" * (3 * 1024 * 1024 + 123)
    mock_hf(monkeypatch, lambda req: httpx.Response(200, content=payload))
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    mm._run_download(job)
    assert job.status == "done", job.error
    assert (isolated_models / "m.gguf").read_bytes() == payload
    assert not (isolated_models / "m.gguf.part").exists()
    assert job.downloaded == job.total == len(payload)


def test_download_resumes_partial(monkeypatch, isolated_models):
    full = bytes(range(256)) * 8192  # 2 MiB
    (isolated_models / "m.gguf.part").write_bytes(full[:1000])
    seen = {}

    def handler(req: httpx.Request):
        seen["range"] = req.headers.get("range")
        start = int(req.headers["range"].split("=")[1].rstrip("-"))
        return httpx.Response(206, content=full[start:])

    mock_hf(monkeypatch, handler)
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    mm._run_download(job)
    assert seen["range"] == "bytes=1000-"
    assert job.status == "done", job.error
    assert (isolated_models / "m.gguf").read_bytes() == full


def test_download_416_restarts(monkeypatch, isolated_models):
    full = b"abc" * 1000
    (isolated_models / "m.gguf.part").write_bytes(b"stale")
    calls = []

    def handler(req: httpx.Request):
        calls.append(req.headers.get("range"))
        if req.headers.get("range"):
            return httpx.Response(416)
        return httpx.Response(200, content=full)

    mock_hf(monkeypatch, handler)
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    mm._run_download(job)
    assert job.status == "done", job.error
    assert calls == ["bytes=5-", None]
    assert (isolated_models / "m.gguf").read_bytes() == full


def test_download_http_error(monkeypatch):
    mock_hf(monkeypatch, lambda req: httpx.Response(404))
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    mm._run_download(job)
    assert job.status == "error" and "404" in job.error


def test_download_cancel_keeps_part(monkeypatch, isolated_models):
    mock_hf(monkeypatch, lambda req: httpx.Response(200, content=b"x" * (3 << 20)))
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    assert mm.cancel_download(job.id) is True
    mm._run_download(job)
    assert job.status == "cancelled"
    assert not (isolated_models / "m.gguf").exists()
    assert mm.cancel_download(job.id) is False  # no longer active


def test_download_not_enough_disk(monkeypatch):
    mock_hf(monkeypatch, lambda req: httpx.Response(200, content=b"x" * 100))
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: type("U", (), {"free": 10})())
    job = mm.start_download("a/b", "m.gguf", threaded=False)
    mm._run_download(job)
    assert job.status == "error" and "disk" in job.error
