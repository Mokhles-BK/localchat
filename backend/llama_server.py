"""Manage the llama-server subprocess.

llama.cpp ships a prebuilt Windows CUDA binary (`llama-server.exe`) that we
run as a managed subprocess. This module is responsible for:

  * starting it with GPU offload,
  * waiting until its HTTP `/health` endpoint answers,
  * shutting it down cleanly on normal exit, SIGTERM, or SIGINT.

We deliberately do NOT write our own inference engine and we do NOT use
llama-cpp-python (Windows CUDA build issues).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TextIO

import httpx

# How long (seconds) to wait for llama-server to report healthy after launch.
HEALTH_TIMEOUT_SECONDS = 120
# Poll interval for the health check.
HEALTH_POLL_INTERVAL = 0.5

# Where llama-server's stdout/stderr go so startup failures are diagnosable.
LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "llama-server.log"


class LlamaServer:
    """Wraps a running `llama-server` process and its HTTP client."""

    def __init__(
        self,
        binary: Path,
        model: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = 999,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self.binary = Path(binary)
        self.model = Path(model)
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = list(extra_args or [])

        self._proc: Optional[subprocess.Popen] = None
        self._client = httpx.Client(timeout=None)
        self._log_fp: Optional[TextIO] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Launch llama-server and block until it reports healthy."""
        if not self.binary.exists():
            raise FileNotFoundError(f"llama-server binary not found: {self.binary}")
        if not self.model.exists():
            raise FileNotFoundError(f"model file not found: {self.model}")

        ctx_size = int(os.environ.get("LLAMA_CTX_SIZE", "4096"))
        cmd = [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(ctx_size),
        ]
        cmd.extend(self.extra_args)

        # CUDA runtime DLLs ship alongside the binary; make sure the DLL
        # directory is on PATH so Windows can find cublas64_12.dll etc.
        env = dict(os.environ)
        bin_dir = str(self.binary.parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

        # Log file: capture stdout/stderr so an unknown-argument error or a
        # CUDA failure is visible instead of silently swallowed.
        self._log_fp: TextIO = open(LOG_FILE, "a", encoding="utf-8")
        self._log_fp.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self._log_fp.write(" ".join(cmd) + "\n")
        self._log_fp.flush()

        # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT to the whole
        # process group (llama-server and any children it spawns). CREATE_NO_WINDOW
        # keeps the console window offscreen.
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fp,
            stderr=self._log_fp,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        self._wait_for_health()

    def _wait_for_health(self) -> None:
        """Poll llama-server's /health until it responds OK or we time out."""
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        last_err: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self._proc.returncode}"
                )
            try:
                r = self._client.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001 - transient while booting
                last_err = exc
            time.sleep(HEALTH_POLL_INTERVAL)
        raise RuntimeError(
            f"llama-server did not become healthy within {HEALTH_TIMEOUT_SECONDS}s"
            + (f" (last error: {last_err!r})" if last_err else "")
        )

    def stop(self) -> None:
        """Shut llama-server down cleanly (CTRL_BREAK, then terminate, then kill)."""
        if self._proc is None:
            return
        if self.is_running:
            # CTRL_BREAK_EVENT targets the process group (works because we
            # launched with CREATE_NEW_PROCESS_GROUP).
            try:
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            except Exception:
                pass

            if self.is_running:
                # Fall back to terminate() (graceful) then kill() (hard).
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                except Exception:
                    pass

            if self.is_running:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

        try:
            if self._log_fp is not None:
                self._log_fp.write("=== stopped ===\n")
                self._log_fp.flush()
                self._log_fp.close()
                self._log_fp = None
        except Exception:
            pass
        self._proc = None

    def close(self) -> None:
        self.stop()
        try:
            self._client.close()
        except Exception:
            pass

    @property
    def log_path(self) -> Path:
        return LOG_FILE

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    def chat_stream(self, messages, **kwargs):
        """Yield raw SSE `data:` payloads from llama-server's completion stream.

        This is a *plain* (sync) generator. FastAPI's StreamingResponse runs it
        in a threadpool, so it does not block the event loop — /api/health
        stays responsive while a reply is streaming.
        """
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", -1),
        }
        with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                # llama.cpp / OpenAI-compatible SSE: lines look like
                # "data: {json}" and a final "data: [DONE]".
                if line.startswith("data: "):
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    yield data

    async def chat_stream_async(self, messages, **kwargs):
        """Async generator of raw SSE payloads, *cancellable*.

        The upstream httpx stream is wrapped in an ``async with`` so that when
        the downstream client disconnects (Stop), the async generator is
        closed via GeneratorExit and the context manager's ``__aexit__``
        closes the response.  llama-server — seeing the connection drop —
        cancels the in-flight generation on the GPU.  No process restart, no
        POST /api/chat/stop endpoint.

        If ``sink`` is given (a list), the decoded assistant text is appended
        to it as it streams, so the caller can persist a partial reply if the
        stream is cancelled mid-generation.
        """
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", -1),
        }
        sink: list[str] = kwargs.pop("sink", [])
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: ") :]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        except (json.JSONDecodeError, IndexError, KeyError):
                            text = ""
                        if text:
                            sink.append(text)
                        yield data


def register_exit_handler(server: LlamaServer) -> None:
    """Make sure llama-server is stopped when this process exits."""
    import atexit

    atexit.register(server.close)

    def _handler(signum, frame):  # noqa: ARG001
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)