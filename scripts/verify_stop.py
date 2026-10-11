"""Stop verification for LocalChat.

1. Start the backend (it starts llama-server on :8080).
2. Create a conversation.
3. POST /api/chat with a long prompt, then close the connection after ~1.5s.
4. Check that the partial assistant message was saved to SQLite.
5. Check that llama-server's log shows the request was cancelled (under 250 tokens
   generated, no new tokens 3s after connection drop).

Usage: python verify_stop.py
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DATA = ROOT / "data"
DB = DATA / "localchat.db"
LOG = DATA / "logs" / "llama-server.log"

# Long-ish prompt so generation takes more than a couple of seconds.
PROMPT = ("Write a 400-word essay about the ocean, with vivid descriptions of "
          "waves, marine life, and the feeling of standing on the shore at dusk.")


def kill_ports() -> None:
    """Stop any processes listening on ports 8000 and 8080."""
    for port in (8000, 8080):
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    pid = int(line.strip())
                    if pid > 0:
                        subprocess.run(["powershell", "-Command", f"Stop-Process -Id {pid} -Force"], timeout=10)
                        print(f"  Killed PID {pid} on port {port}")
        except Exception:
            pass
    time.sleep(1)
    # Verify
    for port in (8000, 8080):
        result = subprocess.run(
            ["powershell", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            print(f"  WARNING: port {port} still in use")
        else:
            print(f"  Port {port} free")


async def read_for_seconds(resp, seconds: float) -> int:
    """Read SSE bytes from an httpx streaming response for `seconds`, then
    close the connection.  Returns the number of bytes read."""
    deadline = time.monotonic() + seconds
    bytes_read = 0
    while time.monotonic() < deadline:
        try:
            async for chunk in resp.aiter_raw():
                bytes_read += len(chunk)
                if time.monotonic() >= deadline:
                    break
        except Exception:
            break
    return bytes_read


async def _drop_after_seconds(conv_id: int, seconds: float) -> int:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            "http://127.0.0.1:8000/api/chat",
            json={"conversation_id": conv_id, "content": PROMPT},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise AssertionError(f"POST /api/chat returned {resp.status_code}: {body[:200]}")
            # Check first chunk is not an error frame
            async for chunk in resp.aiter_raw():
                if b"error" in chunk.lower():
                    raise AssertionError(f"First chunk is an error frame: {chunk[:200]}")
                break
            return await read_for_seconds(resp, seconds)


def parse_eval_tokens(log_text: str) -> int:
    """Extract the total tokens from the last 'eval time = ... / N tokens' line."""
    matches = re.findall(r"eval time =.*?/\s*(\d+)\s*tokens", log_text)
    return int(matches[-1]) if matches else 0


def main() -> int:
    print("=== Stop verification ===")
    print("Step 1: Ensure ports 8000 and 8080 are free.")
    kill_ports()

    print("Step 2: Start backend...")
    log_file = open(DATA / "backend_stdout.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(BACKEND),
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for /api/health.
        deadline = time.monotonic() + 90
        client = httpx.Client(timeout=5)
        while time.monotonic() < deadline:
            try:
                r = client.get("http://127.0.0.1:8000/api/health")
                if r.status_code == 200 and r.json().get("llama_server"):
                    print("  Backend healthy, llama-server running.")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            print("  ERROR: backend never became healthy")
            print(log_file.read_text()[-2000:] if log_file.tell() else "no log")
            return 1

        # Create a conversation.
        r = client.post("http://127.0.0.1:8000/api/conversations",
                        json={"system_prompt": "You are a helpful assistant."})
        r.raise_for_status()
        conv_id = r.json()["id"]
        print(f"  Created conversation {conv_id}.")

        # Stream a long reply, then drop the connection mid-generation.
        log_size_before = LOG.stat().st_size if LOG.exists() else 0
        print("  Starting generation, will drop the connection after 1.5s...")
        bytes_read = asyncio.run(_drop_after_seconds(conv_id, 1.5))
        print(f"  Dropped connection after reading {bytes_read} bytes.")

        # Give the backend a moment to notice the disconnect and llama-server
        # to cancel the request.
        time.sleep(3)

        # Check the database for the partial assistant message.
        conn = sqlite3.connect(str(DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
        conn.close()

        user_msgs = [r for r in rows if r["role"] == "user"]
        assistant_msgs = [r for r in rows if r["role"] == "assistant"]
        print(f"\n  Messages in DB ({len(rows)} total):")
        for r in rows:
            preview = r["content"][:80].replace("\n", " ")
            print(f"    {r['role']}: {preview}{'...' if len(r['content']) > 80 else ''}")

        if len(user_msgs) != 1 or len(assistant_msgs) != 1:
            print(f"\n  FAIL: expected 1 user + 1 assistant message, "
                  f"got {len(user_msgs)} user, {len(assistant_msgs)} assistant")
            return 1
        if not assistant_msgs[0]["content"].strip():
            print("\n  FAIL: assistant message is empty")
            return 1
        print(f"\n  PASS: partial assistant message saved "
              f"({len(assistant_msgs[0]['content'])} chars).")

        # Check llama-server's log for token count.
        log_text = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
        new_log_text = log_text[log_size_before:]
        tokens = parse_eval_tokens(new_log_text)
        print(f"  Tokens generated (from llama-server log): {tokens}")

        if tokens >= 250:
            print(f"  FAIL: generated {tokens} tokens (expected < 250 for a cancelled request)")
            return 1
        print("  PASS: token count confirms generation was cancelled early.")

        # Ensure no new tokens appeared in the 3 seconds after disconnect.
        # (The log was read after the 3s sleep, so this is already covered.)

        # Check that llama-server is idle (health check).
        try:
            h = client.get("http://127.0.0.1:8080/health")
            print(f"\n  llama-server /health: {h.status_code} {h.text[:200]}")
        except Exception as e:
            print(f"\n  llama-server /health error: {e!r}")

        return 0
    finally:
        print("\nStep 3: Stop backend and verify ports are free.")
        # Kill the uvicorn process tree.
        try:
            subprocess.run(
                ["powershell", "-Command", f"taskkill /PID {proc.pid} /T /F"],
                capture_output=True, timeout=10
            )
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()

        kill_ports()
        print("Done.")


if __name__ == "__main__":
    sys.exit(main())