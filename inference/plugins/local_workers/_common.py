"""Minimal line-delimited JSON server shared by the local model workers.

The workers run inside the model's own virtualenv (they must not import the
``inference`` package).  Protocol, one request per Unix-socket connection:

    client -> worker : one JSON line  {"op": "...", ...}
    worker -> client : zero or more JSON lines (streamed results), then a
                       final line containing ``"done": true`` or ``"error"``.

GPU work runs in a background thread under a single process-wide lock.  When
the client disconnects mid-stream, ``emit`` starts returning False so the
handler can stop early (barge-in / interruption).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable

STREAM_LIMIT = 64 * 1024 * 1024
_GPU_LOCK = threading.Lock()

Emit = Callable[[dict], bool]
Handler = Callable[[dict, Emit], dict | None]


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{name}] %(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def _watch_parent_stdin() -> None:
    """Exit when the parent (inference server) closes our stdin or dies."""

    def _run() -> None:
        try:
            while sys.stdin.buffer.read(1024):
                pass
        except Exception:
            pass
        os._exit(0)

    if not sys.stdin.isatty():
        threading.Thread(target=_run, name="parent-watch", daemon=True).start()


async def _handle(reader, writer, handler: Handler, info: dict) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()
    cancelled = threading.Event()

    def emit(msg: dict) -> bool:
        if cancelled.is_set():
            return False
        loop.call_soon_threadsafe(queue.put_nowait, msg)
        return True

    def run(req: dict) -> None:
        try:
            with _GPU_LOCK:
                if cancelled.is_set():
                    return
                final = handler(req, emit) or {}
            loop.call_soon_threadsafe(queue.put_nowait, {"done": True, **final})
        except Exception as exc:  # noqa: BLE001
            logging.exception("worker request failed")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"error": f"{type(exc).__name__}: {exc}"}
            )

    try:
        line = await reader.readline()
        if not line:
            return
        req = json.loads(line)
        if req.get("op") == "ping":
            msg = {"done": True, "pong": True, **info}
            writer.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            await writer.drain()
            return
        loop.run_in_executor(None, run, req)
        while True:
            msg = await queue.get()
            writer.write((json.dumps(msg, ensure_ascii=False) + "\n").encode())
            await writer.drain()
            if msg.get("done") or "error" in msg:
                break
    except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        cancelled.set()
        try:
            writer.close()
        except Exception:
            pass


def run_server(socket_path: str, handler: Handler, info: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(socket_path)), exist_ok=True)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    _watch_parent_stdin()

    async def main() -> None:
        server = await asyncio.start_unix_server(
            lambda r, w: _handle(r, w, handler, info),
            path=socket_path,
            limit=STREAM_LIMIT,
        )
        # The parent waits for this exact prefix on stdout.
        print("READY " + json.dumps(info, ensure_ascii=False), flush=True)
        async with server:
            await server.serve_forever()

    asyncio.run(main())


def now_ms() -> float:
    return time.perf_counter() * 1000.0