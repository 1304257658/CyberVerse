"""Client + process manager for local model workers (SenseVoice, CosyVoice3).

Local ASR/TTS models run in their own virtualenvs because their pinned
dependencies (e.g. CosyVoice: transformers 4.51 / diffusers 0.29 / numpy 1.26)
conflict with the avatar stack in the inference venv.  A plugin spawns its
worker as a child process at ``initialize`` and talks to it over a Unix socket
using line-delimited JSON (see ``inference/plugins/local_workers/_common.py``).

If ``python`` is empty the plugin only connects to an already running worker
at ``socket_path`` (useful when the worker is launched separately).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

STREAM_LIMIT = 64 * 1024 * 1024
WORKERS_DIR = Path(__file__).resolve().parent / "local_workers"


class LocalModelWorker:
    def __init__(
        self,
        name: str,
        script: str,
        args: list[str],
        socket_path: str,
        python: str = "",
        startup_timeout_s: float = 900.0,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.script = str(WORKERS_DIR / script) if not os.path.isabs(script) else script
        self.args = [str(a) for a in args]
        self.socket_path = socket_path
        self.python = python
        self.startup_timeout_s = startup_timeout_s
        self.env = env or {}
        self.cwd = cwd
        self.info: dict[str, Any] = {}
        self._proc: asyncio.subprocess.Process | None = None
        self._drain_task: asyncio.Task | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=40)

    async def start(self) -> dict[str, Any]:
        if not self.python:
            self.info = await self.call({"op": "ping"})
            logger.info("%s: connected to external worker %s", self.name, self.socket_path)
            return self.info

        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileNotFoundError):
            os.unlink(self.socket_path)
        env = os.environ.copy()
        # Do not leak the inference venv into the worker's interpreter.
        for key in ("VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(key, None)
        env.setdefault("TQDM_DISABLE", "1")
        env["PYTHONUNBUFFERED"] = "1"
        env.update({k: str(v) for k, v in self.env.items()})
        cmd = [self.python, self.script, *self.args, "--socket", self.socket_path]
        logger.info("%s: starting worker: %s", self.name, " ".join(cmd))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=self.cwd,
            limit=STREAM_LIMIT,
        )
        try:
            await asyncio.wait_for(self._wait_ready(), timeout=self.startup_timeout_s)
        except BaseException:
            await self.stop()
            raise
        self._drain_task = asyncio.create_task(self._drain_output())
        logger.info("%s: worker ready pid=%s info=%s", self.name, self._proc.pid, self.info)
        return self.info

    async def _wait_ready(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                rc = await self._proc.wait()
                tail = "\n".join(self._tail)
                raise RuntimeError(f"{self.name} worker exited rc={rc} before READY:\n{tail}")
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith("READY "):
                with suppress(json.JSONDecodeError):
                    self.info = json.loads(line[6:])
                return
            self._tail.append(line)
            logger.info("%s| %s", self.name, line)

    async def _drain_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            logger.info("%s| %s", self.name, raw.decode("utf-8", "replace").rstrip())
        rc = await self._proc.wait()
        logger.warning("%s: worker exited rc=%s", self.name, rc)

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Send one request and yield every reply line (the last has done=True)."""
        reader, writer = await asyncio.open_unix_connection(self.socket_path, limit=STREAM_LIMIT)
        try:
            writer.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            await writer.drain()
            while True:
                raw = await reader.readline()
                if not raw:
                    raise RuntimeError(f"{self.name} worker closed the connection")
                msg = json.loads(raw)
                if "error" in msg:
                    raise RuntimeError(f"{self.name} worker error: {msg['error']}")
                yield msg
                if msg.get("done"):
                    return
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: dict[str, Any] = {}
        async for msg in self.stream(payload):
            last = msg
        return last

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            with suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
        if self._drain_task is not None:
            self._drain_task.cancel()
            with suppress(BaseException):
                await self._drain_task
            self._drain_task = None
        if self.python:
            with suppress(FileNotFoundError):
                os.unlink(self.socket_path)