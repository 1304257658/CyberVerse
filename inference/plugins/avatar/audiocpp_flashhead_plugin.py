"""SoulX-FlashHead Lite avatar plugin backed by audio.cpp (GGUF, cuDNN VAE).

Same ``AvatarPlugin`` contract and gRPC surface as ``flash_head_plugin``; it is
selected purely by config (``inference.avatar.default: audiocpp_flashhead``).

Residency: a long-lived child process (``audiocpp_flashhead_worker.py``) loads
the GGUF once through audio.cpp's C API and keeps one streaming session warm.
The reference portrait is encoded once per ``set_avatar`` (skipped when the
same image bytes are set again).  Every utterance is pushed block by block
(0.96 s audio -> 24 frames @ 25 fps); there is no per-utterance cold start.

Streaming + adaptive start-up buffer ("buffer a bit, then play while
generating"):

* generation rate g is measured continuously (EWMA of frames/s per block);
* before releasing the first chunk of an utterance the plugin holds video
  until the buffered lead reaches
      lead = D * (1 - min(g, input_rate) / 25) + chunk + safety + bias
  clamped to [min_lead_s, max_lead_s]; D is the utterance length when the
  audio stream already ended, else max(audio received, rolling estimate);
* during playback an emulated playback clock detects underruns.  Audio and
  video are always delivered together (the Go side pairs PCM to video frame
  counts), so an underrun is a joint pause: the client holds the last frame,
  audio pauses with it and A/V sync is preserved.  After an underrun the
  plugin rebuffers ``rebuffer_s`` before continuing (one longer hold instead
  of many micro-stalls) and raises ``bias`` for the next utterance; clean
  utterances decay the bias.  Every adjustment is logged.
* bounded queues only (audio blocks and ready video chunks); one generation
  at a time on the GPU (single worker thread + single worker process);
  barge-in cancels the reader/generator tasks and drops in-flight output.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from inference.core.types import AudioChunk, PluginConfig, VideoChunk
from inference.plugins.avatar.base import AvatarPlugin

logger = logging.getLogger(__name__)

PLAYBACK_FPS = 25
SAMPLE_RATE = 16000
WORKER_SCRIPT = Path(__file__).resolve().parent / "audiocpp_flashhead_worker.py"


def _audio_bytes_to_float32_mono(data: bytes, format_hint: str) -> np.ndarray:
    fmt = (format_hint or "").strip().lower()
    if fmt in ("float32", "f32", "pcm_f32le"):
        b = data[: len(data) - (len(data) % 4)]
        return np.frombuffer(b, dtype="<f4").astype(np.float32) if b else np.zeros(0, np.float32)
    b = data[: len(data) - (len(data) % 2)]
    if not b:
        return np.zeros(0, np.float32)
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or src_sr <= 0 or src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_dst = max(int(round(x.shape[0] * dst_sr / src_sr)), 1)
    t_src = np.arange(x.shape[0], dtype=np.float64) / float(src_sr)
    t_dst = np.linspace(0.0, (x.shape[0] - 1) / float(src_sr), n_dst, dtype=np.float64)
    return np.interp(t_dst, t_src, x.astype(np.float64)).astype(np.float32)


def _gray_png(width: int, height: int, value: int = 128) -> bytes:
    """Minimal RGB PNG without PIL (the inference venv may not ship Pillow)."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    row = b"\x00" + bytes([value]) * (width * 3)
    raw = zlib.compress(row * height)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


class _WorkerProcess:
    """Blocking client for the binary stdin/stdout protocol (one call at a time)."""

    def __init__(self, cmd: list[str], env: dict[str, str], startup_timeout_s: float) -> None:
        self.cmd = cmd
        self.env = env
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None
        self.info: dict = {}
        self._lock = threading.Lock()
        self._log_thread: threading.Thread | None = None

    def start(self) -> dict:
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, bufsize=0,
        )
        self._log_thread = threading.Thread(target=self._pump_stderr, name="audiocpp-avatar-log", daemon=True)
        self._log_thread.start()
        result: dict = {}

        def _wait_ready() -> None:
            result["h"], _ = self._read()

        t = threading.Thread(target=_wait_ready, daemon=True)
        t.start()
        t.join(self.startup_timeout_s)
        if t.is_alive() or "h" not in result:
            self.stop()
            raise RuntimeError("audio.cpp FlashHead worker did not become ready")
        if not result["h"].get("ok"):
            self.stop()
            raise RuntimeError(f"audio.cpp FlashHead worker failed: {result['h'].get('error')}")
        self.info = result["h"]
        return self.info

    def _pump_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            logger.info("audiocpp-worker| %s", raw.decode("utf-8", "replace").rstrip())

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_exact(self, n: int) -> bytes:
        assert self.proc is not None and self.proc.stdout is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("audio.cpp FlashHead worker closed its output")
            buf += chunk
        return bytes(buf)

    def _read(self) -> tuple[dict, bytes]:
        (hlen,) = struct.unpack(">I", self._read_exact(4))
        header = json.loads(self._read_exact(hlen))
        n = int(header.get("nbytes", 0) or 0)
        return header, (self._read_exact(n) if n else b"")

    def call(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        with self._lock:
            if not self.alive():
                raise RuntimeError("audio.cpp FlashHead worker is not running")
            assert self.proc is not None and self.proc.stdin is not None
            header = dict(header)
            header["nbytes"] = len(payload)
            raw = json.dumps(header).encode()
            self.proc.stdin.write(struct.pack(">I", len(raw)) + raw)
            if payload:
                self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            reply, data = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"audio.cpp FlashHead worker error: {reply.get('error')}")
        return reply, data

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
            proc.wait()


class AdaptiveLead:
    """Runtime estimate of how much video to buffer before playback starts."""

    def __init__(self, cfg: dict, chunk_s: float) -> None:
        self.chunk_s = chunk_s
        self.min_lead_s = float(cfg.get("min_lead_s", 0.3))
        self.max_lead_s = float(cfg.get("max_lead_s", 8.0))
        self.safety_s = float(cfg.get("safety_s", 0.3))
        self.alpha = float(cfg.get("gen_fps_ewma_alpha", 0.3))
        self.gen_fps = float(cfg.get("initial_gen_fps", 20.0))
        self.utt_alpha = float(cfg.get("utterance_ewma_alpha", 0.3))
        self.utterance_s = float(cfg.get("initial_utterance_s", 6.0))
        self.unknown_cap_s = float(cfg.get("unknown_duration_cap_s", 20.0))
        self.bias_s = 0.0
        self.bias_gain = float(cfg.get("underrun_bias_gain", 1.0))
        self.bias_step_s = float(cfg.get("underrun_bias_step_s", 0.5))
        self.bias_max_s = float(cfg.get("bias_max_s", 4.0))
        self.bias_decay = float(cfg.get("bias_decay", 0.7))
        self.first_chunk_ms_ewma: float | None = None

    def observe_chunk(self, frames: int, gen_s: float) -> float:
        inst = frames / max(gen_s, 1e-3)
        self.gen_fps = (1 - self.alpha) * self.gen_fps + self.alpha * inst
        return inst

    def observe_first_chunk_latency(self, ms: float) -> None:
        if self.first_chunk_ms_ewma is None:
            self.first_chunk_ms_ewma = ms
        else:
            self.first_chunk_ms_ewma = 0.7 * self.first_chunk_ms_ewma + 0.3 * ms

    def required(self, received_s: float, ended: bool, input_rt: float | None) -> tuple[float, dict]:
        if ended:
            dur = received_s
        else:
            dur = min(max(received_s, self.utterance_s), self.unknown_cap_s)
        rate_rt = self.gen_fps / PLAYBACK_FPS
        if not ended and input_rt is not None:
            rate_rt = min(rate_rt, input_rt)
        raw = dur * max(0.0, 1.0 - rate_rt) + self.chunk_s + self.safety_s + self.bias_s
        lead = min(max(raw, self.min_lead_s), self.max_lead_s)
        return lead, {"dur_s": dur, "ended": ended, "gen_fps": self.gen_fps, "rate_rt": rate_rt,
                      "bias_s": self.bias_s, "raw_s": raw}

    def end_utterance(self, dur_s: float, stall_s: float, underruns: int, low_marks: int,
                      clamped: bool = False) -> str:
        if dur_s > 0:
            self.utterance_s = (1 - self.utt_alpha) * self.utterance_s + self.utt_alpha * dur_s
        old = self.bias_s
        if (underruns or stall_s > 0) and clamped:
            # The lead was capped at max_lead_s, so the stall was predicted and a
            # bigger bias cannot prevent it; the measured gen rate already drives D*(1-g/p).
            why = f"underruns={underruns} stall={stall_s:.2f}s with lead capped at max_lead_s (bias unchanged)"
            logger.info("audiocpp avatar lead: %s", why)
        elif underruns or stall_s > 0:
            self.bias_s = min(self.bias_max_s, self.bias_s + self.bias_gain * stall_s + self.bias_step_s)
            why = f"underruns={underruns} stall={stall_s:.2f}s"
        elif low_marks:
            self.bias_s = min(self.bias_max_s, self.bias_s + self.bias_step_s / 2)
            why = f"low_watermark_hits={low_marks}"
        else:
            self.bias_s = self.bias_s * self.bias_decay
            if self.bias_s < 0.05:
                self.bias_s = 0.0
            why = "clean"
        if abs(self.bias_s - old) > 1e-3:
            logger.info("audiocpp avatar lead bias %.2fs -> %.2fs (%s)", old, self.bias_s, why)
        return why


class AudioCppFlashHeadAvatarPlugin(AvatarPlugin):
    name = "avatar.audiocpp_flashhead"

    def __init__(self) -> None:
        self.infer_params: dict = {}
        self._worker: _WorkerProcess | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audiocpp-avatar")
        self._params: dict = {}
        self._buf_cfg: dict = {}
        self._block_samples = 15360
        self._chunk_s = 0.96
        self._avatar_key: str | None = None
        self._avatar_path: str | None = None
        self._lead: AdaptiveLead | None = None
        self._chunk_counter = 0
        self._metrics_path: str = ""
        # The utterance currently streaming. grpc.aio may stop iterating a cancelled
        # RPC without closing our async generator, so a new GenerateStream supersedes
        # (cancels) any still-open previous one instead of waiting on a lock forever.
        self._active: dict | None = None
        self._placeholder: str | None = None

    # ------------------------------------------------------------------ setup
    async def initialize(self, config: PluginConfig) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._init_sync, config)

    def _init_sync(self, config: PluginConfig) -> None:
        p = dict(config.params)
        if int(p.get("world_size", 1) or 1) != 1:
            raise ValueError("audiocpp_flashhead supports world_size=1 only")
        self._params = p
        self._buf_cfg = dict(p.get("buffering") or {})
        self._metrics_path = str(p.get("metrics_log", "") or "")
        height = int(p.get("height", 512))
        width = int(p.get("width", 512))
        env = os.environ.copy()
        for key in ("VIRTUAL_ENV", "PYTHONHOME"):
            env.pop(key, None)
        if p.get("ld_library_path"):
            env["LD_LIBRARY_PATH"] = str(p["ld_library_path"]) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        if p.get("cuda_visible_devices") is not None and str(p.get("cuda_visible_devices")).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(p["cuda_visible_devices"])
        worker_args = {
            "lib": str(p["lib_path"]),
            "model": str(p["model_path"]),
            "exec_plan": str(p.get("exec_plan", "") or ""),
            "backend": str(p.get("backend", "cuda")),
            "device": int(p.get("device", 0)),
            "threads": int(p.get("threads", 8)),
        }
        python = str(p.get("python") or sys.executable)
        cmd = [python, str(WORKER_SCRIPT), json.dumps(worker_args)]
        self._worker = _WorkerProcess(cmd, env, float(p.get("startup_timeout_s", 300)))
        t0 = time.perf_counter()
        info = self._worker.start()
        self._block_samples = int(info.get("block_samples", 15360))
        frames_per_chunk = int(round(self._block_samples * PLAYBACK_FPS / SAMPLE_RATE))
        self._chunk_s = frames_per_chunk / PLAYBACK_FPS
        motion = 9
        self.infer_params = {
            "frame_num": frames_per_chunk + motion,
            "motion_frames_num": motion,
            "tgt_fps": PLAYBACK_FPS,
            "sample_rate": SAMPLE_RATE,
            "height": height,
            "width": width,
        }
        self._lead = AdaptiveLead(self._buf_cfg, self._chunk_s)
        logger.info("audiocpp FlashHead worker ready in %.1fs: %s", time.perf_counter() - t0, info)

        # Placeholder avatar + warmup so the first real turn hits warm graphs.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(_gray_png(width, height))
        tmp.close()
        self._placeholder = tmp.name
        self._set_avatar_sync(str(p.get("default_avatar") or tmp.name), False)
        if bool(p.get("warmup", True)):
            silent = np.zeros(self._block_samples, dtype=np.float32).tobytes()
            for i in range(int(p.get("warmup_blocks", 2))):
                h, _ = self._worker.call({"op": "push"}, silent)
                if i > 0:  # first block includes graph/cuDNN plan build
                    self._lead.observe_chunk(int(h["frames"]), float(h["gen_ms"]) / 1000.0)
                logger.info("audiocpp FlashHead warmup block %d: %.0f ms", i, float(h["gen_ms"]))
            self._worker.call({"op": "reset"})

    # ----------------------------------------------------------- avatar state
    async def set_avatar(self, image_path: str, use_face_crop: bool = False) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._set_avatar_sync, image_path, use_face_crop)

    def _set_avatar_sync(self, image_path: str, use_face_crop: bool) -> None:
        assert self._worker is not None
        if use_face_crop:
            logger.info("audiocpp FlashHead: use_face_crop is not supported; using the image as-is")
        with open(image_path, "rb") as f:
            key = hashlib.sha1(f.read()).hexdigest()
        if key == self._avatar_key:
            logger.info("audiocpp FlashHead avatar unchanged (sha1=%s); reference encode skipped", key[:12])
            return
        p = self._params
        h, _ = self._worker.call({
            "op": "avatar", "path": image_path,
            "height": int(p.get("height", 512)), "width": int(p.get("width", 512)),
            "steps": int(p.get("num_inference_steps", 4)), "seed": int(p.get("seed", 42)),
            "color": float(p.get("color_correction_strength", 1.0)),
        })
        self._avatar_key = key
        self._avatar_path = image_path
        logger.info("audiocpp FlashHead avatar set: %s encode=%.0fms", image_path, float(h.get("encode_ms", 0)))

    async def reset(self) -> None:
        # Keep the encoded reference and motion state; nothing per-utterance is cached here.
        self._chunk_counter = 0

    def get_fps(self) -> int:
        return PLAYBACK_FPS

    def get_output_dimensions(self) -> tuple[int, int]:
        return int(self.infer_params.get("width", 512)), int(self.infer_params.get("height", 512))

    # ---------------------------------------------------------------- stream
    def _push_sync(self, block: np.ndarray) -> tuple[dict, bytes]:
        assert self._worker is not None
        return self._worker.call({"op": "push"}, block.astype(np.float32, copy=False).tobytes())

    async def generate_stream(self, audio_stream: AsyncIterator[AudioChunk]) -> AsyncIterator[VideoChunk]:
        assert self._lead is not None
        prev = self._active
        if prev is not None and not prev["finished"]:
            logger.warning("audiocpp avatar: previous stream still open (cancelled RPC / barge-in); superseding it")
            prev["superseded"] = True
            for t in prev["tasks"]:
                t.cancel()
            prev["ready"].clear()  # free buffered RGB now (bounded memory)
            prev["finalize"]()
        async for vc in self._generate_one(audio_stream):
            yield vc

    async def _generate_one(self, audio_stream: AsyncIterator[AudioChunk]) -> AsyncIterator[VideoChunk]:
        lead = self._lead
        assert lead is not None
        cfg = self._buf_cfg
        loop = asyncio.get_running_loop()
        block_n = self._block_samples
        chunk_s = self._chunk_s
        max_ready = max(2, int(cfg.get("max_buffer_chunks", math.ceil(lead.max_lead_s / chunk_s) + 2)))
        max_ready = max(max_ready, math.ceil(lead.max_lead_s / chunk_s) + 1)
        blocks: asyncio.Queue = asyncio.Queue(maxsize=int(cfg.get("max_pending_audio_blocks", 32)))
        ready: collections.deque = collections.deque()
        cond = asyncio.Condition()
        st = {
            "received_s": 0.0, "reader_done": False, "gen_done": False, "outstanding": 0,
            "first_audio_t": None, "user_final_ms": 0, "error": None, "gen_frames": 0, "gen_time": 0.0,
            "first_chunk_t": None, "reply_id": "",
        }

        async def notify() -> None:
            async with cond:
                cond.notify_all()

        async def reader() -> None:
            pending = np.zeros(0, np.float32)
            try:
                async for ac in audio_stream:
                    if st["first_audio_t"] is None:
                        st["first_audio_t"] = time.perf_counter()
                        st["user_final_ms"] = int(getattr(ac, "user_final_unix_ms", 0) or 0)
                        st["reply_id"] = getattr(ac, "reply_id", "") or ""
                    x = _audio_bytes_to_float32_mono(ac.data, ac.format)
                    x = _resample_linear(x, int(ac.sample_rate or SAMPLE_RATE), SAMPLE_RATE)
                    if x.size:
                        st["received_s"] += x.shape[0] / SAMPLE_RATE
                        pending = x if pending.size == 0 else np.concatenate([pending, x])
                    while pending.shape[0] >= block_n:
                        st["outstanding"] += 1
                        await blocks.put(pending[:block_n].copy())
                        pending = pending[block_n:]
                    if ac.is_final and pending.size:
                        tail = np.zeros(block_n, np.float32)
                        tail[: pending.shape[0]] = pending
                        st["outstanding"] += 1
                        await blocks.put(tail)
                        pending = np.zeros(0, np.float32)
                    await notify()
                if pending.size:
                    tail = np.zeros(block_n, np.float32)
                    tail[: pending.shape[0]] = pending
                    st["outstanding"] += 1
                    await blocks.put(tail)
            except Exception as exc:  # noqa: BLE001
                st["error"] = exc
            finally:
                st["reader_done"] = True
                await blocks.put(None)
                await notify()

        async def generator() -> None:
            try:
                while True:
                    blk = await blocks.get()
                    if blk is None:
                        break
                    async with cond:
                        await cond.wait_for(lambda: len(ready) < max_ready)
                    h, rgb = await loop.run_in_executor(self._executor, self._push_sync, blk)
                    nf, w, hh = int(h["frames"]), int(h["width"]), int(h["height"])
                    gen_s = float(h["gen_ms"]) / 1000.0
                    if nf > 0:
                        lead.observe_chunk(nf, gen_s)
                        st["gen_frames"] += nf
                        st["gen_time"] += gen_s
                        frames = np.frombuffer(rgb, dtype=np.uint8).reshape(nf, hh, w, 3)
                        if st["first_chunk_t"] is None:
                            st["first_chunk_t"] = time.perf_counter()
                        ready.append((frames, time.perf_counter()))
                    st["outstanding"] -= 1
                    await notify()
            except Exception as exc:  # noqa: BLE001
                st["error"] = exc
            finally:
                st["gen_done"] = True
                await notify()

        def input_rt() -> float | None:
            t0 = st["first_audio_t"]
            if t0 is None:
                return None
            el = time.perf_counter() - t0
            return None if el < 0.5 else st["received_s"] / el

        reader_task = asyncio.create_task(reader())
        gen_task = asyncio.create_task(generator())
        stats = {"underruns": 0, "stall_s": 0.0, "low_marks": 0, "rebuffers": 0, "chunks": 0}
        target = 0.0
        detail: dict = {}
        release_t = 0.0
        first_release_t = 0.0
        me: dict = {"finished": False, "superseded": False, "tasks": [reader_task, gen_task], "ready": ready}

        def finalize() -> None:
            if me["finished"]:
                return
            me["finished"] = True
            dur = st["received_s"]
            why = lead.end_utterance(dur, stats["stall_s"], stats["underruns"], stats["low_marks"],
                                     clamped=float(detail.get("raw_s", 0.0)) > lead.max_lead_s)
            avg_fps = st["gen_frames"] / st["gen_time"] if st["gen_time"] > 0 else 0.0
            fa = st["first_audio_t"]
            rec = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "reply_id": st["reply_id"], "audio_s": round(dur, 3),
                "chunks": stats["chunks"], "target_lead_s": round(target, 3),
                "raw_lead_s": round(float(detail.get("raw_s", 0.0)), 3),
                "first_frame_after_audio_ms": round((first_release_t - fa) * 1000) if (first_release_t and fa) else None,
                "first_chunk_gen_ms": round((st["first_chunk_t"] - fa) * 1000) if (st["first_chunk_t"] and fa) else None,
                "gen_fps_utt": round(avg_fps, 2), "gen_fps_ewma": round(lead.gen_fps, 2),
                "underruns": stats["underruns"], "rebuffers": stats["rebuffers"], "stall_s": round(stats["stall_s"], 3),
                "low_watermark_hits": stats["low_marks"], "bias_next_s": round(lead.bias_s, 3), "adjust": why,
                "cancelled": not st.get("completed", False), "superseded": me["superseded"],
            }
            logger.info("audiocpp avatar utterance: %s", json.dumps(rec))
            if self._metrics_path:
                try:
                    with open(self._metrics_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                except OSError:
                    pass

        me["finalize"] = finalize
        self._active = me
        try:
            # ---- start-up gate: buffer `target` seconds of video, adaptively.
            async with cond:
                while True:
                    target, detail = lead.required(st["received_s"], st["reader_done"], input_rt())
                    buffered = len(ready) * chunk_s
                    if st["error"] is not None:
                        break
                    if ready and (buffered + 1e-6 >= target or len(ready) >= max_ready
                                  or (st["reader_done"] and st["outstanding"] == 0)):
                        break
                    if st["gen_done"] and not ready:
                        break
                    await cond.wait()
            if st["error"] is not None and not ready:
                raise st["error"]
            release_t = time.perf_counter()
            first_release_t = release_t
            first_ms = None
            if st["first_audio_t"] is not None:
                first_ms = (release_t - st["first_audio_t"]) * 1000
                if st["first_chunk_t"] is not None:
                    lead.observe_first_chunk_latency((st["first_chunk_t"] - st["first_audio_t"]) * 1000)
            logger.info(
                "audiocpp avatar release: target_lead=%.2fs buffered=%.2fs audio_received=%.2fs ended=%s "
                "gen_fps=%.1f rate_rt=%.2f bias=%.2fs first_frame_after_audio=%sms",
                target, len(ready) * chunk_s, st["received_s"], st["reader_done"], detail.get("gen_fps", 0),
                detail.get("rate_rt", 0), detail.get("bias_s", 0), f"{first_ms:.0f}" if first_ms else "-",
            )
            low_wm = float(cfg.get("low_watermark_s", 0.5))
            rebuffer_s = float(cfg.get("rebuffer_s", 2 * chunk_s))
            tol = float(cfg.get("underrun_tolerance_s", 0.08))
            k = 0
            while True:
                async with cond:
                    await cond.wait_for(lambda: bool(ready) or st["gen_done"])
                    if not ready or me["superseded"]:
                        break
                    # Lateness is judged by when the chunk became ready, not when we got
                    # to pop it: a blocked yield (Go publisher still busy playing earlier
                    # segments) is backpressure, not an underrun.
                    ready_t = ready[0][1]
                    deadline = release_t + k * chunk_s
                    if ready_t > deadline + tol:
                        late = ready_t - deadline
                        stats["underruns"] += 1
                        stats["stall_s"] += late
                        stats["rebuffers"] += 1
                        # One adaptive hold instead of many micro-stalls: size the rebuffer with
                        # the same lead formula for the audio still left to play.
                        remaining = max(0.0, st["received_s"] - k * chunk_s)
                        rb_target, _ = lead.required(remaining, st["reader_done"], input_rt())
                        rb = max(rebuffer_s, rb_target)
                        logger.warning("audiocpp avatar underrun: chunk=%d late=%.2fs remaining=%.1fs -> hold last "
                                       "frame (A/V paused together), rebuffer %.2fs", k, late, remaining, rb)
                        t_hold = time.perf_counter()
                        await cond.wait_for(lambda: len(ready) * chunk_s >= rb or st["gen_done"]
                                            or len(ready) >= max_ready)
                        held = time.perf_counter() - t_hold
                        stats["stall_s"] += held
                        release_t = time.perf_counter() - k * chunk_s
                    frames, _ready_t = ready.popleft()
                    is_last = not ready and st["reader_done"] and st["outstanding"] == 0
                    cond.notify_all()
                # lead = media emitted (incl. this chunk) + still buffered - played so far
                played = time.perf_counter() - release_t
                cur_lead = (k + 1 + len(ready)) * chunk_s - played
                if not is_last and cur_lead < low_wm and k > 0:
                    stats["low_marks"] += 1
                    logger.info("audiocpp avatar low watermark: lead=%.2fs < %.2fs at chunk %d", cur_lead, low_wm, k)
                self._chunk_counter += 1
                since_ms = -1
                if k == 0 and st["user_final_ms"] > 0:
                    since_ms = max(0, int(time.time() * 1000) - st["user_final_ms"])
                stats["chunks"] += 1
                yield VideoChunk(
                    frames=frames, fps=PLAYBACK_FPS, chunk_index=self._chunk_counter, is_final=is_last,
                    trace_generation_started_since_user_final_ms=since_ms,
                )
                k += 1
                if is_last:
                    st["completed"] = True
                    break
            if st["error"] is not None:
                logger.error("audiocpp avatar stream error: %s", st["error"])
        finally:
            for t in (reader_task, gen_task):
                if not t.done():
                    t.cancel()
            for t in (reader_task, gen_task):
                try:
                    await t
                except BaseException:  # noqa: BLE001
                    pass
            finalize()
            if self._active is me:
                self._active = None

    async def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._executor.shutdown(wait=False)
        if self._placeholder:
            try:
                os.unlink(self._placeholder)
            except OSError:
                pass
