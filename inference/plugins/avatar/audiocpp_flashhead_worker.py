"""Long-lived SoulX-FlashHead worker on top of audio.cpp's C API (ctypes).

Runs as a child process of the CyberVerse inference server.  It loads the
GGUF once, keeps one *streaming* audio.cpp session resident (cuDNN VAE), and
encodes the reference portrait once per avatar (``audiocpp_stream_start``).
Every utterance is then fed block by block with ``audiocpp_stream_push``; the
stream is never finished between utterances, so motion continuity and the
8 s audio window persist exactly like the PyTorch plugin's deque.

Wire protocol on stdin/stdout (binary, one request in flight at a time):

    frame = u32_be(len(header)) + header_json_utf8 + payload
    header["nbytes"] gives the payload length (0 when absent)

Requests:  {"op": "avatar", "path": ..., "height", "width", "steps", "seed", "color"}
           {"op": "push", "nbytes": 4*block_samples}   payload = float32 mono 16 kHz
           {"op": "reset"}                              restart stream (re-encodes ref)
           {"op": "stats"}
Replies:   {"ok": true, ...} or {"ok": false, "error": "..."}
           push reply carries RGB24 frames as payload with frames/width/height.

The first frame written is {"op": "ready", ...}.  Logs go to stderr.
Exits when stdin closes (parent died).
"""

from __future__ import annotations

import ctypes as C
import json
import os
import struct
import sys
import time

import numpy as np


def log(msg: str) -> None:
    sys.stderr.write(f"[audiocpp-flashhead-worker] {time.strftime('%H:%M:%S')} {msg}\n")
    sys.stderr.flush()


def read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return bytes(buf)


def read_frame(stream) -> tuple[dict, bytes]:
    (hlen,) = struct.unpack(">I", read_exact(stream, 4))
    header = json.loads(read_exact(stream, hlen))
    n = int(header.get("nbytes", 0) or 0)
    payload = read_exact(stream, n) if n else b""
    return header, payload


def write_frame(stream, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["nbytes"] = len(payload)
    raw = json.dumps(header).encode()
    stream.write(struct.pack(">I", len(raw)))
    stream.write(raw)
    if payload:
        stream.write(payload)
    stream.flush()


class Backend(C.Structure):
    _fields_ = [("backend", C.c_char_p), ("device", C.c_int), ("threads", C.c_int)]


class ModelConfig(C.Structure):
    _fields_ = [
        ("family_hint", C.c_char_p),
        ("config_id", C.c_char_p),
        ("weight_id", C.c_char_p),
        ("model_spec_override", C.c_char_p),
    ]


class AudioCpp:
    def __init__(self, lib_path: str) -> None:
        L = C.CDLL(lib_path)
        vp = C.c_void_p
        L.audiocpp_last_error.restype = C.c_char_p
        L.audiocpp_build_version.restype = C.c_char_p
        L.audiocpp_options_create.restype = vp
        L.audiocpp_options_set.argtypes = [vp, C.c_char_p, C.c_char_p]
        L.audiocpp_options_free.argtypes = [vp]
        L.audiocpp_registry_create.argtypes = [C.c_char_p, C.POINTER(vp)]
        L.audiocpp_model_load.argtypes = [vp, C.c_char_p, C.POINTER(ModelConfig), vp, C.POINTER(vp)]
        L.audiocpp_session_create.argtypes = [vp, C.c_char_p, C.c_char_p, C.POINTER(Backend), vp, C.POINTER(vp)]
        L.audiocpp_stream_policy.argtypes = [vp, vp, vp, C.POINTER(C.c_int64), C.POINTER(C.c_double)]
        L.audiocpp_request_create.restype = vp
        L.audiocpp_request_free.argtypes = [vp]
        L.audiocpp_request_set_option.argtypes = [vp, C.c_char_p, C.c_char_p]
        L.audiocpp_stream_start.argtypes = [vp, vp]
        L.audiocpp_stream_push.argtypes = [vp, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int, C.c_int64, C.POINTER(vp)]
        L.audiocpp_stream_next_event.argtypes = [vp, C.POINTER(vp)]
        L.audiocpp_event_free.argtypes = [vp]
        L.audiocpp_event_as_result.argtypes = [vp]
        L.audiocpp_event_as_result.restype = vp
        L.audiocpp_result_artifact_count.argtypes = [vp]
        L.audiocpp_result_artifact_count.restype = C.c_size_t
        L.audiocpp_result_artifact.argtypes = [vp, C.c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_char_p), C.POINTER(vp), C.POINTER(C.c_size_t)]
        L.audiocpp_result_artifact_meta_count.argtypes = [vp, C.c_size_t]
        L.audiocpp_result_artifact_meta_count.restype = C.c_size_t
        L.audiocpp_result_artifact_meta.argtypes = [vp, C.c_size_t, C.c_size_t, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)]
        self.L = L

    def check(self, status: int, what: str) -> None:
        if status != 0:
            err = self.L.audiocpp_last_error() or b""
            raise RuntimeError(f"{what} failed status={status}: {err.decode('utf-8', 'replace')}")


class Worker:
    def __init__(self, args: dict) -> None:
        self.args = args
        self.api = AudioCpp(args["lib"])
        L = self.api.L
        vp = C.c_void_p
        t0 = time.perf_counter()
        self.registry = vp()
        self.api.check(L.audiocpp_registry_create(None, C.byref(self.registry)), "registry_create")
        cfg = ModelConfig(b"soulx_flashhead", None, None, None)
        self.model = vp()
        self.api.check(
            L.audiocpp_model_load(self.registry, args["model"].encode(), C.byref(cfg), None, C.byref(self.model)),
            "model_load",
        )
        opts = L.audiocpp_options_create()
        if args.get("exec_plan"):
            self.api.check(L.audiocpp_options_set(opts, b"soulx_flashhead.exec_plan", args["exec_plan"].encode()), "options_set")
        backend = Backend(args.get("backend", "cuda").encode(), int(args.get("device", 0)), int(args.get("threads", 8)))
        self.session = vp()
        self.api.check(
            L.audiocpp_session_create(self.model, b"gen", b"streaming", C.byref(backend), opts, C.byref(self.session)),
            "session_create",
        )
        L.audiocpp_options_free(opts)
        block = C.c_int64(0)
        secs = C.c_double(0)
        self.api.check(L.audiocpp_stream_policy(self.session, None, None, C.byref(block), C.byref(secs)), "stream_policy")
        self.block_samples = int(block.value)
        self.load_s = time.perf_counter() - t0
        self.request_opts: dict | None = None
        self.start_sample = 0
        self.pushes = 0
        log(f"loaded model in {self.load_s:.1f}s block_samples={self.block_samples} "
            f"build={(L.audiocpp_build_version() or b'').decode()}")

    def _start(self, opts: dict) -> float:
        L = self.api.L
        req = L.audiocpp_request_create()
        try:
            for k, v in opts.items():
                self.api.check(L.audiocpp_request_set_option(req, k.encode(), str(v).encode()), f"set_option {k}")
            t0 = time.perf_counter()
            self.api.check(L.audiocpp_stream_start(self.session, req), "stream_start")
            dt = time.perf_counter() - t0
        finally:
            L.audiocpp_request_free(req)
        self.request_opts = dict(opts)
        self.start_sample = 0
        return dt

    def op_avatar(self, h: dict) -> dict:
        opts = {
            "reference_image_file": h["path"],
            "height": int(h.get("height", 512)),
            "width": int(h.get("width", 512)),
            "num_inference_steps": int(h.get("steps", 4)),
            "seed": int(h.get("seed", 42)),
            "color_correction_strength": float(h.get("color", 1.0)),
        }
        dt = self._start(opts)
        log(f"avatar set (reference encoded) in {dt*1000:.0f}ms: {h['path']}")
        return {"ok": True, "encode_ms": dt * 1000}

    def op_reset(self, h: dict) -> dict:
        if not self.request_opts:
            return {"ok": True, "noop": True}
        dt = self._start(self.request_opts)
        return {"ok": True, "encode_ms": dt * 1000}

    def op_push(self, h: dict, payload: bytes) -> tuple[dict, bytes]:
        L = self.api.L
        if self.request_opts is None:
            raise RuntimeError("no avatar set")
        audio = np.frombuffer(payload, dtype="<f4")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        ev = C.c_void_p()
        t0 = time.perf_counter()
        self.api.check(
            L.audiocpp_stream_push(self.session, audio.ctypes.data_as(C.POINTER(C.c_float)), audio.shape[0],
                                   16000, 1, self.start_sample, C.byref(ev)),
            "stream_push",
        )
        gen_s = time.perf_counter() - t0
        self.start_sample += int(audio.shape[0])
        self.pushes += 1
        frames_total = 0
        width = height = 0
        parts: list[bytes] = []
        events = [ev] if ev.value else []
        while True:
            more = C.c_void_p()
            self.api.check(L.audiocpp_stream_next_event(self.session, C.byref(more)), "next_event")
            if not more.value:
                break
            events.append(more)
        for e in events:
            res = L.audiocpp_event_as_result(e)
            n = L.audiocpp_result_artifact_count(res)
            for i in range(n):
                kind = C.c_int(0)
                aid = C.c_char_p()
                pl = C.c_void_p()
                nb = C.c_size_t(0)
                self.api.check(L.audiocpp_result_artifact(res, i, C.byref(kind), C.byref(aid), C.byref(pl), C.byref(nb)), "artifact")
                meta = {}
                for m in range(L.audiocpp_result_artifact_meta_count(res, i)):
                    k = C.c_char_p()
                    v = C.c_char_p()
                    self.api.check(L.audiocpp_result_artifact_meta(res, i, m, C.byref(k), C.byref(v)), "artifact_meta")
                    meta[(k.value or b"").decode()] = (v.value or b"").decode()
                if meta.get("format", "rgb24") != "rgb24":
                    continue
                parts.append(C.string_at(pl.value, nb.value))
                frames_total += int(meta.get("frames", 0))
                width = int(meta.get("width", 0))
                height = int(meta.get("height", 0))
            L.audiocpp_event_free(e)
        return (
            {"ok": True, "frames": frames_total, "width": width, "height": height, "fps": 25,
             "gen_ms": gen_s * 1000, "push_index": self.pushes},
            b"".join(parts),
        )


def main() -> None:
    args = json.loads(sys.argv[1])
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # Anything native libraries print must not corrupt the binary channel.
    out_fd = os.dup(1)
    os.dup2(2, 1)
    stdout = os.fdopen(out_fd, "wb", buffering=0)
    try:
        w = Worker(args)
    except Exception as exc:  # noqa: BLE001
        log(f"init failed: {exc}")
        write_frame(stdout, {"op": "ready", "ok": False, "error": str(exc)})
        sys.exit(1)
    write_frame(stdout, {"op": "ready", "ok": True, "block_samples": w.block_samples, "load_s": w.load_s,
                         "pid": os.getpid()})
    while True:
        try:
            h, payload = read_frame(stdin)
        except EOFError:
            log("stdin closed; exiting")
            return
        op = h.get("op")
        try:
            if op == "push":
                rh, rp = w.op_push(h, payload)
                write_frame(stdout, rh, rp)
            elif op == "avatar":
                write_frame(stdout, w.op_avatar(h))
            elif op == "reset":
                write_frame(stdout, w.op_reset(h))
            elif op == "stats":
                write_frame(stdout, {"ok": True, "pushes": w.pushes, "block_samples": w.block_samples})
            else:
                write_frame(stdout, {"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001
            log(f"op {op} failed: {exc}")
            write_frame(stdout, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
