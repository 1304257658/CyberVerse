"""SenseVoiceSmall ASR worker (runs in its own venv with funasr installed).

Request:  {"op": "asr", "audio": <base64>, "format": "pcm_s16le"|"float32",
           "sample_rate": 16000, "language": "auto"|"zh"|"en"|"yue"|"ja"|"ko"}
Reply:    {"done": true, "text": ..., "language": ..., "emotion": ...,
           "event": ..., "elapsed_ms": ..., "audio_ms": ...}
"""

from __future__ import annotations

import argparse
import base64
import re

import numpy as np

from _common import now_ms, run_server, setup_logging

log = setup_logging("sensevoice")
_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LANGS = {"auto", "zh", "en", "yue", "ja", "ko", "nospeech"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-itn", default="1")
    args = ap.parse_args()

    import torch
    from funasr import AutoModel

    t0 = now_ms()
    model = AutoModel(model=args.model_dir, device=args.device, disable_update=True)
    use_itn = args.use_itn not in ("0", "false", "False")
    # Warm up kernels so the first real utterance is fast.
    model.generate(input=np.zeros(16000, dtype=np.float32), cache={}, language="auto", use_itn=use_itn)
    load_ms = now_ms() - t0
    log.info("SenseVoice loaded from %s on %s in %.0f ms", args.model_dir, args.device, load_ms)

    def handle(req: dict, emit) -> dict:
        if req.get("op") != "asr":
            raise ValueError(f"unknown op {req.get('op')!r}")
        raw = base64.b64decode(req.get("audio", ""))
        fmt = str(req.get("format", "pcm_s16le")).lower()
        if fmt in ("float32", "f32", "pcm_f32le"):
            audio = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sr = int(req.get("sample_rate", 16000))
        if sr != 16000 and audio.size:
            import torchaudio.functional as AF

            audio = AF.resample(torch.from_numpy(audio), sr, 16000).numpy()
        audio_ms = audio.size / 16.0
        if audio.size < 1600:
            return {"text": "", "language": "", "audio_ms": audio_ms, "elapsed_ms": 0.0}
        lang = str(req.get("language") or "auto").lower().split("-")[0].split("_")[0]
        if lang not in _LANGS:
            lang = "auto"
        t = now_ms()
        res = model.generate(input=audio, cache={}, language=lang, use_itn=use_itn, batch_size_s=60)
        elapsed = now_ms() - t
        raw_text = res[0].get("text", "") if res else ""
        tags = _TAG_RE.findall(raw_text)
        text = _TAG_RE.sub("", raw_text).strip()
        return {
            "text": text,
            "language": tags[0] if len(tags) > 0 else "",
            "emotion": tags[1] if len(tags) > 1 else "",
            "event": tags[2] if len(tags) > 2 else "",
            "elapsed_ms": round(elapsed, 1),
            "audio_ms": round(audio_ms, 1),
        }

    run_server(args.socket, handle, {"model": "SenseVoiceSmall", "device": args.device, "load_ms": round(load_ms)})


if __name__ == "__main__":
    main()