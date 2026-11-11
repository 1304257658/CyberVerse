"""Fun-CosyVoice3-0.5B streaming TTS worker (runs in the CosyVoice venv).

The reference voice is registered once at startup with ``add_zero_shot_spk``
(prompt speech tokens / features / speaker embedding cached in memory), so each
request only pays for text tokenization + generation.

Request:  {"op": "tts", "text": ..., "speed": 1.0, "spk_id": optional}
Replies:  {"audio": <base64 float32 mono @ target_sr>, "sample_rate": N} ...
          {"done": true, "first_packet_ms", "total_ms", "audio_ms", "rtf", "packets"}
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

from _common import now_ms, run_server, setup_logging

log = setup_logging("cosyvoice3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--repo", required=True, help="CosyVoice source checkout")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text", required=True)
    ap.add_argument("--spk-id", default="cyberverse_default")
    ap.add_argument("--target-sample-rate", type=int, default=16000)
    ap.add_argument("--fp16", default="0")
    ap.add_argument("--warmup-text", default="你好，很高兴见到你。")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, "third_party", "Matcha-TTS"))
    import torch
    import torchaudio.functional as AF
    from cosyvoice.cli.cosyvoice import AutoModel

    fp16 = args.fp16 not in ("0", "false", "False", "")
    t0 = now_ms()
    cv = AutoModel(model_dir=args.model_dir, fp16=fp16)
    sr = int(cv.sample_rate)
    target_sr = int(args.target_sample_rate) or sr
    t1 = now_ms()
    cv.add_zero_shot_spk(args.prompt_text, args.prompt_wav, args.spk_id)
    spk_ms = now_ms() - t1
    for _ in cv.inference_zero_shot(args.warmup_text, "", "", zero_shot_spk_id=args.spk_id, stream=True):
        pass
    load_ms = now_ms() - t0
    log.info(
        "CosyVoice3 loaded (fp16=%s sr=%d -> %d), speaker '%s' cached in %.0f ms, total %.0f ms",
        fp16, sr, target_sr, args.spk_id, spk_ms, load_ms,
    )

    def handle(req: dict, emit) -> dict:
        if req.get("op") != "tts":
            raise ValueError(f"unknown op {req.get('op')!r}")
        text = str(req.get("text", "")).strip()
        if not text:
            return {"packets": 0, "audio_ms": 0.0}
        spk = req.get("spk_id") or args.spk_id
        speed = float(req.get("speed", 1.0) or 1.0)
        t = now_ms()
        first = None
        samples = 0
        packets = 0
        gen = cv.inference_zero_shot(text, "", "", zero_shot_spk_id=spk, stream=True, speed=speed)
        try:
            for out in gen:
                wav = out["tts_speech"].float().cpu()
                if target_sr != sr:
                    wav = AF.resample(wav, sr, target_sr)
                pcm = wav.reshape(-1).clamp(-1.0, 1.0).numpy().astype("float32")
                if first is None:
                    first = now_ms() - t
                samples += pcm.size
                packets += 1
                if not emit({"audio": base64.b64encode(pcm.tobytes()).decode(), "sample_rate": target_sr}):
                    log.info("client went away; stopping synthesis early")
                    break
        finally:
            gen.close()
        total = now_ms() - t
        audio_ms = samples * 1000.0 / target_sr
        return {
            "first_packet_ms": round(first or 0.0, 1),
            "total_ms": round(total, 1),
            "audio_ms": round(audio_ms, 1),
            "rtf": round(total / audio_ms, 3) if audio_ms else None,
            "packets": packets,
        }

    run_server(
        args.socket,
        handle,
        {"model": os.path.basename(args.model_dir.rstrip("/")), "sample_rate": target_sr,
         "native_sample_rate": sr, "fp16": fp16, "spk_id": args.spk_id, "load_ms": round(load_ms),
         "cuda": torch.cuda.is_available()},
    )


if __name__ == "__main__":
    main()