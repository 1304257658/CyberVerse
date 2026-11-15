#!/usr/bin/env python3
"""On-device FlashHead resolution benchmark (requires CUDA + libaudiocpp).

Records peak process RSS (proxy; not GPU VRAM without nvidia-smi), generation FPS,
first-frame latency, and underrun-related lead metrics emitted by the plugin worker.

This script does NOT run meaningfully in cloud agents without a GPU. Exit code 2 when
libaudiocpp is missing so CI can skip GPU steps.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
import time
from pathlib import Path

# Allow running from repo root without install.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from inference.core.types import AudioChunk, PluginConfig
from inference.plugins.avatar.audiocpp_flashhead_plugin import AudioCppFlashHeadAvatarPlugin


async def _run_benchmark(params: dict, blocks: int) -> dict:
    plugin = AudioCppFlashHeadAvatarPlugin()
    await plugin.initialize(PluginConfig(plugin_name="avatar.audiocpp_flashhead", params=params, shared={}))
    block_samples = 15360
    silent = bytes(block_samples * 4)

    async def audio_stream():
        for i in range(blocks):
            yield AudioChunk(data=silent, sample_rate=16000, is_final=i == blocks - 1)
            await asyncio.sleep(0)

    t0 = time.perf_counter()
    first_frame_ms = None
    frames = 0
    gen_times = []
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    async for chunk in plugin.generate_stream(audio_stream()):
        peak_rss_mb = max(peak_rss_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if chunk.frames is not None and chunk.frames.size:
            if first_frame_ms is None:
                first_frame_ms = (time.perf_counter() - t0) * 1000.0
            frames += int(chunk.frames.shape[0])

    await plugin.shutdown()
    elapsed = time.perf_counter() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    return {
        "height": int(params.get("height", 512)),
        "width": int(params.get("width", 512)),
        "blocks": blocks,
        "frames": frames,
        "elapsed_s": round(elapsed, 3),
        "generation_fps": round(fps, 3),
        "first_frame_ms": round(first_frame_ms or -1, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "note": "peak_rss_mb is host RSS, not GPU VRAM; run nvidia-smi alongside on QAQ for VRAM peaks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark audiocpp FlashHead resolutions on local GPU")
    parser.add_argument("--config-json", type=Path, required=True, help="JSON object of avatar plugin params")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("audiocpp_resolution_benchmark.json"))
    args = parser.parse_args()

    if not args.config_json.exists():
        print(f"config not found: {args.config_json}", file=sys.stderr)
        return 1
    params = json.loads(args.config_json.read_text(encoding="utf-8"))
    lib_path = Path(str(params.get("lib_path", "")))
    if not lib_path.exists():
        print(f"libaudiocpp not found at {lib_path}; run on QAQ with audio.cpp built", file=sys.stderr)
        return 2

    result = asyncio.run(_run_benchmark(params, args.blocks))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())