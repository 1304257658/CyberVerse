"""Reload selected runtime params from disk without restarting inference workers."""

from __future__ import annotations

import logging
from typing import Any

from inference.core.config import load_config

logger = logging.getLogger(__name__)

# Keys that need a new worker / process if changed (log only; never apply at runtime).
AUDIOCPP_STARTUP_LOCKED_KEYS = frozenset(
    {
        "plugin_class",
        "lib_path",
        "model_path",
        "exec_plan",
        "backend",
        "device",
        "threads",
        "python",
        "ld_library_path",
        "startup_timeout_s",
        "default_avatar",
        "world_size",
        "cuda_visible_devices",
        "warmup",
        "warmup_blocks",
    }
)

# Top-level avatar model keys safe to apply on set_avatar / new utterance (audiocpp_flashhead).
AUDIOCPP_HOT_TOP_LEVEL_KEYS = frozenset(
    {
        "height",
        "width",
        "allow_resolution_test_profile",
        "num_inference_steps",
        "seed",
        "color_correction_strength",
        "metrics_log",
    }
)

BUFFERING_PREFIX = "buffering."


def load_active_avatar_params(config_path: str, model_name: str) -> dict[str, Any]:
    """Merge cyberverse.yaml + model dirs the same way as inference server bootstrap."""
    config = load_config(config_path)
    inference = config.get("inference") if isinstance(config.get("inference"), dict) else {}
    avatar = inference.get("avatar") if isinstance(inference.get("avatar"), dict) else {}
    params = dict(avatar.get(model_name) or {}) if isinstance(avatar.get(model_name), dict) else {}
    runtime = avatar.get("runtime")
    if isinstance(runtime, dict):
        params = {**runtime, **params}
    return params


def snapshot_startup_locked(params: dict[str, Any]) -> dict[str, Any]:
    return {k: params.get(k) for k in AUDIOCPP_STARTUP_LOCKED_KEYS if k in params}


def log_startup_locked_drift(
    model_name: str,
    baseline: dict[str, Any],
    fresh: dict[str, Any],
) -> list[str]:
    """Return keys that changed but cannot be hot-applied."""
    drifted: list[str] = []
    for key in AUDIOCPP_STARTUP_LOCKED_KEYS:
        if key not in fresh and key not in baseline:
            continue
        old = baseline.get(key)
        new = fresh.get(key)
        if old != new:
            drifted.append(key)
            logger.warning(
                "%s: %s changed in config (%r -> %r); restart inference to apply",
                model_name,
                key,
                old,
                new,
            )
    return drifted


def apply_audiocpp_hot_params(current: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Merge hot-reload fields from fresh into a copy of current params."""
    out = dict(current)
    for key in AUDIOCPP_HOT_TOP_LEVEL_KEYS:
        if key in fresh:
            out[key] = fresh[key]
    fresh_buf = fresh.get("buffering")
    if isinstance(fresh_buf, dict):
        cur_buf = dict(out.get("buffering") or {})
        cur_buf.update(fresh_buf)
        out["buffering"] = cur_buf
    return out


def encode_signature(params: dict[str, Any]) -> tuple[int, int, int, int, float]:
    return (
        int(params.get("height", 512)),
        int(params.get("width", 512)),
        int(params.get("num_inference_steps", 4)),
        int(params.get("seed", 42)),
        float(params.get("color_correction_strength", 1.0)),
    )