"""MediaPipe face-detection compatibility for FlashHead face crop.

FlashHead originally used ``mp.solutions.face_detection``. MediaPipe >=0.10.30 removed
``mp.solutions``; we use the official Tasks ``FaceDetector`` API when legacy symbols
are absent. Either path is supported; face crop fails loudly when MediaPipe cannot run.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

MEDIAPIPE_LEGACY_SOLUTIONS_MAX = "0.10.21"
FACE_CROP_MEDIAPIPE_INSTALL_HINT = (
    "FlashHead face crop needs MediaPipe with legacy mp.solutions or Tasks FaceDetector. "
    "Install: pip install 'mediapipe>=0.10.13' (cyberverse[flash_head]). "
    f"If you must use legacy-only code paths, pin mediapipe<={MEDIAPIPE_LEGACY_SOLUTIONS_MAX} "
    "in a dedicated inference venv (may conflict with grpcio/protobuf in cyberverse[all])."
)

_BLAZE_FACE_MODEL_URLS = {
    0: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    1: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
    ),
}


class FaceCropDependencyError(RuntimeError):
    """MediaPipe or face-detection runtime is missing or incompatible."""


def mediapipe_legacy_solutions_available() -> bool:
    try:
        import mediapipe as mp
    except ImportError:
        return False
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")


def mediapipe_tasks_vision_available() -> bool:
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except ImportError:
        return False
    return True


def assert_mediapipe_face_crop_runtime() -> None:
    """Raise if neither legacy nor Tasks face detection can be used."""
    try:
        import mediapipe as mp  # noqa: F401
    except ImportError as exc:
        raise FaceCropDependencyError(
            f"MediaPipe is not installed. {FACE_CROP_MEDIAPIPE_INSTALL_HINT}"
        ) from exc
    if mediapipe_legacy_solutions_available() or mediapipe_tasks_vision_available():
        return
    version = getattr(mp, "__version__", "unknown")
    raise FaceCropDependencyError(
        f"MediaPipe {version} exposes neither mp.solutions.face_detection nor Tasks vision. "
        f"{FACE_CROP_MEDIAPIPE_INSTALL_HINT}"
    )


def assert_mediapipe_legacy_solutions() -> None:
    """Backward-compatible check used in tests; prefer assert_mediapipe_face_crop_runtime."""
    if mediapipe_legacy_solutions_available():
        return
    if mediapipe_tasks_vision_available():
        return
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise FaceCropDependencyError(
            f"MediaPipe is not installed. {FACE_CROP_MEDIAPIPE_INSTALL_HINT}"
        ) from exc
    version = getattr(mp, "__version__", "unknown")
    raise FaceCropDependencyError(
        f"MediaPipe {version} does not expose mp.solutions.face_detection. "
        f"{FACE_CROP_MEDIAPIPE_INSTALL_HINT}"
    )


def _default_model_cache_dir() -> str:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache_root, "cyberverse", "mediapipe_models")


def _ensure_blaze_face_model(model_selection: int) -> str:
    if model_selection not in _BLAZE_FACE_MODEL_URLS:
        raise ValueError(f"Invalid model_selection {model_selection}; expected 0 (short) or 1 (full).")
    cache_dir = _default_model_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    name = "blaze_face_short_range.tflite" if model_selection == 0 else "blaze_face_full_range.tflite"
    dest = os.path.join(cache_dir, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    url = _BLAZE_FACE_MODEL_URLS[model_selection]
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise FaceCropDependencyError(
            f"Failed to download BlazeFace model from {url}: {exc}. "
            "Ensure outbound HTTPS or place the .tflite file manually at "
            f"{dest}."
        ) from exc
    return dest


@dataclass
class _TasksDetection:
    score: list[float]
    location_data: Any


@dataclass
class _TasksProcessResult:
    detections: list[_TasksDetection] | None


class _TasksFaceDetection:
    """Shim matching mp.solutions.face_detection.FaceDetection.process()."""

    def __init__(self, model_selection: int = 1, min_detection_confidence: float = 0.0):
        from mediapipe.tasks.python.core import base_options as base_options_module
        from mediapipe.tasks.python.vision import face_detector as face_detector_module
        from mediapipe.tasks.python.vision.core import image as image_module

        model_path = _ensure_blaze_face_model(model_selection)
        options = face_detector_module.FaceDetectorOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_detection_confidence,
        )
        try:
            self._detector = face_detector_module.FaceDetector.create_from_options(options)
        except OSError as exc:
            raise FaceCropDependencyError(
                "MediaPipe Tasks FaceDetector failed to load native libraries "
                f"({exc}). On headless servers install EGL/OSMesa deps, or use "
                f"mediapipe<={MEDIAPIPE_LEGACY_SOLUTIONS_MAX} with mp.solutions."
            ) from exc
        self._image_module = image_module

    def process(self, image):
        import numpy as np

        if not isinstance(image, np.ndarray):
            raise TypeError("Face detection expects an RGB numpy array")
        mp_image = self._image_module.Image(
            image_format=self._image_module.ImageFormat.SRGB,
            data=np.ascontiguousarray(image),
        )
        result = self._detector.detect(mp_image)
        if not result.detections:
            return _TasksProcessResult(detections=None)
        height, width = image.shape[:2]
        detections: list[_TasksDetection] = []
        for det in result.detections:
            box = det.bounding_box
            xmin = box.origin_x / width
            ymin = box.origin_y / height
            w = box.width / width
            h = box.height / height
            score = det.categories[0].score if det.categories else 0.0
            rel_box = SimpleNamespace(xmin=xmin, ymin=ymin, width=w, height=h)
            location = SimpleNamespace(relative_bounding_box=rel_box)
            detections.append(_TasksDetection(score=[score], location_data=location))
        return _TasksProcessResult(detections=detections)


def create_face_detection(model_selection: int = 1, min_detection_confidence: float = 0.0):
    assert_mediapipe_face_crop_runtime()
    if mediapipe_legacy_solutions_available():
        import mediapipe as mp

        return mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )
    return _TasksFaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )