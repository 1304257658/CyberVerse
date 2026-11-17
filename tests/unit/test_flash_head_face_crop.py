import sys
import types
from unittest.mock import patch

import pytest

MODELS_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2] / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))


def test_mediapipe_without_solutions_or_tasks_raises_actionable_error():
    fake_mp = types.ModuleType("mediapipe")
    fake_mp.__version__ = "0.10.35"
    fake_tasks = types.ModuleType("mediapipe.tasks")
    fake_tasks_python = types.ModuleType("mediapipe.tasks.python")
    with patch.dict(
        sys.modules,
        {
            "mediapipe": fake_mp,
            "mediapipe.tasks": fake_tasks,
            "mediapipe.tasks.python": fake_tasks_python,
        },
    ):
        from flash_head.utils import mediapipe_compat

        with pytest.raises(mediapipe_compat.FaceCropDependencyError) as exc:
            mediapipe_compat.assert_mediapipe_legacy_solutions()
        assert "mp.solutions" in str(exc.value)
        assert "mediapipe" in str(exc.value).lower()


def test_cpu_face_handler_missing_solutions_uses_tasks_shim():
    import flash_head.utils.mediapipe_compat as mediapipe_compat

    class FakeDetector:
        def process(self, image):
            import numpy as np

            h, w = image.shape[:2]
            rel = type("B", (), {"xmin": 0.1, "ymin": 0.1, "width": 0.5, "height": 0.5})()
            loc = type("L", (), {"relative_bounding_box": rel})()
            det = type("D", (), {"score": [0.9], "location_data": loc})()
            return type("R", (), {"detections": [det]})()

    with patch.object(mediapipe_compat, "create_face_detection", return_value=FakeDetector()):
        from flash_head.utils.cpu_face_handler import CPUFaceHandler

        handler = CPUFaceHandler()
        import numpy as np

        boxes, scores = handler(np.zeros((100, 100, 3), dtype=np.uint8))
        assert len(boxes) == 1
        assert scores == [0.9]


def test_get_cond_image_dict_face_crop_failure_propagates():
    from flash_head.utils import cond_image as cond_image_mod

    class CropFailed(Exception):
        pass

    with patch.object(cond_image_mod, "process_image", side_effect=CropFailed("no crop")):
        with pytest.raises(CropFailed):
            cond_image_mod.get_cond_image_dict("/tmp/fake.png", use_face_crop=True)


def test_get_cond_image_dict_no_face_crop_skips_mediapipe(monkeypatch, tmp_path):
    from PIL import Image
    from flash_head.utils import cond_image as cond_image_mod

    img_path = tmp_path / "avatar.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)

    def boom(*_args, **_kwargs):
        raise AssertionError("process_image must not run when use_face_crop is false")

    monkeypatch.setattr(cond_image_mod, "process_image", boom)
    result = cond_image_mod.get_cond_image_dict(str(img_path), use_face_crop=False)
    assert len(result) == 1
    assert result["avatar"].size == (64, 64)