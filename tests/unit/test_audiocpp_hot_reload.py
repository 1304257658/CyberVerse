from unittest.mock import MagicMock, patch

import pytest
import yaml

from inference.core.hot_reload import (
    apply_audiocpp_hot_params,
    load_active_avatar_params,
    log_startup_locked_drift,
)
from inference.plugins.avatar.audiocpp_flashhead_plugin import (
    AudioCppFlashHeadAvatarPlugin,
    _validate_resolution,
)


def test_apply_hot_params_merges_buffering(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "cyberverse.yaml").write_text(
        yaml.dump(
            {
                "inference": {
                    "avatar": {
                        "default": "audiocpp_flashhead",
                        "model_config_dir": "avatar_models",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    model_dir = cfg_dir / "avatar_models"
    model_dir.mkdir()
    (model_dir / "audiocpp_flashhead.yaml").write_text(
        yaml.dump(
            {
                "audiocpp_flashhead": {
                    "height": 720,
                    "width": 720,
                    "allow_resolution_test_profile": True,
                    "buffering": {"min_lead_s": 0.3},
                }
            }
        ),
        encoding="utf-8",
    )
    path = str(cfg_dir / "cyberverse.yaml")
    loaded = load_active_avatar_params(path, "audiocpp_flashhead")
    merged = apply_audiocpp_hot_params({"height": 512, "width": 512, "buffering": {"max_lead_s": 10}}, loaded)
    assert merged["height"] == 720
    assert merged["width"] == 720
    assert merged["buffering"]["min_lead_s"] == 0.3
    assert merged["buffering"]["max_lead_s"] == 10


def test_startup_locked_drift_logs_warning(caplog):
    with caplog.at_level("WARNING"):
        drifted = log_startup_locked_drift(
            "audiocpp_flashhead",
            {"lib_path": "/a.so"},
            {"lib_path": "/b.so"},
        )
    assert drifted == ["lib_path"]
    assert "restart inference" in caplog.text


def test_set_avatar_reencode_when_only_height_changes(tmp_path):
    plugin = AudioCppFlashHeadAvatarPlugin()
    plugin._worker = MagicMock()
    plugin._worker.call.return_value = ({"encode_ms": 1.0}, b"")
    plugin._params = {
        "height": 512,
        "width": 512,
        "num_inference_steps": 4,
        "seed": 42,
        "color_correction_strength": 1.0,
    }
    img = tmp_path / "face.png"
    img.write_bytes(b"same-image-bytes")
    plugin._set_avatar_sync(str(img), False)
    assert plugin._avatar_key is not None
    assert plugin._last_encode_sig == (512, 512, 4, 42, 1.0)

    with patch.object(plugin, "_reload_runtime_config_sync") as reload:
        def bump_height():
            plugin._params["height"] = 720
            plugin._params["width"] = 720
            plugin._params["allow_resolution_test_profile"] = True

        reload.side_effect = bump_height
        plugin._set_avatar_sync(str(img), False)

    assert plugin._worker.call.call_count == 2
    assert plugin._last_encode_sig[0] == 720


def test_hot_reload_720_without_flag_raises():
    plugin = AudioCppFlashHeadAvatarPlugin()
    plugin._params = {"height": 512, "width": 512}
    plugin._config_path = ""
    with pytest.raises(ValueError, match="allow_resolution_test_profile"):
        _validate_resolution(720, 720, {})