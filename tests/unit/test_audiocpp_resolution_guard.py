import pytest

from inference.plugins.avatar.audiocpp_flashhead_plugin import _validate_resolution


def test_rejects_non_square_resolution():
    with pytest.raises(ValueError, match="square"):
        _validate_resolution(720, 1280, {})


def test_rejects_720_without_test_profile_flag():
    with pytest.raises(ValueError, match="allow_resolution_test_profile"):
        _validate_resolution(720, 720, {})


def test_allows_720_candidate_profile():
    _validate_resolution(720, 720, {"allow_resolution_test_profile": True})


def test_default_512_always_allowed():
    _validate_resolution(512, 512, {})