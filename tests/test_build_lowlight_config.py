from pathlib import Path

import pytest

from scripts.autodl.build_lowlight_config import build


BASE = Path(__file__).parents[1] / "configs/unlanedet/clrnet_r50_hardlane.py"


def test_build_lowlight_config_enables_conditional_gamma(tmp_path):
    output = tmp_path / "lowlight.py"
    build(BASE, output, 42.0, 0.85)
    text = output.read_text(encoding="utf-8")
    base_text = BASE.read_text(encoding="utf-8")
    assert '"enabled": True' in text
    assert '"luma_threshold": 42.0' in text
    assert '"gamma": 0.85' in text
    assert '"conditional_gamma": {\n            "enabled": False' in base_text
    assert '"conditional_gamma": {\n            "enabled": True' in text


def test_build_lowlight_config_rejects_invalid_parameters(tmp_path):
    with pytest.raises(ValueError, match="luma_threshold"):
        build(BASE, tmp_path / "bad.py", -1.0, 0.85)
    with pytest.raises(ValueError, match="gamma"):
        build(BASE, tmp_path / "bad.py", 42.0, 0.0)
