from pathlib import Path

import pytest

from scripts.autodl.build_resolution_config import build


BASE = Path(__file__).parents[1] / "configs/unlanedet/clrnet_r50_hardlane.py"


def test_build_resolution_config_changes_only_preprocess_declarations(tmp_path):
    output = tmp_path / "resolution.py"
    build(BASE, output, 960, 480, 0)
    text = output.read_text(encoding="utf-8")
    assert "img_w = 960" in text
    assert "img_h = 480" in text
    assert "cut_height = 0" in text
    assert "img_w = 800" not in text
    assert "img_h = 320" not in text
    assert "cut_height = 180" not in text


def test_build_matched_fov_risk_on_candidate(tmp_path):
    output = tmp_path / "risk_on_resolution.py"
    build(BASE, output, 960, 384, 180)
    text = output.read_text(encoding="utf-8")
    assert "img_w = 960" in text
    assert "img_h = 384" in text
    assert "cut_height = 180" in text
    assert "img_w = 800" not in text
    assert "img_h = 320" not in text


def test_build_resolution_config_rejects_unexpected_base(tmp_path):
    base = tmp_path / "bad.py"
    base.write_text("img_w = 800\nimg_h = 320\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cut_height"):
        build(base, tmp_path / "out.py", 960, 480, 0)
