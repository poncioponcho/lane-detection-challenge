from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_convnext_screen_config_declares_same_input_contract():
    text = (
        PROJECT_ROOT / "configs/unlanedet/clrnet_convnext_tiny_hardlane.py"
    ).read_text(encoding="utf-8")
    assert "from .clrnet_r50_hardlane import" in text
    assert "backbone=L(ConvNeXt)" in text
    assert "in_channels=[192, 384, 768]" in text
    assert "batch_size = 12" in text
    assert '"max_to_keep": 3' in text


def test_convnext_screen_script_has_explicit_training_and_score_guards():
    text = (
        PROJECT_ROOT / "scripts/autodl/run_convnext_screen.sh"
    ).read_text(encoding="utf-8")
    assert "train.max_iter=7875" in text
    assert "train.init_checkpoint='$INIT_WEIGHT'" in text
    assert '"candidate_export_conf_threshold": scores.get("candidate_export_conf_threshold")' in text
    assert '"positive_class_softmax_probability"' in text
    assert 'prediction_count" -eq 800' in text
