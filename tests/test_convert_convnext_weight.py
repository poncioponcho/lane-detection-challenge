import pytest

from scripts.autodl.convert_convnext_weight import select_backbone_state


def test_existing_backbone_namespace_is_preserved_without_double_prefix():
    state = {
        "backbone.stages.0.0.dwconv.weight": 1,
        "head.cls_layers.weight": 2,
    }
    selected = select_backbone_state(state)
    assert selected == {"backbone.stages.0.0.dwconv.weight": 1}


def test_plain_backbone_namespace_is_prefixed_once():
    selected = select_backbone_state({"stages.0.0.dwconv.weight": 1, "head.x": 2})
    assert selected == {"backbone.stages.0.0.dwconv.weight": 1}


def test_double_prefixed_state_is_rejected():
    with pytest.raises(ValueError, match="double-prefixed"):
        select_backbone_state({"backbone.backbone.stage.weight": 1})
