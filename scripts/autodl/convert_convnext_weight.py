#!/usr/bin/env python3
"""Convert the published full CLRNet-ConvNeXt checkpoint for HardLane.

The UnLanedet release checkpoint already stores backbone parameters under the
``backbone.`` namespace and also contains CULane head parameters. The generic
``convert_weight_ld.py`` helper is for plain backbone checkpoints and would
prepend a second ``backbone.`` prefix here. This converter keeps only the
already-prefixed backbone entries and fails closed on double-prefix output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def select_backbone_state(state: dict) -> dict:
    """Return a challenge-model-compatible backbone-only state dictionary."""
    prefixed = {
        key: value for key, value in state.items() if key.startswith("backbone.")
    }
    if prefixed:
        selected = prefixed
    else:
        selected = {
            f"backbone.{key}": value
            for key, value in state.items()
            if not key.startswith("head.")
        }
    if not selected:
        raise ValueError("source checkpoint contains no backbone parameters")
    if any(key.startswith("backbone.backbone.") for key in selected):
        raise ValueError("refusing to write a double-prefixed backbone state")
    if not all(key.startswith("backbone.") for key in selected):
        raise ValueError("converted keys are not all in the backbone namespace")
    return selected


def convert(source: Path, destination: Path) -> int:
    payload = torch.load(source, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint model payload is not a state dictionary")
    selected = select_backbone_state(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": selected}, destination)
    print(f"converted {len(selected)} backbone tensors: {source} -> {destination}")
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
