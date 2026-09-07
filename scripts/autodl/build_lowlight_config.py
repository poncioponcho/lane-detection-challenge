#!/usr/bin/env python3
"""Build an explicit config for the conditional low-light eval screen."""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {name} declaration, found {count}")
    return text.replace(old, new, 1)


def build(base: Path, output: Path, luma_threshold: float, gamma: float) -> None:
    if not math.isfinite(luma_threshold) or not 0.0 <= luma_threshold <= 255.0:
        raise ValueError("luma_threshold must be finite and in [0, 255]")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    text = base.read_text(encoding="utf-8")
    old = """        \"conditional_gamma\": {
            \"enabled\": False,
            \"luma_threshold\": 42.0,
            \"gamma\": 0.85,
        },"""
    new = """        \"conditional_gamma\": {
            \"enabled\": True,
            \"luma_threshold\": %s,
            \"gamma\": %s,
        },""" % (repr(float(luma_threshold)), repr(float(gamma)))
    text = replace_once(text, old, new, name="conditional_gamma")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--luma-threshold", type=float, default=42.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    args = parser.parse_args()
    build(args.base.resolve(), args.output.resolve(), args.luma_threshold, args.gamma)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
