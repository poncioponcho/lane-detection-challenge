#!/usr/bin/env python3
"""Build an explicit CLRNet config for a resolution-only eval screen.

The production config remains frozen at 800x320/cut180. This small builder
creates a separate, auditable config by changing only the three top-level
preprocessing constants used by the LazyConfig file; all model, data split,
checkpoint, and NMS settings remain inherited from the base text.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {name} declaration, found {count}")
    return text.replace(old, new, 1)


def build(base: Path, output: Path, width: int, height: int, cut_height: int) -> None:
    if width <= 0 or height <= 0 or cut_height < 0:
        raise ValueError("width/height must be positive and cut_height non-negative")
    text = base.read_text(encoding="utf-8")
    text = replace_once(text, "img_w = 800", f"img_w = {width}", name="img_w")
    text = replace_once(text, "img_h = 320", f"img_h = {height}", name="img_h")
    text = replace_once(text, "cut_height = 180", f"cut_height = {cut_height}", name="cut_height")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--cut-height", required=True, type=int)
    args = parser.parse_args()
    build(args.base.resolve(), args.output.resolve(), args.width, args.height, args.cut_height)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
