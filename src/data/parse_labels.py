"""T20: parse all three label formats for one image (ARCHITECTURE §6.2).

The three formats (TASKS.md §0) describe the SAME lanes:
  <image_id>.lines.txt   primary annotation (also the submission format)
  <image_id>.json        secondary
  <image_id>.png         single-channel instance ids (0 = background)

parse_labels() loads whichever formats exist next to the image and returns a
LabelBundle; check_label_consistency (T21) then verifies they agree.

All parsing primitives live in src/common/io_utils.py (ARCHITECTURE §3);
this module only orchestrates file discovery + image_id conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from common.io_utils import (read_instance_png, read_json_lanes,
                              read_lines_txt)


@dataclass
class LabelBundle:
    """All label formats found for one image (missing ones are None)."""
    image_id: str                        # relative path without extension
    lines_lanes: Optional[List[np.ndarray]] = None
    json_lanes: Optional[List[np.ndarray]] = None
    png_lanes: Optional[List[np.ndarray]] = None
    missing: List[str] = field(default_factory=list)   # formats not found


def _find_sibling(base: Path, suffixes) -> Optional[Path]:
    """Find the first existing sibling file with any of the given suffixes.

    Tries both <base><suffix> (CULane: img.jpg + img.jpg.lines.txt style)
    and <stem><suffix> (img.jpg + img.lines.txt style).
    """
    for suf in suffixes:
        cand = Path(str(base) + suf)
        if cand.exists():
            return cand
        cand = base.with_name(base.stem + suf)
        if cand.exists() and cand != base:
            return cand
    return None


def parse_labels(image_path: str | Path,
                 img_size=None) -> LabelBundle:
    """Load every label format that exists for one image.

    image_path: path to the IMAGE (e.g. .../clip_0007/00042.jpg). Label files
    are looked up as siblings with .lines.txt / .json / .png suffixes.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    bundle = LabelBundle(image_id=str(p.with_suffix("")))
    base = p  # sibling search base = image path itself

    lines_path = _find_sibling(base, (".lines.txt",))
    if lines_path is not None:
        bundle.lines_lanes = read_lines_txt(lines_path)
    else:
        bundle.missing.append("lines.txt")

    json_path = _find_sibling(base, (".json",))
    if json_path is not None:
        try:
            bundle.json_lanes = read_json_lanes(json_path)
        except ValueError:
            # unknown schema: keep None, T21 will flag it for probing
            bundle.missing.append("json(unparsed)")
    else:
        bundle.missing.append("json")

    png_path = _find_sibling(base, (".png",))
    if png_path is not None and png_path != Path(image_path):
        bundle.png_lanes = read_instance_png(png_path, img_size=img_size)
    else:
        bundle.missing.append("png")

    return bundle
