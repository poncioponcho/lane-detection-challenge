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

from common.io_utils import (read_instance_png, read_json_lanes, read_lane_mask,
                              read_lines_txt)


@dataclass
class LabelBundle:
    """All label formats found for one image (missing ones are None)."""
    image_id: str                        # relative path without extension
    lines_lanes: Optional[List[np.ndarray]] = None
    json_lanes: Optional[List[np.ndarray]] = None
    png_lanes: Optional[List[np.ndarray]] = None
    png_mask: Optional[np.ndarray] = None
    missing: List[str] = field(default_factory=list)   # formats not found


def _competition_paths(image_path: Path) -> tuple[Path, Path, Path] | None:
    """Resolve the official JPEGImages/anno_txt/Json/Annotations layout."""
    parts = image_path.parts
    try:
        marker = parts.index("JPEGImages")
    except ValueError:
        return None
    relative = Path(*parts[marker + 1:])
    if len(relative.parts) != 2:
        return None
    lane_root = Path(*parts[:marker])
    clip, image_name = relative.parts
    stem = Path(image_name).stem
    return (
        lane_root / "anno_txt" / clip / f"{stem}.lines.txt",
        lane_root / "Json" / clip / f"{image_name}.json",
        lane_root / "Annotations" / clip / f"{stem}.png",
    )


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
                 img_size=None, *, extract_png_instances: bool = True) -> LabelBundle:
    """Load every label format that exists for one image.

    image_path: path to the IMAGE (e.g. .../clip_0007/00042.jpg). Label files
    are looked up as siblings with .lines.txt / .json / .png suffixes.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    bundle = LabelBundle(image_id=str(p.with_suffix("")))
    official = _competition_paths(p)
    if official is None:
        lines_path = _find_sibling(p, (".lines.txt",))
        json_path = _find_sibling(p, (".json",))
        png_path = _find_sibling(p, (".png",))
    else:
        lines_path, json_path, png_path = official
        lines_path = lines_path if lines_path.is_file() else None
        json_path = json_path if json_path.is_file() else None
        png_path = png_path if png_path.is_file() else None

    if lines_path is not None:
        bundle.lines_lanes = read_lines_txt(lines_path)
    else:
        bundle.missing.append("lines.txt")

    if json_path is not None:
        try:
            bundle.json_lanes = read_json_lanes(json_path)
        except ValueError:
            # unknown schema: keep None, T21 will flag it for probing
            bundle.missing.append("json(unparsed)")
    else:
        bundle.missing.append("json")

    if png_path is not None and png_path != Path(image_path):
        bundle.png_mask = read_lane_mask(png_path)
        if extract_png_instances:
            bundle.png_lanes = read_instance_png(png_path, img_size=img_size)
    else:
        bundle.missing.append("png")

    return bundle
