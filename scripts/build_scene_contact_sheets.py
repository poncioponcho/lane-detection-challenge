"""Build deterministic contact sheets for clip-level scene annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = (
    ROOT / "data" / "raw" / "dataset" / "_extract" /
    "train_full" / "Lane" / "JPEGImages"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "scene_contact_sheets"
SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
THUMB_W, THUMB_H = 273, 144
HEADER_H = 38
CLIPS_PER_PAGE = 5


def sample_paths(clip_dir: Path) -> list[Path]:
    frames = sorted(clip_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"clip has no JPEG frames: {clip_dir}")
    indices = [round(fraction * (len(frames) - 1)) for fraction in SAMPLE_FRACTIONS]
    return [frames[index] for index in indices]


def render_clip_row(clip_dir: Path, ordinal: int) -> tuple[np.ndarray, dict]:
    paths = sample_paths(clip_dir)
    header = np.full((HEADER_H, THUMB_W * len(paths), 3), 245, dtype=np.uint8)
    cv2.putText(
        header, f"#{ordinal:02d}  {clip_dir.name}", (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (15, 15, 15), 1, cv2.LINE_AA,
    )
    thumbnails = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image: {path}")
        thumb = cv2.resize(image, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        label = path.stem
        cv2.rectangle(thumb, (0, 0), (68, 22), (0, 0, 0), thickness=-1)
        cv2.putText(
            thumb, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    return np.vstack([header, np.hstack(thumbnails)]), {
        "ordinal": ordinal,
        "clip_id": clip_dir.name,
        "spot_frames": [path.stem for path in paths],
    }


def build(image_root: Path, output_dir: Path) -> dict:
    clips = sorted(path for path in image_root.iterdir() if path.is_dir())
    if len(clips) != 71:
        raise ValueError(f"expected 71 training clips, found {len(clips)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, records, pages = [], [], []
    for ordinal, clip_dir in enumerate(clips, start=1):
        row, record = render_clip_row(clip_dir, ordinal)
        rows.append(row)
        records.append(record)
        if len(rows) == CLIPS_PER_PAGE or ordinal == len(clips):
            if len(rows) < CLIPS_PER_PAGE:
                blank = np.full_like(rows[0], 230)
                rows.extend([blank] * (CLIPS_PER_PAGE - len(rows)))
            page_number = len(pages) + 1
            page_path = output_dir / f"page_{page_number:02d}.jpg"
            if not cv2.imwrite(str(page_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise OSError(f"cannot write contact sheet: {page_path}")
            pages.append(str(page_path))
            rows = []

    index = {
        "image_root": str(image_root),
        "sample_fractions": list(SAMPLE_FRACTIONS),
        "clips_per_page": CLIPS_PER_PAGE,
        "pages": pages,
        "records": records,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    index = build(args.image_root, args.output_dir)
    print(json.dumps({
        "clips": len(index["records"]), "pages": len(index["pages"]),
        "index": str(args.output_dir / "index.json"),
    }))


if __name__ == "__main__":
    main()
