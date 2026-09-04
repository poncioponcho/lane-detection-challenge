#!/usr/bin/env python3
"""Merge horizontal-flip TTA predictions into the base prediction set.

Design: neither the shared evaluator nor UnLanedet are touched (the val F1
replay path must stay byte-stable). The flip pass runs through the SAME
infer_testA.py pipeline over manifest_testA_flip.jsonl (images pre-flipped on
disk under JPEGImages/<clip>_hflip/). This script therefore works purely on
exported .lines.txt files:

  1. un-flip every flip-pass lane: x -> (CANVAS_W - 1) - x, y unchanged;
  2. for each un-flipped flip lane, match it against base lanes by mean |dx|
     over the overlapping y-range (both passes export on the same fixed
     sample_y grid, so per-y comparison is well defined);
  3. drop the flip lane when matched within --dup-dist pixels, else append it
     (recall boost: lanes only visible in the flipped view survive).

Precision stays anchored on the base set (conf 0.5 arm). No scores exist in
the exported files, so the merge is geometric-only by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CANVAS_W = 1366


def load_lanes(path: Path) -> list[np.ndarray]:
    lanes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) < 4 or len(tokens) % 2:
            continue
        lanes.append(np.asarray(tokens, dtype=np.float64).reshape(-1, 2))
    return lanes


def unflip(lane: np.ndarray) -> np.ndarray:
    out = lane.copy()
    out[:, 0] = (CANVAS_W - 1) - out[:, 0]
    return out


def mean_dx(lane_a: np.ndarray, lane_b: np.ndarray, samples: int = 36) -> float | None:
    """Mean |dx| between two y-monotone lanes over their overlapping y-range."""
    a = lane_a[np.argsort(lane_a[:, 1])]
    b = lane_b[np.argsort(lane_b[:, 1])]
    lo = max(a[0, 1], b[0, 1])
    hi = min(a[-1, 1], b[-1, 1])
    if hi - lo < 2.0:
        return None
    ys = np.linspace(lo, hi, samples)
    dx = np.interp(ys, a[:, 1], a[:, 0]) - np.interp(ys, b[:, 1], b[:, 0])
    return float(np.mean(np.abs(dx)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path,
                        help="Base predictions dir (clip/frame.lines.txt).")
    parser.add_argument("--flip-dir", required=True, type=Path,
                        help="Flip-pass predictions dir (clip_hflip/frame.lines.txt).")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Original (non-flip) testA manifest defining the file set.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dup-dist", type=float, default=24.0,
                        help="Mean |dx| (px) below which a flip lane counts as duplicate.")
    args = parser.parse_args()

    expected = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            expected.append((row["clip_id"], row["frame_id"], row["pred_rel_path"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "base_lanes": 0, "added_from_flip": 0,
             "dup_matched": 0, "empty_files": 0}
    for clip, frame, rel in expected:
        base_lanes = load_lanes(args.base_dir / rel)
        flip_lanes = [unflip(lane) for lane in
                      load_lanes(args.flip_dir / f"{clip}_hflip/{frame}.lines.txt")]
        kept = [lane for lane in base_lanes]
        for flip_lane in flip_lanes:
            dists = [mean_dx(flip_lane, base_lane) for base_lane in base_lanes]
            dists = [d for d in dists if d is not None]
            if dists and min(dists) < args.dup_dist:
                stats["dup_matched"] += 1
                continue
            kept.append(flip_lane)
            stats["added_from_flip"] += 1

        out_path = args.out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [" ".join(f"{v:.5f}" for v in lane.reshape(-1)) for lane in kept]
        out_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["files"] += 1
        stats["base_lanes"] += len(base_lanes)
        if not kept:
            stats["empty_files"] += 1

    print(json.dumps(stats | {"out_dir": str(args.out_dir),
                              "dup_dist_px": args.dup_dist}))


if __name__ == "__main__":
    main()
