"""Run the real-data local-vs-Oracle differential audit for T1.2.

This is a conformance tool, not an evaluation entry point. It directly loads
the frozen module solely to compare implementations. Production/final scoring
must go through ``oracle_runner.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval.matching import match_image  # type: ignore
    from eval.rasterize import parse_lines_txt, rasterize_lanes  # type: ignore
else:
    from .matching import match_image
    from .rasterize import parse_lines_txt, rasterize_lanes


ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = Path(__file__).resolve().parent / "official_oracle" / "score.py"
DEFAULT_LANE_ROOT = ROOT / "data" / "raw" / "dataset" / "_extract" / "train_full" / "Lane"


def _load_oracle():
    spec = importlib.util.spec_from_file_location("frozen_score_for_diff_audit", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load frozen Oracle: {ORACLE_PATH}")
    spec.loader.exec_module(module)
    return module


def _task_relpaths(list_path: Path) -> list[Path]:
    relpaths = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = Path(raw).parts
        relpaths.append(Path(parts[-2]) / f"{Path(parts[-1]).stem}.lines.txt")
    return relpaths


def run_real_identity(lane_root: Path, limit: int | None = None) -> dict:
    """Compare both implementations on every official training GT file."""
    oracle = _load_oracle()
    list_path = lane_root / "data" / "train.txt"
    gt_root = lane_root / "anno_txt"
    tasks = _task_relpaths(list_path)
    if limit is not None:
        tasks = tasks[:limit]

    started = time.monotonic()
    totals = {"tp": 0, "fp": 0, "fn": 0, "lanes": 0}
    for index, relpath in enumerate(tasks, start=1):
        gt_path = gt_root / relpath
        local_lanes = parse_lines_txt(gt_path)
        oracle_lanes = oracle.parse_lines_txt(str(gt_path))
        if len(local_lanes) != len(oracle_lanes):
            raise AssertionError(f"parse lane-count mismatch: {relpath}")
        for local_lane, oracle_lane in zip(local_lanes, oracle_lanes):
            np.testing.assert_array_equal(
                local_lane, np.asarray(oracle_lane, dtype=np.float64),
                err_msg=f"parse coordinate mismatch: {relpath}",
            )

        local_masks = rasterize_lanes(local_lanes)
        local_result = match_image(local_masks, local_masks)
        oracle_result = oracle.culane_metric_single(oracle_lanes, oracle_lanes)[0.5]
        local_triplet = [local_result[0], local_result[1] - local_result[0],
                         local_result[2] - local_result[0]]
        if local_triplet != oracle_result:
            raise AssertionError(
                f"metric mismatch at {relpath}: local={local_triplet}, oracle={oracle_result}"
            )
        totals["tp"] += local_triplet[0]
        totals["fp"] += local_triplet[1]
        totals["fn"] += local_triplet[2]
        totals["lanes"] += len(local_lanes)

        if index % 500 == 0:
            print(f"checked {index}/{len(tasks)} images", file=sys.stderr, flush=True)

    elapsed = time.monotonic() - started
    return {
        "status": "pass",
        "images": len(tasks),
        **totals,
        "f1": 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"])
        if totals["tp"] else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "oracle": str(ORACLE_PATH.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-root", type=Path, default=DEFAULT_LANE_ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_real_identity(args.lane_root, args.limit), indent=2))


if __name__ == "__main__":
    main()
