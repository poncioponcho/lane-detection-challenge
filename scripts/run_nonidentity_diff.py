"""Reproduce the real-GT, non-identity cross-environment metric audit.

The coordinator deterministically derives perturbed predictions from real GT,
runs this file as a worker under both the local and official Python
interpreters, and compares every image's TP/FP/FN triplet.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = (
    ROOT / "data" / "raw" / "dataset" / "_extract" /
    "train_full" / "Lane" / "anno_txt"
)
ORACLE_PATH = ROOT / "src" / "eval" / "official_oracle" / "score.py"
_ORACLE_MODULE = None


def _read_lanes(path: Path) -> list[np.ndarray]:
    lanes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            values = np.asarray(line.split(), dtype=np.float64)
            lanes.append(values.reshape(-1, 2))
    return lanes


def _dedup(lanes: list) -> list[list[list[float]]]:
    """Apply official text-parser consecutive deduplication."""
    output = []
    for lane in lanes:
        points = [tuple(point) for point in lane]
        cleaned = [points[0]]
        for point in points[1:]:
            if point != cleaned[-1]:
                cleaned.append(point)
        if len(cleaned) >= 2:
            output.append([[float(x), float(y)] for x, y in cleaned])
    return output


def generate_cases(gt_root: Path, sample_images: int, seed: int) -> dict:
    files = sorted(gt_root.rglob("*.lines.txt"))
    if sample_images > len(files):
        raise ValueError(f"requested {sample_images} images, only {len(files)} available")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(files), size=sample_images, replace=False)
    tasks = []
    shifts = [0.0, 2.0, 5.0, 8.0, 9.5, 10.0, 10.5, 11.0, 14.0, 20.0, 35.0]

    for slot, file_index in enumerate(indices):
        gt = _read_lanes(files[int(file_index)])
        if not gt:
            continue
        pred = []
        for lane in gt:
            if rng.random() < 0.15:
                continue
            shift = float(rng.choice(shifts))
            pred.append(np.round(lane + np.asarray([shift, shift * 0.15]), 1))
        for _ in range(int(rng.integers(0, 3))):
            source = gt[int(rng.integers(0, len(gt)))]
            offset = float(rng.choice([60.0, 120.0, 260.0]))
            pred.append(np.round(source + np.asarray([offset, 0.0]), 1))
        tasks.append({
            "name": f"img{slot}",
            "source": str(files[int(file_index)].relative_to(gt_root)),
            "gt": _dedup(gt),
            "pred": _dedup(pred),
        })
    return {"seed": seed, "requested_images": sample_images, "tasks": tasks}


def _load_oracle():
    global _ORACLE_MODULE
    if _ORACLE_MODULE is not None:
        return _ORACLE_MODULE
    spec = importlib.util.spec_from_file_location("frozen_score_nonidentity", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load Oracle: {ORACLE_PATH}")
    spec.loader.exec_module(module)
    _ORACLE_MODULE = module
    return _ORACLE_MODULE


def _score_local(pred, gt):
    sys.path.insert(0, str(ROOT / "src"))
    from eval.matching import iou_matrix, match_image
    from eval.rasterize import rasterize_lanes

    pred_masks = rasterize_lanes([np.asarray(lane, dtype=np.float64) for lane in pred])
    gt_masks = rasterize_lanes([np.asarray(lane, dtype=np.float64) for lane in gt])
    tp, p, g = match_image(pred_masks, gt_masks, iou_thr=0.5)
    matrix = iou_matrix(pred_masks, gt_masks) if pred_masks and gt_masks else np.zeros((0, 0))
    margin = float(np.abs(matrix - 0.5).min()) if matrix.size else None
    return [tp, p - tp, g - tp], margin


def _score_oracle(pred, gt):
    oracle = _load_oracle()
    pred_tuples = [[tuple(point) for point in lane] for lane in pred]
    gt_tuples = [[tuple(point) for point in lane] for lane in gt]
    triplet = oracle.culane_metric_single(pred_tuples, gt_tuples)[0.5]
    if not pred or not gt:
        return triplet, None
    pred_masks = [oracle.draw_lane_mask(oracle.interp_lane(lane), oracle.LINE_WIDTH)
                  for lane in pred_tuples]
    gt_masks = [oracle.draw_lane_mask(oracle.interp_lane(lane), oracle.LINE_WIDTH)
                for lane in gt_tuples]
    matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for row, pred_mask in enumerate(pred_masks):
        for col, gt_mask in enumerate(gt_masks):
            union = (pred_mask | gt_mask).sum()
            if union:
                matrix[row, col] = (pred_mask & gt_mask).sum() / union
    return triplet, float(np.abs(matrix - 0.5).min())


def run_worker(mode: str, cases_path: Path, output_path: Path) -> None:
    import cv2
    import scipy

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    scorer = _score_local if mode == "local" else _score_oracle
    rows = []
    for task in cases["tasks"]:
        triplet, margin = scorer(task["pred"], task["gt"])
        rows.append({
            "name": task["name"], "tp": int(triplet[0]),
            "fp": int(triplet[1]), "fn": int(triplet[2]), "margin": margin,
        })
    output_path.write_text(json.dumps({
        "mode": mode,
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "opencv": cv2.__version__,
        },
        "rows": rows,
    }, separators=(",", ":")), encoding="utf-8")


def _run_worker(python: Path, mode: str, cases: Path, output: Path) -> None:
    proc = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--worker", mode,
         "--cases", str(cases), "--worker-output", str(output)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(
            f"{mode} worker failed under {python}:\n{proc.stdout}\n{proc.stderr}"
        )


def _totals(rows: dict[str, dict]) -> dict:
    tp = sum(row["tp"] for row in rows.values())
    fp = sum(row["fp"] for row in rows.values())
    fn = sum(row["fn"] for row in rows.values())
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / denominator if denominator else 0.0}


def compare(local_path: Path, oracle_path: Path, cases_path: Path) -> dict:
    local_raw = json.loads(local_path.read_text(encoding="utf-8"))
    oracle_raw = json.loads(oracle_path.read_text(encoding="utf-8"))
    local = {row["name"]: row for row in local_raw["rows"]}
    oracle = {row["name"]: row for row in oracle_raw["rows"]}
    if local.keys() != oracle.keys():
        raise AssertionError("worker image sets differ")
    mismatches = []
    for name in local:
        local_triplet = [local[name][key] for key in ("tp", "fp", "fn")]
        oracle_triplet = [oracle[name][key] for key in ("tp", "fp", "fn")]
        if local_triplet != oracle_triplet:
            mismatches.append({"name": name, "local": local_triplet,
                               "oracle": oracle_triplet})
    margins = sorted(
        (row["margin"], name) for name, row in local.items()
        if row["margin"] is not None
    )
    cases_bytes = cases_path.read_bytes()
    cases = json.loads(cases_bytes)
    return {
        "status": "pass" if not mismatches else "fail",
        "seed": cases["seed"],
        "requested_images": cases["requested_images"],
        "scored_images": len(local),
        "gt_lanes": sum(len(task["gt"]) for task in cases["tasks"]),
        "local_environment": local_raw["environment"],
        "oracle_environment": oracle_raw["environment"],
        "local_totals": _totals(local),
        "oracle_totals": _totals(oracle),
        "per_image_triplet_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "closest_iou_margin": margins[0][0] if margins else None,
        "cases_sha256": hashlib.sha256(cases_bytes).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-python", type=Path)
    parser.add_argument("--oracle-python", type=Path)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--sample-images", type=int, default=600)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "reports" / "nonidentity_diff.json")
    parser.add_argument("--worker", choices=("local", "oracle"))
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.cases is None or args.worker_output is None:
            parser.error("worker mode requires --cases and --worker-output")
        run_worker(args.worker, args.cases, args.worker_output)
        return
    if args.local_python is None or args.oracle_python is None:
        parser.error("coordinator mode requires --local-python and --oracle-python")

    with tempfile.TemporaryDirectory(prefix="lane-nonidentity-") as temp:
        temp = Path(temp)
        cases_path = temp / "cases.json"
        local_path = temp / "local.json"
        oracle_path = temp / "oracle.json"
        cases = generate_cases(args.gt_root, args.sample_images, args.seed)
        cases_path.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
        _run_worker(args.local_python, "local", cases_path, local_path)
        _run_worker(args.oracle_python, "oracle", cases_path, oracle_path)
        report = compare(local_path, oracle_path, cases_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
