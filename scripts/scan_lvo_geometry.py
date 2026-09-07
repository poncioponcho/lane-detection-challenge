#!/usr/bin/env python3
"""Pre-screen geometry-only post-processing on video-disjoint LVO OOF.

The LVO export contains only polylines, not candidate confidence scores.  This
script therefore never interprets line order as confidence.  It evaluates a
small, auditable grid of geometry-only duplicate suppression and conservative
lane-count caps with the frozen official Oracle, using video-cluster paired
bootstrap against the untouched OOF export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# Allow direct execution from any working directory.  The project packages
# intentionally live under src/ and are not installed as a wheel.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import ManifestRecord, read_manifest
from eval.oracle_runner import run_official_eval


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id(clip_id: str) -> str:
    marker = "_1_0_"
    if marker not in clip_id:
        raise ValueError(f"invalid clip id: {clip_id}")
    return clip_id.split(marker, 1)[0]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom else 0.0


def prediction_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.lines.txt")
        if path.is_file()
    }


def prediction_tree_sha256(root: Path, rels: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(rels):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(root / rel)))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Lane:
    points: tuple[tuple[float, float], ...]
    y_min: float
    y_max: float
    y_span: float
    length: float


def parse_lanes(path: Path) -> list[Lane]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    lanes: list[Lane] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 4 or len(fields) % 2:
            raise ValueError(f"{path}:{line_no}: invalid coordinate count")
        points = tuple(
            (float(fields[index]), float(fields[index + 1]))
            for index in range(0, len(fields), 2)
        )
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in points):
            raise ValueError(f"{path}:{line_no}: non-finite coordinate")
        y_values = [point[1] for point in points]
        length = sum(
            math.hypot(points[index + 1][0] - points[index][0],
                       points[index + 1][1] - points[index][1])
            for index in range(len(points) - 1)
        )
        y_min, y_max = min(y_values), max(y_values)
        lanes.append(Lane(points, y_min, y_max, y_max - y_min, length))
    return lanes


def x_at_y(lane: Lane, ys: np.ndarray) -> np.ndarray:
    # Predictions are normally y-monotone, but sorting makes the diagnostic
    # robust to a future exporter without changing the original points.
    ordered = sorted(lane.points, key=lambda point: point[1])
    y_values = np.asarray([point[1] for point in ordered], dtype=float)
    x_values = np.asarray([point[0] for point in ordered], dtype=float)
    unique_y, unique_indices = np.unique(y_values, return_index=True)
    return np.interp(ys, unique_y, x_values[unique_indices])


def near_duplicate(left: Lane, right: Lane, threshold_px: float) -> bool:
    overlap_low = max(left.y_min, right.y_min)
    overlap_high = min(left.y_max, right.y_max)
    overlap = overlap_high - overlap_low
    if overlap <= 0:
        return False
    min_span = min(left.y_span, right.y_span)
    if min_span <= 0 or overlap / min_span < 0.70:
        return False
    ys = np.linspace(overlap_low, overlap_high, 41)
    distance = np.abs(x_at_y(left, ys) - x_at_y(right, ys))
    return bool(np.median(distance) <= threshold_px and
                np.percentile(distance, 90) <= threshold_px * 1.8)


def suppress_duplicates(lanes: list[Lane], threshold_px: float) -> list[Lane]:
    # Geometry-only quality: visible vertical support, then polyline length.
    # Original index is a deterministic tie-breaker, never a score proxy.
    ranked = sorted(enumerate(lanes), key=lambda item: (
        -item[1].y_span, -item[1].length, item[0]))
    kept: list[tuple[int, Lane]] = []
    for original_index, lane in ranked:
        if any(near_duplicate(lane, other, threshold_px)
               for _, other in kept):
            continue
        kept.append((original_index, lane))
    return [lane for _, lane in sorted(kept, key=lambda item: item[0])]


def geometry_rank(lane: Lane) -> tuple[float, float]:
    return (lane.y_span, lane.length)


def transform(lanes: list[Lane], threshold_px: float | None,
              max_lanes: int | None) -> list[Lane]:
    result = suppress_duplicates(lanes, threshold_px) if threshold_px else list(lanes)
    if max_lanes is not None and len(result) > max_lanes:
        ranked = sorted(enumerate(result), key=lambda item: (
            -geometry_rank(item[1])[0], -geometry_rank(item[1])[1], item[0]))
        selected = {index for index, _ in ranked[:max_lanes]}
        result = [lane for index, lane in enumerate(result) if index in selected]
    return result


def format_lanes(lanes: list[Lane]) -> str:
    lines = []
    for lane in lanes:
        values = []
        for x, y in lane.points:
            values.extend((f"{x:.5f}", f"{y:.5f}"))
        lines.append(" ".join(values))
    return "\n".join(lines) + ("\n" if lines else "")


def write_variant(name: str, records: list[ManifestRecord], source: Path,
                  root: Path, threshold_px: float | None,
                  max_lanes: int | None) -> dict:
    target = root / name
    target.mkdir(parents=True, exist_ok=False)
    total_before = total_after = 0
    empty_before = empty_after = 0
    for record in records:
        source_path = source / record.pred_rel_path
        target_path = target / record.pred_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lanes = parse_lanes(source_path)
        transformed = transform(lanes, threshold_px, max_lanes)
        target_path.write_text(format_lanes(transformed), encoding="utf-8")
        total_before += len(lanes)
        total_after += len(transformed)
        empty_before += not lanes
        empty_after += not transformed
    rels = prediction_set(target)
    expected = {record.pred_rel_path for record in records}
    if rels != expected:
        raise RuntimeError(f"{name}: prediction set mismatch")
    return {
        "name": name,
        "threshold_px": threshold_px,
        "max_lanes": max_lanes,
        "images": len(records),
        "lines_before": total_before,
        "lines_after": total_after,
        "empty_before": empty_before,
        "empty_after": empty_after,
        "prediction_tree_sha256": prediction_tree_sha256(target, rels),
        "path": str(target),
    }


def score_variant(name: str, variant_meta: dict, records: list[ManifestRecord],
                  gt_dir: Path, official_python: Path,
                  output_dir: Path) -> dict:
    pred_dir = Path(variant_meta["path"])
    global_path = output_dir / name / "oracle_global.json"
    result = run_official_eval(
        pred_dir, gt_dir, records, official_python=official_python,
        per_clip=False, output_path=global_path,
    ).to_dict()
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    videos = []
    for video, original_subset in sorted(grouped.items()):
        subset = [replace(record, order=index)
                  for index, record in enumerate(original_subset)]
        video_path = output_dir / name / f"oracle_{video}.json"
        video_result = run_official_eval(
            pred_dir, gt_dir, subset, official_python=official_python,
            per_clip=False, output_path=video_path,
        ).to_dict()["global"]
        videos.append({
            "video": video,
            "tp": int(video_result["tp"]),
            "fp": int(video_result["fp"]),
            "fn": int(video_result["fn"]),
            "precision": float(video_result["precision"]),
            "recall": float(video_result["recall"]),
            "f1": float(video_result["f1"]),
            "oracle_path": str(video_path),
        })
    global_counts = result["global"]
    summed = tuple(sum(row[key] for row in videos) for key in ("tp", "fp", "fn"))
    if summed != (global_counts["tp"], global_counts["fp"], global_counts["fn"]):
        raise RuntimeError(f"{name}: video/global Oracle count mismatch")
    return {
        **variant_meta,
        "global": global_counts,
        "videos": videos,
        "oracle": result,
    }


def paired_bootstrap(base: dict, candidate: dict, *, seed: int,
                     n_bootstrap: int) -> dict:
    base_rows = {row["video"]: row for row in base["videos"]}
    candidate_rows = {row["video"]: row for row in candidate["videos"]}
    videos = sorted(base_rows)
    if set(videos) != set(candidate_rows):
        raise RuntimeError("paired bootstrap video sets differ")
    base_counts = np.asarray([[base_rows[v][key] for key in ("tp", "fp", "fn")]
                              for v in videos], dtype=np.int64)
    candidate_counts = np.asarray([[candidate_rows[v][key] for key in ("tp", "fp", "fn")]
                                   for v in videos], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(videos), size=(n_bootstrap, len(videos)))
    base_sum = base_counts[sample].sum(axis=1)
    candidate_sum = candidate_counts[sample].sum(axis=1)
    base_f1 = 2.0 * base_sum[:, 0] / np.maximum(
        2 * base_sum[:, 0] + base_sum[:, 1] + base_sum[:, 2], 1)
    candidate_f1 = 2.0 * candidate_sum[:, 0] / np.maximum(
        2 * candidate_sum[:, 0] + candidate_sum[:, 1] + candidate_sum[:, 2], 1)
    delta = candidate_f1 - base_f1
    return {
        "unit": "video",
        "paired": True,
        "videos": videos,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "observed_delta_f1": float(candidate["global"]["f1"] - base["global"]["f1"]),
        "observed_delta_pp": float((candidate["global"]["f1"] - base["global"]["f1"]) * 100),
        "ci95_delta_f1": [float(value) for value in np.percentile(delta, [2.5, 97.5])],
        "ci95_delta_pp": [float(value) for value in np.percentile(delta * 100, [2.5, 97.5])],
    }


def write_report(path: Path, value: dict) -> None:
    lines = [
        "# LVO 几何后处理预筛报告",
        "",
        f"> 生成时间（UTC）：{value['created_utc']}",
        "> 目的：在 15ep video-disjoint OOF 上做不含 score 的几何预筛；不是最终训练/榜单裁决。",
        "",
        "## 口径",
        "",
        f"- 输入：{value['records']} 张 OOF 图像、{value['videos']} 个 video cluster。",
        "- baseline 是原始 `.lines.txt`；没有把文件内行序解释为置信度。",
        "- 重复判定：y 重叠至少 70%，重叠区间中位横向距离及 P90 均低于阈值；保留几何支撑更长者。",
        "- 截断按 y 支撑和折线长度排序，只作诊断性上限扫描。",
        "- CI：同一 video cluster 成对重采样，汇总 TP/FP/FN 后计算 F1 差。",
        "",
        "## 结果",
        "",
        "| variant | threshold(px) | cap | lines | empty | F1 | Δpp | paired 95% CI(pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in value["results"]:
        boot = row["paired_bootstrap"]
        ci = boot["ci95_delta_pp"]
        lines.append(
            f"| `{row['name']}` | {row['threshold_px'] if row['threshold_px'] is not None else '—'} | "
            f"{row['max_lanes'] if row['max_lanes'] is not None else '—'} | "
            f"{row['lines_after']} | {row['empty_after']} | {row['global']['f1']:.6f} | "
            f"{boot['observed_delta_pp']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 此扫描只回答“几何去重/截断是否值得进入下一轮”，不能证明 testA 收益；15ep LVO 与 testA 仍有域差异。",
        "- 若 CI 下界不超过 0，不能把点估计的微小正增益当成方向确认。",
        "- 真正的 score-aware C 需要 D 阶段保留 raw candidate confidence；本报告不替代 C。",
        "",
        f"- 详细 JSON：`{value['json_path']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    expected = {record.pred_rel_path for record in records}
    actual = prediction_set(args.pred_dir)
    if actual != expected:
        raise SystemExit(
            f"prediction set mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    variants = [
        ("raw", None, None),
        ("dedup_d05", 5.0, None),
        ("dedup_d10", 10.0, None),
        ("dedup_d15", 15.0, None),
        ("dedup_d20", 20.0, None),
        ("cap4", None, 4),
        ("cap5", None, 5),
        ("dedup_d10_cap4", 10.0, 4),
        ("dedup_d10_cap5", 10.0, 5),
    ]
    metas = [write_variant(name, records, args.pred_dir, args.output_dir / "predictions",
                           threshold, cap)
             for name, threshold, cap in variants]
    scored = [score_variant(meta["name"], meta, records, args.gt_dir,
                            args.official_python, args.output_dir / "oracle")
              for meta in metas]
    base = scored[0]
    for row in scored:
        row["paired_bootstrap"] = paired_bootstrap(
            base, row, seed=args.seed, n_bootstrap=args.bootstrap)
        row.pop("oracle", None)
    value = {
        "status": "pass",
        "protocol": "15ep leave-one-video-out OOF geometry pre-screen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "records": len(records),
        "videos": len({video_id(record.clip_id) for record in records}),
        "source_prediction_tree_sha256": prediction_tree_sha256(args.pred_dir, actual),
        "official_python": str(args.official_python.resolve()),
        "bootstrap": {"n": args.bootstrap, "seed": args.seed, "unit": "video", "paired": True},
        "results": scored,
    }
    json_path = args.output_dir / "geometry_scan.json"
    value["json_path"] = str(json_path)
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "geometry_scan.md", value)
    print(json.dumps({
        "status": "pass",
        "json": str(json_path),
        "markdown": str(args.output_dir / "geometry_scan.md"),
        "variants": len(scored),
        "raw_f1": base["global"]["f1"],
        "best_point_f1": max(row["global"]["f1"] for row in scored),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
