#!/usr/bin/env python3
"""Official-Oracle scan of exported candidate scores across checkpoints.

The remote evaluator exports post-NMS lines plus one confidence per exported
line. This tool only filters those candidates, so threshold scans are local
operations and never reinterpret line order as score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution from any working directory.  The project packages
# intentionally live under src/ and are not installed as a wheel.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import ManifestRecord, read_manifest
from eval.oracle_runner import run_official_eval
from scan_lvo_geometry import paired_bootstrap, prediction_set, video_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_name(value: float) -> str:
    return f"conf_{value:.2f}".replace(".", "p")


def write_filtered_variant(
    name: str,
    records: list[ManifestRecord],
    pred_root: Path,
    score_path: Path,
    threshold: float,
    output_root: Path,
) -> dict:
    scores_payload = json.loads(score_path.read_text(encoding="utf-8"))
    scores_by_image = scores_payload.get("scores_by_image")
    if not isinstance(scores_by_image, dict):
        raise ValueError(f"missing scores_by_image: {score_path}")
    expected_images = {record.image_id for record in records}
    if set(scores_by_image) != expected_images:
        raise ValueError(
            f"{name}: score image set mismatch: "
            f"{len(scores_by_image)} != {len(expected_images)}"
        )
    export_threshold = float(
        scores_payload.get("candidate_export_conf_threshold", 0.0)
    )
    if not math.isfinite(export_threshold):
        raise ValueError(f"{name}: non-finite candidate export threshold")
    if threshold + 1e-12 < export_threshold:
        raise ValueError(
            f"{name}: requested threshold {threshold} is below candidate export "
            f"threshold {export_threshold}; rerun raw export first"
        )
    target = output_root / name
    target.mkdir(parents=True, exist_ok=False)
    before = after = 0
    empty = 0
    for record in records:
        source = pred_root / record.pred_rel_path
        target_path = target / record.pred_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        lines = source.read_text(encoding="utf-8").splitlines() if source.is_file() else []
        scores = scores_by_image.get(record.image_id)
        if not isinstance(scores, list) or len(scores) != len(lines):
            raise ValueError(
                f"{name}: score/line mismatch for {record.image_id}: "
                f"{len(scores) if isinstance(scores, list) else 'missing'} != {len(lines)}"
            )
        numeric_scores = []
        for score in scores:
            if score is None or not math.isfinite(float(score)):
                raise ValueError(
                    f"{name}: every exported line needs a finite score for "
                    f"{record.image_id}"
                )
            numeric_scores.append(float(score))
        kept = [
            line for line, score in zip(lines, numeric_scores)
            if score >= threshold
        ]
        target_path.write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
        )
        before += len(lines)
        after += len(kept)
        empty += not kept
    expected = {record.pred_rel_path for record in records}
    if prediction_set(target) != expected:
        raise ValueError(f"{name}: prediction set mismatch")
    return {
        "name": name,
        "threshold": threshold,
        "candidate_export_threshold": export_threshold,
        "path": str(target),
        "lines_before": before,
        "lines_after": after,
        "empty_images": empty,
        "score_sidecar": str(score_path),
    }


def score_variant(
    meta: dict,
    records: list[ManifestRecord],
    gt_dir: Path,
    official_python: Path,
    oracle_root: Path,
) -> dict:
    name = meta["name"]
    pred_root = Path(meta["path"])
    global_path = oracle_root / name / "oracle_global.json"
    global_result = run_official_eval(
        pred_root, gt_dir, records, official_python=official_python,
        per_clip=False, output_path=global_path,
    ).to_dict()
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    videos = []
    for video, original_subset in sorted(grouped.items()):
        subset = [
            replace(record, order=index)
            for index, record in enumerate(original_subset)
        ]
        path = oracle_root / name / f"oracle_{video}.json"
        counts = run_official_eval(
            pred_root, gt_dir, subset, official_python=official_python,
            per_clip=False, output_path=path,
        ).to_dict()["global"]
        videos.append({
            "video": video,
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "precision": float(counts["precision"]),
            "recall": float(counts["recall"]),
            "f1": float(counts["f1"]),
            "oracle_path": str(path),
        })
    counts = global_result["global"]
    if (
        sum(row["tp"] for row in videos),
        sum(row["fp"] for row in videos),
        sum(row["fn"] for row in videos),
    ) != (counts["tp"], counts["fp"], counts["fn"]):
        raise ValueError(f"{name}: global/video Oracle count mismatch")
    return {**meta, "global": counts, "videos": videos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input", action="append", required=True,
        help="label=prediction_dir=scores_json; repeat for each checkpoint",
    )
    parser.add_argument(
        "--threshold", action="append", required=True, type=float,
        help="confidence threshold; repeat to define the scan grid",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    expected = {record.pred_rel_path for record in records}
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    prediction_root = args.output_dir / "predictions"
    oracle_root = args.output_dir / "oracle"
    entries = []
    for raw in args.input:
        parts = raw.split("=", 2)
        if len(parts) != 3 or not all(parts):
            raise SystemExit(
                f"invalid --input {raw!r}; expected label=pred_dir=scores_json"
            )
        label, pred_dir, score_path = parts
        pred_dir_path, score_path_obj = Path(pred_dir), Path(score_path)
        if prediction_set(pred_dir_path) != expected:
            raise SystemExit(f"{label}: prediction set does not cover manifest")
        entries.append((label, pred_dir_path, score_path_obj))

    results = []
    for label, pred_dir, score_path in entries:
        for threshold in args.threshold:
            name = f"{label}_{threshold_name(threshold)}"
            meta = write_filtered_variant(
                name, records, pred_dir, score_path, threshold, prediction_root
            )
            results.append(score_variant(
                meta, records, args.gt_dir, args.official_python, oracle_root
            ))

    reference_name = "final_conf_0p40"
    references = {row["name"]: row for row in results}
    reference = references.get(reference_name, results[0])
    for row in results:
        row["paired_bootstrap"] = paired_bootstrap(
            reference, row, seed=args.seed, n_bootstrap=args.bootstrap
        )

    value = {
        "status": "pass",
        "protocol": "36ep leave-one-video-out confidence/checkpoint scan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
            "rows": len(records),
        },
        "gt_dir": str(args.gt_dir.resolve()),
        "official_python": str(args.official_python.resolve()),
        "records": len(records),
        "videos": len({video_id(record.clip_id) for record in records}),
        "reference": reference["name"],
        "bootstrap": {
            "unit": "video", "paired": True,
            "n": args.bootstrap, "seed": args.seed,
        },
        "results": results,
    }
    json_path = args.output_dir / "conf_checkpoint_scan.json"
    json_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# D OOF conf / checkpoint 扫描",
        "",
        f"> 生成时间（UTC）：{value['created_utc']}",
        "",
        f"- 参考 variant：{reference['name']}。",
        "- 全部分数来自冻结官方 Oracle；CI 单位为 video，成对重采样并汇总 TP/FP/FN。",
        "- 预测文件内行序未用于排序或打分；过滤依据是 evaluator 导出的 metadata score。",
        "",
        "| variant | export conf | threshold | lines | empty | F1 | Δpp vs ref | paired CI(pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        boot = row["paired_bootstrap"]
        ci = boot["ci95_delta_pp"]
        lines.append(
            f"| {row['name']} | {row['candidate_export_threshold']:.2f} | "
            f"{row['threshold']:.2f} | {row['lines_after']} | {row['empty_images']} | "
            f"{row['global']['f1']:.6f} | {boot['observed_delta_pp']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend([
        "",
        "## 选择规则",
        "",
        "- 只有 paired CI 下界 > 0 的正差异才可作为确认性改进；否则保留参考 variant。",
        "- checkpoint 选择只在 video-disjoint OOF 上进行，禁止回到泄漏 val 选点。",
        "",
        f"- 详细 JSON：{json_path}",
    ])
    (args.output_dir / "conf_checkpoint_scan.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "json": str(json_path),
        "markdown": str(args.output_dir / "conf_checkpoint_scan.md"),
        "variants": len(results),
        "reference": reference["name"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
