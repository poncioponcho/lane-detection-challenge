#!/usr/bin/env python3
"""Score LVO out-of-fold predictions with the frozen Oracle.

The global score is obtained from one full-manifest Oracle call.  Video-level
scores are obtained from one Oracle call per video, and the confidence
interval resamples those eight video clusters (never individual frames).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.manifest import read_manifest
from eval.oracle_runner import run_official_eval
from submit.pack_submit import _normalize_rel


def video_id(clip_id: str) -> str:
    marker = "_1_0_"
    if marker not in clip_id:
        raise ValueError(f"invalid clip id: {clip_id}")
    return clip_id.split(marker, 1)[0]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def prediction_set(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.lines.txt")
        if path.is_file()
    } if root.is_dir() else set()


def prediction_tree_sha256(root: Path, rels: set[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(rels):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashlib.sha256((root / rel).read_bytes()).hexdigest()))
        digest.update(b"\n")
    return digest.hexdigest()


def write_markdown(path: Path, value: dict) -> None:
    oracle = value["global_oracle"]["global"]
    bootstrap = value["video_cluster_bootstrap"]
    lines = [
        "# CLRNet-R50 LVO OOF 评估报告",
        "",
        f"> 生成时间（UTC）：{value['created_utc']}",
        "",
        "## 协议",
        "",
        f"- 模型：`{value['protocol']['model']}`；输入 `{value['protocol']['input']}`；"
        f"`cut_height={value['protocol']['cut_height']}`；15ep。",
        f"- {len(value['videos'])}-fold leave-one-video-out；每折固定使用 `model_final.pth`，不使用留出 video 选点。",
        f"- OOF 预测覆盖 {value['prediction_count']} 张图，每张恰好一次。",
        f"- CI 重采样单位为 {len(value['videos'])} 个 video cluster，不能解释为图像独立样本。",
        "",
        "## 全局 Oracle 结果",
        "",
        f"- F1：**{oracle['f1']:.6f}** ({oracle['f1'] * 100:.3f}pp)",
        f"- TP/FP/FN：`{oracle['tp']}/{oracle['fp']}/{oracle['fn']}`",
        f"- video-cluster bootstrap 95% CI："
        f"`[{bootstrap['ci95_f1'][0]:.6f}, {bootstrap['ci95_f1'][1]:.6f}]` "
        f"（{bootstrap['ci95_pp'][0]:.3f}–{bootstrap['ci95_pp'][1]:.3f}pp）",
        "",
        "## Video 画像",
        "",
        "| video | images | clips | TP | FP | FN | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in value["videos"]:
        lines.append(
            f"| `{row['video']}` | {row['images']} | {row['clips']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['f1']:.6f} |"
        )
    lines.extend([
        "",
        "## 证据与限制",
        "",
        f"- Oracle JSON：`{value['global_oracle_path']}`",
        f"- OOF evidence：`{value['oof_evidence_path']}`",
        f"- 预测树 SHA-256：`{value['prediction_tree_sha256']}`",
        "- 该结果是 15ep 的跨 video 泛化估计；不能直接替代 36ep 最终模型的独立评估。",
        "- 8 个 video 仍然很少，区间应作为不确定性提示，不应人为承诺固定半宽。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--oof-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="clrnet_r50")
    parser.add_argument("--input", default="800x320")
    parser.add_argument("--cut-height", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--expected-videos", type=int, default=8,
        help="expected held-out video count (default: 8)",
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    output_dir = args.output_dir.resolve()
    records = read_manifest(manifest)
    expected = {_normalize_rel(record.pred_rel_path) for record in records}
    actual = prediction_set(pred_dir)
    if actual != expected:
        raise SystemExit(
            f"OOF prediction set mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    oof_evidence = json.loads(args.oof_evidence.read_text(encoding="utf-8"))
    if oof_evidence.get("status") != "pass" or int(oof_evidence.get("prediction_count", -1)) != len(records):
        raise SystemExit("OOF evidence is not a passing 7100-row artifact")

    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "oracle_global_per_clip.json"
    global_result = run_official_eval(
        pred_dir, gt_dir, records, official_python=args.official_python,
        per_clip=True, output_path=global_path,
    ).to_dict()

    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    if len(grouped) != args.expected_videos:
        raise SystemExit(
            f"expected {args.expected_videos} videos, got {len(grouped)}"
        )

    video_rows = []
    video_oracle_paths = {}
    for video in sorted(grouped):
        subset = [replace(record, order=index)
                  for index, record in enumerate(grouped[video])]
        video_path = output_dir / f"oracle_{video}.json"
        result = run_official_eval(
            pred_dir, gt_dir, subset, official_python=args.official_python,
            per_clip=False, output_path=video_path,
        ).to_dict()
        counts = result["global"]
        video_oracle_paths[video] = str(video_path)
        video_rows.append({
            "video": video,
            "images": len(subset),
            "clips": len({record.clip_id for record in subset}),
            "tp": int(counts["tp"]), "fp": int(counts["fp"]), "fn": int(counts["fn"]),
            "precision": float(counts["precision"]),
            "recall": float(counts["recall"]),
            "f1": float(counts["f1"]),
            "oracle_path": str(video_path),
        })

    if sum(row["images"] for row in video_rows) != len(records):
        raise SystemExit("video image counts do not cover the manifest")
    tp = sum(row["tp"] for row in video_rows)
    fp = sum(row["fp"] for row in video_rows)
    fn = sum(row["fn"] for row in video_rows)
    if not np.isclose(f1_from_counts(tp, fp, fn), global_result["global"]["f1"], atol=1e-12):
        raise SystemExit("video count sum does not match global Oracle F1")

    n_video = len(video_rows)
    rng = np.random.default_rng(args.seed)
    samples = rng.integers(0, n_video, size=(args.bootstrap, n_video))
    tps = np.asarray([row["tp"] for row in video_rows], dtype=np.int64)[samples].sum(axis=1)
    fps = np.asarray([row["fp"] for row in video_rows], dtype=np.int64)[samples].sum(axis=1)
    fns = np.asarray([row["fn"] for row in video_rows], dtype=np.int64)[samples].sum(axis=1)
    boot_f1 = 2.0 * tps / np.maximum(2 * tps + fps + fns, 1)
    ci = np.percentile(boot_f1, [2.5, 97.5]).tolist()
    value = {
        "status": "pass",
        "protocol": {
            "name": "leave-one-video-out",
            "model": args.model,
            "input": args.input,
            "cut_height": args.cut_height,
            "epochs": args.epochs,
            "checkpoint_policy": "fixed model_final; no holdout selection",
            "bootstrap_unit": "video",
        },
        "manifest": {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "rows": len(records)},
        "prediction_count": len(actual),
        "prediction_tree_sha256": prediction_tree_sha256(pred_dir, actual),
        "global_oracle_path": str(global_path),
        "global_oracle": global_result,
        "videos": video_rows,
        "video_oracle_paths": video_oracle_paths,
        "video_cluster_bootstrap": {
            "n_video": n_video,
            "n_bootstrap": args.bootstrap,
            "seed": args.seed,
            "ci95_f1": ci,
            "ci95_pp": [value * 100.0 for value in ci],
            "observed_f1": float(global_result["global"]["f1"]),
        },
        "oof_evidence_path": str(args.oof_evidence.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_path = output_dir / "lvo_evaluation.json"
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "lvo_evaluation.md", value)
    print(json.dumps({
        "status": "pass", "f1": value["global_oracle"]["global"]["f1"],
        "ci95_f1": value["video_cluster_bootstrap"]["ci95_f1"],
        "videos": len(value["videos"]), "predictions": value["prediction_count"],
        "json": str(json_path), "markdown": str(output_dir / "lvo_evaluation.md"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
