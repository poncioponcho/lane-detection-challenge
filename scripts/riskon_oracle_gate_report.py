#!/usr/bin/env python3
"""Build the final risk-on gate report from frozen Oracle evaluations.

The diagnostic trainer counts are intentionally not used here.  Both input
files must be passing, video-disjoint evaluations produced by
``evaluate_lvo_video_oof.py``.  The image count is read from and cross-checked
against each evaluation artifact rather than hard-coded, so reduced screen
protocols such as a 6300-image run are audited correctly.  The global score
is computed from the single full-OOF Oracle result; video rows are used only
for paired bootstrap and leave-one-video-out diagnostics.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MIN_GLOBAL_DELTA_PP = 1.0
MIN_POSITIVE_VIDEOS = 5
BOOTSTRAP = 10_000
SEED = 42


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2.0 * tp / denominator if denominator else 0.0


def load_eval(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "pass":
        raise SystemExit(f"evaluation is not passing: {path}")
    try:
        prediction_count = int(value["prediction_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"evaluation has no valid prediction count: {path}") from exc
    if prediction_count <= 0:
        raise SystemExit(f"evaluation has an invalid prediction count: {path}")
    rows = value.get("videos")
    if not isinstance(rows, list) or len(rows) != 8:
        raise SystemExit(f"expected 8 video rows: {path}")
    by_video = {row["video"]: row for row in rows}
    if len(by_video) != 8:
        raise SystemExit(f"duplicate video rows: {path}")
    try:
        image_count = sum(int(row["images"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"video rows have no valid image counts: {path}") from exc
    if image_count != prediction_count:
        raise SystemExit(
            f"prediction/video image count mismatch: {prediction_count} != {image_count}: {path}"
        )
    global_counts = value["global_oracle"]["global"]
    summed = {
        key: sum(int(row[key]) for row in rows)
        for key in ("tp", "fp", "fn")
    }
    if summed != {key: int(global_counts[key]) for key in ("tp", "fp", "fn")}:
        raise SystemExit(f"global/video count mismatch: {path}")
    return value | {"by_video": by_video, "prediction_count": prediction_count}


def load_fold_names(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in value["folds"]:
        fold = row["fold_name"]
        result[fold.split("_", 2)[2]] = fold
    return result


def counts(row: dict) -> tuple[int, int, int]:
    return tuple(int(row[key]) for key in ("tp", "fp", "fn"))


def paired_bootstrap(cand: dict, base: dict, n: int, seed: int) -> dict:
    videos = sorted(cand["by_video"])
    cand_counts = np.asarray([counts(cand["by_video"][v]) for v in videos], dtype=np.int64)
    base_counts = np.asarray([counts(base["by_video"][v]) for v in videos], dtype=np.int64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(videos), size=(n, len(videos)))
    csum = cand_counts[sample].sum(axis=1)
    bsum = base_counts[sample].sum(axis=1)
    cf1 = 2.0 * csum[:, 0] / np.maximum(2 * csum[:, 0] + csum[:, 1] + csum[:, 2], 1)
    bf1 = 2.0 * bsum[:, 0] / np.maximum(2 * bsum[:, 0] + bsum[:, 1] + bsum[:, 2], 1)
    delta_pp = (cf1 - bf1) * 100.0
    lo, hi = np.percentile(delta_pp, [2.5, 97.5])
    return {
        "unit": "video",
        "paired": True,
        "videos": videos,
        "n_bootstrap": n,
        "seed": seed,
        "mean_delta_pp": float(delta_pp.mean()),
        "sd_delta_pp": float(delta_pp.std(ddof=1)),
        "ci95_delta_pp": [float(lo), float(hi)],
        "p_delta_gt_0": float(np.mean(delta_pp > 0.0)),
    }


def loco(cand: dict, base: dict) -> list[dict]:
    rows = []
    for omitted in sorted(cand["by_video"]):
        csum = {
            key: sum(int(cand["by_video"][v][key]) for v in cand["by_video"] if v != omitted)
            for key in ("tp", "fp", "fn")
        }
        bsum = {
            key: sum(int(base["by_video"][v][key]) for v in base["by_video"] if v != omitted)
            for key in ("tp", "fp", "fn")
        }
        delta_pp = (f1(**csum) - f1(**bsum)) * 100.0
        rows.append({
            "omitted_video": omitted,
            "cand_f1": f1(**csum),
            "base_f1": f1(**bsum),
            "delta_pp": delta_pp,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--folds-candidate", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--autodl-status", required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cand = load_eval(args.candidate.resolve())
    base = load_eval(args.baseline.resolve())
    if cand["prediction_count"] != base["prediction_count"]:
        raise SystemExit(
            "candidate and baseline prediction counts differ: "
            f"{cand['prediction_count']} != {base['prediction_count']}"
        )
    if set(cand["by_video"]) != set(base["by_video"]):
        raise SystemExit("candidate and baseline video sets differ")
    fold_names = load_fold_names(args.folds_candidate.resolve())
    if set(fold_names) != set(cand["by_video"]):
        raise SystemExit("fold/video sets differ")

    cg = cand["global_oracle"]["global"]
    bg = base["global_oracle"]["global"]
    delta_pp = (float(cg["f1"]) - float(bg["f1"])) * 100.0
    boot = paired_bootstrap(cand, base, args.bootstrap, args.seed)
    loco_rows = loco(cand, base)
    positive = sum(
        float(cand["by_video"][video]["f1"]) > float(base["by_video"][video]["f1"])
        for video in cand["by_video"]
    )
    gates = {
        "global_dF1_ge_1pp": delta_pp >= MIN_GLOBAL_DELTA_PP,
        "bootstrap_ci_lower_gt_0": boot["ci95_delta_pp"][0] > 0.0,
        "positive_videos_ge_5": positive >= MIN_POSITIVE_VIDEOS,
    }
    verdict = "GREEN" if all(gates.values()) else "RED"

    per_video = []
    for video in sorted(cand["by_video"]):
        c = cand["by_video"][video]
        b = base["by_video"][video]
        per_video.append({
            "video": video,
            "fold": fold_names[video],
            "cand_f1": float(c["f1"]),
            "base_f1": float(b["f1"]),
            "delta_pp": (float(c["f1"]) - float(b["f1"])) * 100.0,
            "cand_counts": {key: int(c[key]) for key in ("tp", "fp", "fn")},
            "base_counts": {key: int(b[key]) for key in ("tp", "fp", "fn")},
        })

    payload = {
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "metric": "frozen official Oracle",
            "records": cand["prediction_count"],
            "videos": len(cand["by_video"]),
            "global_score": "one full-OOF Oracle call per experiment",
            "bootstrap": "paired video-cluster resampling",
        },
        "autodl_status": args.autodl_status,
        "candidate_evaluation": str(args.candidate.resolve()),
        "baseline_evaluation": str(args.baseline.resolve()),
        "global": {
            "candidate": {key: float(cg[key]) if key == "f1" else int(cg[key]) for key in ("f1", "tp", "fp", "fn")},
            "baseline": {key: float(bg[key]) if key == "f1" else int(bg[key]) for key in ("f1", "tp", "fp", "fn")},
            "delta_pp": delta_pp,
        },
        "per_video": per_video,
        "paired_bootstrap": boot,
        "loco": loco_rows,
        "positive_videos": positive,
        "gates": gates,
        "verdict": verdict,
        "disposition": {
            "train_36ep": False,
            "package": False,
            "submit_a榜": False,
            "incumbent": "714962 / 0.73444 / conf=0.50 unchanged",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def result(ok: bool) -> str:
        return "通过" if ok else "未通过"

    lines = [
        f"> AutoDL：{args.autodl_status}",
        "",
        "# risk-on 960×384 + cut_height=180：冻结 Oracle 终审",
        "",
        f"- 生成时间（UTC）：{payload['created_utc']}",
        f"- 评估协议：{cand['prediction_count']} 张 OOF 图像、{len(cand['by_video'])} 个 held-out video；全局 F1 来自一次完整 OOF 官方 Oracle 调用；禁止用 per-video F1 平均代替全局分数。",
        "- 候选：CLRNet-R50，960×384，`cut_height=180`，15ep；基线：CLRNet-R50，800×320，15ep。",
        "",
        f"## 最终裁决：**{verdict}**",
        "",
        "## 逐折 Oracle 对照",
        "",
        "| 折 | 基线 F1 | 基线 TP/FP/FN | 候选 F1 | 候选 TP/FP/FN | ΔF1(pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_video:
        c = row["cand_counts"]
        b = row["base_counts"]
        lines.append(
            f"| `{row['fold']}` | {row['base_f1']:.9f} | {b['tp']}/{b['fp']}/{b['fn']} | "
            f"{row['cand_f1']:.9f} | {c['tp']}/{c['fp']}/{c['fn']} | {row['delta_pp']:+.3f} |"
        )
    lines.extend([
        "",
        "## 计数加权全局 Oracle",
        "",
        "| 方案 | TP | FP | FN | F1 |",
        "|---|---:|---:|---:|---:|",
        f"| 基线 | {bg['tp']} | {bg['fp']} | {bg['fn']} | {bg['f1']:.9f} |",
        f"| 候选 | {cg['tp']} | {cg['fp']} | {cg['fn']} | {cg['f1']:.9f} |",
        f"| 候选−基线 | {int(cg['tp']) - int(bg['tp']):+d} | {int(cg['fp']) - int(bg['fp']):+d} | {int(cg['fn']) - int(bg['fn']):+d} | **{delta_pp:+.6f}pp** |",
        "",
        "## Paired video bootstrap",
        "",
        f"- 单位：{len(cand['by_video'])} 个 held-out video；重采样 {boot['n_bootstrap']} 次；seed={boot['seed']}。",
        f"- ΔF1 95% CI：`[{boot['ci95_delta_pp'][0]:+.3f}, {boot['ci95_delta_pp'][1]:+.3f}]pp`；均值 `{boot['mean_delta_pp']:+.3f}pp`；P(Δ>0) `{boot['p_delta_gt_0']:.3f}`。",
        "",
        "## LOCO（逐一留出 video）",
        "",
        "| 留出 video | ΔF1(pp) |",
        "|---|---:|",
    ])
    for row in loco_rows:
        lines.append(f"| `{row['omitted_video']}` | {row['delta_pp']:+.3f} |")
    lines.extend([
        f"| 最小/最大 | **{min(row['delta_pp'] for row in loco_rows):+.3f} / {max(row['delta_pp'] for row in loco_rows):+.3f}** |",
        "",
        "## 三项放行门",
        "",
        "| 门 | 判据 | 实测 | 结果 |",
        "|---|---|---:|:--:|",
        f"| 效应量 | 全局 ΔF1 ≥ +{MIN_GLOBAL_DELTA_PP:.1f}pp | {delta_pp:+.3f}pp | {result(gates['global_dF1_ge_1pp'])} |",
        f"| 稳定性 | paired bootstrap CI 下界 > 0 | {boot['ci95_delta_pp'][0]:+.3f}pp | {result(gates['bootstrap_ci_lower_gt_0'])} |",
        f"| 一致性 | ≥{MIN_POSITIVE_VIDEOS}/{len(cand['by_video'])} 个 video 为正 | {positive}/{len(cand['by_video'])} | {result(gates['positive_videos_ge_5'])} |",
        "",
        "## 处置",
        "",
        "- 三项门未全部通过，最终为 **RED**；不训练 36ep、不打包、不提交 A 榜。",
        "- incumbent `714962 / 0.73444 / conf=0.50` 保持不变。",
        "- AutoDL 训练已结束；如不再继续实验，应手动关机/释放实例以停止空闲服务成本。",
        "",
        f"- 机器可读裁决：`{args.output_json.resolve()}`",
    ])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "pass", "verdict": verdict, "delta_pp": delta_pp,
        "ci95_delta_pp": boot["ci95_delta_pp"], "positive": f"{positive}/8",
        "markdown": str(args.output_md.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
