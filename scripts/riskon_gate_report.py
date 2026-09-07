#!/usr/bin/env python3
"""Gate verdict for a risk-on LVO screen vs the 15ep R50 baseline.

Diagnostic-metric (trainer-native) fold counts are the input. It is the fast
verdict path: it does not need the 7100-row OOF download, so it can run the
moment the eighth fold lands. The frozen Oracle remains the authority for any
reported score; this script only decides whether the screen is worth promoting.

Outputs a JSON verdict plus a Markdown report.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

N_BOOTSTRAP = 10000
SEED = 42
GATE_MIN_DF1_PP = 1.0
GATE_MIN_POSITIVE_VIDEOS = 5
GATE_WORST_LOCO_FLOOR_PP = -0.50


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2.0 * tp / denom if denom else 0.0


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload["folds"] if isinstance(payload, dict) else payload
    rows = []
    for item in folds:
        if item.get("status") != "pass":
            raise SystemExit(f"fold not pass: {item.get('fold_name')} -> {item.get('status')}")
        g = item["diagnostic"]
        rows.append(
            {
                "fold": item["fold_name"],
                "video": item["fold_name"].split("_", 2)[2],
                "f1": g["F1"],
                "tp": g["TP"],
                "fp": g["FP"],
                "fn": g["FN"],
                "g": g["TP"] + g["FN"],
            }
        )
    rows.sort(key=lambda r: r["fold"])
    return rows


def aggregate(rows: list[dict]) -> tuple[float, int, int, int]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return f1(tp, fp, fn), tp, fp, fn


def paired_bootstrap(cand: list[dict], base: list[dict], n: int, seed: int) -> dict:
    """Resample the 8 video clusters with replacement; never frames."""
    rng = random.Random(seed)
    index = list(range(len(cand)))
    deltas = []
    for _ in range(n):
        pick = [rng.choice(index) for _ in index]
        ct = sum(cand[i]["tp"] for i in pick)
        cf = sum(cand[i]["fp"] for i in pick)
        cn = sum(cand[i]["fn"] for i in pick)
        bt = sum(base[i]["tp"] for i in pick)
        bf = sum(base[i]["fp"] for i in pick)
        bn = sum(base[i]["fn"] for i in pick)
        deltas.append((f1(ct, cf, cn) - f1(bt, bf, bn)) * 100.0)
    deltas.sort()
    lo = deltas[int(0.025 * n)]
    hi = deltas[int(0.975 * n) - 1]
    return {
        "n_bootstrap": n,
        "seed": seed,
        "unit": "video",
        "ci95_pp": [lo, hi],
        "mean_pp": sum(deltas) / n,
    }


def loco(cand: list[dict], base: list[dict]) -> list[dict]:
    out = []
    for i in range(len(cand)):
        keep = [r for j, r in enumerate(cand) if j != i]
        keepb = [r for j, r in enumerate(base) if j != i]
        cf, _, _, _ = aggregate(keep)
        bf, _, _, _ = aggregate(keepb)
        out.append(
            {
                "video": cand[i]["video"],
                "delta_pp": (cf - bf) * 100.0,
                "cand_f1": cf,
                "base_f1": bf,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True, type=Path, help="candidate folds summary json")
    ap.add_argument("--base", required=True, type=Path, help="baseline folds summary json")
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    ap.add_argument("--label", default="risk-on 960x384")
    ap.add_argument("--baseline-label", default="15ep R50 800x320 baseline")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    cand, base = load(args.cand), load(args.base)
    if len(cand) != len(base):
        raise SystemExit(f"fold count mismatch: cand={len(cand)} base={len(base)}")

    per_fold = []
    for c, b in zip(cand, base):
        per_fold.append(
            {
                "fold": c["fold"],
                "video": c["video"],
                "cand_f1": c["f1"],
                "base_f1": b["f1"],
                "delta_pp": (c["f1"] - b["f1"]) * 100.0,
                "d_tp": c["tp"] - b["tp"],
                "d_fp": c["fp"] - b["fp"],
            }
        )

    cand_g, ctp, cfp, cfn = aggregate(cand)
    base_g, btp, bfp, bfn = aggregate(base)
    delta_pp = (cand_g - base_g) * 100.0
    boot = paired_bootstrap(cand, base, args.bootstrap, args.seed)
    loco_rows = loco(cand, base)
    positive = sum(1 for r in per_fold if r["delta_pp"] > 0)
    worst = min(r["delta_pp"] for r in loco_rows)

    gates = {
        "global_dF1_ge_1pp": delta_pp >= GATE_MIN_DF1_PP,
        "bootstrap_ci_lower_gt_0": boot["ci95_pp"][0] > 0,
        "positive_videos_ge_5": positive >= GATE_MIN_POSITIVE_VIDEOS,
        "worst_loco_ge_minus_0p50pp": worst >= GATE_WORST_LOCO_FLOOR_PP,
    }
    verdict = "GREEN" if all(gates.values()) else "RED"

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cand_label": args.label,
        "base_label": args.baseline_label,
        "folds": len(cand),
        "global": {
            "cand_f1": cand_g,
            "base_f1": base_g,
            "delta_pp": delta_pp,
            "cand_counts": {"tp": ctp, "fp": cfp, "fn": cfn},
            "base_counts": {"tp": btp, "fp": bfp, "fn": bfn},
        },
        "per_fold": per_fold,
        "paired_bootstrap": boot,
        "loco": loco_rows,
        "positive_videos": positive,
        "worst_loco_pp": worst,
        "gates": gates,
        "verdict": verdict,
        "metric": "diagnostic (trainer-native); Oracle confirmation required before any reported score",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {args.label} LVO 闸门裁决（诊断口径）",
        "",
        f"> 生成时间（UTC）：{payload['created_utc']}",
        "",
        f"## 结论：**{verdict}**",
        "",
        "| 项 | 基线 | 候选 | 差异 |",
        "|---|---:|---:|---:|",
        f"| 全局 F1（计数加权，{len(cand)} 折） | {base_g:.6f} | {cand_g:.6f} | {delta_pp:+.3f}pp |",
        f"| TP | {btp} | {ctp} | {ctp - btp:+d} |",
        f"| FP | {bfp} | {cfp} | {cfp - bfp:+d} |",
        f"| FN | {bfn} | {cfn} | {cfn - bfn:+d} |",
        "",
        "## 逐折对照",
        "",
        "| 折 | video | 基线 F1 | 候选 F1 | Δ(pp) | ΔTP | ΔFP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_fold:
        lines.append(
            f"| {r['fold']} | `{r['video']}` | {r['base_f1']:.6f} | {r['cand_f1']:.6f} | "
            f"{r['delta_pp']:+.3f} | {r['d_tp']:+d} | {r['d_fp']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 稳定性证据",
            "",
            f"- {len(cand)} 个 video cluster、{boot['n_bootstrap']} 次 **paired** bootstrap、seed={boot['seed']}。",
            f"- ΔF1 95% CI：`[{boot['ci95_pp'][0]:+.3f}, {boot['ci95_pp'][1]:+.3f}]pp`；下界"
            f"{'大于' if boot['ci95_pp'][0] > 0 else '不大于'} 0。",
            f"- 正向 video：`{positive}/{len(cand)}`。",
            f"- 最差 LOCO：`{worst:+.3f}pp`。",
            "",
            "## 三项放行门",
            "",
            "| 门 | 判据 | 实测 | 结果 |",
            "|---|---|---:|:--:|",
            f"| 效应量 | ΔF1 ≥ +{GATE_MIN_DF1_PP:.2f}pp | {delta_pp:+.3f}pp | {'通过' if gates['global_dF1_ge_1pp'] else '未通过'} |",
            f"| 稳定性 | paired bootstrap CI 下界 > 0 | {boot['ci95_pp'][0]:+.3f}pp | {'通过' if gates['bootstrap_ci_lower_gt_0'] else '未通过'} |",
            f"| 一致性 | ≥{GATE_MIN_POSITIVE_VIDEOS}/{len(cand)} 个 video 为正 | {positive}/{len(cand)} | {'通过' if gates['positive_videos_ge_5'] else '未通过'} |",
            "",
        ]
    )
    if verdict == "RED":
        lines.extend(
            [
                "## 处置",
                "",
                "- 不训练 36ep、不生成新 testA 候选、不消耗 A 榜额度。",
                "- 生产 incumbent 继续保持记录 `714962` / A 榜 `0.73444` / `conf=0.50`。",
                "- 本结论基于诊断口径；若需对外引用分数，仍须以冻结 Oracle 的全局单次调用为准。",
            ]
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{verdict}: global dF1={delta_pp:+.3f}pp CI=[{boot['ci95_pp'][0]:+.3f},{boot['ci95_pp'][1]:+.3f}] "
          f"positive={positive}/{len(cand)} worst_loco={worst:+.3f}pp")


if __name__ == "__main__":
    main()
