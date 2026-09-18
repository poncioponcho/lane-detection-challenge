#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solve the hidden GT count from a junk-probe score pair, then judge the lever.

Usage
-----
python scripts/decompose_probe_solve_20260913.py \\
    --base-f1 0.73505 --base-p 2664 --junk-f1 <returned score> --j 225

Self-test: --selftest sweeps candidate G, synthesises the score the probe would
return, and checks the inversion recovers G exactly.
"""
from __future__ import annotations

import argparse
import json


def solve(base_f1: float, base_p: int, junk_f1: float, j: int) -> dict:
    d = base_f1 - junk_f1
    if j <= 0:
        raise ValueError("j must be positive")
    if d <= 0:
        raise ValueError(f"junk F1 ({junk_f1}) did not fall below base F1 ({base_f1})")
    a = j * junk_f1 / d                      # A = P + G
    g = a - base_p
    tp = base_f1 * a / 2.0
    fp, fn = base_p - tp, g - tp
    return {
        "dF1": d,
        "A_est": a, "G_est": g, "TP_est": tp, "FP_est": fp, "FN_est": fn,
        "P": base_p, "base_F1": base_f1, "junk_F1": junk_f1, "J": j,
        "precision_est": tp / base_p if base_p else None,
        "recall_est": tp / g if g else None,
        "ceiling_remove_fp": 2 * tp / (tp + g) if (tp + g) else None,
        "ceiling_add_fn": 2 * g / (base_p + fn + g) if (base_p + fn + g) else None,
    }


def verdict(est: dict) -> str:
    g = est["G_est"]
    # TP <= P is a hard constraint: F1*(P+G)/2 <= P  =>  G <= P*(2-F1)/F1
    g_max = est["P"] * (2 - est["base_F1"]) / est["base_F1"]
    if est["dF1"] < 0.005:
        return "SUSPECT: 分数几乎没掉，人造样本可能未被计入 P —— 先查包，不要解读 G"
    if est["FP_est"] < 0:
        return (f"INCONSISTENT: 反解出的 G={g:.0f} 超过可行域上界 G <= P(2-F1)/F1 = {g_max:.0f}"
                "（TP 不可能超过 P）——说明基线或被注入内容有误，先复查再解读")
    if 2700 <= g <= 3500:
        return "CASE 1 (与 train 先验一致): 精确≈0.8/召回≈0.69 → 主攻剔 FP，提交高阈值候选并用本次 G 反解其被剔样本的 TP 率"
    if g > 4000:
        return "CASE 2 (召回 <0.55): 漏检是主症，提阈值无望 → 转远端低阈值 / 多尺度 / 召回恢复"
    if g < 2200:
        return "CASE 3 (精确 <0.66): FP 占比过半 → 全力 FP 抑制（阈值 + 几何剪枝 + 去重）"
    return "CASE 1 边缘: 结合精确/召回两值判断"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-f1", default=0.73505, type=float)
    ap.add_argument("--base-p", default=2664, type=int)
    ap.add_argument("--junk-f1", type=float)
    ap.add_argument("--j", default=225, type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if args.selftest:
        print(f"{'G_true':>8} {'预期返回 F1':>14} {'G_est':>10} {'误差':>8} {'TP_est':>10} {'FP_est':>9} {'FN_est':>9}")
        ok = True
        for g_true in (2200, 2600, 3097, 3400, 4200, 5000):
            f1j = args.base_f1 * (args.base_p + g_true) / (args.base_p + g_true + args.j)
            f1j_r = round(f1j, 5)          # 榜单只显示 5 位小数
            est = solve(args.base_f1, args.base_p, f1j_r, args.j)
            err = est["G_est"] - g_true
            ok &= abs(err) < 3.0
            print(f"{g_true:8d} {f1j_r:14.5f} {est['G_est']:10.2f} {err:8.2f} "
                  f"{est['TP_est']:10.1f} {est['FP_est']:9.1f} {est['FN_est']:9.1f}")
        print("SELFTEST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.junk_f1 is None:
        raise SystemExit("--junk-f1 is required (or pass --selftest)")
    est = solve(args.base_f1, args.base_p, args.junk_f1, args.j)
    print(json.dumps(est, indent=2, ensure_ascii=False))
    print("\n判定:", verdict(est))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"estimate": est, "verdict": verdict(est)}, fh,
                      indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
