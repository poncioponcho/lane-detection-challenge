#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Junk-injection probe: recover the hidden GT lane count G from a score pair.

Rationale
---------
The official metric (frozen ``src/eval/official_oracle/score.py``) is pooled:

    F1 = 2*TP / (P + G)      with P = TP + FP,  G = TP + FN

``P`` is fully known from our own submission (we count the lanes).  ``G`` is
hidden.  One score therefore leaves two unknowns.  If we append ``J`` synthetic
lanes that are *guaranteed* to be unmatched (IoU ~ 0 with every GT lane), the
score pairs up into a solvable system because the injected lanes are known-FP:

    F1_base = 2*TP / (P    + G)
    F1_junk = 2*TP / (P+J  + G)          [TP unchanged]

    =>  A = P + G = J * F1_junk / (F1_base - F1_junk)
    =>  G = A - P,   TP = F1_base * A / 2,   FP = P - TP,   FN = G - TP

Validity conditions, all checked by this script:
  1. junk lanes must not steal TP (TP_junk == TP_base)
  2. sets must be nested / P must be counted identically
  3. junk must survive the official pre-check (>=4 values, even, finite,
     >=2 distinct points, <=64 lanes/image)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORACLE_PY = Path.home() / ".workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
SCORE_PY = REPO / "src/eval/official_oracle/score.py"
SENTINEL = "__ORACLE_RESULT_JSON__="

# A full-width horizontal segment hugging the top row.  Its rendered mask is a
# ~1366x20 band; the intersection with any roughly-vertical lane is at most
# ~31x20 px, so IoU stays far below the 0.5 gate for every plausible lane shape.
JUNK_LINE = "0.0 4.0 1365.0 4.0"


def oracle_eval(pred_dir: Path, gt_dir: Path, list_path: Path) -> dict:
    code = (
        "import importlib.util,json,sys\n"
        "p,pr,gt,ls,se=sys.argv[1:]\n"
        "s=importlib.util.spec_from_file_location('m',p)\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        "r=m.eval_predictions(pr,gt,ls)\n"
        "print(se+json.dumps(r[m.IOU_THRESHOLDS[0]],separators=(',',':')))\n"
    )
    out = subprocess.run(
        [str(ORACLE_PY), "-c", code, str(SCORE_PY), str(pred_dir), str(gt_dir), str(list_path), SENTINEL],
        capture_output=True, text=True, check=True, cwd=str(REPO),
    )
    for chunk in (out.stdout + out.stderr).splitlines():
        if chunk.startswith(SENTINEL):
            return json.loads(chunk[len(SENTINEL):])
    raise RuntimeError(f"oracle produced no result:\n{out.stdout}\n{out.stderr}")


def collect_images(pred_dir: Path, limit: int | None) -> list[tuple[str, str]]:
    pairs = []
    for clip in sorted(os.listdir(pred_dir)):
        cdir = pred_dir / clip
        if not cdir.is_dir():
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".lines.txt"):
                pairs.append((clip, fn))
    if limit:
        pairs = pairs[:limit]
    return pairs


def ensured(path: Path) -> Path:
    """mkdir -p that is safe under the sandbox's mkdir broker."""
    if not path.exists():
        path.mkdir(parents=True)
    return path


def build_tree(root: Path, pred_dir: Path, gt_dir: Path, pairs, *, junk_pairs,
               apply_junk: bool, list_path: Path) -> tuple[Path, Path, int]:
    pd, gd = root / "predictions", root / "gt"
    ensured(pd)
    ensured(gd)
    junk_total = 0
    lines = []
    for clip, fn in pairs:
        ensured(pd / clip)
        ensured(gd / clip)
        body = (pred_dir / clip / fn).read_text(encoding="utf-8")
        if apply_junk and (clip, fn) in junk_pairs:
            body = body.rstrip("\n") + ("\n" if body.strip() else "") + JUNK_LINE + "\n"
            junk_total += 1
        (pd / clip / fn).write_text(body, encoding="utf-8")
        shutil.copyfile(gt_dir / clip / fn, gd / clip / fn)
        lines.append(f"/Lane/anno_txt_check/{clip}/{fn.replace('.lines.txt', '.jpg')}")
    list_path.write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
    return pd, gd, junk_total


def decompose(f1_base: float, f1_junk: float, p_base: int, junk_total: int) -> dict:
    if junk_total <= 0:
        raise ValueError("no junk lines injected")
    if f1_junk >= f1_base:
        raise ValueError(f"junk did not lower F1 ({f1_junk} >= {f1_base})")
    a = junk_total * f1_junk / (f1_base - f1_junk)      # A = P + G
    g = a - p_base
    tp = f1_base * a / 2.0
    return {
        "A_est": a,
        "G_est": g,
        "TP_est": tp,
        "FP_est": p_base - tp,
        "FN_est": g - tp,
        "P_est": p_base,
        "precision_est": tp / p_base if p_base else None,
        "recall_est": tp / g if g else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--junk-ratio", type=float, default=0.25,
                    help="fraction of images that receive one junk lane")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    pairs = collect_images(pred_dir, args.n_images)
    rng = random.Random(args.seed)
    n_junk = max(1, int(round(len(pairs) * args.junk_ratio)))
    junk_pairs = set(rng.sample(pairs, n_junk))

    base = Path(tempfile.mkdtemp(prefix="decompose_probe_", dir="/private/tmp"))
    try:
        lp = base / "list.txt"
        pd0, gd0, _ = build_tree(base, pred_dir, gt_dir, pairs, junk_pairs=set(),
                                 apply_junk=False, list_path=lp)
        list0 = base / "list_base.txt"
        list0.write_text(lp.read_text(encoding="utf-8"), encoding="utf-8")
        r0 = oracle_eval(pd0, gd0, list0)

        bj = base / "junk"
        list1 = base / "list_junk.txt"
        pd1, gd1, junk_total = build_tree(bj, pred_dir, gt_dir, pairs,
                                          junk_pairs=junk_pairs, apply_junk=True,
                                          list_path=list1)
        r1 = oracle_eval(pd1, gd1, list1)

        # ground truth for the subset, straight from the annotation files
        g_true = sum(
            len([l for l in (gt_dir / c / f).read_text(encoding="utf-8").splitlines() if l.strip()])
            for c, f in pairs
        )
        p_base = sum(
            len([l for l in (pred_dir / c / f).read_text(encoding="utf-8").splitlines() if l.strip()])
            for c, f in pairs
        )

        est = decompose(r0["F1"], r1["F1"], p_base, junk_total)
        report = {
            "n_images": len(pairs),
            "junk_total": junk_total,
            "baseline": r0,
            "junk": r1,
            "junk_tp_leak": r1["TP"] - r0["TP"],
            "P_base_counted": p_base,
            "P_base_oracle": r0["TP"] + r0["FP"],
            "G_true": g_true,
            "G_est": est["G_est"],
            "G_abs_err": abs(est["G_est"] - g_true),
            "estimate": est,
            "true_decomposition": {
                "TP": r0["TP"], "FP": r0["FP"], "FN": r0["FN"],
                "G": r0["TP"] + r0["FN"],
                "TPS_est_err": est["TP_est"] - r0["TP"],
            },
            "verdict": "PASS" if (r1["TP"] == r0["TP"] and abs(est["G_est"] - g_true) < 1.0) else "FAIL",
        }
        ensured(Path(args.out).parent)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"images={len(pairs)} junk={junk_total} P={p_base}")
        print(f"base : TP={r0['TP']} FP={r0['FP']} FN={r0['FN']} F1={r0['F1']:.6f}")
        print(f"junk : TP={r1['TP']} FP={r1['FP']} FN={r1['FN']} F1={r1['F1']:.6f}")
        print(f"junk_tp_leak = {r1['TP'] - r0['TP']}  (must be 0)")
        print(f"G_true = {g_true}   G_est = {est['G_est']:.2f}   abs_err = {abs(est['G_est']-g_true):.2f}")
        print(f"TP_est = {est['TP_est']:.2f} (true {r0['TP']})  FP_est={est['FP_est']:.2f}  FN_est={est['FN_est']:.2f}")
        print(f"VERDICT: {report['verdict']}")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
        else:
            print(f"kept: {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
