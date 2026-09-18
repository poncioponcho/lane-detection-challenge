#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Size the cross-model consensus filter on testA.

With G known (3155.8) the break-even rule for a *removal* candidate set is

    beneficial  <=>  (TP removed) / (lines removed)  <  TP/(P+G) = 0.3675

A line that no independent model reproduces is a plausible FP, so the
unsupported subset is the natural removal candidate.  This script only measures
its SIZE and composition; the TP rate can only be settled by an A-board probe.

No GT is used: this is pure inference on the unlabeled testA split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_oracle():
    spec = importlib.util.spec_from_file_location(
        "frozen_score", REPO / "src/eval/official_oracle/score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lines_from_text(text: str):
    out = []
    for raw in text.splitlines():
        toks = raw.split()
        if len(toks) < 4:
            continue
        out.append([(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks), 2)])
    return out


def load_zip(path: Path) -> dict[str, list]:
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            rel = name[len("submit/"):] if name.startswith("submit/") else name
            out[rel] = lines_from_text(z.read(name).decode("utf-8"))
    return out


def load_tree(root: Path) -> dict[str, list]:
    out = {}
    for clip in sorted(root.iterdir()):
        if not clip.is_dir():
            continue
        for f in sorted(clip.iterdir()):
            if f.name.endswith(".lines.txt"):
                out[f"{clip.name}/{f.name}"] = lines_from_text(f.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-zip", required=True)
    ap.add_argument("--support-dir", action="append", required=True)
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracle = load_oracle()
    ref = load_zip(Path(args.ref_zip))
    supports = [load_tree(Path(d)) for d in args.support_dir]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    total_ref = 0
    unsupported = {t: 0 for t in thresholds}
    per_set_only = {i: 0 for i in range(len(supports))}
    n_images = 0

    for rel, ref_lanes in ref.items():
        if not ref_lanes:
            continue
        n_images += 1
        total_ref += len(ref_lanes)
        ref_masks = [oracle.draw_lane_mask(oracle.interp_lane(l), oracle.LINE_WIDTH) for l in ref_lanes]
        sup_masks = []
        for s in supports:
            lanes = s.get(rel, [])
            sup_masks.append([oracle.draw_lane_mask(oracle.interp_lane(l), oracle.LINE_WIDTH) for l in lanes])

        for i, rm in enumerate(ref_masks):
            # best IoU achieved by each support set (not a running global best:
            # we need "how many independent models reproduce this line")
            best_per_set = []
            for masks in sup_masks:
                best = 0.0
                for sm in masks:
                    union = int((rm | sm).sum())
                    if not union:
                        continue
                    iou = float((rm & sm).sum()) / union
                    if iou > best:
                        best = iou
                best_per_set.append(best)
            for t in thresholds:
                if not any(b > t for b in best_per_set):
                    unsupported[t] += 1
            n_ok = sum(1 for b in best_per_set if b > 0.5)
            per_set_only[n_ok] = per_set_only.get(n_ok, 0) + 1

    report = {
        "ref_zip": args.ref_zip,
        "support_dirs": args.support_dir,
        "images": n_images,
        "ref_lanes": total_ref,
        "unsupported": {str(t): unsupported[t] for t in thresholds},
        "unsupported_pct": {str(t): 100.0 * unsupported[t] / total_ref for t in thresholds},
        "ref_lanes_by_supporter_count": {str(k): v for k, v in sorted(per_set_only.items())},
        "break_even_tp_rate": 0.3675,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"参考集: {total_ref} 条 / {n_images} 图")
    for t in thresholds:
        print(f"  无任何支持集在 IoU>{t} 命中: {unsupported[t]:5d} 条 ({100.0*unsupported[t]/total_ref:5.1f}%)")
    print("  按支持模型数分布:", {k: v for k, v in sorted(per_set_only.items())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
