#!/usr/bin/env python3
"""Measure the marginal true-rate of another model's unique lanes, honestly.

Context
-------
The testA decomposition (docs/a_board_decompose_probe_20260913.md §11) shows
the domain tax is a recall tax: TP 2138.9 / FP 525.1 / FN 1016.9, i.e. FN is
1.93x FP. Adding a lane helps iff its marginal precision exceeds F1/2
(0.3675 on testA, 0.3888 on the OOF). The conf axis was proved to add lanes
at only 23% precision, which is why lowering the gate loses.

The remaining recall shot is the *union*: keep our lanes and add lanes that
other models find and we do not (`submit_testA_union_trim0_20260914.zip`,
+271 lanes). Whether those 271 are worth it is the one thing an A-board
submission would tell us -- but it can be measured locally instead, because
the honest LVO OOF has ground truth and ships a second OOF tree
(15ep, cls_loss_weight=3.0) that can play the role of the support model.

Method: build the same union construction (rasterised mask IoU novelty test)
with the baseline OOF as reference and the clsweight3 OOF as support, score
with the frozen Oracle, and invert

    TP = F1 * (P + G) / 2        (G = 24435 for the 7100-image OOF)

to get the number of added lanes that turned out to be true.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from build_testA_variants_20260914 import (  # noqa: E402
    IMG_HEIGHT,
    IMG_WIDTH,
    IOU_THRESHOLD,
    load_oracle,
    parse_line,
)
from exp_bottom_trim_20260913 import OFFICIAL_PY  # noqa: E402

ORACLE = load_oracle()
_MASK_CACHE: dict[tuple, "np.ndarray"] = {}


def mask_of(points):
    """Byte-frozen official rasterisation (same call the builder uses)."""
    key = tuple(points)
    cached = _MASK_CACHE.get(key)
    if cached is None:
        interp = ORACLE.interp_lane(list(points))
        cached = ORACLE.draw_lane_mask(interp, ORACLE.LINE_WIDTH)
        _MASK_CACHE[key] = cached
    return cached

REF = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
SUPPORT = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_clsweight3_20260907/oof/predictions"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
WORK = PROJECT_ROOT / "outputs/exp_union_true_rate_20260913"
G = 24435
P_REF = 22248
F1_REF = 0.7776278302594092


def build_union(ref: Path, support: Path, dst: Path) -> int:
    if dst.exists():
        raise SystemExit(f"destination exists: {dst}")
    added = 0
    for ref_file in sorted(ref.rglob("*.lines.txt")):
        rel = ref_file.relative_to(ref)
        sup_file = support / rel
        out_file = dst / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        kept_pts = [parse_line(t) for t in ref_file.read_text(encoding="utf-8").splitlines()
                    if t.strip()]
        kept_masks = [mask_of(p) for p in kept_pts]
        if sup_file.is_file():
            for text in sup_file.read_text(encoding="utf-8").splitlines():
                text = text.strip()
                if not text:
                    continue
                cand = parse_line(text)
                cand_mask = mask_of(cand)
                dup = False
                for km in kept_masks:
                    union = int(np.logical_or(cand_mask, km).sum())
                    if union == 0:
                        continue
                    if int(np.logical_and(cand_mask, km).sum()) / union > IOU_THRESHOLD:
                        dup = True
                        break
                if dup:
                    continue
                kept_pts.append(cand)
                kept_masks.append(cand_mask)
                added += 1
        out_file.write_text(
            "\n".join(" ".join(f"{x:.5f} {y:.5f}" for x, y in p) for p in kept_pts)
            + ("\n" if kept_pts else ""),
            encoding="utf-8",
        )
    return added


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")

    t0 = time.time()
    g0 = run_official_eval(REF, GT_DIR, records, official_python=OFFICIAL_PY,
                           per_clip=False).to_dict()["global"]
    print(f"ref     : TP={g0['tp']} FP={g0['fp']} FN={g0['fn']} F1={g0['f1']:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    dst = WORK / "union"
    added = build_union(REF, SUPPORT, dst)
    print(f"union built: +{added} novel lanes -> P={P_REF + added}", flush=True)

    t0 = time.time()
    g1 = run_official_eval(dst, GT_DIR, records, official_python=OFFICIAL_PY,
                           per_clip=False).to_dict()["global"]
    print(f"union   : TP={g1['tp']} FP={g1['fp']} FN={g1['fn']} F1={g1['f1']:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    tp_ref = F1_REF * (P_REF + G) / 2
    tp_uni = g1["f1"] * (P_REF + added + G) / 2
    t = tp_uni - tp_ref
    rate = t / added if added else float("nan")
    need = tp_ref / (P_REF + G)
    print("\n=== marginal true-rate of the support model's unique lanes ===")
    print(f"added N          = {added}")
    print(f"true among them  = {t:.1f}")
    print(f"marginal rate    = {rate:.3f}")
    print(f"break-even rate  = {need:.3f}   (F1/2 on this scale)")
    print(f"verdict          = {'WORTH ADDING' if rate > need else 'NOT worth adding'}")
    print(f"dF1              = {(g1['f1'] - F1_REF) * 100:+.3f}pp")
    (WORK / "result.json").write_text(json.dumps({
        "added": added, "true_among_added": t, "marginal_rate": rate,
        "break_even": need, "f1_ref": F1_REF, "f1_union": g1["f1"],
        "iou_threshold": IOU_THRESHOLD,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
