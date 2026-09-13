#!/usr/bin/env python3
"""Does consensus gating lift a union's marginal true-rate above F1/2?

Why this exists
---------------
`exp_union_true_rate_20260913.py` measured a single-support union on the honest
OOF: 789 added lanes, 304.0 true, marginal rate 0.3853 against a break-even of
0.3888 -- exactly neutral. That support model (15ep, cls_loss_weight=3.0) is
highly correlated with the reference, which is the worst case for a union.

The obvious next idea -- only add a lane that TWO independent support models
agree on -- could not be measured then because only one honest OOF tree was
available. There are in fact two: the clsweight3 OOF (71 clips) and the os_v4
OOF (63 clips, leave-one-video-out per its manifest_build.json). On the 63-clip
intersection both are honest and video-disjoint, so the gated variant can be
scored against ground truth.

Variants scored on the same 6300-image subset:
  ref        baseline 15ep OOF
  unionA     ref + lanes unique to clsweight3
  unionB     ref + lanes unique to os_v4
  consensus  ref + lanes unique to clsweight3 AND confirmed by os_v4

Marginal true-rate is recovered from TP = F1 * (P + G) / 2, with G counted
directly from the GT tree for the subset.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from build_testA_variants_20260914 import (  # noqa: E402
    IOU_THRESHOLD,
    load_oracle,
    parse_line,
)
from exp_bottom_trim_20260913 import OFFICIAL_PY  # noqa: E402

REF = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
SUP_A = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_clsweight3_20260907/oof/predictions"
SUP_B = PROJECT_ROOT / "outputs/lvo_os_v4_15ep_20260909/oof/predictions"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
WORK = PROJECT_ROOT / "outputs/exp_union_consensus_20260913"

ORACLE = load_oracle()
_CACHE: dict[tuple, "np.ndarray"] = {}


def mask_of(points):
    key = tuple(points)
    hit = _CACHE.get(key)
    if hit is None:
        interp = ORACLE.interp_lane(list(points))
        hit = ORACLE.draw_lane_mask(interp, ORACLE.LINE_WIDTH)
        _CACHE[key] = hit
    return hit


def _read(path: Path) -> list:
    if not path.is_file():
        return []
    return [parse_line(t) for t in path.read_text(encoding="utf-8").splitlines()
            if t.strip()]


def _matches(a, b) -> bool:
    ma, mb = mask_of(a), mask_of(b)
    union = int(np.logical_or(ma, mb).sum())
    if union == 0:
        return False
    return int(np.logical_and(ma, mb).sum()) / union > IOU_THRESHOLD


def count_gt(clips: set[str]) -> int:
    total = 0
    for f in (GT_DIR).rglob("*.lines.txt"):
        if f.parent.name in clips:
            total += len([l for l in f.read_text(encoding="utf-8").splitlines()
                          if l.strip()])
    return total


def build(dst: Path, rels: list[Path], mode: str) -> int:
    """mode: ref | unionA | unionB | consensus"""
    added = 0
    for rel in rels:
        ref_lines = _read(REF / rel)
        kept = list(ref_lines)
        kept_masks = [mask_of(p) for p in kept]
        if mode in {"unionA", "consensus"}:
            b_novel = []
            if mode == "consensus":
                for cand in _read(SUP_B / rel):
                    if not any(_matches(cand, k) for k in kept):
                        b_novel.append(cand)
            for cand in _read(SUP_A / rel):
                if any(_matches(cand, k) for k in kept):
                    continue
                if mode == "consensus" and not any(_matches(cand, b) for b in b_novel):
                    continue
                kept.append(cand)
                kept_masks.append(mask_of(cand))
                added += 1
        elif mode == "unionB":
            for cand in _read(SUP_B / rel):
                if any(_matches(cand, k) for k in kept):
                    continue
                kept.append(cand)
                kept_masks.append(mask_of(cand))
                added += 1
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(" ".join(f"{x:.5f} {y:.5f}" for x, y in p) for p in kept)
            + ("\n" if kept else ""), encoding="utf-8")
    return added


def main() -> None:
    all_records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    b_clips = {p.parent.name for p in SUP_B.rglob("*.lines.txt")}
    records = [replace(r, order=i)
               for i, r in enumerate(r for r in all_records if r.clip_id in b_clips)]
    print(f"subset clips={len(b_clips)} images={len(records)}")
    G = count_gt(b_clips)
    print(f"GT lanes in subset: {G}")

    rels = [Path(r.pred_rel_path) for r in records]
    P_REF = sum(len(_read(REF / rel)) for rel in rels)
    print(f"ref lanes: {P_REF}")

    WORK.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()
    g0 = run_official_eval(REF, GT_DIR, records, official_python=OFFICIAL_PY,
                           per_clip=False).to_dict()["global"]
    print(f"ref      : TP={g0['tp']} FP={g0['fp']} FN={g0['fn']} "
          f"F1={g0['f1']:.6f} ({time.time()-t0:.0f}s)", flush=True)
    results["ref"] = {"f1": g0["f1"], "p": P_REF, "added": 0}

    tp_ref = g0["f1"] * (P_REF + G) / 2
    need = g0["f1"] / 2
    print(f"break-even marginal rate = {need:.4f}\n")

    for mode in ("unionA", "unionB", "consensus"):
        dst = WORK / mode
        added = build(dst, rels, mode)
        t0 = time.time()
        g = run_official_eval(dst, GT_DIR, records, official_python=OFFICIAL_PY,
                              per_clip=False).to_dict()["global"]
        tp = g["f1"] * (P_REF + added + G) / 2
        t = tp - tp_ref
        rate = t / added if added else float("nan")
        print(f"{mode:>10}: added={added:5d} true={t:7.1f} rate={rate:.4f} "
              f"F1={g['f1']:.6f} dF1={(g['f1']-g0['f1'])*100:+.3f}pp "
              f"({time.time()-t0:.0f}s)", flush=True)
        results[mode] = {"f1": g["f1"], "p": P_REF + added, "added": added,
                         "true": t, "rate": rate, "dF1_pp": (g["f1"]-g0["f1"])*100}

    print("\n=== verdict ===")
    for mode in ("unionA", "unionB", "consensus"):
        r = results[mode]
        flag = "WORTH ADDING" if r["rate"] > need else "not worth adding"
        print(f"{mode:>10}: rate={r['rate']:.4f} vs {need:.4f} -> {flag}")
    (WORK / "result.json").write_text(json.dumps(
        {"G": G, "P_ref": P_REF, "break_even": need, "results": results}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
