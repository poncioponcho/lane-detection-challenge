#!/usr/bin/env python3
"""Does averaging the agreeing support lanes beat taking one representative?

The consensus-gated union (exp_union_consensus_20260913.py) adds a novel lane
when two independent support trees propose it, and keeps the first tree's
geometry. But both trees estimate the same lane with independent lateral
error, and the official TP gate is a lateral tolerance (~10.3px in original
coordinates). Averaging the two x(y) curves should cut that error by about
sqrt(2) and lift the marginal true-rate above the measured 0.5507.

This script scores, on the same honest 63-clip OOF subset:
  consensus      first representative (already measured: 0.5507)
  consensus_avg  pointwise mean of the two agreeing lanes over their common
                 y range (20 samples, linear interpolation in x)
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
WORK = PROJECT_ROOT / "outputs/exp_union_consensus_avg_20260913"

ORACLE = load_oracle()
_CACHE: dict[tuple, "np.ndarray"] = {}
SAMPLES = 20


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


def x_at(lane, y: float) -> float:
    pts = sorted(lane, key=lambda p: p[1])
    if y <= pts[0][1]:
        return pts[0][0]
    if y >= pts[-1][1]:
        return pts[-1][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= y <= y1:
            if y1 == y0:
                return x0
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return pts[-1][0]


def averaged(a, b) -> list | None:
    lo = max(min(p[1] for p in a), min(p[1] for p in b))
    hi = min(max(p[1] for p in a), max(p[1] for p in b))
    if hi - lo < 10.0:
        return None
    out = []
    for k in range(SAMPLES):
        y = lo + (hi - lo) * k / (SAMPLES - 1)
        out.append(((x_at(a, y) + x_at(b, y)) / 2.0, y))
    return out


def count_gt(clips: set[str]) -> int:
    return sum(len([l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
               for f in GT_DIR.rglob("*.lines.txt") if f.parent.name in clips)


def build(dst: Path, rels: list[Path], mode: str) -> int:
    added = 0
    for rel in rels:
        base = _read(REF / rel)
        kept = list(base)
        b_novel = [c for c in _read(SUP_B / rel)
                   if not any(_matches(c, k) for k in base)]
        for cand in _read(SUP_A / rel):
            if any(_matches(cand, k) for k in kept):
                continue
            match = next((b for b in b_novel if _matches(cand, b)), None)
            if match is None:
                continue
            lane = averaged(cand, match) if mode == "avg" else cand
            if lane is None:
                continue
            kept.append(lane)
            added += 1
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(" ".join(f"{x:.5f} {y:.5f}" for x, y in p) for p in kept)
            + "\n", encoding="utf-8")
    return added


def main() -> None:
    all_records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    b_clips = {p.parent.name for p in SUP_B.rglob("*.lines.txt")}
    records = [replace(r, order=i)
               for i, r in enumerate(r for r in all_records if r.clip_id in b_clips)]
    G = count_gt(b_clips)
    rels = [Path(r.pred_rel_path) for r in records]
    P_REF = sum(len(_read(REF / rel)) for rel in rels)
    print(f"subset images={len(records)} G={G} P_ref={P_REF}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    g0 = run_official_eval(REF, GT_DIR, records, official_python=OFFICIAL_PY,
                           per_clip=False).to_dict()["global"]
    print(f"ref: F1={g0['f1']:.6f}", flush=True)
    tp_ref = g0["f1"] * (P_REF + G) / 2
    need = g0["f1"] / 2
    print(f"break-even={need:.4f}", flush=True)

    out = {}
    for mode in ("rep", "avg"):
        dst = WORK / mode
        added = build(dst, rels, mode)
        t0 = time.time()
        g = run_official_eval(dst, GT_DIR, records, official_python=OFFICIAL_PY,
                              per_clip=False).to_dict()["global"]
        t = g["f1"] * (P_REF + added + G) / 2 - tp_ref
        rate = t / added if added else float("nan")
        print(f"{mode}: added={added} true={t:.1f} rate={rate:.4f} "
              f"F1={g['f1']:.6f} dF1={(g['f1']-g0['f1'])*100:+.3f}pp "
              f"({time.time()-t0:.0f}s)", flush=True)
        out[mode] = {"added": added, "true": t, "rate": rate,
                     "f1": g["f1"], "dF1_pp": (g["f1"] - g0["f1"]) * 100}
    (WORK / "result.json").write_text(json.dumps(
        {"break_even": need, "G": G, "P_ref": P_REF, "results": out}, indent=1),
        encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
