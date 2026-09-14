#!/usr/bin/env python3
"""Calibrate the consensus gate: agreement count -> marginal true-rate.

Why this exists
---------------
The 9/14 A-board shot used a consensus gate of ">=2 of 10 support trees" and
scored 0.73429 (incumbent 0.73574): the 164 added lanes came in at a 0.3414
marginal true-rate against a 0.3675 break-even. The local measurement that
justified the construction had required BOTH of TWO supports -- a 100%
agreement condition -- and measured 0.5688. So the controlling variable is the
AGREEMENT FRACTION, not the vote count, and the gate was set far too loose.

Six independent, honest OOF trees exist locally (all leave-one-video-out, each
video predicted only by the fold that held it out):

  ref       lvo_clrnet_r50_15ep_20260904          (7100)
  support1  lvo_clrnet_r50_15ep_clsweight3        (7100)
  support2  lvo_os_v4_15ep                        (6300, 63-clip split)
  support3  lvo_clrnet_r50_36ep_c_export /midpoint(7100)
  support4  lvo_clrnet_r50_36ep_c_export /final   (7100)
  support5  riskon_res960x384_lvo                 (7100, different resolution)

That is enough to measure the whole curve. This script, in ONE pass over the
63-clip intersection (6300 images, so every tree is available), writes one
union tree per gate k >= 1..5, scores each with the frozen Oracle, and
differences consecutive gates to recover the marginal true-rate of each
agreement band. The band where the rate crosses F1/2 is the gate to use.

Masks are cached per image only; a global cache would hold ~10^5 full-frame
boolean masks and blow up.
"""
from __future__ import annotations

import argparse
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
# Frozen order (dedup is order sensitive; the first tree owns the geometry).
SUPPORTS = [
    ("clsweight3", "outputs/lvo_clrnet_r50_15ep_clsweight3_20260907/oof/predictions"),
    ("os_v4", "outputs/lvo_os_v4_15ep_20260909/oof/predictions"),
    ("36ep_mid", "outputs/lvo_clrnet_r50_36ep_c_export_20260906/midpoint/predictions"),
    ("36ep_final", "outputs/lvo_clrnet_r50_36ep_c_export_20260906/final/predictions"),
    ("res960", "outputs/riskon_res960x384_lvo_20260907/oof/predictions"),
]
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
# rec.gt_path already carries the "anno_txt/" prefix, so the GT root is the
# lane root, not GT_DIR itself.
LANE_ROOT = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane"
WORK = PROJECT_ROOT / "outputs/exp_consensus_gate_calibration_20260914"

ORACLE = load_oracle()


def make_mask_of():
    cache: dict[tuple, "np.ndarray"] = {}

    def mask_of(points):
        key = tuple(points)
        hit = cache.get(key)
        if hit is None:
            interp = ORACLE.interp_lane(list(points))
            hit = ORACLE.draw_lane_mask(interp, ORACLE.LINE_WIDTH)
            cache[key] = hit
        return hit

    return mask_of, cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    all_records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    clip_sets = [{p.parent.name for p in (PROJECT_ROOT / p).rglob("*.lines.txt")}
                 for _, p in SUPPORTS]
    clips = set.intersection(*clip_sets)
    records = [replace(r, order=i)
               for i, r in enumerate(r for r in all_records if r.clip_id in clips)]
    if args.max_images:
        records = records[: args.max_images]
    print(f"clips={len(clips)} images={len(records)} supports={len(SUPPORTS)}", flush=True)

    # per-tree file cache keyed by (tree, rel); lanes are small, unlike masks
    file_cache: dict[tuple[str, str], list] = {}

    def read_tree(tag: str, root: Path, rel: Path) -> list:
        key = (tag, str(rel))
        hit = file_cache.get(key)
        if hit is None:
            f = root / rel
            hit = [parse_line(t) for t in f.read_text(encoding="utf-8").splitlines()
                   if t.strip()] if f.is_file() else []
            file_cache[key] = hit
        return hit

    gates = list(range(1, len(SUPPORTS) + 1))
    dsts = {k: WORK / f"k{k}" for k in gates}
    for d in dsts.values():
        d.mkdir(parents=True, exist_ok=True)

    counts_hist: dict[int, int] = {}
    t0 = time.time()
    for idx, rec in enumerate(records):
        rel = Path(rec.pred_rel_path)
        mask_of, cache = make_mask_of()
        base = read_tree("ref", REF, rel)
        base_masks = [mask_of(p) for p in base]

        def novel_of(root: Path, tag: str) -> list:
            out = []
            for cand in read_tree(tag, root, rel):
                cm = mask_of(cand)
                dup = False
                for bm in base_masks:
                    u = int(np.logical_or(cm, bm).sum())
                    if u and int(np.logical_and(cm, bm).sum()) / u > IOU_THRESHOLD:
                        dup = True
                        break
                if not dup:
                    out.append(cand)
            return out

        novel = [novel_of(PROJECT_ROOT / p, name) for name, p in SUPPORTS]

        def agrees(a, b) -> bool:
            ma, mb = mask_of(a), mask_of(b)
            u = int(np.logical_or(ma, mb).sum())
            return bool(u) and int(np.logical_and(ma, mb).sum()) / u > IOU_THRESHOLD

        # one representative per distinct novel lane, with a support count
        chosen: list[tuple[list, int]] = []
        for ti, cands in enumerate(novel):
            for cand in cands:
                if any(agrees(cand, c) for c, _ in chosen):
                    continue
                k = 1 + sum(1 for tj, other in enumerate(novel) if tj != ti
                            and any(agrees(cand, o) for o in other))
                chosen.append((cand, k))

        for c, k in chosen:
            counts_hist[k] = counts_hist.get(k, 0) + 1

        for gate, dst in dsts.items():
            keep = list(base) + [c for c, k in chosen if k >= gate]
            text = "\n".join(" ".join(f"{x:.5f} {y:.5f}" for x, y in lane)
                             for lane in keep)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n" if text else "", encoding="utf-8")

        if idx % 500 == 0:
            print(f"  {idx}/{len(records)}  {time.time()-t0:.0f}s", flush=True)

    print(f"pass done in {time.time()-t0:.0f}s; support-count histogram:", flush=True)
    print("  " + str(dict(sorted(counts_hist.items()))), flush=True)

    # G must be counted over the EVALUATED images, not over all clips of the
    # subset -- otherwise --max-images produces a mismatch and every derived
    # quantity (tp, band rate) is garbage.
    G = 0
    for rec in records:
        gt = LANE_ROOT / rec.gt_path if rec.gt_path else None
        if gt and gt.is_file():
            G += len([l for l in gt.read_text(encoding="utf-8").splitlines() if l.strip()])
    P_ref = sum(len(read_tree("ref", REF, Path(r.pred_rel_path))) for r in records)
    print(f"G={G} P_ref={P_ref}", flush=True)

    results = {"G": G, "P_ref": P_ref, "supports": [n for n, _ in SUPPORTS],
               "gates": {}, "counts_hist": {str(k): v for k, v in counts_hist.items()}}
    g_ref = run_official_eval(REF, GT_DIR, records, official_python=OFFICIAL_PY,
                              per_clip=False).to_dict()["global"]
    print(f"ref: F1={g_ref['f1']:.6f} P={P_ref} TP={g_ref['tp']}", flush=True)
    results["ref"] = {"f1": g_ref["f1"], "P": P_ref, "tp": g_ref["tp"]}

    scored: dict[int, dict] = {}
    for gate in gates:
        g = run_official_eval(dsts[gate], GT_DIR, records, official_python=OFFICIAL_PY,
                              per_clip=False).to_dict()["global"]
        P = g["tp"] + g["fp"]
        scored[gate] = {"f1": g["f1"], "P": P, "tp": g["tp"]}
        print(f"k>={gate}: P={P:6d} TP={g['tp']:6d} F1={g['f1']:.6f} "
              f"dF1={(g['f1']-g_ref['f1'])*100:+.3f}pp", flush=True)

    # The k>=j tree is a SUBSET of the k>=(j-1) tree, so the lanes carrying
    # exactly j votes are the difference between consecutive gates, and the
    # strictly-lower gate must be subtracted from the current one.
    print("\n=== marginal true-rate by agreement band ===", flush=True)
    print(f"break-even = F1/2 = {g_ref['f1']/2:.4f}\n", flush=True)
    for gate in gates:
        if gate < len(gates):
            n = scored[gate]["P"] - scored[gate + 1]["P"]
            t = scored[gate]["tp"] - scored[gate + 1]["tp"]
        else:
            n = scored[gate]["P"] - P_ref
            t = scored[gate]["tp"] - g_ref["tp"]
        rate = t / n if n else float("nan")
        frac = gate / len(SUPPORTS)
        print(f"exactly {gate}/{len(SUPPORTS)} votes ({frac:.0%} agreement): "
              f"n={n:5d} true={t:4d} rate={rate:.4f} "
              f"{'CLEARS' if rate > g_ref['f1']/2 else 'below'}", flush=True)
        scored[gate]["band"] = {"n": n, "true": t, "rate": rate}

    results["gates"] = {str(k): v for k, v in scored.items()}
    results["break_even"] = g_ref["f1"] / 2
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "result.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
