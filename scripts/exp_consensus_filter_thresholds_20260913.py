#!/usr/bin/env python3
"""Pick the short-lane filter threshold by measurement, not by argument.

The testA main shot filters consensus-added lanes with span < 80px. That number
was reasoned from the IoU algebra (a stub contained in a GT lane scores at most
span_pred/span_GT) but never measured. It can be measured: the honest 63-clip
OOF already carries a built consensus-union tree, so filtering it at several
thresholds and scoring each with the frozen Oracle costs a few minutes.

Only base-unmatched lanes are filtered, so the reference geometry is untouched.
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

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from exp_bottom_trim_20260913 import OFFICIAL_PY  # noqa: E402
from filter_short_lanes import matches_base, parse_line  # noqa: E402

REF = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
SRC = PROJECT_ROOT / "outputs/exp_union_consensus_avg_20260913/avg"
GT_DIR = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
WORK = PROJECT_ROOT / "outputs/exp_consensus_filter_thresholds_20260913"
G = 21780


def build_filtered(dst: Path, rels: list[Path], min_span: float) -> tuple[int, int, int]:
    base_kept = added_kept = dropped = 0
    for rel in rels:
        bfile = REF / rel
        base_lanes = [parse_line(t) for t in bfile.read_text(encoding="utf-8").splitlines()
                      if t.strip()] if bfile.is_file() else []
        rows = []
        for line in (SRC / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lane = parse_line(line)
            ys = [p[1] for p in lane]
            if matches_base(lane, base_lanes, 30.0):
                rows.append(line.strip())
                base_kept += 1
            elif max(ys) - min(ys) >= min_span:
                rows.append(line.strip())
                added_kept += 1
            else:
                dropped += 1
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
    return base_kept, added_kept, dropped


def main() -> None:
    all_records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    b_clips = {p.parent.name for p in SRC.rglob("*.lines.txt")}
    records = [replace(r, order=i)
               for i, r in enumerate(r for r in all_records if r.clip_id in b_clips)]
    rels = [Path(r.pred_rel_path) for r in records]
    print(f"images={len(records)}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for min_span in (0.0, 60.0, 80.0, 100.0, 120.0):
        name = "unfiltered" if min_span == 0 else f"span{int(min_span)}"
        dst = WORK / name
        if min_span == 0:
            bk = ak = dr = -1
            pred_dir = SRC
        else:
            bk, ak, dr = build_filtered(dst, rels, min_span)
            pred_dir = dst
        t0 = time.time()
        g = run_official_eval(pred_dir, GT_DIR, records, official_python=OFFICIAL_PY,
                              per_clip=False).to_dict()["global"]
        print(f"{name:>10}: base={bk} added_kept={ak} dropped={dr} "
              f"F1={g['f1']:.6f} P={g['tp']+g['fp']} ({time.time()-t0:.0f}s)", flush=True)
        out[name] = {"min_span": min_span, "base_kept": bk, "added_kept": ak,
                     "dropped": dr, "f1": g["f1"], "P": g["tp"] + g["fp"]}
    (WORK / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
