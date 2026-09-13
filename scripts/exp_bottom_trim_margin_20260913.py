#!/usr/bin/env python3
"""Bottom-trim margin sweep on the honest 15ep LVO OOF (follow-up).

`exp_bottom_trim_20260913.py` found gtcond+0 = +1.955pp and gtcond+40 =
+4.224pp. The margin is monotone so far, and margin -> infinity collapses to
the untrimmed baseline (0.777628), so the optimum is interior. This sweep
scans 60/80/100/140 to locate it before the rule is frozen for the B board.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from exp_bottom_trim_20260913 import build_variant, OFFICIAL_PY  # noqa: E402


def main() -> None:
    records = read_manifest(PROJECT_ROOT / "data/processed/manifest_train.jsonl")
    src = PROJECT_ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
    gt = PROJECT_ROOT / "outputs/a_gt_train_20260905/Lane/anno_txt"
    work = PROJECT_ROOT / "outputs/exp_bottom_trim_margin_20260913"
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for margin in (60.0, 80.0, 100.0, 140.0):
        name = f"gtcond+{margin:.0f}"
        dst = work / name
        build_variant(src, dst, "gt_cond", margin)
        t0 = time.time()
        g = run_official_eval(
            dst, gt, records, official_python=OFFICIAL_PY, per_clip=False
        ).to_dict()["global"]
        print(f"{name}: TP={g['tp']} FP={g['fp']} FN={g['fn']} "
              f"F1={g['f1']:.6f} ({time.time()-t0:.0f}s)", flush=True)
        out.append({"variant": name, "tp": g["tp"], "fp": g["fp"],
                    "fn": g["fn"], "f1": g["f1"]})
    (work / "margin_results.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
