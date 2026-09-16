#!/usr/bin/env python3
"""Score an arbitrary prediction tree with the frozen official Oracle.

The existing scanners can only score variants they build themselves (threshold
sweeps). Comparing hand-built trees -- a union, a trim variant, a consensus
output -- needed throwaway code each time, which is exactly how paste-edit
mistakes get into an evidence trail. This is the generic entry point.

All scores come from ONE global Oracle call plus per-video calls for the
paired bootstrap. Nothing here talks to the platform.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import read_manifest                      # noqa: E402
from eval.oracle_runner import run_official_eval             # noqa: E402
from scan_lvo_geometry import paired_bootstrap, video_id     # noqa: E402


def per_video(records, gt_dir, official_python, pred_root, oracle_root, name):
    grouped: dict[str, list] = {}
    for record in records:
        grouped.setdefault(video_id(record.clip_id), []).append(record)
    videos = []
    for video, original in sorted(grouped.items()):
        subset = [replace(r, order=i) for i, r in enumerate(original)]
        path = oracle_root / name / f"oracle_{video}.json"
        counts = run_official_eval(
            pred_root, gt_dir, subset, official_python=official_python,
            per_clip=False, output_path=path,
        ).to_dict()["global"]
        videos.append({
            "video": video,
            "tp": int(counts["tp"]), "fp": int(counts["fp"]), "fn": int(counts["fn"]),
            "precision": float(counts["precision"]), "recall": float(counts["recall"]),
            "f1": float(counts["f1"]),
        })
    return videos


def score_tree(name, pred_root, records, gt_dir, official_python, oracle_root):
    global_path = oracle_root / name / "oracle_global.json"
    result = run_official_eval(
        pred_root, gt_dir, records, official_python=official_python,
        per_clip=False, output_path=global_path,
    ).to_dict()
    counts = result["global"]
    row = {
        "name": name,
        "path": str(pred_root),
        "global": counts,
        "lines": int(counts["tp"] + counts["fp"]),
    }
    row["videos"] = per_video(
        records, gt_dir, official_python, pred_root, oracle_root, name
    )
    if (
        sum(v["tp"] for v in row["videos"]),
        sum(v["fp"] for v in row["videos"]),
        sum(v["fn"] for v in row["videos"]),
    ) != (counts["tp"], counts["fp"], counts["fn"]):
        raise ValueError(f"{name}: global/video Oracle count mismatch")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--reference", type=Path,
                    help="tree to compare against (per-video bootstrap)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--official-python", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    oracle_root = args.output_dir / "oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)

    cand = score_tree(args.label, args.pred, records, args.gt_dir,
                      args.official_python, oracle_root)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "gt_dir": str(args.gt_dir.resolve()),
        "official_python": str(args.official_python.resolve()),
        "records": len(records),
        "candidate": cand,
    }
    if args.reference is not None:
        ref = score_tree("reference", args.reference, records, args.gt_dir,
                         args.official_python, oracle_root)
        cand["paired_bootstrap"] = paired_bootstrap(
            ref, cand, seed=args.seed, n_bootstrap=args.bootstrap
        )
        payload["reference"] = ref
        boot = cand["paired_bootstrap"]
        # observed_delta_pp and ci95_delta_pp are ALREADY in percentage points.
        print(f"{args.label}: lines={cand['lines']} F1={cand['global']['f1']:.6f} "
              f"({boot['observed_delta_pp']:+.3f}pp vs reference) "
              f"CI=[{boot['ci95_delta_pp'][0]:+.3f}, {boot['ci95_delta_pp'][1]:+.3f}]")
    else:
        print(f"{args.label}: lines={cand['lines']} F1={cand['global']['f1']:.6f}")

    out = args.output_dir / f"score_{args.label}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"json: {out}")


if __name__ == "__main__":
    main()
