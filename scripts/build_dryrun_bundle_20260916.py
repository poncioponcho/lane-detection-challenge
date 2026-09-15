#!/usr/bin/env python3
"""Build a fake "testB" bundle out of testA trees so the B-board candidate
pipeline can be exercised end to end before the real data lands.

Why: `build_testB_candidates.sh` had never been run against real data. A bug
found at 16:00 on 9/17 would cost the whole board; a bug found now costs
nothing.

SAFETY: everything this script and the pipeline produce is testA content.
Run `bash scripts/build_testB_candidates.sh <bundle>` and then IMMEDIATELY
quarantine the results (see docs/action_testB_20260916.md). Never submit
anything produced from this bundle.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/dryrun_bundle_20260916")

TREES = {
    "base54": ROOT / "outputs/testA_54ep_raw/testA/predictions",
    "seed42": ROOT / "outputs/testA_night_20260915/testA_36ep_s42/testA/predictions",
    "seed101": ROOT / "outputs/testA_support_trees/seed101_36ep",
    "seed202": ROOT / "outputs/testA_support_trees/seed202_36ep",
    "seed303": ROOT / "outputs/testA_support_trees/seed303_36ep",
    "clrernet36": ROOT / "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions",
    "hires": ROOT / "outputs/testA_hires/testA_hires_conf0.40/testA/predictions",
    "swa4": ROOT / "outputs/testA_night_20260915/pred_swa4_54ep",
    "occlude": ROOT / "outputs/testA_night_20260915/testA_occlude50/testA/predictions",
}
ZIP_TREES = {"base54_c55": ROOT / "outputs/submit_testA_night_conf55_m0.zip"}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data_processed").mkdir(parents=True)

    missing = []
    for tag, src in TREES.items():
        dst = OUT / f"testB_{tag}/testB/predictions"
        if not src.is_dir():
            missing.append(str(src))
            continue
        n = 0
        for f in src.rglob("*.lines.txt"):
            rel = f.relative_to(src)
            p = dst / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, p)
            n += 1
        print(f"  testB_{tag:12s} <- {src.name} ({n} files)")

    for tag, zp in ZIP_TREES.items():
        dst = OUT / f"testB_{tag}/testB/predictions"
        n = 0
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                if not nm.endswith(".lines.txt"):
                    continue
                rel = nm.split("submit/", 1)[-1]
                p = dst / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(zf.read(nm))
                n += 1
        print(f"  testB_{tag:12s} <- {zp.name} ({n} files)")

    if missing:
        print("MISSING (skipped):", missing, file=sys.stderr)

    # manifest + official list, straight from the testA manifest
    recs = [json.loads(l) for l in
            (ROOT / "data/processed/manifest_testA.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    (OUT / "data_processed/manifest_testB.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    (OUT / "data_processed/manifest_testB.list.txt").write_text(
        "".join("/" + r["image_path"] + "\n" for r in recs), encoding="utf-8")
    print(f"  manifest rows={len(recs)} (real testB expects >=1000)")

    tgz = Path("/tmp/dryrun_bundle_20260916.tgz")
    import subprocess
    subprocess.run(["tar", "czf", str(tgz), "-C", "/tmp", "dryrun_bundle_20260916"], check=True)
    print(f"\nbundle: {tgz}  ({tgz.stat().st_size} bytes)")
    print("REMINDER: contents are testA. Quarantine every product; never submit.")


if __name__ == "__main__":
    main()
