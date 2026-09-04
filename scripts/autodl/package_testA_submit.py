#!/usr/bin/env python3
"""Package unlabeled-split predictions into a verified submit.zip (A/B 榜).

Chain position: infer_testA.py produces predictions -> THIS script packs and
verifies -> upload. Uses the frozen contract modules so the checks here are the
same ones that gate the real submission:

  * src/submit/pack_submit.pack_submit: zip layout submit/<clip>/<frame>.lines.txt,
    driven by the manifest's pred_rel_path set (missing -> empty file, never skipped);
  * src/submit/verify_submit.verify_submit: single root, exact file set, per-line
    official geometry smoke, <=64 lanes/image, <=200 MB.

No F1 is computed here: unlabeled splits carry no ground truth (any score would
be fabricated; scoring belongs to the frozen Oracle or the competition server).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from run_training import required_absolute_env
from validate_run import manifest_prediction_paths


def _bootstrap_src_imports(project_root: Path) -> None:
    """verify_submit imports common/eval/submit as top-level packages from src/."""
    src_root = project_root / "src"
    for path in (str(project_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", default="testA")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/<split>_infer/<split>/predictions.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/<split>_infer/submit_<split>.zip.",
    )
    args = parser.parse_args()

    project_root = Path(required_absolute_env("HARDLANE_PROJECT_ROOT"))
    manifest = args.manifest or (
        project_root / "data/processed" / f"manifest_{args.split}.jsonl"
    )
    if not manifest.is_file():
        raise SystemExit(f"manifest missing: {manifest}")
    prediction_root = args.prediction_root or (
        args.run_dir / f"{args.split}_infer" / args.split / "predictions"
    )
    if not prediction_root.is_dir():
        raise SystemExit(f"prediction root missing: {prediction_root}")
    output_zip = args.output_zip or (
        args.run_dir / f"{args.split}_infer" / f"submit_{args.split}.zip"
    )

    expected = manifest_prediction_paths(manifest)
    if not expected:
        raise SystemExit(f"manifest declares no predictions: {manifest}")

    present = {
        path.relative_to(prediction_root).as_posix()
        for path in prediction_root.rglob("*.lines.txt")
        if path.is_file()
    }
    missing_on_disk = sorted(expected - present)
    extra_on_disk = sorted(present - expected)

    _bootstrap_src_imports(project_root)
    from submit.pack_submit import pack_submit
    from submit.verify_submit import verify_submit

    pack_submit(prediction_root, expected, output_zip)
    ok, report = verify_submit(
        output_zip, expected, report_path=output_zip.with_suffix(".verify.md")
    )

    digest = hashlib.sha256(output_zip.read_bytes()).hexdigest()
    payload = {
        "status": "pass" if ok else "fail",
        "split": args.split,
        "zip": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "sha256": digest,
        "expected_count": len(expected),
        "empty_files": sum(
            1
            for rel in expected
            if not (prediction_root / rel).is_file()
            or (prediction_root / rel).stat().st_size == 0
        ),
        "missing_on_disk": missing_on_disk,
        "extra_on_disk": extra_on_disk,
        "verify_report": report.strip().splitlines()[-1] if report else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    output_zip.with_suffix(".package_evidence.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not ok:
        raise SystemExit(f"submit.zip failed contract verification:\n{report}")


if __name__ == "__main__":
    main()
