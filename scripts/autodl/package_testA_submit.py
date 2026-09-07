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

import numpy as np

from run_training import required_absolute_env
from validate_run import manifest_prediction_paths


def _bootstrap_src_imports(project_root: Path) -> None:
    """verify_submit imports common/eval/submit as top-level packages from src/."""
    src_root = project_root / "src"
    for path in (str(project_root), str(src_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def recontract_predictions(prediction_root: Path, expected: set[str],
                           staged_root: Path) -> dict:
    """Rewrite predictions into the submission contract format (1 decimal).

    The diagnostic evaluator exports 5-decimal coordinates for the F1 replay;
    the official submission contract is strictly 1-decimal
    (verify_submit._ONE_DECIMAL). Rounding happens HERE, in the packaging layer,
    so the replay path and its byte-level F1 guard stay untouched — the boxed
    caveat being that the server scores the rounded geometry, so the 5-decimal
    diagnostic F1 is an upper-bound-style estimate, not the submission score.

    Each lane goes through export_lines.lane_to_line: 1-decimal tokens, bounds
    + non-normalized assertions, and consecutive-duplicate removal AFTER
    formatting. Lanes that collapse below 2 points post-rounding are dropped
    (a lane the official parser would reject anyway) and counted.
    """
    from submit.export_lines import lane_to_line

    staged_root.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "lanes": 0, "dropped_lanes": 0, "empty_files": 0}
    for rel in sorted(expected):
        source = prediction_root / rel
        target = staged_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if source.is_file():
            for line in source.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                if len(tokens) % 2 or len(tokens) < 4:
                    raise ValueError(f"{rel}: malformed lane ({len(tokens)} tokens)")
                points = np.asarray(tokens, dtype=np.float64).reshape(-1, 2)
                try:
                    lines.append(lane_to_line(points))
                    stats["lanes"] += 1
                except ValueError:
                    stats["dropped_lanes"] += 1
        target.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["files"] += 1
        if not lines:
            stats["empty_files"] += 1
    return stats


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

    staged_root = output_zip.parent / f"{args.split}_submit_fmt"
    fmt_stats = recontract_predictions(prediction_root, expected, staged_root)

    pack_submit(staged_root, expected, output_zip)
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
        "recontract": fmt_stats,
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
