"""Canonicalize raw model predictions, pack, verify, and optionally score.

UnLanedet diagnostic predictions intentionally retain five decimals. The
competition archive requires exactly one decimal, so every non-empty lane must
pass through :func:`submit.export_lines.lane_to_line` before packaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from data.manifest import read_manifest
    from eval.oracle_runner import run_official_eval
    from submit.export_lines import export_lines
    from submit.pack_submit import _normalize_rel, pack_submit, read_expected
    from submit.verify_submit import verify_submit
except ModuleNotFoundError:  # ``python src/submit/prepare_submit.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.manifest import read_manifest
    from eval.oracle_runner import run_official_eval
    from submit.export_lines import export_lines
    from submit.pack_submit import _normalize_rel, pack_submit, read_expected
    from submit.verify_submit import verify_submit


def read_raw_prediction(path: str | Path) -> list[np.ndarray]:
    """Read whitespace-separated finite coordinates at arbitrary precision."""
    lanes: list[np.ndarray] = []
    path = Path(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        tokens = raw.split()
        if len(tokens) < 4 or len(tokens) % 2:
            raise ValueError(
                f"{path}:L{line_number}: expected an even coordinate count >=4"
            )
        try:
            values = np.asarray(tokens, dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"{path}:L{line_number}: non-numeric token") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{path}:L{line_number}: NaN/Inf")
        lanes.append(values.reshape(-1, 2))
    return lanes


def expected_from_manifest(path: str | Path) -> list[str]:
    """Return the exact ordered prediction paths from a validated manifest."""
    return [record.pred_rel_path for record in read_manifest(path)]


def canonicalize_prediction_dir(
    raw_pred_dir: str | Path,
    canonical_dir: str | Path,
    expected: Iterable[str],
    *,
    missing_as_empty: bool = False,
) -> dict:
    """Rewrite an exact expected prediction set through the 1-decimal exporter."""
    raw_pred_dir = Path(raw_pred_dir)
    canonical_dir = Path(canonical_dir)
    if not raw_pred_dir.is_dir():
        raise FileNotFoundError(f"raw prediction directory does not exist: {raw_pred_dir}")

    rels = [_normalize_rel(value) for value in expected]
    if len(rels) != len(set(rels)):
        raise ValueError("expected prediction paths contain duplicates after normalization")
    actual = {
        path.relative_to(raw_pred_dir).as_posix()
        for path in raw_pred_dir.rglob("*.lines.txt")
        if path.is_file()
    }
    extra = sorted(actual.difference(rels))
    if extra:
        raise ValueError(f"raw prediction directory contains {len(extra)} extra files: {extra[:3]}")

    canonical_existing = {
        path.relative_to(canonical_dir).as_posix()
        for path in canonical_dir.rglob("*.lines.txt")
        if path.is_file()
    } if canonical_dir.is_dir() else set()
    stale = sorted(canonical_existing.difference(rels))
    if stale:
        raise ValueError(
            "canonical prediction directory contains "
            f"{len(stale)} stale files: {stale[:3]}"
        )

    total_lanes = 0
    total_points = 0
    empty_files = 0
    missing_files: list[str] = []
    for rel in rels:
        source = raw_pred_dir / rel
        destination = canonical_dir / rel
        if not source.is_file():
            if not missing_as_empty:
                raise FileNotFoundError(f"missing raw prediction: {rel}")
            missing_files.append(rel)
            lanes: list[np.ndarray] = []
        else:
            lanes = read_raw_prediction(source)
        try:
            export_lines(lanes, destination, ndigits=1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"failed to canonicalize {rel}: {exc}") from exc
        total_lanes += len(lanes)
        total_points += sum(len(lane) for lane in lanes)
        empty_files += not lanes

    return {
        "expected_files": len(rels),
        "source_files": len(actual),
        "empty_files": empty_files,
        "missing_as_empty": missing_files,
        "lanes": total_lanes,
        "points": total_points,
    }


def prepare_submission(
    raw_pred_dir: str | Path,
    canonical_dir: str | Path,
    expected: Iterable[str],
    out_zip: str | Path,
    *,
    report_path: str | Path | None = None,
    missing_as_empty: bool = False,
    manifest: str | Path | None = None,
    gt_dir: str | Path | None = None,
    official_python: str | Path | None = None,
    oracle_output: str | Path | None = None,
    enforce_official_env: bool = True,
) -> dict:
    """Run raw→canonical→zip→verify and optional frozen-Oracle evaluation."""
    expected = [_normalize_rel(value) for value in expected]
    conversion = canonicalize_prediction_dir(
        raw_pred_dir, canonical_dir, expected, missing_as_empty=missing_as_empty
    )
    archive = pack_submit(
        canonical_dir, expected, out_zip, missing_as_empty=False
    )
    ok, verify_report = verify_submit(archive, expected)
    if not ok:
        raise RuntimeError(f"canonical submission failed final verification:\n{verify_report}")

    oracle = None
    oracle_args = (manifest, gt_dir, official_python)
    if any(value is not None for value in oracle_args):
        if not all(value is not None for value in oracle_args):
            raise ValueError(
                "manifest, gt_dir, and official_python are all required for Oracle evaluation"
            )
        oracle_result = run_official_eval(
            canonical_dir,
            gt_dir,
            manifest,
            official_python=official_python,
            per_clip=False,
            output_path=oracle_output,
            enforce_official_env=enforce_official_env,
        )
        oracle = oracle_result.to_dict()

    result = {
        "status": "pass",
        "raw_pred_dir": str(Path(raw_pred_dir)),
        "canonical_dir": str(Path(canonical_dir)),
        "archive": str(Path(archive)),
        "conversion": conversion,
        "verify": verify_report,
        "oracle": oracle,
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-pred-dir", required=True, type=Path)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--out-zip", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--expected-list", type=Path)
    parser.add_argument("--missing-as-empty", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--official-python", type=Path)
    parser.add_argument("--oracle-output", type=Path)
    args = parser.parse_args()

    expected = (
        expected_from_manifest(args.manifest)
        if args.manifest is not None
        else read_expected(args.expected_list)
    )
    result = prepare_submission(
        args.raw_pred_dir,
        args.canonical_dir,
        expected,
        args.out_zip,
        report_path=args.report,
        missing_as_empty=args.missing_as_empty,
        manifest=args.manifest if args.gt_dir is not None else None,
        gt_dir=args.gt_dir,
        official_python=args.official_python,
        oracle_output=args.oracle_output,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
