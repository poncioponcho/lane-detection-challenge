#!/usr/bin/env python3
"""Run a trained checkpoint over an unlabeled split (testA/testB) and export
predictions for packaging.

Unlike evaluate_selected.py, this never compares against a diagnostic F1: the
prediction-only splits ship no ground truth, so any score produced here would
be fabricated. Scoring is left to the frozen Oracle (local rehearsal) or the
competition server (real submission).

The dataset/evaluator unlabeled support lives in
src/integrations/unlanedet_hardlane.py (LABELED_SPLITS / UNLABELED_SPLITS).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from run_training import (
    CONFIGS,
    PINNED_UNLANEDET_COMMIT,
    assert_tracked_worktree_clean,
    git_head,
    inspect_cuda_python,
    override,
    required_absolute_env,
    stream_command,
)
from validate_run import manifest_prediction_paths, sha256_file

# Must stay in sync with src/integrations/unlanedet_hardlane.py::UNLABELED_SPLITS;
# tests/test_unlanedet_hardlane.py guards the pairing.
UNLABELED_SPLITS = {"testA", "testB"}
# The checked-in CLRNet config remains at 0.4 for the training/LVO contract.
# Production testA/testB inference must always record and pass this explicit
# decode-time override so a rehearsal cannot silently fall back to 0.4.
PRODUCTION_CONF_THRESHOLD = 0.50


def manifest_for_split(project_root: Path, split: str) -> Path:
    return project_root / "data/processed" / f"manifest_{split}.jsonl"


def existing_evidence_passes(
    path: Path,
    checkpoint_sha: str,
    manifest_sha: str,
    split: str,
    conf_threshold: float,
    expected_count: int,
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value.get("status") == "pass"
            and value.get("split") == split
            and value.get("selected_checkpoint_sha256") == checkpoint_sha
            and value.get("manifest_sha256") == manifest_sha
            and math.isclose(
                float(value.get("conf_threshold")), conf_threshold, abs_tol=1e-12
            )
            and int(value.get("prediction_count", -1)) == expected_count
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="testA",
        help="Unlabeled split to run; manifest_<split>.jsonl must exist.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=PRODUCTION_CONF_THRESHOLD,
        help="Override model.head.cfg.test_parameters.conf_threshold "
        f"(defaults to frozen production override {PRODUCTION_CONF_THRESHOLD:.2f}; "
        "pass another value only for a named threshold experiment).",
    )
    parser.add_argument("--skip-if-complete", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.conf_threshold) or not 0.0 <= args.conf_threshold <= 1.0:
        raise SystemExit(
            f"--conf-threshold must be a finite probability in [0, 1], got {args.conf_threshold!r}"
        )

    if args.split not in UNLABELED_SPLITS:
        raise SystemExit(
            f"{args.split!r} is not an unlabeled split; "
            f"use evaluate_selected.py for labeled val replay"
        )

    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT")
    required_absolute_env("HARDLANE_DATA_ROOT")
    unlanedet_root = required_absolute_env("UNLANEDET_ROOT")
    required_absolute_env("HARDLANE_WEIGHTS_ROOT")
    required_absolute_env("HARDLANE_OUTPUT_ROOT")
    python_bin = os.environ.get("HARDLANE_PYTHON", sys.executable)
    assert_tracked_worktree_clean(project_root)
    project_head = git_head(project_root)
    if git_head(unlanedet_root) != PINNED_UNLANEDET_COMMIT:
        raise SystemExit("UnLanedet HEAD does not match the pinned commit")
    inspect_cuda_python(python_bin)

    run_dir = args.run_dir.resolve()
    run_evidence_path = run_dir / "run_evidence.json"
    run_evidence = json.loads(run_evidence_path.read_text(encoding="utf-8"))
    if run_evidence.get("status") != "pass":
        raise SystemExit(f"run evidence is not pass: {run_evidence_path}")
    model = run_evidence.get("model")
    if model not in CONFIGS:
        raise SystemExit(f"unknown model in run evidence: {model!r}")
    # Deliberately NOT requiring run_evidence["project_git_head"] == current head.
    # evaluate_selected.py needs that equality because it must reproduce the
    # training-time F1, which any code change can legitimately alter. Here there
    # is no score to reproduce: the artifact identity is the checkpoint SHA
    # (verified below), and inference should run under the CURRENT code --
    # otherwise no new code could ever export predictions from an existing
    # checkpoint. Provenance is preserved by recording both commits.
    training_head = run_evidence.get("project_git_head")
    if not isinstance(training_head, str) or len(training_head) != 40:
        raise SystemExit(f"run evidence has no usable project_git_head: {run_evidence_path}")
    selected = run_evidence["selected_best_checkpoint"]
    checkpoint = Path(selected["path"]).resolve()
    if checkpoint.parent != run_dir:
        raise SystemExit(f"selected checkpoint escaped run directory: {checkpoint}")
    if not checkpoint.is_file():
        raise SystemExit(f"selected checkpoint missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != selected["sha256"]:
        raise SystemExit("selected checkpoint SHA changed after run validation")

    manifest = (args.manifest or manifest_for_split(project_root, args.split)).resolve()
    if not manifest.is_file():
        raise SystemExit(f"manifest missing: {manifest}")
    manifest_sha = sha256_file(manifest)
    # testA declares 900 paths; only the val split is pinned to 800.
    expected_paths = manifest_prediction_paths(manifest, expected_count=None)

    output_dir = (args.output_dir or (run_dir / f"{args.split.lower()}_infer")).resolve()
    evidence_path = output_dir / "infer_evidence.json"
    if args.skip_if_complete and existing_evidence_passes(
        evidence_path,
        checkpoint_sha,
        manifest_sha,
        args.split,
        args.conf_threshold,
        len(expected_paths),
    ):
        if _prediction_set(output_dir, args.split) == expected_paths:
            print(json.dumps({"status": "already_complete", "evidence": str(evidence_path)}))
            return

    config = project_root / "configs/unlanedet" / CONFIGS[model]
    train_net = unlanedet_root / "tools/train_net.py"
    if not config.is_file() or not train_net.is_file():
        raise SystemExit("config or train_net.py is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        str(train_net),
        "--eval-only",
        "--config-file",
        str(config),
        "--num-gpus",
        "1",
        # Path/string values must reuse run_training.override: the pinned
        # apply_overrides literal_evals every value, so bare strings crash with
        # SyntaxError (same failure mode fixed in dc49dea / 06f1da8).
        override("train.init_checkpoint", checkpoint),
        override("train.output_dir", output_dir),
        override("dataloader.evaluator.output_basedir", output_dir / args.split),
        override("dataloader.test.dataset.manifest_path", manifest),
        override("dataloader.test.dataset.split", args.split),
        "train.seed=42",
        "train.cudnn_benchmark=False",
    ]
    # The head reads the decode-time confidence gate from model.head.cfg (the
    # shared param_config), so the production override path is explicit even
    # when the caller omitted --conf-threshold.
    command.append(
        override("model.head.cfg.test_parameters.conf_threshold", args.conf_threshold)
    )
    (output_dir / "infer_command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "status": "running",
                "model": model,
                "split": args.split,
                "project_git_head": project_head,
                "training_git_head": training_head,
                "selected_checkpoint_sha256": checkpoint_sha,
                "manifest_sha256": manifest_sha,
                "conf_threshold": args.conf_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return_code = stream_command(command, output_dir / "infer.log")
    if return_code:
        raise SystemExit(f"{args.split} inference failed with exit code {return_code}")

    actual_paths = _prediction_set(output_dir, args.split)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise SystemExit(
            f"{args.split} prediction set differs from manifest: "
            f"{len(missing)} missing (e.g. {missing[:3]}), "
            f"{len(extra)} unexpected (e.g. {extra[:3]})"
        )

    basedir = output_dir / args.split
    summary_path = basedir / "unlabeled_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"unlabeled evaluator summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "predictions_only":
        raise SystemExit(f"evaluator did not take the predictions-only path: {summary}")
    if int(summary.get("images", -1)) != len(expected_paths):
        raise SystemExit(
            f"evaluator exported {summary.get('images')} predictions, "
            f"manifest expects {len(expected_paths)}"
        )
    if (basedir / "diagnostic_metric.json").exists():
        # No GT exists for this split; a score here can only be fabricated.
        raise SystemExit(
            f"unexpected diagnostic_metric.json for unlabeled split {args.split}"
        )

    evidence = {
        "status": "pass",
        "model": model,
        "split": args.split,
        "run_evidence": str(run_evidence_path),
        "project_git_head": project_head,
        "training_git_head": training_head,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "conf_threshold": args.conf_threshold,
        "prediction_count": len(actual_paths),
        "prediction_root": str(basedir / "predictions"),
        "selected_best_f1": run_evidence["f1_history"]["best"]["f1"],
        "artifacts": {
            "command_sha256": sha256_file(output_dir / "infer_command.json"),
            "summary": summary,
        },
    }
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "model": model,
        "split": args.split,
        "prediction_count": len(actual_paths),
        "prediction_root": str(basedir / "predictions"),
        "evidence": str(evidence_path),
    }))


def _prediction_set(output_dir: Path, split: str) -> set[str]:
    prediction_root = output_dir / split / "predictions"
    if not prediction_root.is_dir():
        return set()
    return {
        path.relative_to(prediction_root).as_posix()
        for path in prediction_root.rglob("*.lines.txt")
        if path.is_file()
    }


if __name__ == "__main__":
    main()
