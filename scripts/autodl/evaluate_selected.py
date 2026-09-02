#!/usr/bin/env python3
"""Replay a run's selected best checkpoint and verify its validation F1."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from run_training import (
    CONFIGS,
    PINNED_UNLANEDET_COMMIT,
    assert_tracked_worktree_clean,
    git_head,
    inspect_cuda_python,
    required_absolute_env,
    stream_command,
)
from validate_run import (expected_prediction_paths, sha256_file,
                          validate_diagnostic)


def existing_evidence_passes(
    path: Path,
    checkpoint_sha: str,
    run_evidence_sha: str,
    expected_f1: float,
    expected_model: str,
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        diagnostic = path.parent / "val/diagnostic_metric.json"
        command = path.parent / "eval_command.json"
        return (
            value.get("status") == "pass"
            and value.get("model") == expected_model
            and value.get("selected_checkpoint_sha256") == checkpoint_sha
            and value.get("run_evidence_sha256") == run_evidence_sha
            and int(value.get("validation_images", -1)) == 800
            and diagnostic.is_file()
            and command.is_file()
            and (path.parent / "eval.log").is_file()
            and (path.parent / "config.yaml").is_file()
            and value.get("artifacts", {}).get("diagnostic_sha256")
            == sha256_file(diagnostic)
            and value.get("artifacts", {}).get("command_sha256") == sha256_file(command)
            and math.isclose(
                float(value["diagnostic"]["F1"]), expected_f1,
                rel_tol=0.0, abs_tol=1e-12,
            )
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--skip-if-complete", action="store_true")
    args = parser.parse_args()

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
    if run_evidence.get("project_git_head") != project_head:
        raise SystemExit("run evidence was produced by a different project commit")
    selected = run_evidence["selected_best_checkpoint"]
    checkpoint = Path(selected["path"]).resolve()
    if checkpoint.parent != run_dir:
        raise SystemExit(f"selected checkpoint escaped run directory: {checkpoint}")
    if not checkpoint.is_file():
        raise SystemExit(f"selected checkpoint missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != selected["sha256"]:
        raise SystemExit("selected checkpoint SHA changed after run validation")
    expected_f1 = float(run_evidence["f1_history"]["best"]["f1"])
    if int(selected["metric_iteration"]) != int(
        run_evidence["f1_history"]["best"]["iteration"]
    ):
        raise SystemExit("selected checkpoint metric iteration differs from history best")
    run_evidence_sha = sha256_file(run_evidence_path)

    manifest = project_root / "data/processed/manifest_val_v1_seed42.jsonl"
    expected_paths = expected_prediction_paths(manifest)

    eval_dir = run_dir / "selected_best_eval"
    eval_evidence_path = eval_dir / "eval_evidence.json"
    if args.skip_if_complete and existing_evidence_passes(
        eval_evidence_path, checkpoint_sha, run_evidence_sha, expected_f1, model
    ):
        prediction_root = eval_dir / "val/predictions"
        actual_paths = {
            path.relative_to(prediction_root).as_posix()
            for path in prediction_root.rglob("*.lines.txt")
            if path.is_file()
        } if prediction_root.is_dir() else set()
        if actual_paths == expected_paths:
            print(json.dumps({"status": "already_complete", "evidence": str(eval_evidence_path)}))
            return
    eval_dir.mkdir(parents=True, exist_ok=True)
    config = project_root / "configs/unlanedet" / CONFIGS[model]
    train_net = unlanedet_root / "tools/train_net.py"
    if not config.is_file() or not train_net.is_file():
        raise SystemExit("selected-best eval config or train_net.py is missing")
    command = [
        python_bin,
        str(train_net),
        "--eval-only",
        "--config-file",
        str(config),
        "--num-gpus",
        "1",
        f"train.init_checkpoint={checkpoint}",
        f"train.output_dir={eval_dir}",
        f"dataloader.evaluator.output_basedir={eval_dir / 'val'}",
        "train.seed=42",
        "train.cudnn_benchmark=False",
    ]
    (eval_dir / "eval_command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    eval_evidence_path.write_text(
        json.dumps({
            "status": "running",
            "model": model,
            "selected_checkpoint_sha256": checkpoint_sha,
            "run_evidence_sha256": run_evidence_sha,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return_code = stream_command(command, eval_dir / "eval.log")
    if return_code:
        raise SystemExit(f"selected checkpoint eval failed with exit code {return_code}")

    diagnostic_path = eval_dir / "val/diagnostic_metric.json"
    diagnostic = validate_diagnostic(
        json.loads(diagnostic_path.read_text(encoding="utf-8"))
    )
    if not math.isclose(diagnostic["F1"], expected_f1, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"selected checkpoint replay F1 {diagnostic['F1']} != history best {expected_f1}"
        )
    prediction_root = eval_dir / "val/predictions"
    actual_paths = {
        path.relative_to(prediction_root).as_posix()
        for path in prediction_root.rglob("*.lines.txt")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise SystemExit("selected checkpoint replay prediction set differs from manifest")

    evidence = {
        "status": "pass",
        "model": model,
        "run_evidence": str(run_evidence_path),
        "run_evidence_sha256": run_evidence_sha,
        "project_git_head": project_head,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
        "expected_best_f1": expected_f1,
        "diagnostic": diagnostic,
        "validation_images": len(actual_paths),
        "artifacts": {
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "command_sha256": sha256_file(eval_dir / "eval_command.json"),
        },
    }
    eval_evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass", "model": model, "F1": diagnostic["F1"],
        "evidence": str(eval_evidence_path),
    }))


if __name__ == "__main__":
    main()
