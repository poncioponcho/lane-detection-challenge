#!/usr/bin/env python3
"""Launch or resume one gated UnLanedet training run on AutoDL."""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_run import (
    checkpoint_payload,
    expected_prediction_paths,
    read_metrics,
    sha256_file,
    summarize_f1,
    validate_diagnostic,
    validate_run,
)


ITERATIONS_PER_EPOCH = 525
CONFIGS = {
    "clrnet_r50": "clrnet_r50_hardlane.py",
    "adnet_r34": "adnet_r34_hardlane.py",
}
PINNED_UNLANEDET_COMMIT = "03921844220adb2e65c840de2d9759478d5c3d4c"
SAFE_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")


def override(key: str, value) -> str:
    """Format one LazyConfig CLI override for UnLanedet's pinned apply_overrides.

    pinned unlanedet/config/lazy.py calls ast.literal_eval(value) with no
    SyntaxError fallback, so bare path strings like train.output_dir=/hy-tmp/...
    crash the launch. Values that already parse as Python literals (ints, bools)
    pass through bare; anything else is single-quoted so literal_eval yields the
    original string.
    """
    text = str(value)
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return f"{key}='{text}'"
    return f"{key}={text}"


def required_absolute_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be absolute, got {value!r}")
    return path


def evidence_is_complete(
    path: Path, model: str, target: int, project_head: str | None = None
) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value.get("status") == "pass"
            and value.get("model") == model
            and (project_head is None or value.get("project_git_head") == project_head)
            and int(value.get("completed_iteration", -1)) + 1 >= target
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return False


def stream_command(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def assert_tracked_worktree_clean(path: Path) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
        text=True,
    ).strip()
    if status:
        raise SystemExit(f"tracked project files are dirty; commit before AutoDL training:\n{status}")


def validate_launch_history(path: Path, model: str, project_head: str) -> None:
    if not path.is_file():
        return
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            launch = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid launch history at {path}:L{line_number}") from exc
        if launch.get("model") != model:
            raise SystemExit(f"run directory belongs to another model: {path}:L{line_number}")
        if launch.get("project_git_head") != project_head:
            raise SystemExit(
                f"refusing cross-commit resume: {path}:L{line_number} was launched at "
                f"{launch.get('project_git_head')}, current HEAD is {project_head}"
            )


def inspect_cuda_python(python_bin: str) -> dict:
    source = (
        "import json,platform,sys,torch; "
        "print(json.dumps({'python':sys.version,'platform':platform.platform(),"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    value = json.loads(subprocess.check_output([python_bin, "-c", source], text=True))
    if not value["cuda_available"]:
        raise SystemExit("CUDA is unavailable in HARDLANE_PYTHON")
    if not str(value["torch"]).startswith("2.1.2") or value["torch_cuda"] != "11.8":
        raise SystemExit(f"unexpected AutoDL torch environment: {value}")
    return value


def validate_training_prerequisites(
    output_root: Path,
    weights_root: Path,
    model: str,
    project_head: str,
) -> dict:
    weight_path = output_root / "weight_probe.json"
    smoke_path = output_root / "smoke/dataloader_loss_smoke.json"
    try:
        weight = json.loads(weight_path.read_text(encoding="utf-8"))
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit("weight probe and smoke evidence must pass before training") from exc
    for label, value in (("weight probe", weight), ("smoke", smoke)):
        if value.get("status") != "pass" or value.get("project_git_head") != project_head:
            raise SystemExit(f"{label} evidence is stale or invalid for project {project_head}")
        if value.get("pinned_unlanedet_commit") != PINNED_UNLANEDET_COMMIT:
            raise SystemExit(f"{label} evidence uses a different UnLanedet commit")
    expected_checkpoint = weights_root / f"adapted_{model}_hardlane.pth"
    adapted = weight.get("adapted_checkpoints", {}).get(model, {})
    if Path(adapted.get("path", "")).resolve() != expected_checkpoint.resolve():
        raise SystemExit(f"weight probe points to an unexpected {model} checkpoint")
    if not expected_checkpoint.is_file():
        raise SystemExit(f"adapted checkpoint is missing: {expected_checkpoint}")
    checkpoint_sha = sha256_file(expected_checkpoint)
    if adapted.get("sha256") != checkpoint_sha:
        raise SystemExit(f"adapted {model} checkpoint SHA differs from weight probe")
    smoke_model = smoke.get("models", {}).get(model, {})
    if smoke_model.get("checkpoint_sha256") != checkpoint_sha:
        raise SystemExit(f"{model} smoke evidence does not match the adapted checkpoint")
    return {
        "weight_probe_sha256": sha256_file(weight_path),
        "smoke_sha256": sha256_file(smoke_path),
        "adapted_checkpoint": str(expected_checkpoint),
        "adapted_checkpoint_sha256": checkpoint_sha,
    }


def read_last_checkpoint(torch, run_dir: Path) -> tuple[Path, dict]:
    pointer = run_dir / "last_checkpoint"
    name = pointer.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name:
        raise SystemExit(f"unsafe last_checkpoint value: {name!r}")
    checkpoint = run_dir / name
    if not checkpoint.is_file():
        raise SystemExit(f"last checkpoint target is missing: {checkpoint}")
    payload = checkpoint_payload(torch, checkpoint)
    top_iteration = int(payload.get("iteration", -1))
    trainer = payload.get("trainer")
    if (
        not isinstance(trainer, dict)
        or int(trainer.get("iteration", -1)) != top_iteration
        or not isinstance(trainer.get("optimizer"), dict)
    ):
        raise SystemExit(f"last checkpoint has inconsistent trainer state: {checkpoint}")
    return checkpoint, payload


def recover_final_evaluation(
    python_bin: str,
    train_net: Path,
    config: Path,
    run_dir: Path,
    checkpoint: Path,
    project_root: Path,
    metric_iteration: int,
) -> dict:
    recovery_dir = run_dir / "recovered_final_eval"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_bin,
        str(train_net),
        "--eval-only",
        "--config-file",
        str(config),
        "--num-gpus",
        "1",
        override("train.init_checkpoint", checkpoint),
        override("train.output_dir", recovery_dir),
        override("dataloader.evaluator.output_basedir", recovery_dir / "val"),
        override("train.seed", 42),
        override("train.cudnn_benchmark", False),
    ]
    command_path = recovery_dir / "eval_command.json"
    command_path.write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return_code = stream_command(command, recovery_dir / "eval.log")
    if return_code:
        raise SystemExit(f"final-eval recovery failed with exit code {return_code}")
    source_diagnostic = recovery_dir / "val/diagnostic_metric.json"
    diagnostic = validate_diagnostic(
        json.loads(source_diagnostic.read_text(encoding="utf-8"))
    )
    manifest = project_root / "data/processed/manifest_val_v1_seed42.jsonl"
    expected_paths = expected_prediction_paths(manifest)
    source_predictions = recovery_dir / "val/predictions"
    actual_paths = {
        path.relative_to(source_predictions).as_posix()
        for path in source_predictions.rglob("*.lines.txt")
        if path.is_file()
    } if source_predictions.is_dir() else set()
    if actual_paths != expected_paths:
        raise SystemExit("final-eval recovery prediction set differs from manifest")

    destination_predictions = run_dir / "val/predictions"
    for relative in sorted(expected_paths):
        destination = destination_predictions / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_predictions / relative, destination)
    destination_diagnostic = run_dir / "val/diagnostic_metric.json"
    destination_diagnostic.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_diagnostic, destination_diagnostic)
    recovery = {
        "status": "pass",
        "reason": "training checkpoint reached max_iter before final eval evidence was complete",
        "metric_iteration": metric_iteration,
        "checkpoint_iteration": metric_iteration - 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "diagnostic": diagnostic,
        "validation_images": len(actual_paths),
        "command_sha256": sha256_file(command_path),
        "project_git_head": git_head(project_root),
    }
    (run_dir / "final_eval_recovery.json").write_text(
        json.dumps(recovery, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = run_dir / "metrics.json"
    existing_metrics = metrics_path.read_bytes()
    with metrics_path.open("ab") as handle:
        if existing_metrics and not existing_metrics.endswith(b"\n"):
            handle.write(b"\n")
        row = {"iteration": metric_iteration, **diagnostic, "_recovered_final_eval": True}
        handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
    return recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-iter", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if not SAFE_RUN_NAME.fullmatch(args.run_name):
        raise SystemExit("--run-name must match [a-z0-9][a-z0-9_.-]{0,79}")
    target_iter = args.max_iter or args.epochs * ITERATIONS_PER_EPOCH
    if target_iter <= 0:
        raise SystemExit("--max-iter must be positive")

    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT")
    required_absolute_env("HARDLANE_DATA_ROOT")
    unlanedet_root = required_absolute_env("UNLANEDET_ROOT")
    weights_root = required_absolute_env("HARDLANE_WEIGHTS_ROOT")
    output_root = required_absolute_env("HARDLANE_OUTPUT_ROOT")
    python_bin = os.environ.get("HARDLANE_PYTHON", sys.executable)
    config = project_root / "configs/unlanedet" / CONFIGS[args.model]
    train_net = unlanedet_root / "tools/train_net.py"
    for path in (config, train_net):
        if not path.is_file():
            raise SystemExit(f"required file missing: {path}")

    environment = None
    project_head = None
    unlanedet_head = None
    prerequisite_evidence = None
    if not args.dry_run:
        assert_tracked_worktree_clean(project_root)
        project_head = git_head(project_root)
        unlanedet_head = git_head(unlanedet_root)
        if unlanedet_head != PINNED_UNLANEDET_COMMIT:
            raise SystemExit(
                f"UnLanedet HEAD {unlanedet_head} != pinned {PINNED_UNLANEDET_COMMIT}"
            )
        patch_source = (
            unlanedet_root / "unlanedet/model/CLRNet/clr_head.py"
        ).read_text(encoding="utf-8")
        condlane_head_source = (
            unlanedet_root / "unlanedet/model/CondlaneNet/head.py"
        ).read_text(encoding="utf-8")
        if (
            "top_k=self.cfg.test_parameters.nms_topk" not in patch_source
            or ".astype(bool)" not in patch_source
            or "predictions[..., 4].clamp(0.01, 0.99)" not in patch_source
            or "torch.autograd.set_detect_anomaly(False)" not in condlane_head_source
        ):
            raise SystemExit("required HardLane UnLanedet patch is not applied")
        environment = inspect_cuda_python(python_bin)
        prerequisite_evidence = validate_training_prerequisites(
            output_root, weights_root, args.model, project_head
        )

    run_dir = output_root / "runs" / args.run_name
    evidence_path = run_dir / "run_evidence.json"
    if not args.dry_run and args.skip_if_complete and evidence_is_complete(
        evidence_path, args.model, target_iter, project_head
    ):
        validate_run(run_dir, args.model, target_iter, project_root)
        print(json.dumps({
            "status": "already_complete",
            "validated": True,
            "evidence": str(evidence_path),
        }))
        return

    has_checkpoint = (run_dir / "last_checkpoint").is_file()
    has_existing_files = run_dir.exists() and any(run_dir.iterdir())
    if args.auto_resume and has_existing_files and not has_checkpoint and not args.dry_run:
        raise SystemExit(
            f"cannot auto-resume a non-empty run without last_checkpoint: {run_dir}; "
            "move it aside after inspecting the interrupted artifacts"
        )
    resume = args.resume or (args.auto_resume and has_checkpoint)
    if args.resume and not has_checkpoint:
        raise SystemExit(f"--resume requested but last_checkpoint is missing: {run_dir}")
    if not resume and has_existing_files and not args.dry_run:
        raise SystemExit(
            f"refusing a fresh run in non-empty directory: {run_dir}; "
            "move it aside or use --resume/--auto-resume"
        )
    if resume and not args.dry_run:
        validate_launch_history(run_dir / "launches.jsonl", args.model, project_head)
        import torch

        last_checkpoint, last_payload = read_last_checkpoint(torch, run_dir)
        completed_iterations = int(last_payload["iteration"]) + 1
        del last_payload
        if completed_iterations > target_iter:
            raise SystemExit(
                f"checkpoint already completed {completed_iterations} iterations, "
                f"beyond requested target {target_iter}"
            )
        if completed_iterations == target_iter:
            try:
                metric_entries = read_metrics(run_dir / "metrics.json")
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(
                    "cannot recover a completed checkpoint without valid metrics history"
                ) from exc
            try:
                final_iteration = int(summarize_f1(metric_entries)["final"]["iteration"])
            except ValueError as exc:
                if str(exc) != "metrics contain no F1 observations":
                    raise
                final_iteration = -1
            if final_iteration != target_iter:
                recover_final_evaluation(
                    python_bin,
                    train_net,
                    config,
                    run_dir,
                    last_checkpoint,
                    project_root,
                    target_iter,
                )
            result = validate_run(run_dir, args.model, target_iter, project_root)
            print(json.dumps({
                "status": "pass",
                "run_dir": str(run_dir),
                "recovered_without_training": True,
                "best_f1": result["f1_history"]["best"]["f1"],
                "final_f1": result["f1_history"]["final"]["f1"],
            }))
            return
    command = [
        python_bin,
        str(train_net),
        "--config-file",
        str(config),
        "--num-gpus",
        "1",
    ]
    if resume:
        command.append("--resume")
    command.extend(
        [
            override("train.max_iter", target_iter),
            override(
                "train.eval_period",
                ITERATIONS_PER_EPOCH
                if target_iter % ITERATIONS_PER_EPOCH == 0
                else target_iter,
            ),
            override("train.checkpointer.period", ITERATIONS_PER_EPOCH),
            override("train.checkpointer.max_to_keep", 40),
            override("train.output_dir", run_dir),
            override("dataloader.evaluator.output_basedir", run_dir / "val"),
            override("train.seed", 42),
            override("train.cudnn_benchmark", False),
        ]
    )
    launch = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "epochs_label": args.epochs,
        "target_max_iter": target_iter,
        "resume": resume,
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_environment": environment,
        "project_git_head": project_head,
        "unlanedet_git_head": unlanedet_head,
        "prerequisite_evidence": prerequisite_evidence,
    }
    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "launches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(launch, ensure_ascii=False) + "\n")
    print(json.dumps(launch, ensure_ascii=False, indent=2), flush=True)

    return_code = stream_command(command, run_dir / "train.log")
    if return_code:
        raise SystemExit(f"training failed with exit code {return_code}: {run_dir}")
    result = validate_run(run_dir, args.model, target_iter, project_root)
    print(json.dumps({
        "status": "pass",
        "run_dir": str(run_dir),
        "best_f1": result["f1_history"]["best"]["f1"],
        "final_f1": result["f1_history"]["final"]["f1"],
    }))


if __name__ == "__main__":
    main()
