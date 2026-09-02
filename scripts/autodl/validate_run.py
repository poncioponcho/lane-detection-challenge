#!/usr/bin/env python3
"""Validate one completed AutoDL training run and freeze compact evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path, PurePath


EXPECTED_VAL_IMAGES = 800
EXPECTED_VAL_LANES = 2655
KNOWN_MODELS = {"clrnet_r50", "adnet_r34"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metrics JSON at {path}:L{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"metrics row is not an object at {path}:L{line_number}")
        entries.append(value)
    if not entries:
        raise ValueError(f"metrics file is empty: {path}")
    return entries


def read_launch_targets(path: Path) -> set[int]:
    targets = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
            target = int(value["target_max_iter"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid launch history at {path}:L{line_number}") from exc
        if target <= 0:
            raise ValueError(f"invalid target_max_iter at {path}:L{line_number}")
        targets.add(target)
    if not targets:
        raise ValueError(f"launch history is empty: {path}")
    return targets


def checkpoint_iteration_for_metric(
    metric_iteration: int, final_eval_iterations: set[int]
) -> int:
    # UnLanedet runs final EvalHook in after_train after setting storage.iter=max_iter.
    # Therefore a final metric at N describes the model after checkpoint iteration N-1.
    return metric_iteration - 1 if metric_iteration in final_eval_iterations else metric_iteration


def summarize_f1(entries: list[dict]) -> dict:
    points = []
    for entry in entries:
        if "F1" not in entry:
            continue
        f1 = float(entry["F1"])
        iteration = int(entry.get("iteration", -1))
        if iteration < 0:
            raise ValueError(f"F1 observation has invalid iteration {iteration}")
        if not math.isfinite(f1) or not 0.0 <= f1 <= 1.0:
            raise ValueError(f"invalid F1={f1} at iteration {iteration}")
        points.append({"iteration": iteration, "f1": f1})
    if not points:
        raise ValueError("metrics contain no F1 observations")
    return {
        "observations": len(points),
        "best": max(points, key=lambda item: (item["f1"], -item["iteration"])),
        "final": points[-1],
    }


def validate_diagnostic(value: dict) -> dict:
    required = {"F1", "TP", "FP", "FN", "P", "G"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"diagnostic metric missing keys: {sorted(missing)}")
    tp, fp, fn = int(value["TP"]), int(value["FP"]), int(value["FN"])
    predicted, gt = int(value["P"]), int(value["G"])
    f1 = float(value["F1"])
    if min(tp, fp, fn, predicted, gt) < 0:
        raise ValueError("diagnostic counts must be non-negative")
    if predicted != tp + fp or gt != tp + fn:
        raise ValueError("diagnostic TP/FP/FN totals are inconsistent")
    if gt != EXPECTED_VAL_LANES:
        raise ValueError(f"validation GT lane count {gt} != {EXPECTED_VAL_LANES}")
    expected_f1 = 2.0 * tp / (predicted + gt) if predicted + gt else 0.0
    if not math.isclose(f1, expected_f1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"diagnostic F1 {f1} != counts-derived {expected_f1}")
    return {"F1": f1, "TP": tp, "FP": fp, "FN": fn, "P": predicted, "G": gt}


def expected_prediction_paths(manifest_path: Path) -> set[str]:
    expected = set()
    for line_number, raw in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        rel = value.get("pred_rel_path")
        if not isinstance(rel, str) or PurePath(rel).is_absolute() or ".." in PurePath(rel).parts:
            raise ValueError(f"unsafe pred_rel_path at manifest line {line_number}")
        if rel in expected:
            raise ValueError(f"duplicate pred_rel_path in manifest: {rel}")
        expected.add(rel)
    if len(expected) != EXPECTED_VAL_IMAGES:
        raise ValueError(
            f"validation manifest has {len(expected)} paths != {EXPECTED_VAL_IMAGES}"
        )
    return expected


def git_head(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()


def checkpoint_payload(torch, path: Path) -> dict:
    value = torch.load(str(path), map_location="cpu")
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
        raise ValueError(f"checkpoint has no model state: {path}")
    return value


def resolve_best_checkpoint(
    torch, run_dir: Path, checkpoint_iteration: int
) -> tuple[Path, dict]:
    preferred = [
        run_dir / f"model_{checkpoint_iteration:07d}.pth",
        run_dir / "model_final.pth",
        run_dir / "model_best.pth",
    ]
    remaining = sorted(run_dir.glob("model_*.pth"))
    seen = set()
    for candidate in preferred + remaining:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        payload = checkpoint_payload(torch, candidate)
        if int(payload.get("iteration", -1)) == checkpoint_iteration:
            return candidate, payload
        del payload
    raise FileNotFoundError(
        f"no retained checkpoint matches iteration {checkpoint_iteration}: {run_dir}"
    )


def validate_run(
    run_dir: Path,
    model: str,
    expected_max_iter: int,
    project_root: Path,
) -> dict:
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown model: {model}")
    if expected_max_iter <= 0:
        raise ValueError("expected_max_iter must be positive")
    run_dir = run_dir.resolve()
    required = {
        "config": run_dir / "config.yaml",
        "metrics": run_dir / "metrics.json",
        "launches": run_dir / "launches.jsonl",
        "last_pointer": run_dir / "last_checkpoint",
        "best_checkpoint": run_dir / "model_best.pth",
        "diagnostic": run_dir / "val/diagnostic_metric.json",
        "train_log": run_dir / "train.log",
    }
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"run is incomplete; missing {absent}: {run_dir}")

    checkpoint_name = required["last_pointer"].read_text(encoding="utf-8").strip()
    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError(f"unsafe last_checkpoint value: {checkpoint_name!r}")
    last_checkpoint = run_dir / checkpoint_name
    if not last_checkpoint.is_file():
        raise FileNotFoundError(f"last checkpoint target missing: {last_checkpoint}")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("validate_run.py must run in the AutoDL torch environment") from exc
    payload = checkpoint_payload(torch, last_checkpoint)
    checkpoint_iteration = int(payload.get("iteration", -1))
    trainer = payload.get("trainer")
    trainer_iteration = int(trainer.get("iteration", -1)) if isinstance(trainer, dict) else -1
    expected_iteration = expected_max_iter - 1
    if checkpoint_iteration != expected_iteration or trainer_iteration != expected_iteration:
        raise ValueError(
            "checkpoint iteration mismatch: "
            f"top={checkpoint_iteration}, trainer={trainer_iteration}, "
            f"expected={expected_iteration}"
        )
    if not isinstance(trainer, dict) or not isinstance(trainer.get("optimizer"), dict):
        raise ValueError("checkpoint trainer state has no optimizer")

    metric_entries = read_metrics(required["metrics"])
    f1_summary = summarize_f1(metric_entries)
    diagnostic = validate_diagnostic(
        json.loads(required["diagnostic"].read_text(encoding="utf-8"))
    )
    if not math.isclose(
        f1_summary["final"]["f1"], diagnostic["F1"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("final metrics.json F1 differs from diagnostic_metric.json")
    if int(f1_summary["final"]["iteration"]) != expected_max_iter:
        raise ValueError(
            f"final F1 iteration {f1_summary['final']['iteration']} != {expected_max_iter}"
        )

    best_metric_iteration = int(f1_summary["best"]["iteration"])
    if best_metric_iteration > expected_max_iter:
        raise ValueError(
            f"best F1 iteration {best_metric_iteration} exceeds final metric iteration "
            f"{expected_max_iter}"
        )
    final_eval_iterations = read_launch_targets(required["launches"])
    best_checkpoint_iteration = checkpoint_iteration_for_metric(
        best_metric_iteration, final_eval_iterations
    )
    if best_checkpoint_iteration < 0 or best_checkpoint_iteration > expected_iteration:
        raise ValueError("best F1 maps to an invalid checkpoint iteration")
    selected_checkpoint, selected_payload = resolve_best_checkpoint(
        torch, run_dir, best_checkpoint_iteration
    )
    if not selected_payload["model"]:
        raise ValueError("selected best checkpoint has an empty model state")
    (run_dir / "selected_best_checkpoint").write_text(
        selected_checkpoint.name + "\n", encoding="utf-8"
    )

    manifest = project_root / "data/processed/manifest_val_v1_seed42.jsonl"
    expected_paths = expected_prediction_paths(manifest)
    prediction_root = run_dir / "val/predictions"
    actual_paths = {
        path.relative_to(prediction_root).as_posix()
        for path in prediction_root.rglob("*.lines.txt")
        if path.is_file()
    } if prediction_root.is_dir() else set()
    if actual_paths != expected_paths:
        missing = sorted(expected_paths.difference(actual_paths))
        extra = sorted(actual_paths.difference(expected_paths))
        raise ValueError(
            f"prediction set mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )

    evidence = {
        "status": "pass",
        "execution_environment": "AutoDL CUDA",
        "model": model,
        "run_dir": str(run_dir),
        "expected_max_iter": expected_max_iter,
        "completed_iteration": checkpoint_iteration,
        "project_git_head": git_head(project_root),
        "validation_images": len(actual_paths),
        "diagnostic": diagnostic,
        "f1_history": f1_summary,
        "selected_best_checkpoint": {
            "path": str(selected_checkpoint),
            "metric_iteration": best_metric_iteration,
            "checkpoint_iteration": best_checkpoint_iteration,
            "bytes": selected_checkpoint.stat().st_size,
            "sha256": sha256_file(selected_checkpoint),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in {
                "config": required["config"],
                "metrics": required["metrics"],
                "launches": required["launches"],
                "diagnostic": required["diagnostic"],
                "last_checkpoint": last_checkpoint,
                "best_checkpoint": required["best_checkpoint"],
            }.items()
        },
    }
    output = run_dir / "run_evidence.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(KNOWN_MODELS))
    parser.add_argument("--expected-max-iter", required=True, type=int)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    project_root = args.project_root or Path(os.environ["HARDLANE_PROJECT_ROOT"])
    result = validate_run(args.run_dir, args.model, args.expected_max_iter, project_root)
    print(json.dumps({
        "status": result["status"],
        "model": result["model"],
        "best_f1": result["f1_history"]["best"]["f1"],
        "final_f1": result["f1_history"]["final"]["f1"],
        "evidence": str(args.run_dir / "run_evidence.json"),
    }))


if __name__ == "__main__":
    main()
