#!/usr/bin/env python3
"""Create a checked AutoDL handoff archive before releasing the GPU instance."""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

from run_training import CONFIGS, git_head, required_absolute_env
from select_screen_winner import choose_winner
from validate_run import sha256_file


# The gate/screen/baseline collect stages describe the ORIGINAL two-model duel
# (DECISIONS §12 baseline selection / §25 pipeline entry).  CONFIGS later grew
# dispatch-only members (clrnet_r50_vat, clrernet_r50) that are launched by
# dedicated phase scripts with their own evidence chains; they must NOT
# inflate the historical pipeline modes' run requirements.
PIPELINE_MODELS = ("clrnet_r50", "adnet_r34")
assert all(model in CONFIGS for model in PIPELINE_MODELS)


RUN_FILES = (
    "config.yaml",
    "metrics.json",
    "train.log",
    "launches.jsonl",
    "last_checkpoint",
    "run_evidence.json",
    "selected_best_checkpoint",
    "val/diagnostic_metric.json",
    "selected_best_eval/config.yaml",
    "selected_best_eval/eval_command.json",
    "selected_best_eval/eval.log",
    "selected_best_eval/eval_evidence.json",
    "selected_best_eval/val/diagnostic_metric.json",
)


def read_pass_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "pass":
        raise ValueError(f"evidence is not pass: {path}")
    return value


def add_required_file(files: set[Path], path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular file missing: {path}")
    files.add(path.resolve())


def add_required_tree(files: set[Path], root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"required directory missing: {root}")
    found = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    if not found:
        raise FileNotFoundError(f"required directory is empty: {root}")
    files.update(path.resolve() for path in found)
    return found


def add_run(
    files: set[Path],
    run_dir: Path,
    model: str,
    expected_project_head: str | None,
    expected_max_iter: int,
) -> dict:
    evidence_path = run_dir / "run_evidence.json"
    evidence = read_pass_json(evidence_path)
    if evidence.get("model") != model:
        raise ValueError(f"run evidence model mismatch: {evidence_path}")
    if int(evidence.get("completed_iteration", -1)) + 1 != expected_max_iter:
        raise ValueError(f"run did not finish exactly {expected_max_iter} iterations: {run_dir}")
    if expected_project_head and evidence.get("project_git_head") != expected_project_head:
        raise ValueError(f"run evidence belongs to another project commit: {evidence_path}")
    replay = read_pass_json(run_dir / "selected_best_eval/eval_evidence.json")
    if replay.get("model") != model:
        raise ValueError(f"selected-best replay model mismatch: {run_dir}")
    if expected_project_head and replay.get("project_git_head") != expected_project_head:
        raise ValueError(f"selected-best replay belongs to another project commit: {run_dir}")
    if replay.get("run_evidence_sha256") != sha256_file(evidence_path):
        raise ValueError(f"selected-best replay is stale for run evidence: {run_dir}")
    if int(replay.get("validation_images", -1)) != 800:
        raise ValueError(f"selected-best replay did not validate 800 images: {run_dir}")
    selected_sha = evidence.get("selected_best_checkpoint", {}).get("sha256")
    if replay.get("selected_checkpoint_sha256") != selected_sha:
        raise ValueError(f"selected-best replay checkpoint mismatch: {run_dir}")
    for relative in RUN_FILES:
        add_required_file(files, run_dir / relative)
    recovery = run_dir / "final_eval_recovery.json"
    if recovery.exists():
        read_pass_json(recovery)
        add_required_file(files, recovery)
        add_required_tree(files, run_dir / "recovered_final_eval")
    prediction_files = add_required_tree(
        files, run_dir / "selected_best_eval/val/predictions"
    )
    line_files = [path for path in prediction_files if path.name.endswith(".lines.txt")]
    if len(line_files) != 800 or len(line_files) != len(prediction_files):
        raise ValueError(f"selected-best archive prediction set is not exactly 800 files: {run_dir}")
    return evidence


def collect_paths(
    output_root: Path, through: str, expected_project_head: str | None = None
) -> tuple[list[Path], dict | None]:
    output_root = output_root.resolve()
    files: set[Path] = set()
    weight = output_root / "weight_probe.json"
    weight_value = read_pass_json(weight)
    if expected_project_head and weight_value.get("project_git_head") != expected_project_head:
        raise ValueError("weight probe belongs to another project commit")
    add_required_file(files, weight)
    smoke = output_root / "smoke/dataloader_loss_smoke.json"
    smoke_value = read_pass_json(smoke)
    if expected_project_head and smoke_value.get("project_git_head") != expected_project_head:
        raise ValueError("smoke evidence belongs to another project commit")
    add_required_tree(files, output_root / "smoke")

    for model in PIPELINE_MODELS:
        add_run(
            files, output_root / f"runs/gate_{model}_1ep", model,
            expected_project_head, 526,
        )

    decision = None
    if through in {"screen", "baseline"}:
        screen_evidence = {}
        for model in PIPELINE_MODELS:
            run_dir = output_root / f"runs/screen_{model}_15ep"
            screen_evidence[model] = add_run(
                files, run_dir, model, expected_project_head, 15 * 525
            )
        decision_path = output_root / "screen_decision.json"
        decision = read_pass_json(decision_path)
        if decision.get("winner") not in CONFIGS:
            raise ValueError(f"invalid winner in {decision_path}")
        close_margin = float(decision.get("close_margin", -1.0))
        if not 0.0 <= close_margin <= 1.0:
            raise ValueError("screen decision close margin is invalid")
        expected_winner, _ = choose_winner(
            float(screen_evidence["clrnet_r50"]["f1_history"]["best"]["f1"]),
            float(screen_evidence["adnet_r34"]["f1_history"]["best"]["f1"]),
            close_margin,
        )
        if decision["winner"] != expected_winner:
            raise ValueError("screen decision winner differs from frozen run evidence")
        for model in CONFIGS:
            run_dir = output_root / f"runs/screen_{model}_15ep"
            frozen = decision.get("runs", {}).get(model, {})
            if (
                frozen.get("evidence_sha256") != sha256_file(run_dir / "run_evidence.json")
                or frozen.get("selected_best_replay_sha256")
                != sha256_file(run_dir / "selected_best_eval/eval_evidence.json")
            ):
                raise ValueError(f"screen decision evidence SHA is stale for {model}")
        add_required_file(files, decision_path)

    if through == "baseline":
        assert decision is not None
        winner = str(decision["winner"])
        baseline_dir = output_root / f"runs/baseline_{winner}_36ep"
        evidence = add_run(
            files, baseline_dir, winner, expected_project_head, 36 * 525
        )
        selected = Path(evidence["selected_best_checkpoint"]["path"]).resolve()
        if selected.parent != baseline_dir.resolve():
            raise ValueError(f"baseline selected checkpoint escaped run directory: {selected}")
        add_required_file(files, selected)
        if sha256_file(selected) != evidence["selected_best_checkpoint"]["sha256"]:
            raise ValueError("baseline selected checkpoint SHA differs from run evidence")

    for path in files:
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"artifact escaped HARDLANE_OUTPUT_ROOT: {path}") from exc
    return sorted(files), decision


def build_archive(
    output_root: Path,
    project_root: Path,
    through: str,
    archive: Path,
    force: bool,
) -> dict:
    project_head = git_head(project_root)
    files, decision = collect_paths(output_root, through, project_head)
    archive = archive.resolve()
    if archive.exists() and not force:
        raise FileExistsError(f"archive already exists; pass --force to replace: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "pass",
        "through": through,
        "project_git_head": project_head,
        "winner": decision.get("winner") if decision else None,
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    partial = archive.with_suffix(archive.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    with tarfile.open(partial, "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=f"artifacts/{path.relative_to(output_root).as_posix()}")
        info = tarfile.TarInfo("handoff_manifest.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        bundle.addfile(info, io.BytesIO(manifest_bytes))
    os.replace(partial, archive)
    result = {
        "status": "pass",
        "through": through,
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "files": len(files),
        "project_git_head": manifest["project_git_head"],
        "winner": manifest["winner"],
    }
    report = archive.with_suffix(archive.suffix + ".json")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=("gate", "screen", "baseline"), required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT").resolve()
    output_root = required_absolute_env("HARDLANE_OUTPUT_ROOT").resolve()
    result = build_archive(output_root, project_root, args.through, args.archive, args.force)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
