from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts/autodl"))


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATE = _load("validate_autodl_run", "scripts/autodl/validate_run.py")
SELECT = _load("select_screen_winner", "scripts/autodl/select_screen_winner.py")
RUN = _load("run_autodl_training", "scripts/autodl/run_training.py")
COLLECT = _load("collect_autodl_results", "scripts/autodl/collect_results.py")


def test_metric_evidence_helpers_reject_inconsistent_counts():
    entries = [
        {"iteration": 524, "F1": 0.4},
        {"iteration": 1049, "F1": 0.6},
        {"iteration": 1574, "F1": 0.55},
    ]
    assert VALIDATE.summarize_f1(entries) == {
        "observations": 3,
        "best": {"iteration": 1049, "f1": 0.6},
        "final": {"iteration": 1574, "f1": 0.55},
    }
    diagnostic = {"TP": 1000, "FP": 500, "FN": 1655, "P": 1500, "G": 2655}
    diagnostic["F1"] = 2 * 1000 / (1500 + 2655)
    assert VALIDATE.validate_diagnostic(diagnostic)["TP"] == 1000
    with pytest.raises(ValueError, match="totals are inconsistent"):
        VALIDATE.validate_diagnostic({**diagnostic, "P": 1501})


@pytest.mark.parametrize(
    "clr,ad,winner",
    [
        (0.70, 0.7149, "clrnet_r50"),
        (0.70, 0.7150, "adnet_r34"),
        (0.72, 0.70, "clrnet_r50"),
    ],
)
def test_screen_winner_rule(clr, ad, winner):
    assert SELECT.choose_winner(clr, ad, 0.015)[0] == winner


def _evidence(model: str, best: float, final: float) -> dict:
    return {
        "status": "pass",
        "model": model,
        "completed_iteration": 15 * 525 - 1,
        "selected_best_checkpoint": {"sha256": "a" * 64},
        "f1_history": {
            "best": {"iteration": 100, "f1": best},
            "final": {"iteration": 15 * 525 - 1, "f1": final},
        },
    }


def test_selector_freezes_sources_and_requires_complete_screens(tmp_path):
    clr = tmp_path / "clr/run_evidence.json"
    ad = tmp_path / "ad/run_evidence.json"
    output = tmp_path / "decision.json"
    clr.parent.mkdir()
    ad.parent.mkdir()
    clr.write_text(json.dumps(_evidence("clrnet_r50", 0.70, 0.69)), encoding="utf-8")
    ad.write_text(json.dumps(_evidence("adnet_r34", 0.72, 0.71)), encoding="utf-8")
    for path, model, f1 in (
        (clr, "clrnet_r50", 0.70),
        (ad, "adnet_r34", 0.72),
    ):
        replay_dir = path.parent / "selected_best_eval"
        replay_dir.mkdir()
        (replay_dir / "eval_evidence.json").write_text(
            json.dumps({
                "status": "pass",
                "model": model,
                "run_evidence_sha256": SELECT.sha256_file(path),
                "selected_checkpoint_sha256": "a" * 64,
                "diagnostic": {"F1": f1},
            }),
            encoding="utf-8",
        )
    result = SELECT.select(clr, ad, output, 0.015)
    assert result["winner"] == "adnet_r34"
    assert len(result["runs"]["clrnet_r50"]["evidence_sha256"]) == 64
    assert len(result["runs"]["clrnet_r50"]["selected_best_replay_sha256"]) == 64
    assert "must start fresh" in result["baseline_restart_note"]

    broken = _evidence("adnet_r34", 0.72, 0.71)
    broken["completed_iteration"] -= 1
    ad.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="did not finish exactly"):
        SELECT.select(clr, ad, output, 0.015)


def test_selector_requires_matching_selected_best_replay(tmp_path):
    evidence = tmp_path / "clr/run_evidence.json"
    evidence.parent.mkdir()
    evidence.write_text(
        json.dumps(_evidence("clrnet_r50", 0.70, 0.69)), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        SELECT.read_evidence(evidence, "clrnet_r50", 15 * 525)

    replay_dir = evidence.parent / "selected_best_eval"
    replay_dir.mkdir()
    (replay_dir / "eval_evidence.json").write_text(
        json.dumps({
            "status": "pass",
            "model": "clrnet_r50",
            "run_evidence_sha256": SELECT.sha256_file(evidence),
            "selected_checkpoint_sha256": "a" * 64,
            "diagnostic": {"F1": 0.699},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not confirm"):
        SELECT.read_evidence(evidence, "clrnet_r50", 15 * 525)


class _FakeTorch:
    @staticmethod
    def load(path, map_location=None):
        assert map_location == "cpu"
        return json.loads(Path(path).read_text(encoding="utf-8"))


def _fake_checkpoint(path: Path, iteration: int) -> None:
    path.write_text(
        json.dumps({"model": {"weight": [1]}, "iteration": iteration}),
        encoding="utf-8",
    )


def test_resolve_best_checkpoint_uses_payload_iteration_not_filename(tmp_path):
    _fake_checkpoint(tmp_path / "model_best.pth", 200)
    _fake_checkpoint(tmp_path / "model_final.pth", 300)
    target = tmp_path / "model_0000100.pth"
    _fake_checkpoint(target, 100)
    selected, payload = VALIDATE.resolve_best_checkpoint(_FakeTorch, tmp_path, 100)
    assert selected == target
    assert payload["iteration"] == 100

    with pytest.raises(FileNotFoundError, match="no retained checkpoint"):
        VALIDATE.resolve_best_checkpoint(_FakeTorch, tmp_path, 99)


def test_final_metric_iteration_maps_to_previous_checkpoint_iteration():
    assert VALIDATE.checkpoint_iteration_for_metric(524, {525, 526}) == 524
    assert VALIDATE.checkpoint_iteration_for_metric(525, {525, 526}) == 524
    assert VALIDATE.checkpoint_iteration_for_metric(526, {525, 526}) == 525


def test_evidence_completion_allows_longer_resume_probe():
    value = {
        "status": "pass",
        "model": "clrnet_r50",
        "completed_iteration": 525,
        "project_git_head": "a" * 40,
    }
    class _EvidencePath:
        def read_text(self, encoding):
            return json.dumps(value)

    assert RUN.evidence_is_complete(_EvidencePath(), "clrnet_r50", 525)
    assert RUN.evidence_is_complete(_EvidencePath(), "clrnet_r50", 526)
    assert not RUN.evidence_is_complete(_EvidencePath(), "clrnet_r50", 527)
    assert RUN.evidence_is_complete(_EvidencePath(), "clrnet_r50", 526, "a" * 40)
    assert not RUN.evidence_is_complete(_EvidencePath(), "clrnet_r50", 526, "b" * 40)


def test_training_prerequisites_bind_project_and_checkpoint_sha(tmp_path):
    output = tmp_path / "output"
    weights = tmp_path / "weights"
    smoke_dir = output / "smoke"
    smoke_dir.mkdir(parents=True)
    weights.mkdir()
    checkpoint = weights / "adapted_clrnet_r50_hardlane.pth"
    checkpoint.write_bytes(b"checkpoint")
    sha = RUN.sha256_file(checkpoint)
    head = "a" * 40
    weight = {
        "status": "pass",
        "project_git_head": head,
        "pinned_unlanedet_commit": RUN.PINNED_UNLANEDET_COMMIT,
        "adapted_checkpoints": {
            "clrnet_r50": {"path": str(checkpoint), "sha256": sha}
        },
    }
    smoke = {
        "status": "pass",
        "project_git_head": head,
        "pinned_unlanedet_commit": RUN.PINNED_UNLANEDET_COMMIT,
        "models": {"clrnet_r50": {"checkpoint_sha256": sha}},
    }
    (output / "weight_probe.json").write_text(json.dumps(weight), encoding="utf-8")
    (smoke_dir / "dataloader_loss_smoke.json").write_text(
        json.dumps(smoke), encoding="utf-8"
    )
    result = RUN.validate_training_prerequisites(
        output, weights, "clrnet_r50", head
    )
    assert result["adapted_checkpoint_sha256"] == sha

    checkpoint.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="SHA differs"):
        RUN.validate_training_prerequisites(output, weights, "clrnet_r50", head)


def _fake_collected_run(output: Path, name: str, model: str) -> None:
    run = output / "runs" / name
    run.mkdir(parents=True)
    evidence = {
        "status": "pass",
        "model": model,
        "completed_iteration": 525,
        "project_git_head": "c" * 40,
        "selected_best_checkpoint": {"sha256": "b" * 64},
    }
    evidence_path = run / "run_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    for relative in COLLECT.RUN_FILES:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "run_evidence.json":
            continue
        if relative == "selected_best_eval/eval_evidence.json":
            path.write_text(json.dumps({
                "status": "pass",
                "model": model,
                "project_git_head": "c" * 40,
                "run_evidence_sha256": COLLECT.sha256_file(evidence_path),
                "selected_checkpoint_sha256": "b" * 64,
                "validation_images": 800,
            }), encoding="utf-8")
        else:
            path.write_text("evidence\n", encoding="utf-8")
    predictions = run / "selected_best_eval/val/predictions/clip"
    predictions.mkdir(parents=True)
    for index in range(800):
        (predictions / f"{index:05d}.lines.txt").write_text("", encoding="utf-8")


def test_collect_gate_builds_checked_handoff_archive(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "weight_probe.json").write_text(
        json.dumps({"status": "pass", "project_git_head": "c" * 40}),
        encoding="utf-8",
    )
    smoke = output / "smoke"
    smoke.mkdir()
    (smoke / "dataloader_loss_smoke.json").write_text(
        json.dumps({"status": "pass", "project_git_head": "c" * 40}),
        encoding="utf-8",
    )
    for model in ("clrnet_r50", "adnet_r34"):
        _fake_collected_run(output, f"gate_{model}_1ep", model)
    monkeypatch.setattr(COLLECT, "git_head", lambda _path: "c" * 40)
    archive = tmp_path / "handoff.tar.gz"
    result = COLLECT.build_archive(output, ROOT, "gate", archive, False)
    assert result["status"] == "pass"
    assert result["files"] > 1600
    assert archive.is_file()
    assert archive.with_suffix(".gz.json").is_file()


def test_run_training_dry_run_is_non_mutating_and_pins_seed(tmp_path):
    project = tmp_path / "project"
    unlanedet = tmp_path / "UnLanedet"
    data = tmp_path / "data"
    weights = tmp_path / "weights"
    output = tmp_path / "output"
    config = project / "configs/unlanedet/clrnet_r50_hardlane.py"
    train_net = unlanedet / "tools/train_net.py"
    config.parent.mkdir(parents=True)
    train_net.parent.mkdir(parents=True)
    config.write_text("# fake\n", encoding="utf-8")
    train_net.write_text("# fake\n", encoding="utf-8")
    for directory in (data, weights, output):
        directory.mkdir()
    env = {
        **os.environ,
        "HARDLANE_PROJECT_ROOT": str(project),
        "HARDLANE_DATA_ROOT": str(data),
        "UNLANEDET_ROOT": str(unlanedet),
        "HARDLANE_WEIGHTS_ROOT": str(weights),
        "HARDLANE_OUTPUT_ROOT": str(output),
        "HARDLANE_PYTHON": sys.executable,
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/autodl/run_training.py"),
            "--model", "clrnet_r50",
            "--epochs", "15",
            "--run-name", "screen_clrnet_r50_15ep",
            "--dry-run",
        ],
        env=env,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    launch = json.loads(proc.stdout)
    assert launch["target_max_iter"] == 7875
    assert "train.seed=42" in launch["command"]
    assert "train.cudnn_benchmark=False" in launch["command"]
    assert "train.checkpointer.max_to_keep=40" in launch["command"]
    assert not (output / "runs").exists()
