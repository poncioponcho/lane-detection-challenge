#!/usr/bin/env python3
"""Run one or all deterministic leave-one-video-out CLRNet folds.

This runner deliberately lives outside the normal pipeline contract: it uses
derived manifests, a fixed final checkpoint per fold, and a separate output
tree.  It never edits the frozen v1 manifests or the checked-in config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_set(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.lines.txt")
        if path.is_file()
    }


def prediction_tree_sha256(root: Path, rels: set[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(rels):
        path = root / rel
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts_utc": utc_now(), **event},
                                ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def video_id(clip_id: str) -> str:
    marker = "_1_0_"
    if marker not in clip_id:
        raise ValueError(f"invalid clip id without {marker!r}: {clip_id}")
    value = clip_id.split(marker, 1)[0]
    if not re.fullmatch(r"v[0-9]+", value):
        raise ValueError(f"invalid video id {value!r} from {clip_id!r}")
    return value


def import_project_modules(project_root: Path):
    sys.path.insert(0, str(project_root / "scripts/autodl"))
    sys.path.insert(0, str(project_root / "src"))
    from data.manifest import read_manifest
    from run_training import (
        PINNED_UNLANEDET_COMMIT,
        assert_tracked_worktree_clean,
        git_head,
        inspect_cuda_python,
        override,
        parse_experiment_overrides,
        required_absolute_env,
        stream_command,
    )
    from validate_run import checkpoint_payload, manifest_prediction_paths
    return {
        "read_manifest": read_manifest,
        "PINNED_UNLANEDET_COMMIT": PINNED_UNLANEDET_COMMIT,
        "assert_tracked_worktree_clean": assert_tracked_worktree_clean,
        "git_head": git_head,
        "inspect_cuda_python": inspect_cuda_python,
        "override": override,
        "parse_experiment_overrides": parse_experiment_overrides,
        "required_absolute_env": required_absolute_env,
        "stream_command": stream_command,
        "checkpoint_payload": checkpoint_payload,
        "manifest_prediction_paths": manifest_prediction_paths,
    }


class LVORunner:
    def __init__(self, args: argparse.Namespace):
        self.project_root = args.project_root.resolve()
        self.data_root = args.data_root.resolve()
        self.unlanedet_root = args.unlanedet_root.resolve()
        self.weights_root = args.weights_root.resolve()
        self.output_root = args.output_root.resolve()
        self.experiment_root = args.experiment_root.resolve()
        self.manifests_root = args.manifests_root.resolve()
        self.python_bin = args.python_bin
        self.resume = args.resume
        self.skip_complete = args.skip_complete
        self.mods = import_project_modules(self.project_root)
        self.read_manifest = self.mods["read_manifest"]
        self.override = self.mods["override"]
        self.experiment_overrides = self.mods["parse_experiment_overrides"](
            args.overrides
        )
        self.stream_command = self.mods["stream_command"]
        self.checkpoint_payload = self.mods["checkpoint_payload"]
        self.manifest_prediction_paths = self.mods["manifest_prediction_paths"]
        self.project_head = ""
        self.unlanedet_head = ""
        self.environment: dict = {}
        self.base_config = (
            args.base_config.resolve()
            if args.base_config is not None
            else self.project_root / "configs/unlanedet/clrnet_r50_hardlane.py"
        )
        self.input_width = args.input_width
        self.input_height = args.input_height
        self.cut_height = args.cut_height
        self.eval_workers = args.eval_workers
        self.checkpoint_max_to_keep = args.checkpoint_max_to_keep
        self.iterations_per_epoch_override = args.iterations_per_epoch
        self.train_net = self.unlanedet_root / "tools/train_net.py"
        self.adapted_checkpoint = self.weights_root / "adapted_clrnet_r50_hardlane.pth"
        self.events_path = self.experiment_root / "runner_events.jsonl"

    def prepare(self) -> None:
        for path in (self.project_root, self.data_root, self.unlanedet_root,
                     self.weights_root, self.output_root, self.experiment_root,
                     self.manifests_root, self.base_config, self.train_net,
                     self.adapted_checkpoint):
            if not path.exists():
                raise SystemExit(f"required LVO path is missing: {path}")
        self.mods["assert_tracked_worktree_clean"](self.project_root)
        self.project_head = self.mods["git_head"](self.project_root)
        self.unlanedet_head = self.mods["git_head"](self.unlanedet_root)
        if self.unlanedet_head != self.mods["PINNED_UNLANEDET_COMMIT"]:
            raise SystemExit(
                f"UnLanedet HEAD {self.unlanedet_head} != pinned "
                f"{self.mods['PINNED_UNLANEDET_COMMIT']}"
            )
        self.environment = self.mods["inspect_cuda_python"](self.python_bin)
        config_text = self.base_config.read_text(encoding="utf-8")
        required_config_fragments = (
            f"img_w = {self.input_width}",
            f"img_h = {self.input_height}",
            f"cut_height = {self.cut_height}",
            "manifest_train_v1_seed42.jsonl",
            "manifest_val_v1_seed42.jsonl",
        )
        missing = [needle for needle in required_config_fragments
                   if needle not in config_text]
        if missing:
            raise SystemExit(f"base config is not the frozen 800x320/cut180 config: {missing}")
        # train_net and the pinned UnLanedet config resolve a few imports
        # relative to the UnLanedet checkout, matching run_pipeline.sh.
        os.chdir(self.unlanedet_root)
        folds = self.discover_folds()
        source = self.manifests_root / "source_manifest_train.jsonl"
        if not source.is_file():
            raise SystemExit(f"source manifest missing: {source}")
        source_records = self.read_manifest(source)
        source_rows = len(source_records)
        if source_rows <= 0:
            raise SystemExit("source manifest is empty")
        source_ids = {record.image_id for record in source_records}
        all_holdout_ids: list[str] = []
        fold_evidence = []
        for fold in folds:
            train_records = self.read_manifest(fold["train_manifest"])
            holdout_records = self.read_manifest(fold["holdout_manifest"])
            train_ids = {record.image_id for record in train_records}
            holdout_ids = {record.image_id for record in holdout_records}
            train_videos = {video_id(record.clip_id) for record in train_records}
            holdout_videos = {video_id(record.clip_id) for record in holdout_records}
            if train_ids & holdout_ids:
                raise SystemExit(f"{fold['name']}: train/holdout image overlap")
            if train_ids | holdout_ids != source_ids:
                raise SystemExit(f"{fold['name']}: fold does not cover source image set")
            if holdout_videos != {fold["video"]}:
                raise SystemExit(f"{fold['name']}: holdout is not exactly one video")
            all_holdout_ids.extend(record.image_id for record in holdout_records)
            fold_evidence.append({
                "fold": fold["index"],
                "name": fold["name"],
                "video": fold["video"],
                "train_manifest": {
                    "path": str(fold["train_manifest"]),
                    "sha256": sha256_file(fold["train_manifest"]),
                    "rows": len(train_records),
                    "clips": len({record.clip_id for record in train_records}),
                },
                "holdout_manifest": {
                    "path": str(fold["holdout_manifest"]),
                    "sha256": sha256_file(fold["holdout_manifest"]),
                    "rows": len(holdout_records),
                    "clips": len({record.clip_id for record in holdout_records}),
                },
                "train_videos": sorted(train_videos),
                "holdout_videos": sorted(holdout_videos),
            })
        if (len(all_holdout_ids) != source_rows
                or len(set(all_holdout_ids)) != source_rows):
            raise SystemExit("LVO holdout union is not a unique cover of the source rows")
        if set(all_holdout_ids) != source_ids:
            raise SystemExit("LVO holdout union differs from source manifest")
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        protocol = {
            "status": "pass",
            "protocol": "leave-one-video-out",
            "model": "clrnet_r50",
            "experiment_overrides": self.experiment_overrides,
            "epochs": 15,
            "batch_size": 12,
            "input": f"{self.input_width}x{self.input_height}",
            "input_width": self.input_width,
            "input_height": self.input_height,
            "cut_height": self.cut_height,
            "eval_workers": self.eval_workers,
            "conf_threshold": 0.4,
            "checkpoint_policy": "fixed model_final after exactly 15 epochs; no holdout selection",
            "project_git_head": self.project_head,
            "unlanedet_git_head": self.unlanedet_head,
            "cuda_environment": self.environment,
            "base_config": {
                "path": str(self.base_config),
                "sha256": sha256_file(self.base_config),
            },
            "adapted_checkpoint": {
                "path": str(self.adapted_checkpoint),
                "sha256": sha256_file(self.adapted_checkpoint),
                "bytes": self.adapted_checkpoint.stat().st_size,
            },
            "source_manifest": {
                "path": str(source),
                "sha256": sha256_file(source),
                "rows": len(source_records),
                "clips": len({record.clip_id for record in source_records}),
                "videos": sorted({video_id(record.clip_id) for record in source_records}),
            },
            "folds": fold_evidence,
            "coverage": {
                "source_rows": source_rows,
                "holdout_rows": len(all_holdout_ids),
                "unique_holdout_rows": len(set(all_holdout_ids)),
                "each_image_held_out_once": True,
            },
            "created_utc": utc_now(),
        }
        write_json(self.experiment_root / "protocol_remote.json", protocol)
        append_event(self.events_path, {
            "event": "protocol_preflight_pass",
            "project_git_head": self.project_head,
            "unlanedet_git_head": self.unlanedet_head,
            "folds": len(folds),
            "source_rows": len(source_records),
        })

    def discover_folds(self) -> list[dict]:
        dirs = sorted(self.manifests_root.glob("fold_*_v*"))
        if len(dirs) != 8:
            raise SystemExit(f"expected 8 LVO fold directories, found {len(dirs)}")
        result = []
        for directory in dirs:
            match = re.fullmatch(r"fold_([0-9]{2})_(v[0-9]+)", directory.name)
            if not match:
                raise SystemExit(f"invalid LVO fold directory name: {directory.name}")
            index, heldout_video = int(match.group(1)), match.group(2)
            train_manifest = directory / "manifest_train.jsonl"
            holdout_manifest = directory / "manifest_holdout.jsonl"
            if not train_manifest.is_file() or not holdout_manifest.is_file():
                raise SystemExit(f"incomplete fold manifests: {directory}")
            result.append({
                "index": index,
                "name": directory.name,
                "video": heldout_video,
                "directory": directory,
                "train_manifest": train_manifest,
                "holdout_manifest": holdout_manifest,
            })
        if [item["index"] for item in result] != list(range(8)):
            raise SystemExit("LVO folds must be numbered 00 through 07")
        return result

    def fold_run_dir(self, fold: dict, smoke: bool) -> Path:
        parent = self.experiment_root / ("smoke" if smoke else "runs")
        return parent / fold["name"]

    def load_fold_records(self, fold: dict):
        train_records = self.read_manifest(fold["train_manifest"])
        holdout_records = self.read_manifest(fold["holdout_manifest"])
        return train_records, holdout_records

    def checkpoint_info(self, checkpoint: Path, target_iter: int) -> dict:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        import torch
        payload = self.checkpoint_payload(torch, checkpoint)
        top_iteration = int(payload.get("iteration", -1))
        trainer = payload.get("trainer")
        trainer_iteration = int(trainer.get("iteration", -1)) if isinstance(trainer, dict) else -1
        if top_iteration != target_iter - 1 or trainer_iteration != target_iter - 1:
            raise ValueError(
                f"checkpoint iteration mismatch for {checkpoint}: "
                f"top={top_iteration}, trainer={trainer_iteration}, expected={target_iter - 1}"
            )
        if not isinstance(trainer, dict) or not isinstance(trainer.get("optimizer"), dict):
            raise ValueError(f"checkpoint has no optimizer state: {checkpoint}")
        return {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "payload_iteration": top_iteration,
            "trainer_iteration": trainer_iteration,
        }

    def validate_diagnostic(self, path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {"F1", "TP", "FP", "FN", "P", "G"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"diagnostic metric missing {sorted(missing)}: {path}")
        tp, fp, fn = int(value["TP"]), int(value["FP"]), int(value["FN"])
        predicted, gt = int(value["P"]), int(value["G"])
        f1 = float(value["F1"])
        if min(tp, fp, fn, predicted, gt) < 0:
            raise ValueError(f"negative diagnostic count: {path}")
        if predicted != tp + fp or gt != tp + fn:
            raise ValueError(f"inconsistent diagnostic counts: {path}")
        expected_f1 = 2.0 * tp / (predicted + gt) if predicted + gt else 0.0
        if not math.isfinite(f1) or not math.isclose(f1, expected_f1, abs_tol=1e-12):
            raise ValueError(f"diagnostic F1 does not match counts: {path}")
        return {"F1": f1, "TP": tp, "FP": fp, "FN": fn, "P": predicted, "G": gt}

    def train_command(self, run_dir: Path, fold: dict, target_iter: int,
                      iter_per_epoch: int, resume: bool) -> list[str]:
        command = [
            self.python_bin, str(self.train_net),
            "--config-file", str(self.base_config),
            "--num-gpus", "1",
        ]
        if resume:
            command.append("--resume")
        command.extend([
            *self.experiment_overrides,
            self.override("train.max_iter", target_iter),
            self.override("train.eval_period", 0),
            self.override("train.checkpointer.period", iter_per_epoch),
            self.override("train.checkpointer.max_to_keep", self.checkpoint_max_to_keep),
            self.override("train.output_dir", run_dir),
            self.override("dataloader.evaluator.output_basedir", run_dir / "train_eval"),
            self.override("dataloader.train.dataset.manifest_path", fold["train_manifest"]),
            self.override("dataloader.train.dataset.split", "train"),
            self.override("dataloader.test.dataset.manifest_path", fold["holdout_manifest"]),
            self.override("dataloader.test.dataset.split", "val"),
            self.override("model.head.cfg.test_parameters.conf_threshold", 0.4),
            self.override("dataloader.test.num_workers", self.eval_workers),
            self.override("dataloader.test.persistent_workers", self.eval_workers > 0),
            "train.seed=42",
            "train.cudnn_benchmark=False",
        ])
        return command

    def eval_command(self, eval_dir: Path, fold: dict, checkpoint: Path) -> list[str]:
        return [
            self.python_bin, str(self.train_net), "--eval-only",
            "--config-file", str(self.base_config), "--num-gpus", "1",
            *self.experiment_overrides,
            self.override("train.init_checkpoint", checkpoint),
            self.override("train.output_dir", eval_dir),
            self.override("dataloader.evaluator.output_basedir", eval_dir / "val"),
            self.override("dataloader.test.dataset.manifest_path", fold["holdout_manifest"]),
            self.override("dataloader.test.dataset.split", "val"),
            self.override("model.head.cfg.test_parameters.conf_threshold", 0.4),
            self.override("dataloader.test.num_workers", self.eval_workers),
            self.override("dataloader.test.persistent_workers", self.eval_workers > 0),
            "train.seed=42", "train.cudnn_benchmark=False",
        ]

    def run_fold(self, fold: dict, *, smoke: bool = False) -> dict:
        train_records, holdout_records = self.load_fold_records(fold)
        batch_size = 12
        iter_per_epoch = (
            self.iterations_per_epoch_override
            if self.iterations_per_epoch_override is not None
            else len(train_records) // batch_size
        )
        if iter_per_epoch <= 0:
            raise ValueError(f"{fold['name']}: no complete training batch")
        target_iter = 2 if smoke else iter_per_epoch * 15
        run_dir = self.fold_run_dir(fold, smoke)
        state_path = run_dir / "fold_state.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        if state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("experiment_overrides", []) != self.experiment_overrides:
                raise SystemExit(
                    f"{fold['name']}: existing fold uses different experiment overrides: "
                    f"expected {self.experiment_overrides}, "
                    f"found {existing.get('experiment_overrides', [])}"
                )
        if self.skip_complete and state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("status") == "pass":
                append_event(self.events_path, {"event": "fold_skip_complete",
                                                "fold": fold["name"], "smoke": smoke})
                return existing
        checkpoint = run_dir / "model_final.pth"
        if checkpoint.exists() and not state_path.is_file() and not self.resume:
            # A complete-looking directory is validated below; an incomplete
            # directory still requires an explicit --resume and is never
            # overwritten.
            try:
                checkpoint_meta = self.checkpoint_info(checkpoint, target_iter)
            except (OSError, RuntimeError, ValueError, KeyError):
                raise SystemExit(
                    f"non-empty incomplete fold directory requires --resume: {run_dir}"
                )
        else:
            checkpoint_meta = None
        meta = {
            "status": "running",
            "fold": fold["index"],
            "fold_name": fold["name"],
            "heldout_video": fold["video"],
            "smoke": smoke,
            "model": "clrnet_r50",
            "experiment_overrides": self.experiment_overrides,
            "epochs": 15,
            "effective_epochs": "smoke-2-iterations" if smoke else 15,
            "batch_size": batch_size,
            "train_rows": len(train_records),
            "holdout_rows": len(holdout_records),
            "train_clips": len({record.clip_id for record in train_records}),
            "holdout_clips": len({record.clip_id for record in holdout_records}),
            "iterations_per_epoch": iter_per_epoch,
            "target_max_iter": target_iter,
            "input": f"{self.input_width}x{self.input_height}",
            "input_width": self.input_width,
            "input_height": self.input_height,
            "cut_height": self.cut_height,
            "eval_workers": self.eval_workers,
            "conf_threshold": 0.4,
            "checkpoint_policy": "fixed model_final; no holdout selection",
            "checkpoint_max_to_keep": self.checkpoint_max_to_keep,
            "project_git_head": self.project_head,
            "unlanedet_git_head": self.unlanedet_head,
            "started_utc": utc_now(),
        }
        write_json(state_path, meta)
        append_event(self.events_path, {"event": "fold_start", "fold": fold["name"],
                                        "smoke": smoke, "target_max_iter": target_iter})
        try:
            if checkpoint_meta is None:
                resume = self.resume and any(run_dir.iterdir())
                train_cmd = self.train_command(run_dir, fold, target_iter,
                                               iter_per_epoch, resume)
                write_json(run_dir / "train_command.json", {
                    "command": train_cmd, "resume": resume,
                    "project_git_head": self.project_head,
                    "created_utc": utc_now(),
                })
                return_code = self.stream_command(train_cmd, run_dir / "train.log")
                if return_code:
                    raise RuntimeError(f"training exit code {return_code}: {run_dir}")
                checkpoint = run_dir / "model_final.pth"
                checkpoint_meta = self.checkpoint_info(checkpoint, target_iter)
            meta["training_checkpoint"] = checkpoint_meta
            meta["train_command_sha256"] = sha256_file(run_dir / "train_command.json")
            if self.mods["git_head"](self.project_root) != self.project_head:
                raise RuntimeError("project HEAD changed during LVO fold")

            eval_dir = run_dir / "holdout_eval"
            eval_cmd = self.eval_command(eval_dir, fold, checkpoint)
            write_json(eval_dir / "eval_command.json", {
                "command": eval_cmd, "project_git_head": self.project_head,
                "created_utc": utc_now(),
            })
            return_code = self.stream_command(eval_cmd, eval_dir / "eval.log")
            if return_code:
                raise RuntimeError(f"holdout eval exit code {return_code}: {eval_dir}")
            pred_root = eval_dir / "val" / "predictions"
            expected = self.manifest_prediction_paths(
                fold["holdout_manifest"], expected_count=len(holdout_records)
            )
            actual = prediction_set(pred_root)
            if actual != expected:
                raise ValueError(
                    f"{fold['name']}: prediction set mismatch; "
                    f"missing={len(expected - actual)} extra={len(actual - expected)}"
                )
            diagnostic_path = eval_dir / "val" / "diagnostic_metric.json"
            diagnostic = self.validate_diagnostic(diagnostic_path)
            meta.update({
                "status": "pass",
                "finished_utc": utc_now(),
                "holdout_eval": {
                    "directory": str(eval_dir),
                    "diagnostic": diagnostic,
                    "diagnostic_sha256": sha256_file(diagnostic_path),
                    "eval_command_sha256": sha256_file(eval_dir / "eval_command.json"),
                    "eval_log_sha256": sha256_file(eval_dir / "eval.log"),
                    "prediction_count": len(actual),
                    "prediction_root": str(pred_root),
                    "prediction_tree_sha256": prediction_tree_sha256(pred_root, actual),
                },
            })
            write_json(state_path, meta)
            append_event(self.events_path, {"event": "fold_pass", "fold": fold["name"],
                                            "smoke": smoke, "f1": diagnostic["F1"]})
            return meta
        except Exception as exc:
            meta.update({"status": "failed", "failed_utc": utc_now(),
                         "error_type": type(exc).__name__, "error": str(exc)})
            write_json(state_path, meta)
            append_event(self.events_path, {"event": "fold_failed", "fold": fold["name"],
                                            "smoke": smoke, "error": str(exc)})
            raise

    def aggregate_oof(self, folds: list[dict]) -> dict:
        oof_root = self.experiment_root / "oof"
        pred_root = oof_root / "predictions"
        pred_root.mkdir(parents=True, exist_ok=True)
        source = self.manifests_root / "source_manifest_train.jsonl"
        source_records = self.read_manifest(source)
        source_by_id = {record.image_id: record for record in source_records}
        index = {}
        for fold in folds:
            state_path = self.fold_run_dir(fold, False) / "fold_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "pass":
                raise SystemExit(f"cannot aggregate incomplete fold: {fold['name']}")
            pred_base = Path(state["holdout_eval"]["prediction_root"])
            holdout_records = self.read_manifest(fold["holdout_manifest"])
            for record in holdout_records:
                rel = record.pred_rel_path
                source_path = pred_base / rel
                destination = pred_root / rel
                if not source_path.is_file():
                    raise FileNotFoundError(f"missing fold prediction: {source_path}")
                if destination.exists():
                    if sha256_file(destination) != sha256_file(source_path):
                        raise ValueError(f"conflicting existing OOF prediction: {destination}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination)
                if record.image_id in index:
                    raise ValueError(f"duplicate OOF image: {record.image_id}")
                index[record.image_id] = {
                    "image_id": record.image_id,
                    "pred_rel_path": rel,
                    "video": video_id(record.clip_id),
                    "fold": fold["index"],
                    "fold_name": fold["name"],
                    "sha256": sha256_file(destination),
                }
        actual = prediction_set(pred_root)
        expected = {record.pred_rel_path for record in source_records}
        if actual != expected or set(index) != set(source_by_id):
            raise ValueError(
                f"OOF coverage mismatch: predictions={len(actual)} expected={len(expected)} "
                f"images={len(index)} source={len(source_by_id)}"
            )
        evidence = {
            "status": "pass",
            "protocol": "leave-one-video-out",
            "source_manifest_sha256": sha256_file(source),
            "source_rows": len(source_records),
            "prediction_count": len(actual),
            "unique_image_count": len(index),
            "each_image_once": True,
            "prediction_root": str(pred_root),
            "prediction_tree_sha256": prediction_tree_sha256(pred_root, actual),
            "index": index,
            "created_utc": utc_now(),
        }
        write_json(oof_root / "oof_evidence.json", evidence)
        append_event(self.events_path, {"event": "oof_aggregate_pass",
                                        "prediction_count": len(actual),
                                        "tree_sha256": evidence["prediction_tree_sha256"]})
        return evidence

    def run_all(self) -> None:
        self.prepare()
        folds = self.discover_folds()
        for fold in folds:
            self.run_fold(fold, smoke=False)
        self.aggregate_oof(folds)
        source_records = self.read_manifest(
            self.manifests_root / "source_manifest_train.jsonl"
        )
        write_json(self.experiment_root / "lvo_training_complete.json", {
            "status": "pass", "folds": len(folds),
            "prediction_count": len(source_records),
            "project_git_head": self.project_head, "completed_utc": utc_now(),
        })

    def run_smoke(self) -> None:
        self.prepare()
        fold = self.discover_folds()[0]
        self.run_fold(fold, smoke=True)
        write_json(self.experiment_root / "lvo_smoke_complete.json", {
            "status": "pass", "fold": fold["name"], "completed_utc": utc_now(),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "all"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--unlanedet-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="optional derived config; defaults to the frozen 800x320/cut180 config",
    )
    parser.add_argument("--input-width", type=int, default=800)
    parser.add_argument("--input-height", type=int, default=320)
    parser.add_argument("--cut-height", type=int, default=180)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="validation DataLoader workers; 0 disables multiprocessing",
    )
    parser.add_argument(
        "--checkpoint-max-to-keep",
        type=int,
        default=40,
        help="periodic checkpoints retained per fold; model_final is retained separately",
    )
    parser.add_argument(
        "--iterations-per-epoch",
        type=int,
        default=None,
        help="optional fixed training budget per epoch, independent of manifest row count",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="single-variable training override; may be repeated",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    if args.input_width <= 0 or args.input_height <= 0 or args.cut_height < 0:
        raise SystemExit("input dimensions must be positive and cut-height non-negative")
    if args.eval_workers < 0:
        raise SystemExit("--eval-workers must be non-negative")
    if args.checkpoint_max_to_keep <= 0:
        raise SystemExit("--checkpoint-max-to-keep must be positive")
    if args.iterations_per_epoch is not None and args.iterations_per_epoch <= 0:
        raise SystemExit("--iterations-per-epoch must be positive")
    runner = LVORunner(args)
    if args.mode == "smoke":
        runner.run_smoke()
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
