#!/usr/bin/env python3
"""AutoDL-only checkpoint/backbone compatibility probe and adapter.

Downloads all three ambiguous release assets, compares their state dictionaries
against CLRNet-R34, CLRNet-R50 and ADNet-R34, then writes head-shape-filtered
initialization checkpoints for the two HardLane configs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path


PINNED_UNLANEDET_COMMIT = "03921844220adb2e65c840de2d9759478d5c3d4c"
ASSETS = {
    "clr_generic": {
        "filename": "clrnet_model_best_culane.pth",
        "url": "https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_culane.pth",
        "bytes": 263_994_164,
    },
    "clr_named_r50": {
        "filename": "clrnet_r50_culane_model_best.pth",
        "url": "https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_r50_culane_model_best.pth",
        "bytes": 292_961_772,
    },
    "adnet_r34": {
        "filename": "adnet_model_best_culane.pth",
        "url": "https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_culane.pth",
        "bytes": 263_254_892,
    },
}


def required_absolute_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"{name} must be absolute, got {value!r}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(spec: dict, destination: Path, allow_download: bool) -> dict:
    expected_bytes = int(spec["bytes"])
    if destination.exists():
        actual_bytes = destination.stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"refusing mismatched existing asset {destination}: "
                f"{actual_bytes} bytes != {expected_bytes}"
            )
    else:
        if not allow_download:
            raise SystemExit(f"missing checkpoint under --no-download: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()
        print(f"Downloading {spec['url']} -> {destination}", flush=True)
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "hardlane-autodl-weight-probe/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        actual_bytes = partial.stat().st_size
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"downloaded size mismatch for {partial}: {actual_bytes} != {expected_bytes}"
            )
        os.replace(partial, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "url": spec["url"],
    }


def checkpoint_state(payload) -> dict:
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    else:
        state = payload
    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint has no string-keyed model state")
    for prefix in ("module.", "model."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


def compatibility_report(model_state: dict, checkpoint: dict) -> tuple[dict, dict]:
    compatible = []
    shape_mismatch = []
    filtered = {}
    for key, value in checkpoint.items():
        if key not in model_state:
            continue
        if not hasattr(value, "shape"):
            shape_mismatch.append(
                {"name": key, "checkpoint_shape": None, "model_shape": list(model_state[key].shape)}
            )
        elif tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(
                {
                    "name": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            compatible.append(key)
            filtered[key] = value
    missing = sorted(set(model_state).difference(filtered))
    unexpected = sorted(set(checkpoint).difference(model_state))
    matched_numel = sum(int(model_state[key].numel()) for key in compatible)
    model_numel = sum(int(value.numel()) for value in model_state.values())
    report = {
        "compatible": sorted(compatible),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": sorted(shape_mismatch, key=lambda item: item["name"]),
        "compatible_keys": len(compatible),
        "model_keys": len(model_state),
        "matched_numel": matched_numel,
        "model_numel": model_numel,
        "matched_numel_ratio": matched_numel / model_numel if model_numel else 0.0,
    }
    return report, filtered


def load_torch_checkpoint(torch, path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu")
    return checkpoint_state(payload)


def instantiate_model(LazyConfig, instantiate, config_path: Path):
    cfg = LazyConfig.load(str(config_path))
    cfg.model.backbone.pretrained = False
    return cfg, instantiate(cfg.model)


def inspect_model(torch, LazyConfig, instantiate, model_name, config_path, candidates):
    print(f"Instantiating {model_name} from {config_path}", flush=True)
    cfg, model = instantiate_model(LazyConfig, instantiate, config_path)
    model_state = model.state_dict()
    reports = {}
    filtered_by_asset = {}
    for asset_name, checkpoint_path in candidates.items():
        state = load_torch_checkpoint(torch, checkpoint_path)
        report, filtered = compatibility_report(model_state, state)
        reports[asset_name] = report
        filtered_by_asset[asset_name] = filtered
        print(
            f"  {asset_name}: {report['matched_numel_ratio']:.6f} matched numel, "
            f"{len(report['shape_mismatch'])} shape mismatches",
            flush=True,
        )
        del state
    return cfg, model, reports, filtered_by_asset


def save_adapted(torch, destination, source_name, source_info, model, report, filtered):
    incompatible = model.load_state_dict(filtered, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected keys survived filtering: {incompatible.unexpected_keys}")
    if sorted(incompatible.missing_keys) != sorted(report["missing"]):
        raise RuntimeError("load_state_dict missing keys differ from compatibility report")
    payload = {
        "model": filtered,
        "_hardlane_adaptation": {
            "source_asset": source_name,
            "source_sha256": source_info["sha256"],
            "source_bytes": source_info["bytes"],
            "reinitialized_parameters": report["missing"],
            "shape_mismatch": report["shape_mismatch"],
            "unexpected_source_parameters": report["unexpected"],
            "unlanedet_commit": PINNED_UNLANEDET_COMMIT,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(destination))
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **payload["_hardlane_adaptation"],
    }


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = required_absolute_env("HARDLANE_PROJECT_ROOT")
    required_absolute_env("HARDLANE_DATA_ROOT")
    unlanedet_root = required_absolute_env("UNLANEDET_ROOT")
    weights_root = required_absolute_env("HARDLANE_WEIGHTS_ROOT")
    if git_head(unlanedet_root) != PINNED_UNLANEDET_COMMIT:
        raise SystemExit("UnLanedet HEAD does not match the pinned commit")
    clr_head = unlanedet_root / "unlanedet/model/CLRNet/clr_head.py"
    source = clr_head.read_text(encoding="utf-8")
    if (
        "top_k=self.cfg.test_parameters.nms_topk" not in source
        or ".astype(bool)" not in source
        or "torch.softmax(lane[:2], dim=0)[1]" not in source
        or "score_semantics" not in source
    ):
        raise SystemExit("required HardLane UnLanedet patch is not applied")

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(unlanedet_root))
    import torch
    from unlanedet.config import LazyConfig, instantiate

    if not torch.cuda.is_available():
        raise SystemExit("this probe must run on the AutoDL CUDA instance")

    assets = {}
    asset_paths = {}
    for name, spec in ASSETS.items():
        path = weights_root / spec["filename"]
        assets[name] = download_asset(spec, path, not args.no_download)
        asset_paths[name] = path

    project_configs = project_root / "configs/unlanedet"
    model_specs = {
        "clrnet_r34": (
            unlanedet_root / "config/clrnet/resnet34_culane.py",
            {key: asset_paths[key] for key in ("clr_generic", "clr_named_r50")},
        ),
        "clrnet_r50": (
            project_configs / "clrnet_r50_hardlane.py",
            {key: asset_paths[key] for key in ("clr_generic", "clr_named_r50")},
        ),
        "adnet_r34": (
            project_configs / "adnet_r34_hardlane.py",
            {"adnet_r34": asset_paths["adnet_r34"]},
        ),
    }

    compatibility = {}
    retained = {}
    for model_name, (config_path, candidates) in model_specs.items():
        cfg, model, reports, filtered = inspect_model(
            torch, LazyConfig, instantiate, model_name, config_path, candidates
        )
        compatibility[model_name] = reports
        if model_name in {"clrnet_r50", "adnet_r34"}:
            retained[model_name] = (model, filtered)
        else:
            del model, filtered
            gc.collect()
            torch.cuda.empty_cache()
        del cfg

    clr_scores = {
        name: compatibility["clrnet_r50"][name]["matched_numel_ratio"]
        for name in ("clr_generic", "clr_named_r50")
    }
    clr_r50_asset = max(clr_scores, key=clr_scores.get)
    ordered_scores = sorted(clr_scores.values(), reverse=True)
    if ordered_scores[0] < 0.80 or ordered_scores[0] - ordered_scores[1] < 0.01:
        raise SystemExit(f"CLRNet-R50 checkpoint mapping is not decisive: {clr_scores}")
    ad_score = compatibility["adnet_r34"]["adnet_r34"]["matched_numel_ratio"]
    if ad_score < 0.80:
        raise SystemExit(f"ADNet checkpoint compatibility too low: {ad_score:.6f}")

    clr_model, clr_filtered = retained["clrnet_r50"]
    ad_model, ad_filtered = retained["adnet_r34"]
    adapted = {
        "clrnet_r50": save_adapted(
            torch,
            weights_root / "adapted_clrnet_r50_hardlane.pth",
            clr_r50_asset,
            assets[clr_r50_asset],
            clr_model,
            compatibility["clrnet_r50"][clr_r50_asset],
            clr_filtered[clr_r50_asset],
        ),
        "adnet_r34": save_adapted(
            torch,
            weights_root / "adapted_adnet_r34_hardlane.pth",
            "adnet_r34",
            assets["adnet_r34"],
            ad_model,
            compatibility["adnet_r34"]["adnet_r34"],
            ad_filtered["adnet_r34"],
        ),
    }

    report = {
        "status": "pass",
        "execution_environment": "AutoDL CUDA",
        "pinned_unlanedet_commit": PINNED_UNLANEDET_COMMIT,
        "project_git_head": git_head(project_root),
        "unlanedet_head": git_head(unlanedet_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "assets": assets,
        "compatibility": compatibility,
        "resolved_clrnet_r50_asset": clr_r50_asset,
        "adapted_checkpoints": adapted,
    }
    output = args.output or project_root / "outputs/autodl/weight_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "clrnet_r50": clr_r50_asset}))

    del retained, clr_model, ad_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
