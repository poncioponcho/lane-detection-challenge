#!/usr/bin/env python
"""Iteration planner for the overnight optimization loop (§36 tracks).

Bounded, rule-based auto-adjustment — NOT unconstrained HPO.  Given the
tracked score history and the screen results on the instance, decide the
next iteration:

  1. consensus   — T2-A 3-seed consensus fusion evidence (guarded: needs
                   consensus_inputs.json describing seed prediction dirs +
                   sidecars; skipped with a note when absent);
  2. train       — 36-epoch scale-up of the best 15-epoch screen variant
                   when it beats the incumbent baseline by >0.002 F1, or
                   extra seeds (43/44) of the best variant to strengthen
                   the consensus pool;
  3. stop        — nothing worth launching (or past the training freeze).

Training launches are blocked after --freeze-train-epoch and skipped when
the projected wall-clock (epochs x iters x 0.45s + 15% eval overhead)
would overflow the remaining budget.  Output: a JSON iteration spec on
stdout; markers under the state root prevent duplicate work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_tracker import parse_metrics  # noqa: E402

SECONDS_PER_ITER = 0.45
EVAL_OVERHEAD = 1.15

# name -> (model, extra train args) for 36-epoch scale-ups
SCALE_UP = {
    "vat2000_clrnet_r50_15ep": (["--model", "clrnet_r50_vat"], "vat2000_clrnet_r50_36ep"),
    "os_clrnet_r50_15ep": (["--model", "clrnet_r50", "--iters-per-epoch", "525"], "os_clrnet_r50_36ep"),
    "vat2000_os_clrnet_r50_15ep": (["--model", "clrnet_r50_vat", "--iters-per-epoch", "525"], "vat2000_os_clrnet_r50_36ep"),
    "clrernet_r50_15ep": (["--model", "clrernet_r50"], "clrernet_r50_36ep"),
}
# extra args needed when the os manifest must be attached at 36ep
OS_MANIFEST_ARG = ["--train-manifest", "/hy-tmp/lane-outputs/experiments_manifest_train_os_v1_seed42_hard25_x3.jsonl"]

CANDIDATES = list(SCALE_UP)
INCUMBENT_NAME = "screen_clrnet_r50_15ep"
BEAT_MARGIN = 0.002


def load_history_best(history: Path) -> tuple[float | None, set]:
    best = None
    scaled = set()
    if not history.is_file():
        return None, scaled
    for line in history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("best_f1") is not None and (best is None or entry["best_f1"] > best):
            best = entry["best_f1"]
        for marker in entry.get("markers", []):
            scaled.add(marker)
    return best, scaled


def screen_f1(runs_root: Path, name: str) -> float | None:
    parsed = parse_metrics(runs_root / name / "metrics.json")
    if parsed is None:
        return None
    if parsed["final_iter"] < 7875:  # 15 epochs x 525 iters — screen incomplete
        return None
    return parsed["best"]


def screens_pending(runs_root: Path) -> bool:
    """True while any phase-2/T3 screen is still mid-training."""
    for name in CANDIDATES + [INCUMBENT_NAME]:
        parsed = parse_metrics(runs_root / name / "metrics.json")
        if parsed is not None and parsed["final_iter"] < 7875:
            return True
    return False


def projected_seconds(epochs: int, iters_per_epoch: int) -> float:
    return epochs * iters_per_epoch * SECONDS_PER_ITER * EVAL_OVERHEAD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--freeze-train-epoch", type=int, required=True,
                        help="unix seconds; no new training launched after this")
    parser.add_argument("--max-train-minutes", type=float, default=300.0,
                        help="single-iteration training wall-clock budget")
    args = parser.parse_args()

    now = time.time()
    markers_dir = args.state_root / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)

    def done(marker: str) -> bool:
        return (markers_dir / marker).exists()

    # 1. consensus evidence (T2-A) — guarded by an inputs manifest that the
    #    interactive session fills in once seed prediction paths are mapped.
    inputs = args.state_root / "consensus_inputs.json"
    if inputs.is_file() and not done("consensus"):
        return _emit({"type": "consensus", "inputs": str(inputs), "slug": "consensus_t2"})

    # 0. never decide on half-finished screens.
    if screens_pending(args.runs_root):
        return _emit({"type": "wait", "reason": "phase-2/T3 screens still training"})

    # 2. pick the best 15ep screen variant vs the incumbent baseline.
    baseline = screen_f1(args.runs_root, INCUMBENT_NAME) or 0.7840
    scored = []
    for name in CANDIDATES:
        f1 = screen_f1(args.runs_root, name)
        if f1 is not None:
            scored.append((f1, name))
    scored.sort(reverse=True)

    if scored:
        best_f1, best_name = scored[0]
        if best_f1 > baseline + BEAT_MARGIN:
            # 2a. 36-epoch scale-up of the winner.
            if not done(f"scaled_{best_name}") and now < args.freeze_train_epoch:
                model_args, run_name = SCALE_UP[best_name]
                extra = list(model_args)
                if "os" in best_name and "--iters-per-epoch" in extra:
                    extra += OS_MANIFEST_ARG
                seconds = projected_seconds(36, 525)
                if seconds <= args.max_train_minutes * 60:
                    return _emit({
                        "type": "train", "slug": run_name, "run_name": run_name,
                        "epochs": 36, "seed": 42, "extra_args": extra,
                        "marker": f"scaled_{best_name}",
                        "rationale": f"{best_name} best {best_f1:.5f} beats baseline {baseline:.5f}; scaling to 36ep",
                    })
                return _emit({"type": "stop",
                              "reason": f"scale-up of {best_name} projected {seconds/60:.0f}min exceeds budget"})
            # 2b. strengthen the consensus pool with extra seeds of the winner.
            for seed in (43, 44):
                marker = f"seeded_{best_name}_{seed}"
                if not done(marker) and now < args.freeze_train_epoch:
                    model_args, _ = SCALE_UP[best_name]
                    model = model_args[1] if model_args[0] == "--model" else "clrnet_r50"
                    extra = []
                    if "os" in best_name:
                        extra = ["--iters-per-epoch", "525"] + OS_MANIFEST_ARG
                    if model == "clrnet_r50_vat":
                        extra = [a for a in extra if a != "--model"]
                    return _emit({
                        "type": "train", "slug": f"{best_name}_seed{seed}",
                        "run_name": f"{best_name}_seed{seed}", "epochs": 15,
                        "seed": seed,
                        "extra_args": (["--model", model] + extra),
                        "marker": marker,
                        "rationale": f"extra seed {seed} of winner {best_name} for consensus",
                    })

    return _emit({"type": "stop",
                  "reason": "no screen variant beats the baseline (or everything already scaled)"})


def _emit(spec: dict) -> None:
    print(json.dumps(spec))


if __name__ == "__main__":
    main()
