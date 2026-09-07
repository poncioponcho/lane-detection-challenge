#!/usr/bin/env python3
"""Select the 15-epoch backbone winner from validated run evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CLR = "clrnet_r50"
AD = "adnet_r34"
DEFAULT_CLOSE_MARGIN = 0.015


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_evidence(path: Path, expected_model: str, expected_max_iter: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "pass" or value.get("model") != expected_model:
        raise ValueError(f"invalid {expected_model} evidence: {path}")
    if int(value.get("completed_iteration", -1)) + 1 != expected_max_iter:
        raise ValueError(f"screen did not finish exactly {expected_max_iter} iterations: {path}")
    best = float(value["f1_history"]["best"]["f1"])
    final = float(value["f1_history"]["final"]["f1"])
    if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in (best, final)):
        raise ValueError(f"invalid F1 in {expected_model} evidence: {path}")

    selected_sha = value.get("selected_best_checkpoint", {}).get("sha256")
    if not isinstance(selected_sha, str) or len(selected_sha) != 64:
        raise ValueError(f"missing selected-best checkpoint SHA: {path}")
    replay_path = path.parent / "selected_best_eval/eval_evidence.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    try:
        replay_f1 = float(replay["diagnostic"]["F1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid selected-best replay evidence: {replay_path}") from exc
    if (
        replay.get("status") != "pass"
        or replay.get("model") != expected_model
        or replay.get("run_evidence_sha256") != sha256_file(path)
        or replay.get("selected_checkpoint_sha256") != selected_sha
        or not math.isclose(replay_f1, best, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            f"selected-best replay does not confirm {expected_model} history best: "
            f"{replay_path}"
        )
    return {
        "best_f1": best,
        "final_f1": final,
        "evidence": value,
        "replay_path": replay_path,
    }


def choose_winner(clr_f1: float, ad_f1: float, close_margin: float) -> tuple[str, str]:
    delta = ad_f1 - clr_f1
    if abs(delta) < close_margin:
        return CLR, f"absolute best-F1 difference is below the {close_margin:.4f} tie margin"
    if delta > 0:
        return AD, "ADNet best F1 exceeds CLRNet outside the tie margin"
    return CLR, "CLRNet best F1 exceeds ADNet outside the tie margin"


def select(clr_path: Path, ad_path: Path, output: Path, close_margin: float) -> dict:
    expected_max_iter = 15 * 525
    clr = read_evidence(clr_path, CLR, expected_max_iter)
    ad = read_evidence(ad_path, AD, expected_max_iter)
    winner, reason = choose_winner(clr["best_f1"], ad["best_f1"], close_margin)
    decision = {
        "status": "pass",
        "metric": "best validation diagnostic F1 over the fixed 15-epoch screen",
        "close_margin": close_margin,
        "winner": winner,
        "reason": reason,
        "delta_adnet_minus_clrnet": ad["best_f1"] - clr["best_f1"],
        "runs": {
            CLR: {
                "best_f1": clr["best_f1"],
                "final_f1": clr["final_f1"],
                "evidence_path": str(clr_path),
                "evidence_sha256": sha256_file(clr_path),
                "selected_best_replay_sha256": sha256_file(clr["replay_path"]),
            },
            AD: {
                "best_f1": ad["best_f1"],
                "final_f1": ad["final_f1"],
                "evidence_path": str(ad_path),
                "evidence_sha256": sha256_file(ad_path),
                "selected_best_replay_sha256": sha256_file(ad["replay_path"]),
            },
        },
        "baseline_restart_note": (
            "The 36-epoch winner run must start fresh from the adapted CULane checkpoint; "
            "do not resume the completed 15-epoch cosine schedule."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clr-evidence", required=True, type=Path)
    parser.add_argument("--ad-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--close-margin", type=float, default=DEFAULT_CLOSE_MARGIN)
    parser.add_argument("--print-winner", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.close_margin <= 1.0:
        raise SystemExit("--close-margin must be within [0,1]")
    result = select(
        args.clr_evidence, args.ad_evidence, args.output, args.close_margin
    )
    print(result["winner"] if args.print_winner else json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
