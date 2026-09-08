#!/usr/bin/env python
"""Score tracking, comparison and optimization-report generation.

Reads runs/<name>/metrics.json (JSONL with F1/iteration entries), keeps an
append-only score history (JSONL) and renders a per-iteration markdown
report with deltas vs the incumbent baseline and vs the previous best.

Usage:
  score_tracker.py --runs-root /hy-tmp/lane-outputs/runs \
      --names screen_clrnet_r50_15ep,vat2000_clrnet_r50_15ep \
      --history /hy-tmp/lane-outputs/overnight/score_history.jsonl \
      --report /hy-tmp/lane-outputs/overnight/reports/iter_1.md \
      --context optimizer --incumbent 0.7840
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_metrics(metrics_path: Path) -> dict | None:
    best = final = None
    if not metrics_path.is_file():
        return None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "F1" not in entry:
            continue
        f1 = float(entry["F1"])
        iteration = int(entry.get("iteration", 0))
        if final is None or iteration >= final[1]:
            final = (f1, iteration)
        if best is None or f1 > best[0]:
            best = (f1, iteration)
    if best is None:
        return None
    return {"best": best[0], "best_iter": best[1], "final": final[0], "final_iter": final[1]}


def prev_best(history_path: Path) -> float | None:
    best = None
    if not history_path.is_file():
        return None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("best_f1") is not None:
            if best is None or entry["best_f1"] > best:
                best = entry["best_f1"]
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--names", required=True,
                        help="comma-separated run names under runs-root")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--context", default="optimizer")
    parser.add_argument("--incumbent", type=float, default=0.7840,
                        help="baseline F1 to compare against (screen_clrnet_r50_15ep)")
    parser.add_argument("--notes", default="",
                        help="free-text note embedded into the report")
    args = parser.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    runs = {}
    for name in names:
        parsed = parse_metrics(args.runs_root / name / "metrics.json")
        if parsed:
            runs[name] = parsed

    prev = prev_best(args.history)
    best_name = max(runs, key=lambda n: runs[n]["best"]) if runs else None
    best_f1 = runs[best_name]["best"] if best_name else None

    history_entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": args.context,
        "runs": runs,
        "best_name": best_name,
        "best_f1": best_f1,
        "incumbent": args.incumbent,
    }
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_entry) + "\n")

    if not args.report:
        print(json.dumps(history_entry))
        return

    lines = [
        f"# Optimization iteration report — {args.context}",
        "",
        f"* generated: {history_entry['ts']}",
        f"* incumbent baseline (screen_clrnet_r50_15ep): {args.incumbent:.5f}",
        f"* previous tracked best: {prev if prev is not None else 'n/a'}",
        "",
        "| run | best F1 | @iter | final F1 | Δ vs incumbent | Δ vs prev best |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in sorted(runs.items(), key=lambda kv: -kv[1]["best"]):
        d_inc = m["best"] - args.incumbent
        d_prev = (m["best"] - prev) if prev is not None else None
        lines.append(
            f"| {name} | {m['best']:.5f} | {m['best_iter']} | {m['final']:.5f} "
            f"| {d_inc:+.5f} | {('n/a' if d_prev is None else f'{d_prev:+.5f}')} |"
        )
    if not runs:
        lines.append("| (no runs with F1 observations) | | | | | |")
    if best_name:
        lines += ["", f"**tracked best: `{best_name}` = {best_f1:.5f}**"]
    if args.notes:
        lines += ["", "## Notes", "", args.notes]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(history_entry))


if __name__ == "__main__":
    main()
