#!/usr/bin/env python3
"""Step 0 of the span hypothesis: manufacture testA-like short lanes.

testA predictions have median line start y_min=564 whereas train baseline OOF
predictions start at y_min=489. To find out how much of the 4.3pp LVO->A-bang
gap this geometric difference can explain, we truncate every train OOF line to
y >= CUT_Y and score the result with the frozen Oracle.

The prediction set is kept file-for-file identical to the manifest so the Oracle
call stays comparable with the untruncated baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def truncate_line(values: list[float], cut_y: float) -> list[float] | None:
    pts = list(zip(values[0::2], values[1::2]))
    kept = [(x, y) for x, y in pts if y >= cut_y]
    if len(kept) < 2:
        return None
    out: list[float] = []
    for x, y in kept:
        out.append(x)
        out.append(y)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--cut-y", type=float, default=564.0)
    ap.add_argument("--ndigits", type=int, default=5)
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        raise SystemExit(f"destination already exists: {dst}")
    dst.mkdir(parents=True)

    n_files = n_lines_in = n_lines_out = 0
    n_dropped_files = 0
    for path in sorted(src.rglob("*.lines.txt")):
        rel = path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_lines_in += 1
            kept = truncate_line([float(v) for v in line.split()], args.cut_y)
            if kept is None:
                continue
            rows.append(" ".join(f"{v:.{args.ndigits}f}" for v in kept))
            n_lines_out += 1
        out_path.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
        n_files += 1
        if not rows:
            n_dropped_files += 1

    summary = {
        "src": str(src),
        "dst": str(dst),
        "cut_y": args.cut_y,
        "files": n_files,
        "lines_in": n_lines_in,
        "lines_out": n_lines_out,
        "lines_dropped": n_lines_in - n_lines_out,
        "files_became_empty": n_dropped_files,
    }
    (dst.parent / "truncate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
