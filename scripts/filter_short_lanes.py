#!/usr/bin/env python3
"""Drop short consensus lanes from a union tree -- added lanes only.

`build_testA_consensus_union_20260913.py` averages the agreeing support lanes
over their common y range. When two trees' common range is narrow the mean is a
stub: measured on the built k2 tree, **29.4% of the added lanes have a span
below 80px** against 2.5% in the base tree (base span p50 = 155).

A stub cannot clear the 0.5 IoU gate against a normal GT lane: if the
prediction is contained in the GT lane and laterally aligned, IoU is
span_pred/span_GT, so a 26px stub against a 155px GT lane caps at 0.17. Only
~2% of GT lanes are shorter than 90px, so the short added lanes are averaging
artefacts, not real short lanes, and removing them is the F1/2 rule applied
locally: remove a set only when more than 1 - F1/2 of it is false.

`--base` is required on purpose. The incumbent's own lanes are untouchable:
they are the 0.73574 submission, and the base tree does carry a few short lanes
of its own. A lane is kept when it matches a base lane (median lateral
distance below `--keep-dist` over the common y range) or when its span clears
the threshold; only base-unmatched short lanes are dropped.
"""
from __future__ import annotations

import argparse
import shutil
import statistics
from pathlib import Path


def x_at(lane, y: float) -> float:
    pts = sorted(lane, key=lambda p: p[1])
    if y <= pts[0][1]:
        return pts[0][0]
    if y >= pts[-1][1]:
        return pts[-1][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= y <= y1:
            if y1 == y0:
                return x0
            return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    return pts[-1][0]


def matches_base(lane, base_lanes, keep_dist: float,
                 min_overlap: float = 5.0) -> bool:
    """Does `lane` correspond to one of the incumbent's lanes?

    `min_overlap` must stay small. The incumbent tree is untrimmed (lanes run to
    y=719) while the union tree has already been trimmed, and the base tree does
    contain very short lanes (a 694->719 lane trims to 694->713, a 19px common
    range). With the earlier 20px guard those lanes failed to match, were
    reclassified as "added", and were then dropped by the span filter -- the
    audit caught 5 incumbent lanes silently disappearing from the main shot.
    """
    lo, hi = min(p[1] for p in lane), max(p[1] for p in lane)
    for b in base_lanes:
        blo, bhi = min(p[1] for p in b), max(p[1] for p in b)
        o, u = max(lo, blo), min(hi, bhi)
        if u - o < min_overlap:
            continue
        xs = [abs(x_at(lane, o + (u - o) * i / 8) - x_at(b, o + (u - o) * i / 8))
              for i in range(9)] if u > o else [abs(x_at(lane, o) - x_at(b, o))]
        if statistics.median(xs) < keep_dist:
            return True
    return False


def parse_line(text: str):
    vals = [float(t) for t in text.split()]
    return list(zip(vals[0::2], vals[1::2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path,
                    help="incumbent tree; its lanes (matched) are never dropped")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--min-span", type=float, default=80.0)
    ap.add_argument("--keep-dist", type=float, default=30.0)
    args = ap.parse_args()

    src, base, dst = args.src.resolve(), args.base.resolve(), args.dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    kept = dropped = base_kept = files = 0
    base_missing = 0
    for f in sorted(src.rglob("*.lines.txt")):
        rel = f.relative_to(src)
        bfile = base / rel
        if not bfile.is_file():
            base_missing += 1
        base_lanes = [parse_line(t) for t in bfile.read_text(encoding="utf-8").splitlines()
                      if t.strip()] if bfile.is_file() else []
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            lane = parse_line(line)
            ys = [p[1] for p in lane]
            span = max(ys) - min(ys)
            if matches_base(lane, base_lanes, args.keep_dist):
                rows.append(line.strip())
                base_kept += 1
            elif span >= args.min_span:
                rows.append(line.strip())
                kept += 1
            else:
                dropped += 1
        out.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
        files += 1
    # Fail loudly instead of silently downgrading to "every lane is added".
    # When --base does not line up with --src (a very easy mistake: the raw
    # trees under outputs/ carry an extra `testA/predictions/` level while the
    # built consensus trees do not) every lane becomes "added", and the span
    # filter then deletes the incumbent's own short lanes. That cost 136 lanes
    # on the 2026-09-15 gate6 build before this guard existed.
    if base_missing:
        raise SystemExit(
            f"--base {base} has no prediction for {base_missing}/{files} images; "
            "rel paths must match --src exactly (check for an extra "
            "'testA/predictions' level). Refusing to run."
        )
    if base_kept == 0 and (kept or dropped):
        raise SystemExit(
            f"no lane matched --base {base} at all (keep_dist={args.keep_dist}); "
            "the base tree is almost certainly wrong. Refusing to run."
        )
    print(f"{src.name} -> {dst.name}: files={files} base_kept={base_kept} "
          f"added_kept={kept} added_dropped={dropped} (min_span={args.min_span})")


if __name__ == "__main__":
    main()

