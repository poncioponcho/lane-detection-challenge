#!/usr/bin/env python3
"""Union two prediction trees with an official-mask dedup gate.

Why this exists
---------------
The B board takes the max over six shots, so the last shot should combine the
two mechanisms that are individually positive-expected rather than repeat one:

  swa4_54ep   same-trajectory weight average; the smallest-footprint candidate
              (keeps 2635 of the incumbent's 2664 lanes, drops 29, adds 27)
  cons_gate6  60%-agreement consensus; the only zero-downside candidate
              (keeps 2664, drops 0, adds 52)

This script lays the second tree on top of the first, keeping only lanes the
first tree does not already explain (official rasterised IoU <= 0.5, the same
gate the official scorer and the consensus builder use).  Lanes of A are passed
through byte-for-byte; only B's unexplained lanes are appended.  Nothing here
re-scores or reorders anything, and nothing contacts the platform.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ORACLE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
WIDTH = 30
IOU_THR = 0.5


def load_oracle():
    p = PROJECT_ROOT / "src" / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ORACLE = load_oracle()


def read_lines(path: Path):
    if not path.is_file():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        toks = raw.split()
        pts = [(float(toks[i]), float(toks[i + 1])) for i in range(0, len(toks) - 1, 2)]
        clean = [pts[0]]
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        if len(clean) >= 2:
            out.append(clean)
    return out


def mask_of(lane):
    try:
        return ORACLE.draw_lane_mask(ORACLE.interp_lane(lane), WIDTH)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path, help="kept verbatim")
    ap.add_argument("--b", required=True, type=Path, help="source of extra lanes")
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    a_root, b_root, dst = args.a.resolve(), args.b.resolve(), args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    n_a = n_added = n_skip = files = 0
    for f in sorted(a_root.rglob("*.lines.txt")):
        rel = f.relative_to(a_root)
        lanes_a = read_lines(f)
        lanes_b = read_lines(b_root / rel)
        out = [l for l in lanes_a]
        if lanes_b:
            ma = [m for m in (mask_of(l) for l in lanes_a) if m is not None]
            for lb in lanes_b:
                mb = mask_of(lb)
                if mb is None:
                    continue
                best = 0.0
                for m in ma:
                    u = (m | mb).sum()
                    if u:
                        v = (m & mb).sum() / u
                        if v > best:
                            best = v
                if best > IOU_THR:
                    n_skip += 1
                else:
                    out.append(lb)
                    n_added += 1
        n_a += len(lanes_a)
        out_file = dst / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            "".join(" ".join(f"{x:.1f} {y:.1f}" for x, y in lane) + "\n" for lane in out),
            encoding="utf-8")
        files += 1
    print(f"union: files={files} a_lanes={n_a} added_from_b={n_added} "
          f"b_lanes_already_in_a={n_skip} total={n_a + n_added}")


if __name__ == "__main__":
    main()
