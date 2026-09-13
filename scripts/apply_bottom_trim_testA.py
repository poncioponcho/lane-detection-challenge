#!/usr/bin/env python3
"""Apply the frozen bottom-trim rule to a raw testA/testB prediction tree.

The rule is the one measured on the honest 15ep LVO OOF
(`docs/span_bottom_trim_20260913.md`): CLRNet decodes every lane down to the
image bottom (y=719) while GT terminates far higher whenever the lane starts
high in the image. Cutting the near-field overshoot raised OOF F1 from
0.777628 to 0.819870 (+4.224pp) at margin +40 and 0.797181 (+1.955pp) at
margin +0.

On testA every predicted lane starts at y >= 549, where the GT conditional
bottom is ~712, so only margin +0 produces any cut at all (~7px). That is why
this script defaults to margin 0: it is the only setting that is not a no-op
on the A board, and it is the mildest (in-domain it still measures +1.955pp).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exp_bottom_trim_20260913 import build_variant  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--rule", default="gt_cond",
                    choices=["gt_cond", "fixed", "span_frac"])
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        shutil.rmtree(dst)
    n = build_variant(src, dst, args.rule, args.margin)
    print(f"built {dst}: {n} lines (rule={args.rule}, margin={args.margin})")


if __name__ == "__main__":
    main()
