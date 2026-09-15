#!/usr/bin/env python3
"""Screen geometry-only post-processing recipes on the honest local ruler.

Why this exists
---------------
testA has no ground truth, so no B-board candidate can be scored offline. The
8-fold leave-one-video-out OOF *can* be scored: it is video-disjoint, the
frozen Oracle gives one global number, and the per-clip breakdown lets us
bootstrap at the video level (never the frame level).

Everything screened here is a *geometry* transform, because the OOF tree ships
no confidence sidecar -- threshold recipes need a re-run and are out of scope.

Ruler validation first: the bottom-trim margin sweep was already measured on
this tree (0 -> 0.797224, 40 -> 0.819699, 60 -> 0.817685). The sweep is
re-run here so every new number is reported next to a known-good reference. If
the sweep does not reproduce, nothing else in the report means anything.

The transfer caveat is real: OOF predictions are train-shaped (top p5 = 244)
while testA is shaped differently (top p50 = 564). A recipe that only helps
because of the near-field overshoot will not transfer. That is why each recipe
is also reported on the testA-like clip subset (GT top >= 530).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from data.manifest import read_manifest  # noqa: E402
from eval.oracle_runner import run_official_eval  # noqa: E402
from exp_bottom_trim_20260913 import build_variant  # noqa: E402

ORACLE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
MANIFEST = ROOT / "data/processed/manifest_train.jsonl"
GT_DIR = ROOT / "data/gt_train/anno_txt"
OOF = ROOT / "outputs/lvo_clrnet_r50_15ep_20260904/oof/predictions"
WORK = ROOT / "outputs/exp_oof_recipe_screen_20260916"
TOPA_LIKE_GT_TOP = 530.0


# ---------------------------------------------------------------- transforms

def read_tree(root: Path):
    out = {}
    for f in sorted(root.rglob("*.lines.txt")):
        rel = f.relative_to(root).as_posix()
        lanes = []
        for raw in f.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            t = raw.split()
            pts = [(float(t[i]), float(t[i + 1])) for i in range(0, len(t) - 1, 2)]
            if len(pts) >= 2:
                lanes.append(pts)
        out[rel] = lanes
    return out


def write_tree(tree, dst: Path):
    if dst.exists():
        dst = dst.with_name(dst.name + f"_{int(time.time())}")
    for rel, lanes in tree.items():
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "".join(" ".join(f"{x:.1f} {y:.1f}" for x, y in lane) + "\n" for lane in lanes),
            encoding="utf-8")
    return dst


def t_span(tree, min_span):
    out = {}
    for rel, lanes in tree.items():
        keep = []
        for pts in lanes:
            ys = [p[1] for p in pts]
            if max(ys) - min(ys) >= min_span:
                keep.append(pts)
        out[rel] = keep
    return out


def t_ma(tree, k):
    """Moving average of x over the polyline; y is untouched."""
    out = {}
    half = k // 2
    for rel, lanes in tree.items():
        new = []
        for pts in lanes:
            if len(pts) <= k:
                new.append(pts)
                continue
            xs = [p[0] for p in pts]
            sm = []
            for i in range(len(xs)):
                lo, hi = max(0, i - half), min(len(xs), i + half + 1)
                sm.append(sum(xs[lo:hi]) / (hi - lo))
            new.append([(sm[i], pts[i][1]) for i in range(len(pts))])
        out[rel] = new
    return out


def t_trim(src_root: Path, margin: float, dst: Path):
    d = dst if not dst.exists() else dst.with_name(dst.name + f"_{int(time.time())}")
    build_variant(src_root, d, "gt_cond", margin)
    return read_tree(d)


# ---------------------------------------------------------------- scoring

def video_of(clip_id: str) -> str:
    return clip_id.split("_1_0_", 1)[0]


def score(dst_root: Path, records, tag: str, topa_clips):
    res = run_official_eval(dst_root, GT_DIR, records,
                            official_python=ORACLE_PY, per_clip=True,
                            output_path=WORK / f"oracle_{tag}.json").to_dict()
    g = res["global"]
    per_video = defaultdict(lambda: [0, 0, 0])
    topa = [0, 0, 0]
    for clip, v in res["per_clip"].items():
        vid = video_of(clip)
        for i, key in enumerate(("tp", "fp", "fn")):
            per_video[vid][i] += int(v[key])
            if clip in topa_clips:
                topa[i] += int(v[key])
    tp, fp, fn = topa
    topa_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "f1": float(g["f1"]), "tp": int(g["tp"]), "fp": int(g["fp"]), "fn": int(g["fn"]),
        "precision": float(g["precision"]), "recall": float(g["recall"]),
        "lanes": int(g["tp"]) + int(g["fp"]),
        "per_video": {k: v for k, v in per_video.items()},
        "topalike_f1": float(topa_f1),
        "topalike_clips": len(topa_clips),
    }


def paired_bootstrap(base_v, cand_v, n=20000, seed=42):
    """Resample video clusters; both variants see the same resample."""
    vids = sorted(base_v)
    b = np.asarray([base_v[v] for v in vids], dtype=np.int64)
    c = np.asarray([cand_v[v] for v in vids], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vids), size=(n, len(vids)))
    bs, cs = b[idx].sum(axis=1), c[idx].sum(axis=1)

    def f1(m):
        return 2.0 * m[:, 0] / np.maximum(2 * m[:, 0] + m[:, 1] + m[:, 2], 1)

    d = f1(cs) - f1(bs)
    return {
        "delta_pp": float((f1(c.sum(axis=0)[None, :]) - f1(b.sum(axis=0)[None, :]))[0] * 100),
        "ci95_pp": [float(x) * 100 for x in np.percentile(d, [2.5, 97.5])],
        "p_positive": float((d > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--skip-trim-sweep", action="store_true")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    records = read_manifest(MANIFEST)
    print(f"manifest rows={len(records)} gt_dir={GT_DIR}", flush=True)

    # the testA-like clip subset: clips whose GT lanes start low in the image
    topa_clips = set()
    for rec in records:
        p = GT_DIR / rec.gt_path
        if not p.is_file():
            continue
        tops = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            t = raw.split()
            if len(t) >= 4:
                tops.append(min(float(t[i]) for i in range(1, len(t), 2)))
        if tops and min(tops) >= TOPA_LIKE_GT_TOP:
            topa_clips.add(rec.clip_id)
    print(f"testA-like clips (GT top >= {TOPA_LIKE_GT_TOP:.0f}): {len(topa_clips)}", flush=True)

    base = read_tree(OOF)
    src_root = OOF
    variants: list[tuple[str, dict]] = []

    variants.append(("base", base))
    if not args.skip_trim_sweep:
        for m in (0, 40, 60):
            variants.append((f"trim{m}", t_trim(OOF, float(m), WORK / f"tree_trim{m}")))
    for s in (60, 80, 100, 120):
        variants.append((f"span{s}", t_span(base, s)))
    variants.append(("ma5", t_ma(base, 5)))
    variants.append(("span80_trim40",
                     t_trim(write_tree(t_span(base, 80), WORK / "tree_span80"), 40.0,
                            WORK / "tree_span80_trim40")))
    variants.append(("ma5_trim40",
                     t_trim(write_tree(t_ma(base, 5), WORK / "tree_ma5"), 40.0,
                            WORK / "tree_ma5_trim40")))

    results = {}
    base_v = None
    print(f"\n{'variant':16s}{'F1':>10}{'dpp':>9}{'CI95(pp)':>20}{'P(+)%':>8}"
          f"{'P':>7}{'R':>7}{'lanes':>7}{'topA_F1':>9}", flush=True)
    for name, tree in variants:
        dst = write_tree(tree, WORK / f"tree_{name}")
        r = score(dst, records, name, topa_clips)
        results[name] = {k: v for k, v in r.items() if k != "per_video"}
        if base_v is None:
            base_v = r["per_video"]
            print(f"{name:16s}{r['f1']:>10.6f}{'—':>9}{'—':>20}{'—':>8}"
                  f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['lanes']:>7}"
                  f"{r['topalike_f1']:>9.4f}", flush=True)
            continue
        pb = paired_bootstrap(base_v, r["per_video"], n=args.bootstrap)
        results[name]["vs_base"] = pb
        ci = pb["ci95_pp"]
        print(f"{name:16s}{r['f1']:>10.6f}{pb['delta_pp']:>+9.3f}"
              f"{f'[{ci[0]:+.3f},{ci[1]:+.3f}]':>20}{100*pb['p_positive']:>8.1f}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['lanes']:>7}"
              f"{r['topalike_f1']:>9.4f}", flush=True)

    (WORK / "screen.json").write_text(
        json.dumps({"manifest_sha_records": len(records),
                    "topalike_clips": sorted(topa_clips),
                    "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwrote {WORK / 'screen.json'}")


if __name__ == "__main__":
    main()
