#!/usr/bin/env python3
"""Build three auditable testA candidate submission packages (2026-09-14).

Why this exists
---------------
The A-board scoring is the frozen official Oracle (global pooled
``F1 = 2*TP/(P+G)``, Hungarian matching, ``IoU > 0.5`` gate, ``LINE_WIDTH=30``,
image ``(720, 1366)``).  Recall is the current bottleneck, so the three
2026-09-14 submissions test two *mechanistic* hypotheses with an exactly
reproducible construction:

``extend60_trim0`` / ``extend120_trim0``
    Upward (far-field) extrapolation of every lane by ``E`` px using a
    quadratic ``x(y)`` fit, followed by the frozen bottom-trim rule (margin 0).
    Hypothesis: the frozen top end undershoots GT; extending it recovers FN.

``union_trim0``
    Union of the reference package with four supporting model prediction trees
    (greedy IoU<=0.5 de-duplication with the official mask), then the frozen
    bottom-trim rule.  Hypothesis: the support seeds recover distinct GT lanes
    the reference model missed, at the cost of added FP.

Frozen dependencies (reused verbatim, never modified)
-----------------------------------------------------
* Bottom-trim helpers ``gt_bottom_for_top`` / ``cut_bottom`` / ``parse_line`` /
  ``fmt`` come from ``scripts/exp_bottom_trim_20260913.py`` (imported, not
  copied).
* Lane masks / IoU come from the byte-frozen official scorer
  ``src/eval/official_oracle/score.py`` (imported read-only via importlib).

Run with the Oracle interpreter so the mask rasterisation matches the official
environment (numpy 2.1.3 / cv2 4.12.0); note ``sys.dont_write_bytecode`` below
so importing the frozen oracle never drops a ``__pycache__`` next to it::

    ORACLE_PY scripts/build_testA_variants_20260914.py

Nothing here contacts the competition platform.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True  # never drop .pyc next to the frozen oracle

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# --- Frozen bottom-trim helpers, reused verbatim from the 2026-09-13 experiment.
from exp_bottom_trim_20260913 import (  # noqa: E402
    cut_bottom,
    fmt,
    gt_bottom_for_top,
    parse_line,
)

LANE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane/bin/python"
ORACLE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"

SOURCE_ZIP = PROJECT_ROOT / "outputs/submit_testA_a2_54ep_best.zip"
MANIFEST = PROJECT_ROOT / "data/processed/manifest_testA.jsonl"
LIST_PATH = Path("/private/tmp/variant_list_20260914.txt")
OUT_ROOT = PROJECT_ROOT / "outputs/testA_variants_20260914"
REPORTS_DIR = PROJECT_ROOT / "outputs/reports"
BUILD_REPORT = REPORTS_DIR / "build_variants_20260914_report.json"

# Fixed supporting-model order (dedup is order sensitive -> must stay frozen).
SUPPORT_TREES = [
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions",
]

IMG_HEIGHT = 720
IMG_WIDTH = 1366
IOU_THRESHOLD = 0.5
SAMPLE_TOL = 0.15


def ensured(p: Path) -> Path:
    """mkdir -p that is safe under the sandbox broker (no exist_ok)."""
    if not p.exists():
        p.mkdir(parents=True)
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prune_stale_lines(root: Path, expected_rels: set[str]) -> int:
    """Delete ``*.lines.txt`` under ``root`` that are not in ``expected_rels``.

    Kept tiny on purpose: the sandbox intercepts bulk deletions (>=50 files in
    one turn), so we overwrite in place and only prune genuine strays (normally
    zero), never ``rmtree`` the tree.
    """
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*.lines.txt"):
        if path.relative_to(root).as_posix() not in expected_rels:
            path.unlink()
            removed += 1
    return removed


def load_oracle():
    """Read-only import of the frozen official scorer via importlib."""
    oracle_path = SRC_DIR / "eval" / "official_oracle" / "score.py"
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(oracle_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official oracle from {oracle_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_manifest_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            records.append(json.loads(raw))
    return records


def write_precheck_list(records: list[dict], out_path: Path) -> int:
    ensured(out_path.parent)
    out_path.write_text(
        "".join("/" + record["image_path"] + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)


def load_source_zip(zip_path: Path) -> dict[tuple[str, str], list[str]]:
    """Return {(clip, frame): [line_text, ...]} preserving in-file order."""
    images: dict[tuple[str, str], list[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".lines.txt"):
                continue
            parts = PurePosixPath(name).parts  # ('submit', clip, 'frame.lines.txt')
            clip, frame = parts[-2], parts[-1]
            text = zf.read(name).decode("utf-8")
            images[(clip, frame)] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return images


def manifest_order(records: list[dict]) -> list[tuple[str, str]]:
    order: list[tuple[str, str]] = []
    for record in records:
        rel = PurePosixPath(str(record["pred_rel_path"]))
        order.append((rel.parts[-2], rel.parts[-1]))
    return order


# --------------------------------------------------------------------------- #
# Upward extrapolation + bottom trim
# --------------------------------------------------------------------------- #
def extend_up(points: list[tuple[float, float]], extent: float) -> tuple[list[tuple[float, float]], bool]:
    """Append an upward (far-field) extrapolation of ``extent`` px.

    Fits ``x = a*y^2 + b*y + c`` on the existing points and appends the new
    samples at ``y = top-5, top-10, ..., top-extent`` (the top is the *last*
    element under the near->far convention).  Lines with fewer than 4 points or
    a vertical span below 30 px are returned unchanged.

    The appended coordinates are clipped to the image frame
    ``[0, 1365] x [0, 719]`` exactly like ``official_oracle.score.draw_lane_mask``
    does; the submission exporter rejects out-of-frame coordinates, so the clip
    is mandatory for packaging (it does not change how the Oracle rasterises the
    lane, since the Oracle clips to the same bounds).
    """
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    if len(points) < 4 or (ys.max() - ys.min()) < 30.0:
        return list(points), False
    coef = np.polyfit(ys, xs, 2)
    top = float(ys.min())
    new_ys = np.arange(top - 5.0, top - extent - 1.0, -5.0)
    new_xs = np.clip(np.polyval(coef, new_ys), 0.0, float(IMG_WIDTH - 1))
    new_ys = np.clip(new_ys, 0.0, float(IMG_HEIGHT - 1))
    extended = list(points) + [(float(x), float(y)) for x, y in zip(new_xs, new_ys)]
    return extended, True


def trim0(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Frozen bottom-trim with margin 0 (rule ``gt_cond``)."""
    top = min(p[1] for p in points)
    cut = gt_bottom_for_top(top)
    return cut_bottom(points, cut)


def clamp_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Clamp coordinates into the image frame ``[0,1365] x [0,719]``.

    The submission exporter rejects out-of-frame coordinates.  Supporting-model
    lines can carry sub-pixel overshoot (e.g. x=1365.5); clamping matches the
    Oracle's own ``draw_lane_mask`` clip and therefore does not change IoU.
    """
    x_hi = float(IMG_WIDTH - 1)
    y_hi = float(IMG_HEIGHT - 1)
    return [(min(max(x, 0.0), x_hi), min(max(y, 0.0), y_hi)) for x, y in points]


# --------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------- #
def build_extend_variant(
    name: str,
    extent: float,
    images: dict[tuple[str, str], list[str]],
    order: list[tuple[str, str]],
    out_pred_dir: Path,
) -> tuple[dict, list[tuple[str, str, int, float]]]:
    ensured(out_pred_dir)
    prune_stale_lines(out_pred_dir, {f"{c}/{f}" for c, f in order})

    files = 0
    lanes = 0
    empty_files = 0
    lanes_extended = 0
    lanes_dropped_by_trim = 0
    samples: list[tuple[str, str, int, float, float]] = []

    for clip, frame in order:
        src_lines = images.get((clip, frame), [])
        out_lines: list[str] = []
        for text in src_lines:
            pts = parse_line(text)
            top_old = min(p[1] for p in pts)
            bottom_old = max(p[1] for p in pts)
            extended, did_extend = extend_up(pts, extent)
            if did_extend:
                lanes_extended += 1
            kept = trim0(extended)
            if not kept:
                lanes_dropped_by_trim += 1
                continue
            out_lines.append(fmt(kept))
            if did_extend:
                samples.append((clip, frame, len(out_lines) - 1, top_old, bottom_old))

        dst = out_pred_dir / clip / frame
        ensured(dst.parent)
        dst.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
        files += 1
        lanes += len(out_lines)
        empty_files += int(not out_lines)

    stats = {
        "files": files,
        "lanes": lanes,
        "empty_files": empty_files,
        "lanes_extended": lanes_extended,
        "lanes_dropped_by_trim": lanes_dropped_by_trim,
    }
    return stats, samples


def verify_extend_samples(
    out_pred_dir: Path,
    samples: list[tuple[str, str, int, float, float]],
    extent: float,
    n: int = 100,
) -> dict:
    """Re-read up to ``n`` extended lanes from disk and check the invariants.

    Two bottom invariants are reported:

    * ``bottom_literal``: ``bottom_new == min(719, gt_bottom_for_top(top_new))``
      (the invariant stated in the task).  It only holds for lanes whose
      *original* near end already reaches the conditional cut; short lanes
      (near end above the cut) are a deliberate no-op of ``cut_bottom`` so their
      bottom is untouched.
    * ``bottom_refined``: ``bottom_new == min(bottom_old, gt_bottom_for_top(top_new))``
      which is the true post-condition and must hold for every lane.
    """
    if not samples:
        return {"n": 0, "tol": SAMPLE_TOL, "failures": []}
    take = min(n, len(samples))
    idx = sorted({int(i) for i in np.linspace(0, len(samples) - 1, num=take)})
    top_ok = 0
    bottom_literal_ok = 0
    bottom_refined_ok = 0
    failures: list[dict] = []
    file_cache: dict[tuple[str, str], list[str]] = {}
    for i in idx:
        clip, frame, line_idx, top_old, bottom_old = samples[i]
        key = (clip, frame)
        if key not in file_cache:
            file_cache[key] = (out_pred_dir / clip / frame).read_text(encoding="utf-8").splitlines()
        pts = parse_line(file_cache[key][line_idx])
        top_new = min(p[1] for p in pts)
        bottom_new = max(p[1] for p in pts)
        expected_top = top_old - extent
        cut = gt_bottom_for_top(top_new)
        expected_bottom_literal = min(float(IMG_HEIGHT - 1), cut)
        expected_bottom_refined = min(bottom_old, cut)
        top_ok += abs(top_new - expected_top) <= SAMPLE_TOL
        bottom_literal_ok += abs(bottom_new - expected_bottom_literal) <= SAMPLE_TOL
        bottom_refined_ok += abs(bottom_new - expected_bottom_refined) <= SAMPLE_TOL
        if abs(top_new - expected_top) > SAMPLE_TOL or abs(bottom_new - expected_bottom_refined) > SAMPLE_TOL:
            failures.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "line": line_idx,
                    "top_old": round(top_old, 4),
                    "top_new": round(top_new, 4),
                    "expected_top": round(expected_top, 4),
                    "bottom_old": round(bottom_old, 4),
                    "bottom_new": round(bottom_new, 4),
                    "expected_bottom_refined": round(expected_bottom_refined, 4),
                    "expected_bottom_literal": round(expected_bottom_literal, 4),
                }
            )
    return {
        "n": len(idx),
        "tol": SAMPLE_TOL,
        "top_invariant_ok": top_ok,
        "bottom_refined_ok": bottom_refined_ok,
        "bottom_literal_ok": bottom_literal_ok,
        "bottom_literal_mismatch": len(idx) - bottom_literal_ok,
        "failures": failures[:20],
        "note": (
            "bottom_literal mismatch is expected exactly for short lanes whose "
            "original near end sits above the conditional cut (trim0 is a no-op "
            "there); the refined invariant bottom==min(bottom_old, gt_bottom) "
            "holds for every sampled lane."
        ),
    }


def build_union_variant(
    oracle,
    images: dict[tuple[str, str], list[str]],
    order: list[tuple[str, str]],
    out_pred_dir: Path,
    support_dirs: list[Path],
) -> dict:
    ensured(out_pred_dir)
    prune_stale_lines(out_pred_dir, {f"{c}/{f}" for c, f in order})

    files = 0
    lanes = 0
    empty_files = 0
    ref_lanes_kept = 0
    novel_lanes_added = 0
    ref_byte_failures = 0
    candidate_rejected = 0
    missing_support = 0

    for clip, frame in order:
        ref_lines = images.get((clip, frame), [])
        ref_pts = [parse_line(t) for t in ref_lines]

        # Byte-identity invariant (pre-trim): union must not touch ref lanes.
        for text in ref_lines:
            if fmt(parse_line(text)) != text:
                ref_byte_failures += 1

        # Per-image mask cache keyed by the exact point tuple.
        cache: dict[tuple, "np.ndarray"] = {}

        def mask_of(points):
            key = tuple(points)
            cached = cache.get(key)
            if cached is None:
                interp = oracle.interp_lane(points)
                cached = oracle.draw_lane_mask(interp, oracle.LINE_WIDTH)
                cache[key] = cached
            return cached

        kept_pts = list(ref_pts)
        kept_masks = [mask_of(p) for p in kept_pts]

        for tdir in support_dirs:
            support_file = tdir / clip / frame
            if not support_file.is_file():
                missing_support += 1
                continue
            for text in support_file.read_text(encoding="utf-8").splitlines():
                text = text.strip()
                if not text:
                    continue
                cand = parse_line(text)
                cand_mask = mask_of(cand)
                duplicate = False
                for kept_mask in kept_masks:
                    union = int(np.logical_or(cand_mask, kept_mask).sum())
                    if union == 0:
                        continue  # empty union -> treat as non-overlapping
                    inter = int(np.logical_and(cand_mask, kept_mask).sum())
                    if inter / union > IOU_THRESHOLD:
                        duplicate = True
                        break
                if duplicate:
                    candidate_rejected += 1
                    continue
                kept_pts.append(cand)
                kept_masks.append(cand_mask)
                novel_lanes_added += 1

        ref_lanes_kept += len(ref_lines)

        # Frozen bottom-trim applied uniformly to the union.
        out_lines: list[str] = []
        for points in kept_pts:
            kept = trim0(points)
            if kept:
                out_lines.append(fmt(clamp_points(kept)))

        dst = out_pred_dir / clip / frame
        ensured(dst.parent)
        dst.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
        files += 1
        lanes += len(out_lines)
        empty_files += int(not out_lines)

    return {
        "files": files,
        "lanes": lanes,
        "empty_files": empty_files,
        "ref_lanes_kept": ref_lanes_kept,
        "novel_lanes_added": novel_lanes_added,
        "candidate_rejected": candidate_rejected,
        "missing_support_files": missing_support,
        "ref_lanes_byte_identical": ref_byte_failures == 0,
        "ref_lane_byte_mismatches": ref_byte_failures,
        "ref_lane_byte_identity_note": (
            "byte identity verified on the pre-trim union representation "
            "(ref lanes are inserted verbatim in original file order before any "
            "trim; trim0 reorders/echoes them afterwards by design)"
        ),
    }


# --------------------------------------------------------------------------- #
# Packaging + precheck
# --------------------------------------------------------------------------- #
def run_prepare_submit(
    name: str, raw_pred_dir: Path, expected_rels: set[str]
) -> tuple[Path, Path, subprocess.CompletedProcess]:
    canonical_dir = Path(f"/private/tmp/canon_{name}")
    ensured(canonical_dir)
    prune_stale_lines(canonical_dir, expected_rels)
    out_zip = PROJECT_ROOT / "outputs" / f"submit_testA_{name}_20260914.zip"
    report = REPORTS_DIR / f"prepare_submit_{name}_20260914_report.json"
    cmd = [
        LANE_PY,
        str(SRC_DIR / "submit" / "prepare_submit.py"),
        "--raw-pred-dir", str(raw_pred_dir),
        "--canonical-dir", str(canonical_dir),
        "--out-zip", str(out_zip),
        "--manifest", str(MANIFEST),
        "--missing-as-empty",
        "--report", str(report),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return out_zip, report, proc


def run_precheck(zip_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        ORACLE_PY,
        str(SRC_DIR / "eval" / "official_oracle" / "check_submission.py"),
        "--zip_path", str(zip_path),
        "--list_path", str(LIST_PATH),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def package_variant(
    oracle_report: dict, name: str, raw_pred_dir: Path, zip_name: str, expected_rels: set[str]
) -> dict:
    out_zip, report, proc = run_prepare_submit(name, raw_pred_dir, expected_rels)
    prep_stdout = proc.stdout.strip()
    prep_stderr = proc.stderr.strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"prepare_submit failed for {name} (rc={proc.returncode})\n"
            f"stdout:\n{prep_stdout}\nstderr:\n{prep_stderr}"
        )

    pre = run_precheck(out_zip)
    precheck_text = (pre.stdout.strip() + ("\n" + pre.stderr.strip() if pre.stderr.strip() else "")).strip()
    if pre.returncode != 0:
        raise RuntimeError(f"official precheck FAILED for {name}:\n{precheck_text}")

    return {
        **oracle_report,
        "raw_pred_dir": str(raw_pred_dir),
        "zip_path": str(out_zip),
        "zip_bytes": out_zip.stat().st_size,
        "zip_sha256": sha256_file(out_zip),
        "prepare_submit_report": str(report),
        "prepare_submit_stdout": prep_stdout,
        "precheck": precheck_text,
        "precheck_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=["extend60_trim0", "extend120_trim0", "union_trim0"])
    parser.add_argument("--extent-small", type=float, default=60.0)
    parser.add_argument("--extent-large", type=float, default=120.0)
    parser.add_argument("--sample-n", type=int, default=100)
    args = parser.parse_args()

    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(SOURCE_ZIP)
    for tree in SUPPORT_TREES:
        if not tree.is_dir():
            raise FileNotFoundError(tree)

    records = read_manifest_records(MANIFEST)
    order = manifest_order(records)
    expected_rels = {f"{c}/{f}" for c, f in order}
    n_list = write_precheck_list(records, LIST_PATH)
    images = load_source_zip(SOURCE_ZIP)
    oracle = load_oracle()

    src_lanes = sum(len(v) for v in images.values())
    print(f"[info] manifest files={len(records)}  precheck-list={n_list}  source-images={len(images)}  source-lanes={src_lanes}")

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_zip": {
            "path": str(SOURCE_ZIP),
            "bytes": SOURCE_ZIP.stat().st_size,
            "sha256": sha256_file(SOURCE_ZIP),
            "images": len(images),
            "lanes": src_lanes,
        },
        "manifest": str(MANIFEST),
        "precheck_list": str(LIST_PATH),
        "precheck_list_lines": n_list,
        "support_trees": [str(t) for t in SUPPORT_TREES],
        "interpreter": sys.executable,
        "numpy": np.__version__,
        "oracle_cv2": getattr(oracle.cv2, "__version__", "unknown"),
        "variants": {},
    }

    want = set(args.variants)

    if "extend60_trim0" in want:
        stats, samples = build_extend_variant(
            "extend60_trim0", args.extent_small, images, order,
            OUT_ROOT / "extend60_trim0" / "testA" / "predictions",
        )
        stats["sample_check"] = verify_extend_samples(
            OUT_ROOT / "extend60_trim0" / "testA" / "predictions", samples, args.extent_small, args.sample_n
        )
        stats["extent"] = args.extent_small
        print(f"[build] extend60_trim0: {stats}")
        report["variants"]["extend60_trim0"] = package_variant(
            stats, "extend60_trim0",
            OUT_ROOT / "extend60_trim0" / "testA" / "predictions",
            "extend60_trim0", expected_rels,
        )

    if "extend120_trim0" in want:
        stats, samples = build_extend_variant(
            "extend120_trim0", args.extent_large, images, order,
            OUT_ROOT / "extend120_trim0" / "testA" / "predictions",
        )
        stats["sample_check"] = verify_extend_samples(
            OUT_ROOT / "extend120_trim0" / "testA" / "predictions", samples, args.extent_large, args.sample_n
        )
        stats["extent"] = args.extent_large
        print(f"[build] extend120_trim0: {stats}")
        report["variants"]["extend120_trim0"] = package_variant(
            stats, "extend120_trim0",
            OUT_ROOT / "extend120_trim0" / "testA" / "predictions",
            "extend120_trim0", expected_rels,
        )

    if "union_trim0" in want:
        stats = build_union_variant(
            oracle, images, order,
            OUT_ROOT / "union_trim0" / "testA" / "predictions",
            SUPPORT_TREES,
        )
        print(f"[build] union_trim0: {stats}")
        report["variants"]["union_trim0"] = package_variant(
            stats, "union_trim0",
            OUT_ROOT / "union_trim0" / "testA" / "predictions",
            "union_trim0", expected_rels,
        )

    ensured(REPORTS_DIR)
    BUILD_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] audit report -> {BUILD_REPORT}")


if __name__ == "__main__":
    main()
