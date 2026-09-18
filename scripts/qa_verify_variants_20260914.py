#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent QA verification of the three 2026-09-14 testA candidate packages.

Author: software-qa-engineer (Edward).  Nothing here trusts the numbers the
implementer reported: every quantity is recomputed from the raw bytes on disk.

Run with the Oracle interpreter so cv2/numpy/scipy match the frozen scoring env::

    ORACLE_PY scripts/qa_verify_variants_20260914.py

The script never writes bytecode next to the frozen oracle (sys.dont_write_bytecode)
and never modifies any existing product — it only reads ``outputs/*.zip``,
``src/``, ``data/`` and writes a JSON artifact under ``outputs/reports/``.

Checks
------
1. structure          : 900 files, exact path set vs manifest_testA, empty/lane counts
2. official precheck  : run the frozen check_submission.py on all three zips
3. extrapolation      : re-implement extend_up (polyfit) and compare ALL 2664 lines
4. trim correctness   : re-implement gt_bottom_for_top/cut_bottom, check y<=cut
5. union completeness : official-IoU inclusion proof + independent novel recount
6. clip neutrality    : is pre-clipping coords to [0,1365]x[0,719] mask-neutral?
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True  # never drop .pyc next to the frozen oracle

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ORACLE_DIR = SRC_DIR / "eval" / "official_oracle"
MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest_testA.jsonl"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
OUT_JSON = REPORTS_DIR / "qa_candidates_20260914_raw.json"

ORACLE_PY = "/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"
LIST_PATH = Path("/private/tmp/variant_list_20260914.txt")

SOURCE_ZIP = PROJECT_ROOT / "outputs" / "submit_testA_a2_54ep_best.zip"
VARIANT_ZIPS = {
    "extend60_trim0": PROJECT_ROOT / "outputs" / "submit_testA_extend60_trim0_20260914.zip",
    "extend120_trim0": PROJECT_ROOT / "outputs" / "submit_testA_extend120_trim0_20260914.zip",
    "union_trim0": PROJECT_ROOT / "outputs" / "submit_testA_union_trim0_20260914.zip",
}
EXTENT = {"extend60_trim0": 60.0, "extend120_trim0": 120.0}

SUPPORT_TREES = [
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed42_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed43_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/all71_seed44_clrnet_r50_36ep/testA/predictions",
    PROJECT_ROOT / "outputs/testA_full71_infer_20260909/clrernet_r50_15ep/testA/predictions",
]

IMG_HEIGHT = 720
IMG_WIDTH = 1366
TOL = 0.2  # coordinate tolerance (task-specified)


# --------------------------------------------------------------------------- #
# Frozen-oracle re-implementation of the trim helpers (independent copy).
# This mirrors the *semantics* documented in scripts/exp_bottom_trim_20260913.py
# but is written here from scratch; we do NOT import that module.
# --------------------------------------------------------------------------- #
GT_BOTTOM_BY_TOP = [
    (200, 396), (250, 460), (300, 512), (350, 563),
    (400, 615), (450, 710), (500, 712), (550, 712), (600, 713), (650, 713),
]


def gt_bottom_for_top(top: float) -> float:
    pts = GT_BOTTOM_BY_TOP
    if top <= pts[0][0]:
        return float(pts[0][1])
    for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
        if t0 <= top < t1:
            return b0 + (b1 - b0) * (top - t0) / (t1 - t0)
    return float(pts[-1][1])


def reimpl_cut_bottom(points, cut):
    pts = sorted(points, key=lambda p: p[1])
    kept = [p for p in pts if p[1] <= cut]
    if len(kept) < 2:
        return []
    if kept[-1][1] < cut:
        prev = kept[-1]
        nxt = next((p for p in pts if p[1] > cut), None)
        if nxt is not None and nxt[1] != prev[1]:
            r = (cut - prev[1]) / (nxt[1] - prev[1])
            kept = kept + [(prev[0] + (nxt[0] - prev[0]) * r, cut)]
    return kept


def reimpl_trim0(points):
    top = min(p[1] for p in points)
    return reimpl_cut_bottom(points, gt_bottom_for_top(top))


def fmt_round(points, ndigits=1):
    """Mimic the submission's 1-decimal serialization, parsed back to floats."""
    return [(round(x, ndigits), round(y, ndigits)) for x, y in points]


def reimpl_extend_up(points, extent):
    """Independent copy of the implementer's extend_up (see build script L181)."""
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    if len(points) < 4 or (ys.max() - ys.min()) < 30.0:
        return list(points), False, None
    coef = np.polyfit(ys, xs, 2)
    top = float(ys.min())
    new_ys_raw = np.arange(top - 5.0, top - extent - 1.0, -5.0)
    new_xs = np.clip(np.polyval(coef, new_ys_raw), 0.0, float(IMG_WIDTH - 1))
    new_ys = np.clip(new_ys_raw, 0.0, float(IMG_HEIGHT - 1))
    new_pts = [(float(x), float(y)) for x, y in zip(new_xs, new_ys)]
    return list(points) + new_pts, True, {"new_ys_raw": new_ys_raw,
                                          "new_xs": new_xs,
                                          "new_ys_clipped": new_ys,
                                          "top_old": top, "coef": coef}


# --------------------------------------------------------------------------- #
# IO / parsing
# --------------------------------------------------------------------------- #
def parse_line(text):
    vals = [float(v) for v in text.split()]
    return list(zip(vals[0::2], vals[1::2]))


def clamp01(v, lo, hi):
    return min(max(v, lo), hi)


def load_zip_lines(zip_path):
    """Return {(clip, frame): [point_list, ...]} plus per-file raw text."""
    images = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".lines.txt"):
                continue
            parts = name.split("/")
            clip, frame = parts[-2], parts[-1]
            text = zf.read(name).decode("utf-8")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            images[(clip, frame)] = [parse_line(ln) for ln in lines]
    return images


def expected_members():
    members = set()
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        rel = rec["pred_rel_path"]
        clip, frame = rel.split("/")
        members.add("submit/{}/{}".format(clip, frame))
    return members


def zip_members(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return set(n for n in zf.namelist() if n.endswith(".lines.txt"))


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def max_coord_dist_ok(exp, got, tol=TOL):
    """Greedy nearest-point match; return (n_unmatched_exp, worst_dist)."""
    used = [False] * len(got)
    worst = 0.0
    unmatched = 0
    for e in exp:
        best, bd = -1, 1e18
        for j, g in enumerate(got):
            if used[j]:
                continue
            d = max(abs(e[0] - g[0]), abs(e[1] - g[1]))
            if d < bd:
                bd, best = d, j
        if best >= 0 and bd <= tol:
            used[best] = True
            worst = max(worst, bd)
        else:
            unmatched += 1
    return unmatched, worst


def load_oracle():
    import importlib.util
    spec = importlib.util.spec_from_file_location("official_oracle_score", str(ORACLE_DIR / "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lane_mask(oracle, points, cache):
    key = tuple((round(x, 3), round(y, 3)) for x, y in points)
    m = cache.get(key)
    if m is None:
        m = oracle.draw_lane_mask(oracle.interp_lane(points), oracle.LINE_WIDTH)
        cache[key] = m
    return m


def iou(m1, m2):
    u = int(np.logical_or(m1, m2).sum())
    if u == 0:
        return 0.0
    return int(np.logical_and(m1, m2).sum()) / u


# --------------------------------------------------------------------------- #
# CHECK 1 — structure
# --------------------------------------------------------------------------- #
def check_structure():
    exp = expected_members()
    res = {"expected_members": len(exp), "source": {}, "variants": {}}

    def scan(label, zip_path):
        members = zip_members(zip_path)
        missing = exp - members
        extra = members - exp
        with zipfile.ZipFile(zip_path) as zf:
            lanes = 0
            empty = 0
            for n in sorted(members):
                ls = [l for l in zf.read(n).decode("utf-8").splitlines() if l.strip()]
                lanes += len(ls)
                empty += (len(ls) == 0)
        return {"zip": zip_path.name, "files": len(members), "lanes": lanes,
                "empty_files": empty, "missing": len(missing), "extra": len(extra),
                "missing_sample": sorted(missing)[:3], "extra_sample": sorted(extra)[:3],
                "path_set_match": not missing and not extra}

    res["source"] = scan("source", SOURCE_ZIP)
    for name, zp in VARIANT_ZIPS.items():
        res["variants"][name] = scan(name, zp)
    return res


# --------------------------------------------------------------------------- #
# CHECK 2 — official precheck
# --------------------------------------------------------------------------- #
def check_precheck():
    out = {}
    if not LIST_PATH.exists():
        out["_note"] = "list missing"
    for name, zp in [("source", SOURCE_ZIP)] + list(VARIANT_ZIPS.items()):
        proc = subprocess.run(
            [ORACLE_PY, str(ORACLE_DIR / "check_submission.py"),
             "--zip_path", str(zp), "--list_path", str(LIST_PATH)],
            capture_output=True, text=True,
        )
        out[name] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    return out


# --------------------------------------------------------------------------- #
# CHECK 3 — extrapolation correctness (extend60 / extend120)
# --------------------------------------------------------------------------- #
def check_extrapolation():
    src = load_zip_lines(SOURCE_ZIP)
    results = {}
    for name in ("extend60_trim0", "extend120_trim0"):
        ext = EXTENT[name]
        var = load_zip_lines(VARIANT_ZIPS[name])
        n_lines = 0
        full_pipeline_ok = 0
        top_ok = 0
        top_clip_aware_ok = 0
        count_formula_ok = 0
        newpoints_ok = 0
        n_extended = 0
        n_not_extended = 0
        failures = []
        count_mismatch_samples = []
        for (clip, frame), src_lines in src.items():
            var_lines = var.get((clip, frame), [])
            if len(var_lines) != len(src_lines):
                failures.append({"clip": clip, "frame": frame,
                                 "err": "line count mismatch",
                                 "src": len(src_lines), "var": len(var_lines)})
                # still align by index up to min
            for i, sp in enumerate(src_lines):
                if i >= len(var_lines):
                    break
                vp = var_lines[i]
                n_lines += 1
                top_old = min(p[1] for p in sp)
                extended, did, info = reimpl_extend_up(sp, ext)
                expected_full = reimpl_trim0(extended)
                # (a) full-pipeline equivalence (strongest)
                u, _ = max_coord_dist_ok(expected_full, vp)
                if u == 0 and len(expected_full) == len(vp):
                    full_pipeline_ok += 1
                if did:
                    n_extended += 1
                    var_top = min(p[1] for p in vp)
                    exp_top_raw = top_old - ext
                    exp_top_clip = max(top_old - ext, 0.0)
                    if abs(var_top - exp_top_raw) <= TOL:
                        top_ok += 1
                    if abs(var_top - exp_top_clip) <= TOL:
                        top_clip_aware_ok += 1
                    # count formula: src + E/5
                    formula = len(sp) + ext / 5.0
                    if len(vp) == formula:
                        count_formula_ok += 1
                    elif len(count_mismatch_samples) < 8:
                        count_mismatch_samples.append(
                            {"clip": clip, "frame": frame, "line": i,
                             "src": len(sp), "var": len(vp), "formula": formula})
                    # new points present
                    new_pts = list(zip([float(x) for x in info["new_xs"]],
                                       [float(y) for y in info["new_ys_clipped"]]))
                    un, _ = max_coord_dist_ok(new_pts, vp)
                    if un == 0:
                        newpoints_ok += 1
                else:
                    n_not_extended += 1
                # collect concrete failures (full pipeline)
                if not (u == 0 and len(expected_full) == len(vp)):
                    if len(failures) < 10:
                        failures.append({
                            "clip": clip, "frame": frame, "line": i,
                            "extended": bool(did), "src_n": len(sp), "var_n": len(vp),
                            "exp_n": len(expected_full), "unmatched_exp": u,
                            "src_first": sp[0], "var_top": min(p[1] for p in vp),
                            "exp_top": (min(p[1] for p in expected_full) if expected_full else None),
                        })
        results[name] = {
            "extent": ext,
            "n_lines": n_lines,
            "full_pipeline_ok": full_pipeline_ok,
            "full_pipeline_fail": n_lines - full_pipeline_ok,
            "n_extended": n_extended,
            "n_not_extended": n_not_extended,
            "top_eq_top_old_minus_E_ok": top_ok,
            "top_clip_aware_ok": top_clip_aware_ok,
            "count_formula_ok": count_formula_ok,
            "count_formula_note": "src_n + E/5; only expected when trim adds no point",
            "count_formula_mismatch_samples": count_mismatch_samples,
            "newpoints_present_ok": newpoints_ok,
            "failures": failures,
        }
    return results


# --------------------------------------------------------------------------- #
# CHECK 4 — trim correctness (all three)
# --------------------------------------------------------------------------- #
def check_trim():
    out = {}
    for name, zp in VARIANT_ZIPS.items():
        var = load_zip_lines(zp)
        violations = []
        n_lines = 0
        worst_excess = 0.0
        for (clip, frame), lines in var.items():
            for i, p in enumerate(lines):
                n_lines += 1
                top = min(q[1] for q in p)
                cut = gt_bottom_for_top(top)
                # max y after clamp-free reading
                ymax = max(q[1] for q in p)
                if ymax > cut + TOL:
                    excess = ymax - cut
                    worst_excess = max(worst_excess, excess)
                    if len(violations) < 10:
                        violations.append({"clip": clip, "frame": frame, "line": i,
                                           "top": top, "cut": cut, "ymax": ymax})
        out[name] = {"n_lines": n_lines, "violations": len(violations),
                     "worst_excess": worst_excess, "samples": violations}
    return out


# --------------------------------------------------------------------------- #
# CHECK 5 — union completeness
# --------------------------------------------------------------------------- #
def check_union():
    oracle = load_oracle()
    src = load_zip_lines(SOURCE_ZIP)
    uni = load_zip_lines(VARIANT_ZIPS["union_trim0"])
    cache = {}

    literal_ok = 0          # max IoU(src_line, any union line) >= 0.99
    literal_fail = 0
    literal_fail_samples = []
    refined_ok = 0          # union contains round(trim0(src_line)) at IoU>=0.999
    refined_fail = 0
    refined_fail_samples = []
    dropped_refs = 0        # source lines whose trim0 is empty (genuinely absent)
    iou_hist = {"=1.0": 0, "[0.99,1)": 0, "[0.9,0.99)": 0, "[0.5,0.9)": 0, "<0.5": 0}
    n_src_total = 0
    ref_masks_by_img = {}   # key -> list of masks of round(trim0(src))

    for (clip, frame), src_lines in src.items():
        uni_lines = uni.get((clip, frame), [])
        uni_masks = [lane_mask(oracle, p, cache) for p in uni_lines]
        valid_masks = []
        for sp in src_lines:
            n_src_total += 1
            t0 = reimpl_trim0(sp)
            if not t0:
                dropped_refs += 1
                continue
            t0r = fmt_round(t0)
            tm = lane_mask(oracle, t0r, cache)
            valid_masks.append(tm)
            sm = lane_mask(oracle, sp, cache)
            best = 0.0
            for um in uni_masks:
                v = iou(sm, um)
                if v > best:
                    best = v
            if best >= 0.99:
                literal_ok += 1
            else:
                literal_fail += 1
                if len(literal_fail_samples) < 8:
                    literal_fail_samples.append(
                        {"clip": clip, "frame": frame, "best_iou": round(best, 4),
                         "src_n": len(sp), "src_bottom": max(q[1] for q in sp)})
            if best >= 1.0 - 1e-9:
                iou_hist["=1.0"] += 1
            elif best >= 0.99:
                iou_hist["[0.99,1)"] += 1
            elif best >= 0.9:
                iou_hist["[0.9,0.99)"] += 1
            elif best >= 0.5:
                iou_hist["[0.5,0.9)"] += 1
            else:
                iou_hist["<0.5"] += 1
        # refined: does union contain the (rounded) trimmed ref line at IoU>=0.999?
        for sp in src_lines:
            t0 = reimpl_trim0(sp)
            if not t0:
                continue
            tm = lane_mask(oracle, fmt_round(t0), cache)
            best = 0.0
            for um in uni_masks:
                v = iou(tm, um)
                if v > best:
                    best = v
            if best >= 0.999:
                refined_ok += 1
            else:
                refined_fail += 1
                if len(refined_fail_samples) < 8:
                    refined_fail_samples.append(
                        {"clip": clip, "frame": frame, "best_iou": round(best, 4)})
        ref_masks_by_img[(clip, frame)] = valid_masks

    # independent novel recount: union lines not matching any round(trim0(src))
    novel = 0
    ref_survivors_in_union = 0
    novel_samples = []
    novel_best_ious = []
    for (clip, frame), uni_lines in uni.items():
        valid = ref_masks_by_img.get((clip, frame), [])
        for j, ul in enumerate(uni_lines):
            um = lane_mask(oracle, ul, cache)
            best = 0.0
            for tm in valid:
                v = iou(um, tm)
                if v > best:
                    best = v
            if best >= 0.999:
                ref_survivors_in_union += 1
            else:
                novel += 1
                novel_best_ious.append(best)
                if len(novel_samples) < 8:
                    novel_samples.append({"clip": clip, "frame": frame, "line": j,
                                          "max_iou_vs_ref": round(best, 4)})

    return {
        "n_source_lines": n_src_total,
        "ref_lines_with_empty_trim0": dropped_refs,
        "literal_iou_ge_0.99_ok": literal_ok,
        "literal_iou_ge_0.99_fail": literal_fail,
        "literal_fail_samples": literal_fail_samples,
        "literal_iou_hist": iou_hist,
        "refined_trim_eq_found_ok": refined_ok,
        "refined_trim_eq_found_fail": refined_fail,
        "refined_fail_samples": refined_fail_samples,
        "union_total_lines": sum(len(v) for v in uni.values()),
        "union_ref_survivors": ref_survivors_in_union,
        "union_novel_lines": novel,
        "novel_max_iou_max": (round(max(novel_best_ious), 4) if novel_best_ious else None),
        "novel_samples": novel_samples,
    }


# --------------------------------------------------------------------------- #
# CHECK 6 — clip neutrality
# --------------------------------------------------------------------------- #
def check_clip_neutrality():
    oracle = load_oracle()
    cases = []

    def lane_mask_raw(pts):
        return oracle.draw_lane_mask(oracle.interp_lane(pts), oracle.LINE_WIDTH)

    def clip(pts):
        return [(clamp01(x, 0, IMG_WIDTH - 1), clamp01(y, 0, IMG_HEIGHT - 1)) for x, y in pts]

    # ---- synthetic out-of-frame cases -------------------------------------
    synth = {
        "x_over_1404": [(100, 700), (300, 600), (700, 500), (1100, 400), (1404, 200)],
        "x_1365.5": [(100, 700), (300, 600), (700, 500), (1100, 400), (1365.5, 200)],
        "y_over_730": [(100, 730), (200, 600), (300, 500), (400, 300), (500, 100)],
        "y_negative": [(600, 700), (600, 600), (600, 500), (600, 400), (610, -30)],
        "mixed": [(0, 719), (100, 600), (200, 500), (1365.5, 120), (1404, 40), (500, 730)],
        "x_negative": [(-40, 600), (200, 500), (500, 400), (800, 300), (1000, 200)],
    }
    for name, pts in synth.items():
        m_raw = lane_mask_raw(pts)
        m_clip = lane_mask_raw(clip(pts))
        same = bool(np.array_equal(m_raw, m_clip))
        diff = int(np.logical_xor(m_raw, m_clip).sum())
        cases.append({"case": name, "kind": "synthetic", "identical": same,
                      "diff_pixels": diff})

    # ---- real cases from the extend variants (appended points get clipped) --
    for vname in ("extend60_trim0", "extend120_trim0"):
        ext = EXTENT[vname]
        src = load_zip_lines(SOURCE_ZIP)
        var = load_zip_lines(VARIANT_ZIPS[vname])
        real_fail = 0
        real_checked = 0
        real_examples = []
        for (clip_key, frame), src_lines in src.items():
            var_lines = var.get((clip_key, frame), [])
            for i, sp in enumerate(src_lines):
                if i >= len(var_lines):
                    break
                extended, did, info = reimpl_extend_up(sp, ext)
                if not did:
                    continue
                # was any appended point actually clipped (i.e. off-frame pre-clip)?
                raw_ys = info["new_ys_raw"]
                raw_xs = np.polyval(info["coef"], raw_ys)
                clipped = np.any((raw_ys < 0) | (raw_ys > IMG_HEIGHT - 1) |
                                 (raw_xs < 0) | (raw_xs > IMG_WIDTH - 1))
                if not clipped:
                    continue
                real_checked += 1
                m_raw = lane_mask_raw(extended)                 # unclipped
                m_clip = lane_mask_raw(clip(extended))          # pre-clipped
                if not np.array_equal(m_raw, m_clip):
                    real_fail += 1
                    if len(real_examples) < 6:
                        real_examples.append({"clip": clip_key, "frame": frame,
                                              "line": i, "extent": ext})
        cases.append({"case": f"{vname}_real_clipped", "kind": "real",
                      "checked": real_checked, "mask_differs": real_fail,
                      "identical": real_fail == 0, "examples": real_examples})

    return {"cases": cases}


# --------------------------------------------------------------------------- #
# CHECK 5b — full independent union rebuild (definitive accounting)
# --------------------------------------------------------------------------- #
def reimpl_clamp(points):
    return [(min(max(x, 0.0), IMG_WIDTH - 1.0), min(max(y, 0.0), IMG_HEIGHT - 1.0))
            for x, y in points]


def check_union_rebuild():
    """Rebuild the union from scratch and diff it against the delivered zip."""
    oracle = load_oracle()
    src = load_zip_lines(SOURCE_ZIP)
    uni = load_zip_lines(VARIANT_ZIPS["union_trim0"])

    support_cache = {}

    def get_support(tdir, clip, frame):
        key = (str(tdir), clip, frame)
        if key not in support_cache:
            f = tdir / clip / frame
            if f.is_file():
                lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
                support_cache[key] = [parse_line(ln) for ln in lines]
            else:
                support_cache[key] = None
        return support_cache[key]

    novel_pre_trim = 0
    ref_pre_trim = 0
    rejected = 0
    missing_support = 0
    dropped_by_trim = 0
    ref_collisions = 0
    exact_match_images = 0
    mismatch_images = 0
    mismatch_samples = []

    for (clip, frame), ref_lines in src.items():
        cache = {}
        kept = list(ref_lines)
        kept_masks = [lane_mask(oracle, p, cache) for p in kept]
        ref_pre_trim += len(ref_lines)
        for tdir in SUPPORT_TREES:
            lines = get_support(tdir, clip, frame)
            if lines is None:
                missing_support += 1
                continue
            for cand in lines:
                cm = lane_mask(oracle, cand, cache)
                dup = False
                for km in kept_masks:
                    if iou(cm, km) > 0.5:
                        dup = True
                        break
                if dup:
                    rejected += 1
                else:
                    kept.append(cand)
                    kept_masks.append(cm)
                    novel_pre_trim += 1
        out_lines = []
        for points in kept:
            t = reimpl_trim0(points)
            if t:
                out_lines.append(fmt_round(reimpl_clamp(t)))
            else:
                dropped_by_trim += 1
        delivered = [fmt_round(p) for p in uni.get((clip, frame), [])]
        if out_lines == delivered:
            exact_match_images += 1
        else:
            mismatch_images += 1
            if len(mismatch_samples) < 8:
                mismatch_samples.append({"clip": clip, "frame": frame,
                                         "rebuilt_n": len(out_lines),
                                         "delivered_n": len(delivered)})
    return {
        "rebuilt_total_lines": ref_pre_trim + novel_pre_trim - dropped_by_trim,
        "ref_pre_trim": ref_pre_trim,
        "novel_pre_trim": novel_pre_trim,
        "rejected_candidates": rejected,
        "dropped_by_trim": dropped_by_trim,
        "missing_support_files": missing_support,
        "images_exact_match": exact_match_images,
        "images_mismatch": mismatch_images,
        "mismatch_samples": mismatch_samples,
    }


# --------------------------------------------------------------------------- #
# CHECK 6b — clip neutrality on the union's supporting-model lines
# --------------------------------------------------------------------------- #
def check_clip_support_lines():
    oracle = load_oracle()
    checked = 0
    affected = 0
    differs = 0
    examples = []
    for tdir in SUPPORT_TREES:
        for clip_dir in tdir.iterdir():
            if not clip_dir.is_dir():
                continue
            for f in clip_dir.glob("*.lines.txt"):
                for ln in f.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    pts = parse_line(ln)
                    oob = any(x < 0 or x > IMG_WIDTH - 1 or y < 0 or y > IMG_HEIGHT - 1
                              for x, y in pts)
                    if not oob:
                        continue
                    affected += 1
                    m_raw = oracle.draw_lane_mask(oracle.interp_lane(pts), oracle.LINE_WIDTH)
                    m_clip = oracle.draw_lane_mask(
                        oracle.interp_lane(reimpl_clamp(pts)), oracle.LINE_WIDTH)
                    checked += 1
                    if not np.array_equal(m_raw, m_clip):
                        differs += 1
                        if len(examples) < 6:
                            examples.append({"tree": str(tdir.name), "clip": clip_dir.name,
                                             "file": f.name})
    return {"support_lines_with_oob_coords": affected,
            "mask_compared": checked, "mask_differs": differs,
            "identical": differs == 0, "examples": examples}


# --------------------------------------------------------------------------- #
# EXTRA — oracle integrity
# --------------------------------------------------------------------------- #
def check_oracle_integrity():
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    readme = (ORACLE_DIR / "README.md").read_text(encoding="utf-8")
    expected = {
        "score.py": "b2f4c9b21de1083de8a420c106262e799947fac37c938994f7d160b8c62de0d2",
        "check_submission.py": "c7897bdbd81d19b56ee0b6b9a803c0645428403ffa13c133e1ea41a422c02043",
        "requirements_official.txt": "eb92e5ce6ff4b97db3de410bc068837f3038f83561f9bde581b5055530f67c05",
    }
    files = {}
    for name, want in expected.items():
        got = sha(ORACLE_DIR / name)
        files[name] = {"sha256": got, "readme_value": want,
                       "match": got == want, "in_readme": want in readme}
    pycache = ORACLE_DIR / "__pycache__"
    pyc = []
    if pycache.is_dir():
        pyc = sorted(p.name for p in pycache.glob("*.pyc"))
    return {"files": files, "pycache_exists": pycache.is_dir(), "pyc_files": pyc,
            "pycache_mtime": (None if not pycache.is_dir()
                              else __import__("datetime").datetime.fromtimestamp(
                                  pycache.stat().st_mtime).isoformat())}


# --------------------------------------------------------------------------- #
def main():
    res = {}
    print("== CHECK 1 structure ==")
    res["check1_structure"] = check_structure()
    print(json.dumps(res["check1_structure"], ensure_ascii=False, indent=1))

    print("== CHECK 2 official precheck ==")
    res["check2_precheck"] = check_precheck()
    for k, v in res["check2_precheck"].items():
        print(k, "rc=", v.get("returncode"), "->", v.get("stdout", v))

    print("== CHECK 3 extrapolation ==")
    res["check3_extrapolation"] = check_extrapolation()
    for k, v in res["check3_extrapolation"].items():
        print(k, {kk: vv for kk, vv in v.items()
                  if not kk.endswith("samples") and kk != "failures"})

    print("== CHECK 4 trim ==")
    res["check4_trim"] = check_trim()
    for k, v in res["check4_trim"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "samples"})

    print("== CHECK 5 union completeness ==")
    res["check5_union"] = check_union()
    print(json.dumps({k: v for k, v in res["check5_union"].items()
                      if "samples" not in k}, ensure_ascii=False, indent=1))

    print("== CHECK 5b union rebuild ==")
    res["check5b_union_rebuild"] = check_union_rebuild()
    print(json.dumps(res["check5b_union_rebuild"], ensure_ascii=False, indent=1))

    print("== CHECK 6 clip neutrality ==")
    res["check6_clip"] = check_clip_neutrality()
    print(json.dumps(res["check6_clip"], ensure_ascii=False, indent=1))

    print("== CHECK 6b clip neutrality on support lines ==")
    res["check6b_clip_support"] = check_clip_support_lines()
    print(json.dumps(res["check6b_clip_support"], ensure_ascii=False, indent=1))

    print("== EXTRA oracle integrity ==")
    res["extra_oracle"] = check_oracle_integrity()
    print(json.dumps(res["extra_oracle"], ensure_ascii=False, indent=1))

    if not REPORTS_DIR.exists():
        REPORTS_DIR.mkdir(parents=True)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[done] ->", OUT_JSON)


if __name__ == "__main__":
    main()
