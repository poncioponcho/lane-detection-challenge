"""Synthetic tests for the submit pack/verify chain (T1.3 / P0-A03 A04).

Runs without real data: builds a fake pred dir + expected list, packs a
submit.zip, verifies it passes, then injects each error class and confirms
verification fails.

Run:  python tests/test_submit.py
"""
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pytest

from submit.export_lines import export_lines, lane_to_line, assert_valid_coords
from submit.pack_submit import _normalize_rel, pack_submit, read_expected
from submit.prepare_submit import (canonicalize_prediction_dir,
                                   prepare_submission)
from submit.verify_submit import verify_submit, validate_line
from common.checksum import file_meta

CANVAS_W, CANVAS_H = 1366, 720


def make_lanes():
    ys = np.linspace(80, 700, 30, dtype=np.float32)
    return [np.stack([np.full(30, 300.0, np.float32), ys], axis=1),
            np.stack([np.full(30, 700.0, np.float32), ys], axis=1)]


def build_pred_dir(d, rels):
    for rel in rels:
        p = Path(d) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        export_lines(make_lanes(), p)


def test_export_validation():
    # valid
    assert lane_to_line([[10.0, 20.0], [30.5, 40.0]]) == "10.0 20.0 30.5 40.0"
    # odd / too-few / NaN / normalized / out-of-bounds all raise
    for bad, msg in [
        ([[1, 2, 3]], "odd"),
        ([[1.0, 2.0]], "too few"),
        ([[1.0, 2.0], [float("nan"), 4.0]], "nan"),
        ([[0.1, 0.2], [0.3, 0.4]], "normalized"),
        ([[1.0, 2.0], [2000.0, 4.0]], "out of bounds"),
    ]:
        try:
            lane_to_line(bad)
            assert False, f"should raise: {msg}"
        except ValueError:
            pass
    # Distinct source points can collapse at the mandated one-decimal output.
    with pytest.raises(ValueError, match="after serialization"):
        lane_to_line([[100.01, 700.01], [100.04, 700.04]])
    print("[ok] export_lines: strict validation (5 error classes)")


@pytest.mark.parametrize("line, error", [
    ("100.0 700.0 100.0 700.0", "after consecutive-deduplication"),
    ("1,2 3,4 5,6 7,8", "non-numeric"),
    ("100 700 200 500", "not 1-decimal"),
    # 2026-09-17: these two used to be 1366.0 / 720.0 and were expected to fail.
    # They are NOT failures under the organisers' contract: score.py clips x to
    # [0,1365] and y to [0,719] before rasterising, and check_submission.py
    # validates no bounds at all, so a point sitting exactly on the border is a
    # legal submission that the scorer trivially clamps. testB genuinely
    # produces x in (1365, 1366] on 3-21 lanes per tree. The bound is now a 1 px
    # tolerance, so the cases below are moved genuinely past it.
    ("100.0 700.0 1367.0 500.0", "out of bounds"),
    ("100.0 700.0 200.0 721.0", "out of bounds"),
])
def test_verify_line_rejects_only_genuinely_out_of_canvas(line, error):
    assert error in validate_line(line)


@pytest.mark.parametrize("line", [
    # exactly on the right border: the real testB case
    "100.0 700.0 1366.0 500.0",
    "100.0 700.0 1365.9 500.0",
    # exactly on the bottom border
    "100.0 700.0 200.0 720.0",
])
def test_verify_line_accepts_canvas_border_effect(line):
    assert validate_line(line) is None


def test_export_clamps_border_effect_instead_of_raising():
    """A ≤1 px overshoot must be snapped onto the canvas, not rejected.

    Regression guard for 2026-09-17: rejecting it aborted the whole testB
    packaging step, and pack() hides prepare_submit's output, so it failed
    silently into a zero-lane zip.
    """
    assert lane_to_line([[10.0, 20.0], [1366.0, 400.0]]) == "10.0 20.0 1365.0 400.0"
    assert lane_to_line([[10.0, 20.0], [1365.9, 400.0]]) == "10.0 20.0 1365.0 400.0"
    # far outside is still a hard error, not a silent clamp
    with pytest.raises(ValueError, match="out of bounds"):
        lane_to_line([[10.0, 20.0], [1400.0, 400.0]])


def test_verify_rejects_lane_and_point_resource_overflow(tmp_path):
    rel = "clip/00000.lines.txt"
    pred = tmp_path / "pred"
    target = pred / rel
    target.parent.mkdir(parents=True)

    target.write_text("100.0 700.0 200.0 500.0\n" * 65, encoding="utf-8")
    archive = pack_submit(pred, [rel], tmp_path / "too_many_lanes.zip")
    ok, report = verify_submit(archive, [rel])
    assert not ok and "too many lanes 65" in report

    points = " ".join(f"{100 + index * 0.1:.1f} 700.0" for index in range(2049))
    target.write_text(points + "\n", encoding="utf-8")
    archive = pack_submit(pred, [rel], tmp_path / "too_many_points.zip")
    ok, report = verify_submit(archive, [rel])
    assert not ok and "too many points 2049" in report


def test_zip_with_collapsed_lane_fails_final_gate(tmp_path):
    rel = "clip/00000.lines.txt"
    pred = tmp_path / "pred"
    target = pred / rel
    target.parent.mkdir(parents=True)
    target.write_text("100.0 700.0 100.0 700.0\n", encoding="utf-8")
    archive = pack_submit(pred, [rel], tmp_path / "submit.zip")
    ok, report = verify_submit(archive, [rel])
    assert not ok
    assert "after consecutive-deduplication" in report


def test_pack_verify_roundtrip():
    rels = ["clip_0007/00042.lines.txt", "clip_0007/00043.lines.txt",
            "clip_0008/00000.lines.txt"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pred = d / "preds"
        build_pred_dir(pred, rels)
        # one image with NO lane -> empty file
        (pred / "clip_0008/00001.lines.txt").write_text("", encoding="utf-8")
        expected = rels + ["clip_0008/00001.lines.txt"]
        exp_txt = d / "expected.txt"
        exp_txt.write_text("\n".join(expected) + "\n")

        zip_path = pack_submit(pred, read_expected(exp_txt), d / "submit.zip")
        ok, report = verify_submit(zip_path, read_expected(exp_txt),
                                   report_path=d / "verify.md")
        assert ok, report
        assert "PASS" in report
        # root dir is submit/
        with zipfile.ZipFile(zip_path) as zf:
            assert all(n.startswith("submit/") for n in zf.namelist())
    print("[ok] pack -> verify roundtrip PASS (incl. empty no-lane file)")


def test_official_image_paths_normalize_without_weakening_path_safety(tmp_path):
    assert _normalize_rel("/JPEGImages/clip_a/00003.jpg") == \
        "clip_a/00003.lines.txt"
    assert _normalize_rel("JPEGImages/clip_a/00003.jpeg") == \
        "clip_a/00003.lines.txt"
    assert _normalize_rel("./clip_a/00003.lines.txt") == \
        "clip_a/00003.lines.txt"
    for unsafe in ("../../clip_a/00003.jpg", "/clip_a/00003.jpg"):
        with pytest.raises(ValueError, match="unsafe expected path"):
            _normalize_rel(unsafe)

    expected_list = tmp_path / "expected.txt"
    expected_list.write_text("../../clip_a/00003.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe expected path"):
        pack_submit(tmp_path, read_expected(expected_list), tmp_path / "bad.zip")

    official = "/JPEGImages/clip_a/00003.jpg"
    pred = tmp_path / "pred"
    build_pred_dir(pred, ["clip_a/00003.lines.txt"])
    archive = pack_submit(pred, [official], tmp_path / "official-list.zip")
    ok, report = verify_submit(archive, [official])
    assert ok, report


def test_prepare_submission_converts_five_decimals_then_packs_and_verifies(tmp_path):
    rel = "clip_a/00003.lines.txt"
    raw = tmp_path / "raw" / rel
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "100.12345 700.12345 200.56789 500.56789 300.99999 300.00001\n",
        encoding="utf-8",
    )
    canonical = tmp_path / "canonical"
    archive = tmp_path / "submit.zip"
    report = prepare_submission(raw.parents[1], canonical, [rel], archive)

    assert report["status"] == "pass"
    assert report["conversion"]["lanes"] == 1
    expected_text = "100.1 700.1 200.6 500.6 301.0 300.0\n"
    assert (canonical / rel).read_text(encoding="utf-8") == expected_text
    with zipfile.ZipFile(archive) as zf:
        assert zf.read(f"submit/{rel}").decode("utf-8") == expected_text
    assert verify_submit(archive, [rel])[0]


def test_canonicalize_rejects_rounding_collapse_with_relative_path(tmp_path):
    rel = "clip_a/00003.lines.txt"
    raw = tmp_path / "raw" / rel
    raw.parent.mkdir(parents=True)
    raw.write_text(
        "100.01000 700.01000 100.04000 700.04000\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"clip_a/00003\.lines\.txt.*serialization"):
        canonicalize_prediction_dir(
            raw.parents[1], tmp_path / "canonical", [rel]
        )


def test_canonicalize_requires_exact_raw_set_and_rejects_stale_output(tmp_path):
    rel = "clip_a/00003.lines.txt"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing raw prediction"):
        canonicalize_prediction_dir(raw_dir, tmp_path / "canonical", [rel])

    report = canonicalize_prediction_dir(
        raw_dir, tmp_path / "canonical-empty", [rel], missing_as_empty=True
    )
    assert report["missing_as_empty"] == [rel]
    assert (tmp_path / "canonical-empty" / rel).read_bytes() == b""

    extra = raw_dir / "clip_a/99999.lines.txt"
    extra.parent.mkdir(parents=True)
    extra.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="extra files"):
        canonicalize_prediction_dir(
            raw_dir, tmp_path / "canonical-extra", [rel], missing_as_empty=True
        )

    extra.unlink()
    canonical = tmp_path / "canonical-stale"
    stale = canonical / "clip_b/00000.lines.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="stale files"):
        canonicalize_prediction_dir(
            raw_dir, canonical, [rel], missing_as_empty=True
        )


def test_prepare_submission_nonidentity_oracle_rehearsal(tmp_path):
    from data.manifest import build_manifest, write_manifest

    lane_root = tmp_path / "Lane"
    raw_root = tmp_path / "raw"
    rows = []
    for index, frame in enumerate(("00000", "00003")):
        image = lane_root / "JPEGImages" / "clip_a" / f"{frame}.jpg"
        gt = lane_root / "anno_txt" / "clip_a" / f"{frame}.lines.txt"
        pred = raw_root / "clip_a" / f"{frame}.lines.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        gt.parent.mkdir(parents=True, exist_ok=True)
        pred.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"jpeg")
        gt.write_text("100 700 200 500 300 300 400 100\n", encoding="utf-8")
        offset = 0 if index == 0 else 100
        pred.write_text(
            " ".join(
                f"{value:.5f}"
                for value in (
                    100 + offset, 700, 200 + offset, 500,
                    300 + offset, 300, 400 + offset, 100,
                )
            ) + "\n",
            encoding="utf-8",
        )
        rows.append(f"/JPEGImages/clip_a/{frame}.jpg")

    task_list = lane_root / "data/train.txt"
    task_list.parent.mkdir(parents=True)
    task_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
    records = build_manifest(
        task_list, lane_root, "train", enforce_verified_frame_count=False
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, records)

    result = prepare_submission(
        raw_root,
        tmp_path / "canonical",
        [record.pred_rel_path for record in records],
        tmp_path / "submit.zip",
        manifest=manifest,
        gt_dir=lane_root / "anno_txt",
        official_python=sys.executable,
        enforce_official_env=False,
    )
    counts = result["oracle"]["global"]
    assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 1, 1)
    assert counts["f1"] == pytest.approx(0.5)


def test_verify_catches_errors():
    rels = ["clip_0007/00042.lines.txt", "clip_0007/00043.lines.txt"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        pred = d / "preds"
        build_pred_dir(pred, rels)
        exp = d / "expected.txt"

        # (a) missing file: pack with 1 entry, verify expects 2
        pack_exp = d / "pack.txt"
        pack_exp.write_text("clip_0007/00042.lines.txt\n")
        z = pack_submit(pred, read_expected(pack_exp), d / "a.zip")
        verify_exp = d / "verify.txt"
        verify_exp.write_text(
            "clip_0007/00042.lines.txt\nclip_0007/00043.lines.txt\n")
        ok, rep = verify_submit(z, read_expected(verify_exp))
        assert not ok and "missing" in rep

        # (b) extra file in zip (inject directly)
        exp.write_text("\n".join(rels) + "\n")
        z = pack_submit(pred, read_expected(exp), d / "b.zip")
        with zipfile.ZipFile(z, "a") as zf:
            zf.writestr("submit/extra.lines.txt", "")
        ok, rep = verify_submit(z, read_expected(exp))
        assert not ok and "extra" in rep

        # (c) NaN + wrong decimal content
        exp.write_text("clip_0007/00042.lines.txt\n")
        (pred / "clip_0007/00042.lines.txt").write_text(
            "100.0 200.0 300.0 400.0\n1.0 2.0 nan 4.0\n", encoding="utf-8")
        z = pack_submit(pred, read_expected(exp), d / "c.zip")
        ok, rep = verify_submit(z, read_expected(exp))
        assert not ok and ("not 1-decimal" in rep or "NaN" in rep)
    print("[ok] verify catches: missing / extra / NaN+decimal")


def test_checksum():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.bin"
        p.write_bytes(b"hello")
        sha, n = file_meta(p)
        assert n == 5
        assert len(sha) == 64
        import hashlib
        assert sha == hashlib.sha256(b"hello").hexdigest()
    print("[ok] checksum: sha256 + byte count")


if __name__ == "__main__":
    test_export_validation()
    test_pack_verify_roundtrip()
    test_verify_catches_errors()
    test_checksum()
    print("\nALL SUBMIT TESTS PASSED")
