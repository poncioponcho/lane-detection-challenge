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

from submit.export_lines import export_lines, lane_to_line, assert_valid_coords
from submit.pack_submit import pack_submit, read_expected
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
    print("[ok] export_lines: strict validation (5 error classes)")


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
