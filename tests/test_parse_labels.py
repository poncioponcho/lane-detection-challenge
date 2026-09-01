"""Synthetic-data tests for T20 (parsing) + T21 (consistency).

Runs without real data: generates a few lane polylines, writes all three label
formats, parses them back, and verifies cross-format consistency.

Run:  python tests/test_parse_labels.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import cv2

from common.io_utils import (parse_txt_line, read_lines_txt, write_lines_txt,
                             read_json_lanes, read_instance_png)
from data.parse_labels import parse_labels
from data.check_label_consistency import check_image

W, H = 1366, 720


def make_lanes():
    """Three roughly-vertical lanes (lane-like, so PNG centerline is clean)."""
    ys = np.linspace(80, 700, 60, dtype=np.float32)
    lanes = []
    for base_x, slope in [(200.0, 0.0), (683.0, -60.0), (1100.0, 40.0)]:
        xs = (base_x + slope * (ys - 80) / 620.0).astype(np.float32)
        lanes.append(np.stack([xs, ys], axis=1))
    return lanes


def render_png(lanes, path, thickness=8):
    """Render instance-id PNG: each lane stroke filled with its 1-based id."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for i, pts in enumerate(lanes, start=1):
        p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [p], isClosed=False, color=i, thickness=thickness)
    cv2.imwrite(str(path), mask)
    return mask


def test_txt_roundtrip():
    lanes = make_lanes()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.lines.txt"
        write_lines_txt(p, lanes)
        back = read_lines_txt(p)
        assert len(back) == len(lanes), f"{len(back)} != {len(lanes)}"
        for a, b in zip(lanes, back):
            assert np.allclose(a, b, atol=0.05), "txt roundtrip drift"
    print("[ok] txt roundtrip: 3 lanes, 1-decimal")


def test_txt_edges():
    # empty file -> no lanes
    assert parse_txt_line("") is None
    assert parse_txt_line("   \r") is None
    # comment-ish / odd tokens
    try:
        parse_txt_line("1 2 3")
        assert False, "odd tokens should raise"
    except ValueError:
        pass
    # comma separator tolerated
    pts = parse_txt_line("10,20 30,40")
    assert pts.shape == (2, 2) and pts[0, 0] == 10
    print("[ok] txt edge cases: empty/odd/comma")


def test_json_variants():
    lanes = make_lanes()
    import json
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # variant A: {"lanes": [[[x,y],...], ...]}
        a = d / "a.json"
        a.write_text(json.dumps({"lanes": [l.tolist() for l in lanes]}))
        r1 = read_json_lanes(a)
        assert len(r1) == 3
        # variant B: {"Lines": [{"x":[...],"y":[...]}, ...]}
        b = d / "b.json"
        b.write_text(json.dumps({"Lines": [
            {"x": l[:, 0].tolist(), "y": l[:, 1].tolist()} for l in lanes]}))
        r2 = read_json_lanes(b)
        assert len(r2) == 3
        # variant C: raw list of flat lanes
        c = d / "c.json"
        c.write_text(json.dumps([l.reshape(-1).tolist() for l in lanes]))
        r3 = read_json_lanes(c)
        assert len(r3) == 3
        for r in (r1, r2, r3):
            assert np.allclose(r[0][0], lanes[0][0], atol=1e-4)
    print("[ok] json: 3 schema variants parsed")


def test_png_extraction():
    lanes = make_lanes()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        render_png(lanes, p, thickness=8)
        back = read_instance_png(p)
        assert len(back) == 3, f"png extracted {len(back)} lanes"
        # centerline within half stroke width + quantization
        from common.types import mean_lateral_error
        # png lanes are sorted by mean x; re-sort lanes the same way
        lanes_sorted = sorted(lanes, key=lambda l: float(l[:, 0].mean()))
        for a, b in zip(lanes_sorted, back):
            e = mean_lateral_error(a, b)
            assert e < 5.0, f"png centerline error {e:.1f}px too large"
    print("[ok] png: 3 lanes extracted, centerline error < 5px")


def test_consistency_ok():
    lanes = make_lanes()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = d / "img.jpg"
        img.write_bytes(b"")
        write_lines_txt(d / "img.lines.txt", lanes)
        import json
        (d / "img.json").write_text(json.dumps({"lanes": [l.tolist() for l in lanes]}))
        render_png(lanes, d / "img.png", thickness=8)
        res = check_image("img", read_lines_txt(d / "img.lines.txt"),
                          read_json_lanes(d / "img.json"),
                          read_instance_png(d / "img.png"))
        assert res.ok, f"consistency should pass: {res.note}"
        # matched = total pairs across comparisons: 3 lanes x (json + png)
        assert res.matched == 6, f"expected 6 matched pairs, got {res.matched}"
    print("[ok] consistency: 3 formats agree (all green)")


def test_consistency_mismatch():
    lanes = make_lanes()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # png has 2 lanes fewer than txt/json (default count_tol=1 tolerates ±1,
        # so use a 2-lane gap to force the flag)
        res = check_image("img", lanes, lanes, lanes[:1])
        assert not res.ok, "count mismatch should be flagged"
        assert "png count" in res.note
    print("[ok] consistency: count mismatch flagged")


if __name__ == "__main__":
    test_txt_roundtrip()
    test_txt_edges()
    test_json_variants()
    test_png_extraction()
    test_consistency_ok()
    test_consistency_mismatch()
    print("\nALL PARSER TESTS PASSED")
