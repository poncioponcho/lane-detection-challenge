#!/usr/bin/env python3
"""Multi-seed consensus lane fusion (DECISIONS §36.3 / top3_sprint_plan §6 T2-A).

Honest, no-tricks consensus voting across N seed predictions on the SAME image
set.  Mechanism is orthogonal to score-aware NMS (already Red): it kills
seed-specific hallucination FP by requiring a lane to recur in a quorum of
seeds, and it stabilises geometry by averaging across the agreeing members.

Inputs (per seed): a predictions dir of `<clip>/<img>.lines.txt` (one lane per
row, space-separated x y pairs) AND a frozen-export sidecar
`prediction_scores.json` whose `scores_by_image[img_id]` is a list of
per-lane confidences (positive_class_softmax_probability, [0,1], post_nms),
aligned with the .lines.txt row order.

Algorithm (per image):
  1. load every seed's lanes + per-lane confidences;
  2. build a cross-seed match graph: an edge links two lanes from DIFFERENT
     seeds whose mean |dx| over the overlapping y-range < --max-dx;
  3. connected components are consensus clusters; a cluster is KEPT only if it
     contains lanes from >= --quorum distinct seeds;
  4. the kept lane's geometry defaults to the highest-mean-confidence
     member's lane (avoids resampling artefacts); ``--geometry median`` can
     instead denoise the x coordinates pointwise on that member's y grid while
     preserving its endpoint coverage. Its confidence is the MEAN of members.

Single-seed-unique lines (clusters with < quorum seeds) are dropped by design.
Output: <out>/<clip>/<img>.lines.txt + (with --emit-scores) a merged
prediction_scores.json sidecar in the same frozen schema, ready for the Oracle
or the submit packer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CANVAS_W = 1366


def load_lanes(path: Path) -> list[np.ndarray]:
    lanes: list[np.ndarray] = []
    if not path.exists():
        return lanes
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) < 4 or len(tokens) % 2:
            continue
        lanes.append(np.asarray(tokens, dtype=np.float64).reshape(-1, 2))
    return lanes


def mean_dx(lane_a: np.ndarray, lane_b: np.ndarray, samples: int = 36) -> float | None:
    """Mean |dx| between two y-monotone lanes over their overlapping y-range."""
    if len(lane_a) < 2 or len(lane_b) < 2:
        return None
    a = lane_a[np.argsort(lane_a[:, 1])]
    b = lane_b[np.argsort(lane_b[:, 1])]
    lo = max(a[0, 1], b[0, 1])
    hi = min(a[-1, 1], b[-1, 1])
    if hi - lo < 2.0:
        return None
    ys = np.linspace(lo, hi, samples)
    dx = np.interp(ys, a[:, 1], a[:, 0]) - np.interp(ys, b[:, 1], b[:, 0])
    return float(np.mean(np.abs(dx)))


def load_scores(json_path: Path) -> dict[str, list[float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data["scores_by_image"]


def _interpolate_on_base_grid(
    lane: np.ndarray, base_y: np.ndarray, fallback_x: np.ndarray
) -> np.ndarray:
    """Interpolate one lane on the selected member's y grid safely.

    Outside a member's observed y range we keep the selected member's x.  This
    avoids shortening a consensus line merely because one agreeing seed
    emitted fewer endpoint samples.
    """
    ordered = lane[np.argsort(lane[:, 1])]
    y = ordered[:, 1]
    x = ordered[:, 0]
    y, unique = np.unique(y, return_index=True)
    x = x[unique]
    inside = (base_y >= y[0]) & (base_y <= y[-1])
    result = fallback_x.copy()
    if np.any(inside):
        result[inside] = np.interp(base_y[inside], y, x)
    return result


def _fused_geometry(
    member_nodes: list[tuple[int, int, np.ndarray, float]],
    geometry: str,
) -> np.ndarray:
    """Choose or denoise the geometry for one retained consensus cluster."""
    best = max(member_nodes, key=lambda mn: mn[3])
    base = best[2]
    if geometry == "best" or len(member_nodes) == 1:
        return base
    ordered = base[np.argsort(base[:, 1])]
    base_y = ordered[:, 1]
    fallback_x = ordered[:, 0]
    aligned = np.vstack([
        _interpolate_on_base_grid(node[2], base_y, fallback_x)
        for node in member_nodes
    ])
    if geometry == "median":
        fused_x = np.median(aligned, axis=0)
    elif geometry == "mean":
        fused_x = np.mean(aligned, axis=0)
    else:
        raise ValueError(f"unsupported consensus geometry: {geometry!r}")
    fused = np.column_stack((fused_x, base_y))
    # Match the source/export convention (descending y) regardless of the
    # internal interpolation order.
    return fused[np.argsort(-fused[:, 1])]


def fuse_image(
    seed_lanes: list[list[np.ndarray]],
    seed_scores: list[list[float]],
    quorum: int,
    max_dx: float,
    geometry: str = "best",
) -> tuple[list[np.ndarray], list[float], int]:
    """Return (fused_lanes, fused_scores) for one image."""
    n_seeds = len(seed_lanes)
    # nodes: (seed_idx, lane_idx, lane, score)
    nodes: list[tuple[int, int, np.ndarray, float]] = []
    for s, (lanes, scores) in enumerate(zip(seed_lanes, seed_scores)):
        for li, lane in enumerate(lanes):
            score = float(scores[li]) if li < len(scores) else 0.0
            nodes.append((s, li, lane, score))
    if not nodes:
        # Keep the return shape identical to the normal path.  Empty
        # prediction files are valid in HardLane and must remain empty in the
        # fused output rather than aborting the whole 900/1000-image export.
        return [], [], 0

    # union-find over nodes
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # cross-seed edges by mean_dx (O(M^2) but M is tiny per image)
    for i in range(len(nodes)):
        si, _, lane_i, _ = nodes[i]
        for j in range(i + 1, len(nodes)):
            sj, _, lane_j, _ = nodes[j]
            if si == sj:
                continue  # never merge two lanes of the same seed
            d = mean_dx(lane_i, lane_j)
            if d is not None and d < max_dx:
                union(i, j)

    # group by root
    clusters: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        clusters.setdefault(find(i), []).append(i)

    fused_lanes: list[np.ndarray] = []
    fused_scores: list[float] = []
    subquorum_lanes = 0
    for members in clusters.values():
        member_nodes = [nodes[m] for m in members]
        distinct_seeds = {mn[0] for mn in member_nodes}
        if len(distinct_seeds) < quorum:
            subquorum_lanes += len(member_nodes)
            continue  # single-seed-unique or sub-quorum → drop
        fused_lanes.append(_fused_geometry(member_nodes, geometry))
        fused_scores.append(float(np.mean([mn[3] for mn in member_nodes])))

    # sort by descending confidence (matches export convention)
    order = sorted(range(len(fused_lanes)), key=lambda i: fused_scores[i], reverse=True)
    fused_lanes = [fused_lanes[i] for i in order]
    fused_scores = [fused_scores[i] for i in order]
    return fused_lanes, fused_scores, subquorum_lanes


def export_lines(lanes: list[np.ndarray]) -> str:
    lines = []
    for lane in lanes:
        flat = lane.reshape(-1)
        lines.append(" ".join(f"{v:.1f}" for v in flat))
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed", action="append", required=True, metavar="PRED_DIR=SCORES_JSON",
        help="repeatable; each seed as 'pred_dir=scores_json'",
    )
    ap.add_argument("--out", required=True, help="output predictions dir")
    ap.add_argument("--quorum", type=int, default=0, help=">=N seeds; 0 => majority (n//2+1)")
    ap.add_argument("--max-dx", type=float, default=15.0, help="match threshold px (mean |dx|)")
    ap.add_argument(
        "--geometry", choices=("best", "median", "mean"), default="best",
        help="geometry for a retained cluster: best member or pointwise denoising",
    )
    ap.add_argument("--emit-scores", action="store_true", help="write merged prediction_scores.json")
    args = ap.parse_args()

    seeds = []
    for spec in args.seed:
        if "=" not in spec:
            raise SystemExit(f"--seed must be 'pred_dir=scores_json', got {spec!r}")
        pred_dir, scores_json = spec.split("=", 1)
        seeds.append((Path(pred_dir), Path(scores_json)))

    n_seeds = len(seeds)
    quorum = args.quorum if args.quorum > 0 else (n_seeds // 2 + 1)
    if quorum > n_seeds:
        raise SystemExit(f"quorum {quorum} > n_seeds {n_seeds}")
    print(
        f"[consensus] seeds={n_seeds} quorum={quorum} max_dx={args.max_dx} "
        f"geometry={args.geometry}"
    )

    # load all sidecars
    seed_scores = [load_scores(sj) for _, sj in seeds]
    # image universe = intersection of all seeds' score keys (same image set)
    image_ids = set(seed_scores[0].keys())
    for sc in seed_scores[1:]:
        image_ids &= set(sc.keys())
    image_ids = sorted(image_ids)
    print(f"[consensus] images in all seeds: {len(image_ids)}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    merged_scores: dict[str, list[float]] = {}
    kept_total = 0
    input_total = 0
    dropped_total = 0
    for img in image_ids:
        seed_lanes = [load_lanes(pd / f"{img}.lines.txt") for pd, _ in seeds]
        per_img_scores = [seed_scores[s].get(img, []) for s in range(n_seeds)]
        fused_lanes, fused_scores, subquorum = fuse_image(
            seed_lanes, per_img_scores, quorum, args.max_dx, args.geometry)
        kept_total += len(fused_lanes)
        input_total += sum(len(l) for l in seed_lanes)
        dropped_total += subquorum
        rel = Path(f"{img}.lines.txt")
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(export_lines(fused_lanes), encoding="utf-8")
        merged_scores[img] = fused_scores

    if args.emit_scores:
        sidecar = {
            "status": "pass",
            "checkpoint": "consensus_fusion",
            "candidate_export_conf_threshold": 0.0,
            "score_schema_version": 1,
            "score_semantics": "positive_class_softmax_probability",
            "score_range": {"min": 0.0, "max": 1.0, "inclusive": True},
            "post_nms": True,
            "images": len(image_ids),
            "scores_by_image": merged_scores,
        }
        (out_root / "prediction_scores.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"[consensus] kept {kept_total} output lanes from {input_total} input lanes "
        f"(dropped {dropped_total} sub-quorum member lanes) across {len(image_ids)} images"
    )


if __name__ == "__main__":
    main()
