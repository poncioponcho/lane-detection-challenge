"""T22: deterministic label EDA and visual data-health audit.

The report is derived from the ordered manifest and primary ``.lines.txt``
annotations.  It intentionally does not scan directories, infer scene labels,
or silently discard malformed lanes.  A compact contact sheet renders a fixed
set of extrema so numerical findings remain visually auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from common.io_utils import read_lines_txt
from data.manifest import ManifestRecord, read_manifest

CANVAS_WIDTH = 1366
CANVAS_HEIGHT = 720
QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
OVERLAY_CATEGORIES = (
    "empty_gt",
    "max_lanes",
    "highest_reach",
    "shortest_lane",
    "max_points",
    "consecutive_duplicates",
    "mixed_weather",
    "glare",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Sequence[float | int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    result = {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
    for quantile in QUANTILES:
        result[f"p{int(quantile * 100):02d}"] = float(
            np.quantile(array, quantile)
        )
    return result


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items())
    }


def _arc_length(lane: np.ndarray) -> float:
    if len(lane) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(lane.astype(np.float64), axis=0), axis=1).sum())


def _dedup_consecutive(lane: np.ndarray) -> np.ndarray:
    if len(lane) < 2:
        return lane
    keep = np.ones(len(lane), dtype=bool)
    keep[1:] = np.any(lane[1:] != lane[:-1], axis=1)
    return lane[keep]


def _direction(lane: np.ndarray) -> str:
    if len(lane) < 2:
        return "degenerate"
    delta = np.diff(lane[:, 1])
    if np.all(delta >= 0) and np.any(delta > 0):
        return "top_to_bottom"
    if np.all(delta <= 0) and np.any(delta < 0):
        return "bottom_to_top"
    if np.all(delta == 0):
        return "horizontal"
    return "non_monotonic"


def _load_scene_bundle(path: str | Path | None) -> tuple[dict, str | None]:
    if path is None:
        return {}, None
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("scene label file must contain an object-valued 'labels'")
    return labels, _sha256(source)


def _load_split(path: str | Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, None
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    train = payload.get("train_clips", [])
    validation = payload.get("val_clips", [])
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"split contains overlapping clips: {sorted(overlap)}")
    mapping = {clip: "train" for clip in train}
    mapping.update({clip: "val" for clip in validation})
    return mapping, _sha256(source)


def _scene_summary(scene_labels: dict, image_counts: Counter) -> dict:
    if not scene_labels:
        return {}
    categories = {
        "weather": Counter(),
        "illumination": Counter(),
        "artifact": Counter(),
        "geometry": Counter(),
    }
    image_categories = {key: Counter() for key in categories}
    cross = Counter()
    for clip, label in sorted(scene_labels.items()):
        weight = int(image_counts.get(clip, 0))
        weather = label["weather"]
        illumination = label["illumination"]
        categories["weather"][weather] += 1
        categories["illumination"][illumination] += 1
        image_categories["weather"][weather] += weight
        image_categories["illumination"][illumination] += weight
        cross[f"{weather}|{illumination}"] += 1
        for field in ("artifact", "geometry"):
            values = label.get(field) or ["none"]
            for value in values:
                categories[field][value] += 1
                image_categories[field][value] += weight
    return {
        "clip_counts": {
            key: dict(sorted(counter.items())) for key, counter in categories.items()
        },
        "image_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in image_categories.items()
        },
        "weather_x_illumination_clip_counts": dict(sorted(cross.items())),
    }


def analyze_manifest(
    records: Sequence[ManifestRecord],
    lane_root: str | Path,
    *,
    scene_labels: dict | None = None,
    split_by_clip: dict[str, str] | None = None,
) -> tuple[dict, list[dict]]:
    """Analyze an ordered manifest and return ``(report, image_rows)``."""
    lane_root = Path(lane_root)
    scene_labels = scene_labels or {}
    split_by_clip = split_by_clip or {}
    if not records:
        raise ValueError("EDA requires a non-empty manifest")

    manifest_clips = {record.clip_id for record in records}
    if scene_labels and manifest_clips != set(scene_labels):
        missing = sorted(manifest_clips - set(scene_labels))
        extra = sorted(set(scene_labels) - manifest_clips)
        raise ValueError(f"scene labels do not match manifest clips: missing={missing}, extra={extra}")
    if split_by_clip and manifest_clips != set(split_by_clip):
        missing = sorted(manifest_clips - set(split_by_clip))
        extra = sorted(set(split_by_clip) - manifest_clips)
        raise ValueError(f"split does not match manifest clips: missing={missing}, extra={extra}")

    lane_counts: list[int] = []
    point_counts: list[int] = []
    top_y: list[float] = []
    bottom_y: list[float] = []
    y_spans: list[float] = []
    arc_lengths: list[float] = []
    all_x: list[float] = []
    all_y: list[float] = []
    directions = Counter()
    clip_rows: dict[str, list[dict]] = defaultdict(list)
    split_rows: dict[str, list[dict]] = defaultdict(list)
    image_counts = Counter()
    image_rows: list[dict] = []

    nonfinite_points = 0
    outside_points = 0
    lanes_lt_two_points = 0
    lanes_lt_two_after_dedup = 0
    lanes_with_consecutive_duplicates = 0
    duplicate_points_removed = 0
    images_with_consecutive_duplicates = 0
    cut_lane_counts = {180: 0, 330: 0}
    cut_image_counts = {180: 0, 330: 0}

    for record in records:
        if record.gt_path is None:
            raise ValueError(f"EDA needs labeled records, got {record.image_id}")
        gt_path = lane_root / record.gt_path
        lanes = read_lines_txt(gt_path)
        n_lanes = len(lanes)
        lane_counts.append(n_lanes)
        image_counts[record.clip_id] += 1
        row = {
            "order": record.order,
            "image_id": record.image_id,
            "image_path": record.image_path,
            "gt_path": record.gt_path,
            "clip_id": record.clip_id,
            "n_lanes": n_lanes,
            "n_points": int(sum(len(lane) for lane in lanes)),
            "min_top_y": None,
            "min_arc_length": None,
            "max_lane_points": 0,
            "has_consecutive_duplicates": False,
        }
        affected_cuts = set()
        row_top_y: list[float] = []
        row_arc: list[float] = []
        for lane in lanes:
            point_counts.append(len(lane))
            lanes_lt_two_points += int(len(lane) < 2)
            finite = np.isfinite(lane).all(axis=1)
            nonfinite_points += int((~finite).sum())
            finite_lane = lane[finite]
            if finite_lane.size:
                xs = finite_lane[:, 0]
                ys = finite_lane[:, 1]
                all_x.extend(xs.tolist())
                all_y.extend(ys.tolist())
                outside_points += int(np.count_nonzero(
                    (xs < 0) | (xs >= CANVAS_WIDTH) |
                    (ys < 0) | (ys >= CANVAS_HEIGHT)
                ))
                lane_top = float(ys.min())
                lane_bottom = float(ys.max())
                lane_span = lane_bottom - lane_top
                lane_arc = _arc_length(finite_lane)
                top_y.append(lane_top)
                bottom_y.append(lane_bottom)
                y_spans.append(lane_span)
                arc_lengths.append(lane_arc)
                row_top_y.append(lane_top)
                row_arc.append(lane_arc)
                for cut in cut_lane_counts:
                    if lane_top < cut:
                        cut_lane_counts[cut] += 1
                        affected_cuts.add(cut)
            directions[_direction(lane)] += 1
            deduped = _dedup_consecutive(lane)
            removed = len(lane) - len(deduped)
            duplicate_points_removed += removed
            if removed:
                lanes_with_consecutive_duplicates += 1
                row["has_consecutive_duplicates"] = True
            lanes_lt_two_after_dedup += int(len(deduped) < 2)
        for cut in affected_cuts:
            cut_image_counts[cut] += 1
        if row["has_consecutive_duplicates"]:
            images_with_consecutive_duplicates += 1
        row["min_top_y"] = min(row_top_y) if row_top_y else None
        row["min_arc_length"] = min(row_arc) if row_arc else None
        row["max_lane_points"] = max((len(lane) for lane in lanes), default=0)
        image_rows.append(row)
        clip_rows[record.clip_id].append(row)
        if split_by_clip:
            split_rows[split_by_clip[record.clip_id]].append(row)

    total_images = len(records)
    total_lanes = len(point_counts)
    total_points = len(all_x) + nonfinite_points
    empty_images = int(sum(count == 0 for count in lane_counts))

    clip_stats = []
    for clip, rows in sorted(clip_rows.items()):
        counts = [row["n_lanes"] for row in rows]
        clip_stats.append({
            "clip_id": clip,
            "images": len(rows),
            "lanes": int(sum(counts)),
            "mean_lanes_per_image": float(np.mean(counts)),
            "empty_images": int(sum(count == 0 for count in counts)),
            "empty_rate": float(np.mean([count == 0 for count in counts])),
        })
    means = [row["mean_lanes_per_image"] for row in clip_stats]
    empty_rates = [row["empty_rate"] for row in clip_stats]

    split_summary = {}
    for split, rows in sorted(split_rows.items()):
        counts = [row["n_lanes"] for row in rows]
        points = [row["n_points"] for row in rows]
        tops = [row["min_top_y"] for row in rows if row["min_top_y"] is not None]
        split_summary[split] = {
            "clips": len({row["clip_id"] for row in rows}),
            "images": len(rows),
            "lanes": int(sum(counts)),
            "empty_images": int(sum(count == 0 for count in counts)),
            "empty_rate": float(np.mean([count == 0 for count in counts])),
            "mean_lanes_per_image": float(np.mean(counts)),
            "mean_points_per_image": float(np.mean(points)),
            "mean_min_top_y_nonempty": float(np.mean(tops)) if tops else None,
        }

    report = {
        "schema_version": 1,
        "dataset": {
            "images": total_images,
            "clips": len(clip_rows),
            "lanes": total_lanes,
            "points": total_points,
            "empty_images": empty_images,
            "empty_rate": empty_images / total_images,
        },
        "lanes_per_image": {
            "summary": _summary(lane_counts),
            "histogram": _histogram(lane_counts),
        },
        "points_per_lane": {
            "summary": _summary(point_counts),
            "histogram": _histogram(point_counts),
        },
        "geometry": {
            "x": _summary(all_x),
            "y": _summary(all_y),
            "lane_top_y": _summary(top_y),
            "lane_bottom_y": _summary(bottom_y),
            "lane_y_span": _summary(y_spans),
            "lane_arc_length_px": _summary(arc_lengths),
            "point_order": dict(sorted(directions.items())),
            "cut_height_exposure": {
                str(cut): {
                    "affected_lanes": cut_lane_counts[cut],
                    "affected_lane_rate": cut_lane_counts[cut] / total_lanes,
                    "affected_images": cut_image_counts[cut],
                    "affected_image_rate": cut_image_counts[cut] / total_images,
                }
                for cut in sorted(cut_lane_counts)
            },
        },
        "annotation_health": {
            "nonfinite_points": nonfinite_points,
            "outside_canvas_points": outside_points,
            "lanes_with_fewer_than_two_points": lanes_lt_two_points,
            "lanes_with_fewer_than_two_points_after_consecutive_dedup": lanes_lt_two_after_dedup,
            "lanes_with_consecutive_duplicates": lanes_with_consecutive_duplicates,
            "images_with_consecutive_duplicates": images_with_consecutive_duplicates,
            "consecutive_duplicate_points_removed": duplicate_points_removed,
        },
        "clip_level": {
            "image_count_histogram": _histogram(image_counts.values()),
            "mean_lanes_per_image": _summary(means),
            "empty_rate": _summary(empty_rates),
            "lowest_mean_lane_clips": sorted(
                clip_stats, key=lambda row: (row["mean_lanes_per_image"], row["clip_id"])
            )[:5],
            "highest_mean_lane_clips": sorted(
                clip_stats, key=lambda row: (-row["mean_lanes_per_image"], row["clip_id"])
            )[:5],
            "highest_empty_rate_clips": sorted(
                clip_stats, key=lambda row: (-row["empty_rate"], row["clip_id"])
            )[:5],
        },
        "scenes": _scene_summary(scene_labels, image_counts),
        "split_comparison": split_summary,
    }
    return report, image_rows


def select_overlay_rows(image_rows: Sequence[dict], scene_labels: dict | None = None) -> list[dict]:
    """Select one unique deterministic row for each visual audit category."""
    labels = scene_labels or {}
    candidates = {
        "empty_gt": [row for row in image_rows if row["n_lanes"] == 0],
        "max_lanes": sorted(image_rows, key=lambda row: (-row["n_lanes"], row["order"])),
        "highest_reach": sorted(
            (row for row in image_rows if row["min_top_y"] is not None),
            key=lambda row: (row["min_top_y"], row["order"]),
        ),
        "shortest_lane": sorted(
            (row for row in image_rows if row["min_arc_length"] is not None),
            key=lambda row: (row["min_arc_length"], row["order"]),
        ),
        "max_points": sorted(image_rows, key=lambda row: (-row["max_lane_points"], row["order"])),
        "consecutive_duplicates": [
            row for row in image_rows if row["has_consecutive_duplicates"]
        ],
        "mixed_weather": [
            row for row in image_rows
            if labels.get(row["clip_id"], {}).get("weather") == "mixed"
        ],
        "glare": [
            row for row in image_rows
            if "glare" in labels.get(row["clip_id"], {}).get("artifact", [])
        ],
    }
    used = set()
    selected = []
    for category in OVERLAY_CATEGORIES:
        choice = next(
            (row for row in candidates[category] if row["image_id"] not in used),
            None,
        )
        if choice is None:
            continue
        used.add(choice["image_id"])
        selected.append({**choice, "category": category})
    return selected


def render_overlay_contact_sheet(
    selected: Sequence[dict], lane_root: str | Path, output_path: str | Path
) -> None:
    lane_root = Path(lane_root)
    tiles = []
    palette = ((0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
               (0, 128, 255), (255, 255, 0), (128, 255, 0), (255, 0, 128))
    for row in selected:
        image = cv2.imread(str(lane_root / row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read overlay image: {row['image_path']}")
        gt_path = lane_root / row["gt_path"]
        lanes = read_lines_txt(gt_path)
        for index, lane in enumerate(lanes):
            points = np.rint(lane).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, palette[index % len(palette)], 4, cv2.LINE_8)
            for point in points[::max(1, len(points) // 12)]:
                cv2.circle(image, tuple(point[0]), 3, (255, 255, 255), -1, cv2.LINE_8)
        image = cv2.resize(image, (512, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (512, 44), (0, 0, 0), -1)
        cv2.putText(image, row["category"], (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"{row['image_id']} lanes={row['n_lanes']}", (8, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        raise ValueError("no overlay rows selected")
    blank = np.zeros_like(tiles[0])
    while len(tiles) < 8:
        tiles.append(blank.copy())
    sheet = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:8])))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"failed to write {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lane-root", required=True, type=Path)
    parser.add_argument("--scene-labels", type=Path)
    parser.add_argument("--split-config", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    scenes, scenes_sha = _load_scene_bundle(args.scene_labels)
    split, split_sha = _load_split(args.split_config)
    report, rows = analyze_manifest(
        records, args.lane_root, scene_labels=scenes, split_by_clip=split
    )
    report["inputs"] = {
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "scene_labels": str(args.scene_labels) if args.scene_labels else None,
        "scene_labels_sha256": scenes_sha,
        "split_config": str(args.split_config) if args.split_config else None,
        "split_config_sha256": split_sha,
    }
    selected = select_overlay_rows(rows, scenes)
    report["overlay_selection"] = [
        {key: row[key] for key in ("category", "image_id", "n_lanes")}
        for row in selected
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.overlay:
        render_overlay_contact_sheet(selected, args.lane_root, args.overlay)
    print(json.dumps({
        "images": report["dataset"]["images"],
        "lanes": report["dataset"]["lanes"],
        "empty_images": report["dataset"]["empty_images"],
        "max_lanes": report["lanes_per_image"]["summary"]["max"],
        "max_points": report["points_per_lane"]["summary"]["max"],
        "outside_canvas_points": report["annotation_health"]["outside_canvas_points"],
    }))


if __name__ == "__main__":
    main()
