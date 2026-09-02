"""Deterministic clip-level split with multi-label scene stratification.

Scene labels are used only for offline splitting and diagnostics. They are not
an input to inference because test-set scene labels do not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

from common.io_utils import read_lines_txt
from data.manifest import ManifestRecord, read_manifest, write_manifest

WEATHER = frozenset({"rain", "fog", "snow", "clear", "mixed", "unknown"})
ILLUMINATION = frozenset({"normal", "low_light", "backlight", "mixed", "unknown"})
ARTIFACT = frozenset({"glare", "shadow"})
GEOMETRY = frozenset({"curve", "crossroad"})
CONFIDENCE = frozenset({"high", "low"})


@dataclass(frozen=True)
class SceneLabels:
    weather: str
    illumination: str
    artifact: tuple[str, ...] = ()
    geometry: tuple[str, ...] = ()
    confidence: str = "high"
    spot_frames: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SceneLabels":
        return cls(
            weather=str(value["weather"]),
            illumination=str(value["illumination"]),
            artifact=tuple(str(item) for item in value.get("artifact", [])),
            geometry=tuple(str(item) for item in value.get("geometry", [])),
            confidence=str(value.get("confidence", "high")),
            spot_frames=tuple(str(item) for item in value.get("spot_frames", [])),
        )

    def validate(self, clip_id: str) -> None:
        if self.weather not in WEATHER:
            raise ValueError(f"{clip_id}: invalid weather {self.weather!r}")
        if self.illumination not in ILLUMINATION:
            raise ValueError(f"{clip_id}: invalid illumination {self.illumination!r}")
        if self.confidence not in CONFIDENCE:
            raise ValueError(f"{clip_id}: invalid confidence {self.confidence!r}")
        if len(set(self.artifact)) != len(self.artifact) or not set(self.artifact) <= ARTIFACT:
            raise ValueError(f"{clip_id}: invalid/duplicate artifact labels")
        if len(set(self.geometry)) != len(self.geometry) or not set(self.geometry) <= GEOMETRY:
            raise ValueError(f"{clip_id}: invalid/duplicate geometry labels")
        if not self.spot_frames or any(not frame.isdigit() for frame in self.spot_frames):
            raise ValueError(f"{clip_id}: spot_frames must be non-empty digit strings")


def binary_features(label: SceneLabels) -> frozenset[str]:
    """Convert one annotation to dimension/value binary features, including none."""
    features = {f"weather:{label.weather}", f"illumination:{label.illumination}"}
    features.update(f"artifact:{value}" for value in label.artifact)
    features.update(f"geometry:{value}" for value in label.geometry)
    if not label.artifact:
        features.add("artifact:none")
    if not label.geometry:
        features.add("geometry:none")
    return frozenset(features)


def _feature_holders(labels: Mapping[str, SceneLabels]) -> dict[str, set[str]]:
    holders: dict[str, set[str]] = defaultdict(set)
    for clip_id, label in labels.items():
        for feature in binary_features(label):
            holders[feature].add(clip_id)
    return dict(holders)


def load_scene_labels(path: str | Path) -> tuple[dict[str, SceneLabels], dict]:
    """Load and validate the versioned annotation bundle and its QC record."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("labels"), dict):
        raise ValueError("scene label file must use schema_version=1 and contain labels")
    labels = {clip: SceneLabels.from_dict(value) for clip, value in raw["labels"].items()}
    for clip_id, label in labels.items():
        label.validate(clip_id)

    qc = raw.get("quality_control", {})
    reviewed = set(qc.get("second_reviewed_clips", []))
    if not reviewed <= set(labels):
        raise ValueError(f"second review contains unknown clips: {sorted(reviewed - set(labels))}")
    low = {clip for clip, label in labels.items() if label.confidence == "low"}
    if not low <= reviewed:
        raise ValueError(f"low-confidence clips missing second review: {sorted(low - reviewed)}")
    rare_max = int(qc.get("rare_feature_max_holders", 1))
    if rare_max < 1:
        raise ValueError("rare_feature_max_holders must be >= 1")
    holders = _feature_holders(labels)
    rare_features = {
        feature: sorted(clips) for feature, clips in holders.items()
        if len(clips) <= rare_max
    }
    rare_clips = set().union(*(
        clips for clips in holders.values()
        if len(clips) <= rare_max
    ))
    if not rare_clips <= reviewed:
        raise ValueError(
            "rare-feature clips missing second review: "
            f"{sorted(rare_clips - reviewed)}"
        )
    declared_rare = qc.get("rare_features")
    if declared_rare is not None and declared_rare != rare_features:
        raise ValueError("declared rare_features do not match labels")
    singleton_features = {
        feature: clips for feature, clips in rare_features.items() if len(clips) == 1
    }
    declared_singletons = qc.get("singleton_features")
    if declared_singletons is not None and declared_singletons != singleton_features:
        raise ValueError("declared singleton_features do not match labels")
    return labels, raw


def validate_spot_frames_against_manifest(
    labels: Mapping[str, SceneLabels], records: Iterable[ManifestRecord]
) -> None:
    """Ensure audit frame IDs are real tasks from the official manifest."""
    available: dict[str, set[str]] = defaultdict(set)
    for record in records:
        available[record.clip_id].add(record.frame_id)
    if set(available) != set(labels):
        raise ValueError("manifest clips and scene-label clips differ")
    missing = {
        clip_id: sorted(set(label.spot_frames) - available[clip_id])
        for clip_id, label in labels.items()
        if set(label.spot_frames) - available[clip_id]
    }
    if missing:
        raise ValueError(f"spot_frames absent from official manifest: {missing}")


def _distribution_score(
    val_clips: Iterable[str], labels: Mapping[str, SceneLabels]
) -> tuple[float, float]:
    """Return (squared, maximum) binary-feature prevalence deviation."""
    val = tuple(val_clips)
    holders = _feature_holders(labels)
    deviations = []
    for clips in holders.values():
        if len(clips) == 1:  # protected singleton: val is intentionally N/A
            continue
        overall_rate = len(clips) / len(labels)
        val_rate = len(clips.intersection(val)) / len(val)
        deviations.append(abs(val_rate - overall_rate))
    return math.fsum(value * value for value in deviations), max(deviations, default=0.0)


def build_lane_count_histograms(
    records: Iterable[ManifestRecord], lane_root: str | Path
) -> dict[str, dict[int, int]]:
    """Build per-clip image lane-count histograms from primary annotations."""
    lane_root = Path(lane_root)
    histograms: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        if record.gt_path is None:
            raise ValueError(f"lane-count stratification needs GT: {record.image_id}")
        lane_count = len(read_lines_txt(lane_root / record.gt_path))
        histograms[record.clip_id][lane_count] += 1
    return {
        clip: dict(sorted(histogram.items()))
        for clip, histogram in sorted(histograms.items())
    }


def _lane_distribution_score(
    val_clips: Iterable[str], histograms: Mapping[str, Mapping[int, int]] | None
) -> tuple[float, float]:
    """Compare validation/all image-level lane-count distributions."""
    if not histograms:
        return 0.0, 0.0
    val = tuple(val_clips)
    all_counts = Counter()
    val_counts = Counter()
    for clip, histogram in histograms.items():
        all_counts.update(histogram)
        if clip in val:
            val_counts.update(histogram)
    all_total = sum(all_counts.values())
    val_total = sum(val_counts.values())
    if not all_total or not val_total:
        raise ValueError("lane-count histograms must contain images for all and val")
    bins = sorted(set(all_counts) | set(val_counts))
    deviations = [
        abs(val_counts[key] / val_total - all_counts[key] / all_total)
        for key in bins
    ]
    return math.fsum(value * value for value in deviations), max(deviations, default=0.0)


def choose_val_clips(
    labels: Mapping[str, SceneLabels], *, val_size: int = 8, seed: int = 42,
    trials: int = 100_000,
    lane_count_histograms: Mapping[str, Mapping[int, int]] | None = None,
) -> tuple[tuple[str, ...], dict]:
    """Choose a deterministic hold-out by seeded search plus local swap descent.

    Scene prevalence is the primary objective.  When lane-count histograms are
    supplied, their image-level distribution is a secondary tie-breaker among
    equally scene-balanced candidates.  This prevents arbitrary clip-ID ties
    from dropping empty-GT images out of validation without trading away scene
    coverage.
    """
    if not 6 <= val_size <= 10:
        raise ValueError(f"val_size must be in [6, 10], got {val_size}")
    if len(labels) < val_size + 1:
        raise ValueError("not enough clips for a non-empty training split")
    for clip_id, label in labels.items():
        label.validate(clip_id)
    if lane_count_histograms is not None and set(lane_count_histograms) != set(labels):
        raise ValueError("lane-count histograms and scene-label clips differ")

    holders = _feature_holders(labels)
    protected = set().union(*(clips for clips in holders.values() if len(clips) == 1))
    eligible = sorted(set(labels) - protected)
    if len(eligible) < val_size:
        raise ValueError("singleton protection leaves too few validation candidates")

    rng = random.Random(seed)
    best: tuple[str, ...] | None = None
    best_key: tuple[float, float, float, float, tuple[str, ...]] | None = None
    for _ in range(max(1, trials)):
        candidate = tuple(sorted(rng.sample(eligible, val_size)))
        squared, maximum = _distribution_score(candidate, labels)
        lane_squared, lane_maximum = _lane_distribution_score(
            candidate, lane_count_histograms
        )
        key = (squared, maximum, lane_squared, lane_maximum, candidate)
        if best_key is None or key < best_key:
            best, best_key = candidate, key

    assert best is not None and best_key is not None
    # Deterministic one-swap descent makes the sampled result locally optimal.
    while True:
        improved = None
        val_set = set(best)
        for outgoing in best:
            for incoming in eligible:
                if incoming in val_set:
                    continue
                candidate = tuple(sorted((val_set - {outgoing}) | {incoming}))
                squared, maximum = _distribution_score(candidate, labels)
                lane_squared, lane_maximum = _lane_distribution_score(
                    candidate, lane_count_histograms
                )
                key = (squared, maximum, lane_squared, lane_maximum, candidate)
                if key < best_key and (improved is None or key < improved[0]):
                    improved = key, candidate
        if improved is None:
            break
        best_key, best = improved

    val_set = set(best)
    feature_counts = {}
    for feature, clips in sorted(holders.items()):
        feature_counts[feature] = {
            "all": len(clips),
            "train": len(clips - val_set),
            "val": len(clips & val_set),
            "val_status": "N/A" if len(clips) == 1 else "measured",
        }
    audit = {
        "objective_squared_prevalence_deviation": best_key[0],
        "objective_max_prevalence_deviation": best_key[1],
        "secondary_lane_hist_squared_deviation": best_key[2],
        "secondary_lane_hist_max_deviation": best_key[3],
        "search_trials": max(1, trials),
        "singleton_protected_clips": sorted(protected),
        "feature_counts": feature_counts,
    }
    if lane_count_histograms:
        all_histogram = Counter()
        val_histogram = Counter()
        for clip, histogram in lane_count_histograms.items():
            all_histogram.update(histogram)
            if clip in val_set:
                val_histogram.update(histogram)
        audit["lane_count_histograms"] = {
            "all": {str(key): all_histogram[key] for key in sorted(all_histogram)},
            "train": {
                str(key): all_histogram[key] - val_histogram[key]
                for key in sorted(all_histogram)
            },
            "val": {str(key): val_histogram[key] for key in sorted(all_histogram)},
        }
    return best, audit


@dataclass(frozen=True)
class Split:
    name: str
    train_clips: tuple[str, ...]
    val_clips: tuple[str, ...]
    seed: int = 42
    scene_labels_path: str = "data/processed/scene_labels.json"
    audit: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        train, val = set(self.train_clips), set(self.val_clips)
        if len(train) != len(self.train_clips) or len(val) != len(self.val_clips):
            raise ValueError("split contains duplicate clip IDs")
        if train & val:
            raise ValueError(f"train/val clip overlap: {sorted(train & val)}")
        if not train:
            raise ValueError("training split is empty")
        if not 6 <= len(val) <= 10:
            raise ValueError(f"validation clip count out of range: {len(val)}")

    def save(self, path: str | Path) -> None:
        """Save as JSON-compatible YAML (valid YAML 1.2, stdlib-readable)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["train_clips"] = tuple(raw["train_clips"])
        raw["val_clips"] = tuple(raw["val_clips"])
        return cls(**raw)


def validate_split_against_labels(
    split: Split, labels: Mapping[str, SceneLabels]
) -> None:
    """Assert full clip coverage and singleton-feature protection."""
    assigned = set(split.train_clips) | set(split.val_clips)
    if assigned != set(labels):
        raise ValueError(
            f"split/label clip mismatch: missing={sorted(set(labels) - assigned)}, "
            f"extra={sorted(assigned - set(labels))}"
        )
    val = set(split.val_clips)
    violations = {
        feature: sorted(clips & val)
        for feature, clips in _feature_holders(labels).items()
        if len(clips) == 1 and clips & val
    }
    if violations:
        raise ValueError(f"singleton features assigned to validation: {violations}")


def split_manifest(
    records: Iterable[ManifestRecord], split: Split
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    """Derive ordered train/val manifests without randomizing frame order."""
    records = list(records)
    train_set, val_set = set(split.train_clips), set(split.val_clips)
    source_clips = {record.clip_id for record in records}
    expected = train_set | val_set
    if source_clips != expected:
        raise ValueError(
            f"split/source clip mismatch: missing={sorted(source_clips - expected)}, "
            f"extra={sorted(expected - source_clips)}"
        )
    train, val = [], []
    for record in records:
        target, name = (val, "val") if record.clip_id in val_set else (train, "train")
        target.append(replace(record, split=name, order=len(target)))
    if {record.clip_id for record in train} & {record.clip_id for record in val}:
        raise AssertionError("clip leakage after manifest derivation")
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane-root", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--name", default="v1_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--trials", type=int, default=100_000)
    args = parser.parse_args()

    labels, _bundle = load_scene_labels(args.scene_labels)
    records = read_manifest(args.manifest)
    validate_spot_frames_against_manifest(labels, records)
    source_clips = tuple(dict.fromkeys(record.clip_id for record in records))
    if set(source_clips) != set(labels):
        raise ValueError("manifest clips and scene-label clips differ")
    lane_histograms = build_lane_count_histograms(records, args.lane_root)
    val_clips, audit = choose_val_clips(
        labels, val_size=args.val_size, seed=args.seed, trials=args.trials,
        lane_count_histograms=lane_histograms,
    )
    audit["algorithm"] = "seeded_random_search_plus_one_swap_v2"
    audit["objective_definition"] = (
        "sum of squared absolute val-vs-all prevalence deviations over "
        "non-singleton scene features; image-level lane-count histogram "
        "deviation is a lexicographic tie-breaker"
    )
    audit["scene_labels_sha256"] = hashlib.sha256(args.scene_labels.read_bytes()).hexdigest()
    audit["source_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    audit["lane_count_histograms_sha256"] = hashlib.sha256(
        json.dumps(lane_histograms, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    val_set = set(val_clips)
    split = Split(
        name=args.name,
        train_clips=tuple(clip for clip in source_clips if clip not in val_set),
        val_clips=tuple(clip for clip in source_clips if clip in val_set),
        seed=args.seed,
        scene_labels_path=str(args.scene_labels),
        audit=audit,
    )
    validate_split_against_labels(split, labels)
    split.save(args.split_output)
    train, val = split_manifest(records, split)
    write_manifest(args.train_output, train)
    write_manifest(args.val_output, val)
    print(json.dumps({
        "split": split.name,
        "train_clips": len(split.train_clips),
        "val_clips": len(split.val_clips),
        "train_images": len(train),
        "val_images": len(val),
        "objective": audit["objective_squared_prevalence_deviation"],
    }))


if __name__ == "__main__":
    main()
