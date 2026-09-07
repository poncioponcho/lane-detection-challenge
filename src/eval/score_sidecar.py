"""Contract helpers for exported lane confidence sidecars.

CLRNet's decode-time confidence is the positive-class probability produced by
``softmax(logits[:, :2])[:, 1]``.  Keeping that meaning in a small shared
module prevents an offline scan from silently treating a raw logit as a
probability threshold.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


SCORE_SCHEMA_VERSION = 1
POSITIVE_CLASS_SOFTMAX_PROBABILITY = "positive_class_softmax_probability"
UNKNOWN_SCORE_SEMANTICS = "unknown"
PROBABILITY_SCORE_RANGE = {"min": 0.0, "max": 1.0, "inclusive": True}


def probability_score(value: Any, *, context: str) -> float:
    """Return a finite probability or fail with a contextual error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: score is not numeric: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{context}: expected a finite probability in [0, 1], got {number!r}"
        )
    return number


def make_score_sidecar_contract(
    score_semantics: str, candidate_export_conf_threshold: float
) -> dict[str, Any]:
    """Build the invariant metadata portion of a score sidecar."""
    if not isinstance(score_semantics, str) or not score_semantics:
        raise ValueError(f"invalid score semantics: {score_semantics!r}")
    try:
        export_threshold = float(candidate_export_conf_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"candidate export threshold is not numeric: {candidate_export_conf_threshold!r}"
        ) from exc
    if not math.isfinite(export_threshold) or not 0.0 <= export_threshold <= 1.0:
        raise ValueError(
            "candidate export threshold must be a finite probability in [0, 1], "
            f"got {export_threshold!r}"
        )
    return {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "score_semantics": score_semantics,
        "score_range": (
            dict(PROBABILITY_SCORE_RANGE)
            if score_semantics == POSITIVE_CLASS_SOFTMAX_PROBABILITY
            else None
        ),
        "post_nms": True,
        "candidate_export_conf_threshold": export_threshold,
    }


def validate_probability_sidecar(
    payload: Mapping[str, Any], *, name: str = "score sidecar"
) -> float:
    """Validate the exact sidecar contract accepted by offline scans.

    Missing metadata is intentionally an error.  That rejects historical
    sidecars whose ``conf`` values were raw positive-class logits.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name}: payload must be a JSON object")
    if payload.get("score_schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError(
            f"{name}: unsupported or missing score_schema_version; "
            "refusing raw-logit/legacy sidecars"
        )
    if payload.get("score_semantics") != POSITIVE_CLASS_SOFTMAX_PROBABILITY:
        raise ValueError(
            f"{name}: score_semantics must be "
            f"{POSITIVE_CLASS_SOFTMAX_PROBABILITY!r}; refusing raw-logit scores"
        )
    if payload.get("score_range") != PROBABILITY_SCORE_RANGE:
        raise ValueError(
            f"{name}: score_range must be {PROBABILITY_SCORE_RANGE!r}"
        )
    if payload.get("post_nms") is not True:
        raise ValueError(f"{name}: post_nms must be true for offline filtering")
    try:
        export_threshold = float(payload["candidate_export_conf_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: missing or invalid candidate_export_conf_threshold"
        ) from exc
    if not math.isfinite(export_threshold) or not 0.0 <= export_threshold <= 1.0:
        raise ValueError(
            f"{name}: candidate_export_conf_threshold is not in [0, 1]: "
            f"{export_threshold!r}"
        )
    return export_threshold
