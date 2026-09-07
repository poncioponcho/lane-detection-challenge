"""The sole production entry point for the frozen official scoring script."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from data.manifest import (ManifestRecord, iter_clip_groups, read_manifest,
                               validate_manifest)
except ModuleNotFoundError:  # ``python src/eval/oracle_runner.py``
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.manifest import (ManifestRecord, iter_clip_groups, read_manifest,
                               validate_manifest)

from eval.oracle_integrity import (ORACLE_DIR, inspect_official_environment,
                                   verify_official_environment,
                                   verify_oracle_files)

RESULT_SENTINEL = "__ORACLE_RESULT_JSON__="


@dataclass(frozen=True)
class OfficialCounts:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_images: int


@dataclass(frozen=True)
class OfficialEvalResult:
    global_: OfficialCounts
    per_clip: dict[str, OfficialCounts]
    oracle_sha256: dict[str, str]
    env: dict

    def to_dict(self) -> dict:
        return {
            "global": asdict(self.global_),
            "per_clip": {clip: asdict(value) for clip, value in self.per_clip.items()},
            "aggregation_note": (
                "global is the authoritative competition score from one full-list run; "
                "per_clip is diagnostic only and must never be averaged into a score"
            ),
            "oracle_sha256": self.oracle_sha256,
            "env": self.env,
        }


_SUBPROCESS_CODE = r"""
import importlib.util
import json
import sys

score_path, pred_dir, gt_dir, list_path, sentinel = sys.argv[1:]
spec = importlib.util.spec_from_file_location("frozen_official_score", score_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.eval_predictions(pred_dir, gt_dir, list_path)
value = result[module.IOU_THRESHOLDS[0]]
print(sentinel + json.dumps(value, separators=(",", ":")))
"""


def _write_official_list(path: Path, records: list[ManifestRecord]) -> None:
    # score.py consumes only the final clip/file components, but using the
    # manifest image_path preserves the official source rows and their order.
    path.write_text(
        "".join(f"/{record.image_path}\n" for record in records),
        encoding="utf-8",
    )


def _run_once(official_python: str | Path, pred_dir: Path, gt_dir: Path,
              list_path: Path, n_images: int) -> OfficialCounts:
    proc = subprocess.run(
        [str(official_python), "-c", _SUBPROCESS_CODE,
         str(ORACLE_DIR / "score.py"), str(pred_dir), str(gt_dir),
         str(list_path), RESULT_SENTINEL],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(
            f"official Oracle failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            payload = line[len(RESULT_SENTINEL):]
            break
    if payload is None:
        raise RuntimeError(f"official Oracle emitted no structured result: {proc.stdout!r}")
    try:
        value = json.loads(payload)
        return OfficialCounts(
            tp=int(value["TP"]), fp=int(value["FP"]), fn=int(value["FN"]),
            precision=float(value["Precision"]), recall=float(value["Recall"]),
            f1=float(value["F1"]), n_images=n_images,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid official Oracle result: {payload!r}") from exc


def run_official_eval(pred_dir: str | Path, gt_dir: str | Path,
                      manifest: str | Path | list[ManifestRecord], *,
                      official_python: str | Path, per_clip: bool = True,
                      output_path: str | Path | None = None,
                      enforce_official_env: bool = True) -> OfficialEvalResult:
    """Run global scoring once, then optionally score each manifest clip.

    The frozen source is hash-checked before any subprocess starts. Production
    calls enforce Python 3.12 and the exact official pins by default. The
    opt-out exists only for hermetic contract tests and must not be used for a
    reported competition score.
    """
    hashes = verify_oracle_files()
    env = (verify_official_environment(official_python) if enforce_official_env
           else inspect_official_environment(official_python))
    records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else list(manifest)
    validate_manifest(records)
    if any(record.gt_path is None for record in records):
        raise ValueError("Oracle evaluation requires a labeled manifest")

    pred_dir, gt_dir = Path(pred_dir).resolve(), Path(gt_dir).resolve()
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
    # A missing prediction directory is almost certainly a caller error. Files
    # within a valid directory may be missing and score.py treats those as empty.
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"prediction directory does not exist: {pred_dir}")

    with tempfile.TemporaryDirectory(prefix="lane-oracle-") as temp:
        temp_dir = Path(temp)
        global_list = temp_dir / "global.txt"
        _write_official_list(global_list, records)
        global_result = _run_once(
            official_python, pred_dir, gt_dir, global_list, len(records)
        )

        clip_results = {}
        if per_clip:
            for index, (clip_id, clip_records) in enumerate(iter_clip_groups(records)):
                clip_list = temp_dir / f"clip_{index:04d}.txt"
                _write_official_list(clip_list, clip_records)
                clip_results[clip_id] = _run_once(
                    official_python, pred_dir, gt_dir, clip_list, len(clip_records)
                )

    result = OfficialEvalResult(
        global_=global_result, per_clip=clip_results,
        oracle_sha256=hashes, env=env,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--official-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-per-clip", action="store_true")
    args = parser.parse_args()
    result = run_official_eval(
        args.pred_dir, args.gt_dir, args.manifest,
        official_python=args.official_python,
        per_clip=not args.no_per_clip, output_path=args.output,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
