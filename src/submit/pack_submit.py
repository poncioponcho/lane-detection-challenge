"""Pack predictions into submit.zip with root directory `submit/` (P0-A03).

The zip's internal tree mirrors the test set layout exactly; each entry is a
relative path ending in .lines.txt (e.g. "clip_0007/00042.lines.txt"). The
exact relative paths come from the expected-file list (the organizer's
checklist, e.g. testA.txt), NOT from scanning the pred dir — this guarantees
"no missing, no extra".

Missing expected files are emitted as EMPTY .lines.txt (no detection) rather
than raising: the submission must cover every test image even when a model
outputs no lane.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Union


def read_expected(path: Union[str, Path, None]) -> List[str]:
    """Read an expected-file list (one relative path per line).

    Tolerates: blank lines, '#' comments, leading "./", and paths that point at
    images instead of .lines.txt (suffix is added). Real checklist format will
    be calibrated at data-landing time (like probe_json_schema for .json).
    """
    if path is None:
        return []
    out: List[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _normalize_rel(rel: str) -> str:
    """Normalize a checklist row to ``<clip>/<frame>.lines.txt`` safely."""
    rel = rel.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    # Official task lists conventionally prefix image paths with one slash.
    if rel.startswith("/JPEGImages/"):
        rel = rel[1:]
    path = PurePosixPath(rel)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe expected path: {rel!r}")
    if path.parts and path.parts[0] == "JPEGImages":
        path = PurePosixPath(*path.parts[1:])
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        path = path.with_suffix(".lines.txt")
    elif not path.name.endswith(".lines.txt"):
        path = PurePosixPath(str(path) + ".lines.txt")
    if len(path.parts) != 2 or not path.parts[0] or path.name == ".lines.txt":
        raise ValueError(
            f"expected <clip>/<frame>.lines.txt or JPEGImages equivalent, got {rel!r}"
        )
    return path.as_posix()


def pack_submit(pred_dir: Union[str, Path],
                expected: Iterable[str],
                out_zip: Union[str, Path],
                root: str = "submit",
                missing_as_empty: bool = True) -> Path:
    """Pack pred_dir into out_zip with root/ prefix.

    expected: relative paths (image or .lines.txt form) that MUST appear.
    """
    pred_dir = Path(pred_dir)
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    rels = [_normalize_rel(e) for e in expected]

    # dedupe while preserving order (checklists can list an image and its
    # .lines.txt variant separately)
    seen, rels_u = set(), []
    for r in rels:
        if r not in seen:
            seen.add(r)
            rels_u.append(r)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for rel in rels_u:
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = pred_dir / rel
            if src.exists():
                shutil.copyfile(src, dst)
            elif missing_as_empty:
                dst.write_text("", encoding="utf-8")
            else:
                raise FileNotFoundError(f"missing prediction: {rel}")

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in rels_u:
                zf.write(stage / rel, arcname=f"{root}/{rel}")
    return out
