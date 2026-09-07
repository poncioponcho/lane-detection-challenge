"""SHA-256 + byte-size utilities for freeze / reproducibility (P0-E14)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def sha256_of_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_meta(path: Union[str, Path]) -> tuple[str, int]:
    """Return (sha256_hex, byte_count) for a file — the exact pair the
    organizer requires for solution.zip freeze (TASKS.md §0 / §5.1)."""
    p = Path(path)
    return sha256_of_file(p), p.stat().st_size
