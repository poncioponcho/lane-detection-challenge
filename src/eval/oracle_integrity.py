"""Integrity and environment guardrails for the frozen official Oracle."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ORACLE_DIR = Path(__file__).resolve().parent / "official_oracle"
FROZEN_SHA256 = {
    "score.py": "b2f4c9b21de1083de8a420c106262e799947fac37c938994f7d160b8c62de0d2",
    "check_submission.py": "c7897bdbd81d19b56ee0b6b9a803c0645428403ffa13c133e1ea41a422c02043",
    "requirements_official.txt": "eb92e5ce6ff4b97db3de410bc068837f3038f83561f9bde581b5055530f67c05",
}
OFFICIAL_PACKAGES = {
    "numpy": "2.1.3",
    "scipy": "1.15.3",
    "opencv-python": "4.12.0.88",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_oracle_files() -> dict[str, str]:
    """Return hashes after rejecting any missing or changed frozen file."""
    actual = {}
    for name, expected in FROZEN_SHA256.items():
        path = ORACLE_DIR / name
        if not path.is_file():
            raise RuntimeError(f"Oracle frozen file missing: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise RuntimeError(
                f"Oracle frozen file changed: {name}; expected={expected}, actual={actual[name]}"
            )
    return actual


def inspect_official_environment(python: str | Path) -> dict:
    """Inspect an interpreter in a subprocess without importing its packages here."""
    code = (
        "import importlib.metadata as m,json,sys;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'executable':sys.executable,'packages':{n:next((d.version for d in "
        "m.distributions() if (d.metadata['Name'] or '').lower()==n),None) for n in "
        "['numpy','scipy','opencv-python']}}))"
    )
    proc = subprocess.run(
        [str(python), "-c", code], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(
            f"cannot inspect Oracle environment {python}: {proc.stderr.strip()}"
        )
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Oracle environment response: {proc.stdout!r}") from exc


def verify_official_environment(python: str | Path) -> dict:
    """Require Python 3.12 and the three exact package pins."""
    env = inspect_official_environment(python)
    if not env["python"].startswith("3.12."):
        raise RuntimeError(f"Oracle requires Python 3.12, got {env['python']}")
    if env["packages"] != OFFICIAL_PACKAGES:
        raise RuntimeError(
            f"Oracle package pins mismatch: expected={OFFICIAL_PACKAGES}, got={env['packages']}"
        )
    return env
