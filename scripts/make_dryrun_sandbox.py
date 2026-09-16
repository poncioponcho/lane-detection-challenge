#!/usr/bin/env python3
"""Clone build_testB_candidates.sh with every WRITE path redirected into a
throwaway /tmp dir, so a dry run can never leave testA content masquerading as
testB in the real output trees.

Why this exists (see memory section 10): the pipeline writes its manifest to the
REAL data/processed/manifest_testB.jsonl and its packages to the REAL
outputs/submit_testB_*.zip. A dry run done carelessly leaves testA-content files
at both places, and the next real run can silently pick them up. Redirecting by
hand is error-prone and easy to forget, so it is codified here.

Two failure modes this transformer is built to prevent:
  (a) `--out-zip` and `--zip_path` must move TOGETHER. If only one moves, the
      precheck reads a path that was never written and reports a fake
      "PRECHECK FAILED" that looks like a packaging bug.
  (b) The python heredoc is <<'PY', so the shell does NOT expand $DRY inside it.
      DRY is therefore passed as argv[3] instead.

Usage:
    python scripts/make_dryrun_sandbox.py [SANDBOX_DIR]

Prints the clone path and fails loudly if any write escaped the sandbox.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts/build_testB_candidates.sh"

STAMP = date.today().strftime("%Y%m%d")


def build(dry: str) -> Path:
    dst = Path(f"/tmp/_DRYRUN_candidates_{STAMP}.sh")
    if not SRC.exists():
        print(f"FATAL: source script missing: {SRC}", file=sys.stderr)
        raise SystemExit(1)

    text = SRC.read_text(encoding="utf-8")

    replacements: list[tuple[str, str, int]] = [
        # sandbox root and its own scratch files
        ("OUT=outputs/testB_${DAY}", f"OUT={dry}/testB", 1),
        ("MANIFEST=data/processed/manifest_testB.jsonl",
         f"MANIFEST={dry}/manifest_testB.jsonl", 1),
        # pack(): --out-zip and --zip_path MUST move together
        ('--out-zip "outputs/submit_testB_${n}.zip"',
         f'--out-zip "{dry}/outputs/submit_testB_${{n}}.zip"', 1),
        ('--zip_path "outputs/submit_testB_${n}.zip"',
         f'--zip_path "{dry}/outputs/submit_testB_${{n}}.zip"', 1),
        ('--report "outputs/reports/prepare_submit_testB_${n}.json"',
         f'--report "{dry}/outputs/reports/prepare_submit_testB_${{n}}.json"', 1),
        # shot4 margin-0 hedge branch
        ('--out-zip "outputs/submit_testB_shot4_margin0.zip"',
         f'--out-zip "{dry}/outputs/submit_testB_shot4_margin0.zip"', 1),
        ('--zip_path "outputs/submit_testB_shot4_margin0.zip"',
         f'--zip_path "{dry}/outputs/submit_testB_shot4_margin0.zip"', 1),
        ('--report "outputs/reports/prepare_submit_testB_shot4_margin0.json"',
         f'--report "{dry}/outputs/reports/prepare_submit_testB_shot4_margin0.json"', 1),
        # final listing loop
        ("for z in outputs/submit_testB_*.zip",
         f'for z in "{dry}"/outputs/submit_testB_*.zip', 1),
        # shared scratch files (several occurrences each)
        ("/tmp/testB_list_$DAY.txt", f"{dry}/list.txt", 0),
        ("/tmp/testB_supports_$DAY.json", f"{dry}/supports.json", 0),
        # hand DRY into the quoted heredoc as argv[3]
        ("""$PY - "$OUT" "$DAY" <<'PY'""",
         f"""$PY - "$OUT" "$DAY" "{dry}" <<'PY'""", 1),
        ("out, day = pathlib.Path(sys.argv[1]), sys.argv[2]",
         "out, day = pathlib.Path(sys.argv[1]), sys.argv[2]\n"
         "DRY = sys.argv[3]", 1),
        ('json.dump(present, open(f"/tmp/testB_supports_{day}.json", "w"), indent=1)',
         'json.dump(present, open(f"{DRY}/supports.json", "w"), indent=1)', 1),
    ]

    for old, new, expect in replacements:
        n = text.count(old)
        if n == 0:
            print(f"FATAL: pattern not found: {old[:70]!r}", file=sys.stderr)
            raise SystemExit(1)
        if expect and n != expect:
            print(f"FATAL: {old[:60]!r} matched {n}x, expected {expect}",
                  file=sys.stderr)
            raise SystemExit(1)
        text = text.replace(old, new)
        print(f"  {n:>2}x  {old.splitlines()[0][:66]}")

    anchor = "set -u\n"
    if anchor not in text:
        print("FATAL: could not find injection anchor 'set -u'", file=sys.stderr)
        raise SystemExit(1)
    setup = (
        f'DRY={dry}\n'
        'rm -rf "$DRY"; mkdir -p "$DRY/outputs/reports"\n'
        'echo "DRY-RUN SANDBOX: $DRY  (contents are testA -- never submittable)"\n'
    )
    text = text.replace(anchor, anchor + setup, 1)

    dst.write_text(text, encoding="utf-8")
    dst.chmod(0o755)

    leaks = [ln for ln in text.splitlines()
             if "outputs/submit_testB" in ln and dry not in ln]
    print(f"\nwrote {dst}")
    print(f"unsandboxed submit_testB writes remaining: {len(leaks)}")
    for ln in leaks:
        print("  LEAK>", ln.strip())
    if leaks:
        raise SystemExit(1)
    return dst


def main() -> None:
    dry = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/_DRYRUN_OUT_{STAMP}"
    dst = build(dry)
    print(f"sandbox: {dry}")
    print(f"\nnext:  python scripts/build_dryrun_bundle_20260916.py")
    print(f"then:  bash {dst} /tmp/dryrun_bundle_20260916.tgz")
    print("after: quarantine both products; never submit anything from them.")


if __name__ == "__main__":
    main()
