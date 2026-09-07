"""官方评测 Oracle 冻结文件哈希守护（DECISIONS §17.1 / §18.1）。

三份官方文件逐字节冻结，任何一字节改动都必须红灯。
冻结文件不可变：禁止「修改后更新哈希」。官方若发布新版评测脚本，
新增版本目录（official_oracle_v2/）并保留旧版，新旧并存可差分。

哈希唯一事实源：src/eval/official_oracle/README.md 哈希表
（改该表 = 改 spec，须先过 DECISIONS 裁决记录）。
"""
from pathlib import Path
import sys

ORACLE_DIR = Path(__file__).resolve().parent.parent / "src" / "eval" / "official_oracle"
sys.path.insert(0, str(ORACLE_DIR.parents[1]))

from eval.oracle_integrity import FROZEN_SHA256, sha256_file  # noqa: E402


def test_oracle_files_exist():
    for name in FROZEN_SHA256:
        assert (ORACLE_DIR / name).is_file(), f"Oracle 冻结文件缺失: {name}"


def test_oracle_files_sha256_frozen():
    for name, want in FROZEN_SHA256.items():
        got = sha256_file(ORACLE_DIR / name)
        assert got == want, (
            f"Oracle 冻结文件被改动: {name}\n  期望 {want}\n  实际 {got}\n"
            "冻结文件禁止原地修改；官方新版须新增版本目录并保留旧版（DECISIONS §18.1）"
        )
