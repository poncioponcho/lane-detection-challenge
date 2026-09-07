#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contestant-side structural validator for lane result archives."""

import argparse
import math
import zipfile
from pathlib import Path, PurePosixPath


MAX_LANES_PER_IMAGE = 64
MAX_POINTS_PER_LANE = 2048


def load_expected_files(list_path):
    expected = set()
    with Path(list_path).open("r", encoding="utf-8") as list_file:
        for line_number, raw_line in enumerate(list_file, start=1):
            value = raw_line.strip()
            if not value:
                continue
            parts = PurePosixPath(value.lstrip("/")).parts
            if len(parts) < 2 or not parts[-1].lower().endswith(".jpg"):
                raise ValueError("测试清单第 {} 行格式错误".format(line_number))
            member = "submit/{}/{}".format(
                parts[-2], Path(parts[-1]).with_suffix(".lines.txt").name
            )
            if member in expected:
                raise ValueError("测试清单包含重复图像：{}".format(value))
            expected.add(member)
    if not expected:
        raise ValueError("测试清单为空")
    return expected


def validate_text(content, member_name):
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("{} 不是合法 UTF-8".format(member_name)) from exc

    lane_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) < 4 or len(values) % 2:
            raise ValueError(
                "{} 第 {} 行必须包含不少于 4 个且总数为偶数的坐标值".format(
                    member_name, line_number
                )
            )
        points = []
        for index in range(0, len(values), 2):
            try:
                x = float(values[index])
                y = float(values[index + 1])
            except ValueError as exc:
                raise ValueError(
                    "{} 第 {} 行包含非法数值".format(member_name, line_number)
                ) from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    "{} 第 {} 行包含 NaN 或 Inf".format(member_name, line_number)
                )
            point = (x, y)
            if not points or point != points[-1]:
                points.append(point)
        if len(points) < 2:
            raise ValueError(
                "{} 第 {} 行去除相邻重复点后不足 2 个点".format(
                    member_name, line_number
                )
            )
        if len(points) > MAX_POINTS_PER_LANE:
            raise ValueError(
                "{} 第 {} 行超过 {} 个关键点".format(
                    member_name, line_number, MAX_POINTS_PER_LANE
                )
            )
        lane_count += 1
    if lane_count > MAX_LANES_PER_IMAGE:
        raise ValueError(
            "{} 超过 {} 条预测车道线".format(
                member_name, MAX_LANES_PER_IMAGE
            )
        )


def check_submission(zip_path, list_path):
    expected = load_expected_files(list_path)
    with zipfile.ZipFile(str(zip_path), "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("压缩包 CRC 校验失败")
        files = []
        seen = set()
        for info in archive.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/")).as_posix()
            if name in seen:
                raise ValueError("压缩包包含重复成员：{}".format(name))
            seen.add(name)
            if info.is_dir():
                continue
            files.append(name)
        submitted = set(files)
        missing = expected - submitted
        extra = submitted - expected
        if missing or extra:
            raise ValueError(
                "文件集合不匹配：缺少 {} 个，多出 {} 个".format(
                    len(missing), len(extra)
                )
            )
        for member_name in sorted(expected):
            validate_text(archive.read(member_name), member_name)
    print("校验通过：{} 个结果文件的路径和文本格式均合法。".format(len(expected)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--list_path", required=True)
    args = parser.parse_args()
    try:
        check_submission(args.zip_path, args.list_path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print("校验失败：{}".format(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
