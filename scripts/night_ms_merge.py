"""保守三尺度坐标合并（夜间 TTA 假设验证）。

策略（吸取 flip-union 教训，绝不引入新线）：
- 基准尺度（800x320）预测为锚；
- 对每条基准线，在 ms880/ms720 预测中按 y-重叠段 mean|dx| < 15px 贪心找唯一匹配；
- 仅当 ms880 与 ms720 的匹配线彼此也一致（互差 < 15px）时，把基准线坐标
  与两个尺度在基准 y 采样点的 x 做三点平均；否则原样保留基准线。
输出合并后的 .lines.txt（同目录结构）。
"""
import os
import sys
from pathlib import Path

BASE = Path("/tmp/ms_preds/a2seed42_base/val/predictions")
M880 = Path("/tmp/ms_preds/a2seed42_ms880/val/predictions")
M720 = Path("/tmp/ms_preds/a2seed42_ms720/val/predictions")
OUT = Path("/tmp/ms_preds/a2seed42_merged/val/predictions")
DX_THR = 15.0


def read_lines(p: Path):
    rows = []
    if not p.is_file():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(t) for t in line.split()]
        xs = vals[0::2]
        ys = vals[1::2]
        rows.append(list(zip(xs, ys)))
    return rows


def write_lines(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for pts in rows:
            f.write(" ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + "\n")


def resample_x(pts, ys_target):
    """把折线按 y 重采样到 ys_target 的 x（线性插值，出界用端点外推）"""
    pts = sorted(pts, key=lambda t: t[1])
    xs = [t[0] for t in pts]
    ys = [t[1] for t in pts]
    out = []
    for yt in ys_target:
        if yt <= ys[0]:
            out.append(xs[0])
        elif yt >= ys[-1]:
            out.append(xs[-1])
        else:
            for i in range(len(ys) - 1):
                if ys[i] <= yt <= ys[i + 1]:
                    t = (yt - ys[i]) / (ys[i + 1] - ys[i] + 1e-9)
                    out.append(xs[i] + t * (xs[i + 1] - xs[i]))
                    break
    return out


def y_overlap_mean_dx(a, b):
    ya = [t[1] for t in a]
    yb = [t[1] for t in b]
    lo = max(min(ya), min(yb))
    hi = min(max(ya), max(yb))
    if hi - lo < 0.5 * (max(max(ya), max(yb)) - min(min(ya), min(yb))):
        return None
    ys_t = [y for y in ya if lo <= y <= hi]
    if len(ys_t) < 4:
        return None
    xb = resample_x(b, ys_t)
    xa = resample_x(a, ys_t)
    dx = sum(abs(u - v) for u, v in zip(xa, xb)) / len(ys_t)
    return dx


def match(base_lines, other_lines):
    """贪心一对一：返回 [idx 或 None] * len(base_lines)"""
    used = set()
    out = []
    for bl in base_lines:
        best_j, best_dx = None, DX_THR
        for j, ol in enumerate(other_lines):
            if j in used:
                continue
            dx = y_overlap_mean_dx(bl, ol)
            if dx is not None and dx < best_dx:
                best_dx, best_j = dx, j
        if best_j is not None:
            used.add(best_j)
        out.append(best_j)
    return out


def main():
    n_imgs = 0
    n_adj = 0
    n_kept = 0
    for base_f in sorted(BASE.rglob("*.lines.txt")):
        rel = base_f.relative_to(BASE)
        b = read_lines(base_f)
        m8 = read_lines(M880 / rel)
        m7 = read_lines(M720 / rel)
        j8 = match(b, m8)
        j7 = match(b, m7)
        merged = []
        for i, bl in enumerate(b):
            ys_t = [t[1] for t in bl]
            if j8[i] is not None and j7[i] is not None:
                x8 = resample_x(m8[j8[i]], ys_t)
                x7 = resample_x(m7[j7[i]], ys_t)
                if sum(abs(u - v) for u, v in zip(x8, x7)) / len(ys_t) < DX_THR:
                    xs_b = resample_x(bl, ys_t)
                    pts = [( (xs_b[k] + x8[k] + x7[k]) / 3.0, ys_t[k]) for k in range(len(ys_t))]
                    merged.append(pts)
                    n_adj += 1
                    continue
            merged.append(bl)
            n_kept += 1
        write_lines(OUT / rel, merged)
        n_imgs += 1
    print(f"images={n_imgs} lines_adjusted={n_adj} lines_kept={n_kept}")


if __name__ == "__main__":
    main()
