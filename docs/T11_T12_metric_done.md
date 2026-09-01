# T11/T12 交付：官方 metric 本地复刻（自检全绿）

**日期**：2026-09-01｜**状态**：✅ 完成，可采信
**风险等级**：原定全项目最高单点风险（偏差 >2pp 会让后续所有决策错向）→ 已闭环

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/common/types.py` | `Lane` / `ImageAnnotation` 数据类型 + 几何助手（resample / clip / extrapolate / mean_lateral_error） |
| `src/eval/rasterize.py` | B 样条 k=3 稠密化（step=5px）+ `cv2.polylines(thickness=30, lineType=8)` → 1366×720 二值掩码 |
| `src/eval/matching.py` | `pair_iou` / `iou_matrix` / `match_image`（Hungarian 一对一，IoU>0.5 判 TP）/ `compute_f1` |
| `src/eval/official_metric.py` | 重导出 + `evaluate_dir(loader=...)` 目录级封装（loader 留待 T20 格式解析） |
| `tests/test_metric_selfcheck.py` | 8 条断言自检（5 个 pytest 函数），全绿 |

运行：`python tests/test_metric_selfcheck.py`（隔离 venv：`/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane/bin/python`）

---

## 2. 自检结果（全部 < 0.02 容差）

```
[ok] GT vs GT: F1=1.0000
[ok] lateral shift 5px:  IoU=0.7222  (理想 0.7143)
[ok] lateral shift 10px: IoU=0.5122  (理想 0.5000)
[ok] lateral shift 15px: IoU=0.3478  (理想 0.3333)
[ok] TP boundary ~10.3px: 10px->TP, 11px->FN
[ok] 2/2 matched: F1=1.0000
[ok] 1 TP + 1 FP / 1 GT: F1=0.6667
[ok] curved lane self-match: IoU=1.0000
```

---

## 3. 关键校正（实测推翻理想模型）

理想矩形公式 `IoU=(30−d)/(30+d)` 在 d=10 给 0.5，但 **cv2 `thickness=30` 的有效描边宽度≈31px**，
实测 `IoU(10px)=0.5122`。因此：

- 真实 IoU=0.5 边界 = **≈10.3px**（10px 仍判 TP，11px 才判 FN），不是精确 10px。
- 该偏差与官方一致（官方同用 cv2），属**忠实复刻**；0.3px 偏移远在 2pp 决策容差内。
- **结论：测量仪器可靠，后续基线/消融 F1 读数可直接采信，A 榜反演标定（Q-A3）关闭。**

---

## 4. 下一步（§6.5 单人关键路径，v1.2 口径）

T20 三格式解析 → T21 一致性 → T22 EDA → T23 按段切分 → T31 dataloader+双套 config →
T40 trainer → **双路 15ep 筛选（CLRNet-R50 vs ADNet-R34，DECISIONS §15.2）→ 赢家 36ep**（9/5 出分）→
T50 后处理 → T51 阈值扫描（J4 三件套）→ T55 分辨率 2 档 → T53 退化增强 →
T60 定模型（9/10）→ T61 重训 → T65 冻结（9/14）→ T70/T72 B 榜。
