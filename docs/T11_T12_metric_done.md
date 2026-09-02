# T11/T12 交付：本地诊断 metric 对齐官方 Oracle（差分全绿）

**日期**：2026-09-01 初版；2026-09-02 差分闭环｜**状态**：✅ **完成**。官方 score.py 到手后发现的三处偏差（匈牙利 cost、稠密化方式、float32/异常回退）均已修复；本地 metric 定位保持为**诊断/扫描层**，最终裁决仍一律以冻结 Oracle 为准。
**风险等级**：原定全项目最高单点风险（偏差 >2pp 会让后续所有决策错向）→ Oracle 双层结构建立后降为「诊断层偏差可控」

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/common/types.py` | `Lane` / `ImageAnnotation` 数据类型 + 几何助手（resample / clip / extrapolate / mean_lateral_error） |
| `src/eval/rasterize.py` | Oracle 语义诊断实现：float64 B 样条、`(N−1)*5+1` 参数均匀采样、逐段 `cv2.line`、异常直抛 |
| `src/eval/matching.py` | `cost=1−IoU` 全局 Hungarian，一对一指派后再以 IoU>0.5 判 TP；缺预测按空预测 |
| `src/eval/official_metric.py` | 重导出 + `evaluate_dir(loader=...)` 目录级封装（loader 留待 T20 格式解析） |
| `src/eval/diff_test.py` | 真实训练全集本地 vs Oracle 差分审计工具 |
| `scripts/run_nonidentity_diff.py` | 真实 GT 派生的非 identity 跨环境整图差分（确定性 seed + 用例 SHA-256） |
| `tests/test_metric_differential.py` | 边界、线数、空缺文件、重复点、折返、越界、匈牙利反例与异常传播差分 |
| `tests/test_official_oracle_hash.py` | 三份官方冻结文件 SHA-256 守护 |

运行：`<lane-venv>/bin/python -m pytest tests/ -q`；真实全集：`<lane-venv>/bin/python src/eval/diff_test.py`。

2026-09-02 T11/T12 批次实测为 **66 passed**；§20 后 74，§21 后当前全套为 **76 passed**。真实 identity 全集 **71 段 / 7100 图 / 24435 条线**，TP=24435、FP=0、FN=0、F1=1.0。另补 non-identity 跨环境审计：1034 个渲染用例逐比特一致；576 图/2009 线逐图 TP/FP/FN 零分歧，总计 TP=1488、FP=812、FN=521、F1=0.690647。identity 只证明解析与链路，non-identity 结果才覆盖渲染、阈值与指派判别力。

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
- **2026-09-01 的“旧实现忠实复刻”结论已被 DECISIONS §17.1 推翻**；2026-09-02 完成的是受差分套件守门的本地诊断层，不改变“最终裁决经 `oracle_runner` 调用冻结官方 Oracle”的制度。

---

## 4. 下一步（§6.5 单人关键路径，v1.4 口径）

✅ T11/T12 本地 metric 对齐 + 差分套件 → ✅ T19 manifest → ✅ T18 oracle_runner →
T20/T21/T22 解析、一致性与 EDA →（T24 场景标注、T23 按段切分已于 9/2 完成）→ T31 dataloader+双套 config →
T40 trainer → **双路 15ep 筛选（CLRNet-R50 vs ADNet-R34，DECISIONS §15.2）→ 赢家 36ep**（9/5 出分）→
T50 后处理 → T51 阈值扫描（J4 三件套）→ T55 分辨率 2 档 → T53 退化增强 →
T60 定模型（9/10）→ T61 重训 → T65 冻结（9/14）→ T70/T72 B 榜。
