# LVO 定向低照度筛选（2026-09-06）

## 结论

条件化低照度 gamma 方案 **NO-GO**，不进入生产候选，也不消耗 A 榜额度。生产推理继续冻结为 CLRNet-R50 36ep、`t=0.50`，条件化 gamma 保持关闭。

## 实验协议

- 评测：8-fold leave-one-video-out，共 7100 张图、8 个 video cluster；冻结官方 Oracle 汇总 TP/FP/FN。
- 模型：同一份 CLRNet-R50 36ep 权重，eval-only，不重新训练。
- 基线预处理：`800×320`、`cut_height=180`。
- 候选预处理：仅在下半幅 road ROI 平均亮度 `≤42` 时启用 `gamma=0.85`；亮图保持原图。
- ROI：原始 BGR 图像的下半幅，起点为 `max(height//2, cut_height)`。
- 置信度：导出阈值 `0.0`，离线筛选使用 post-NMS 正类 softmax probability；旧 raw-logit sidecar 不参与本实验。
- 置信区间：video-level paired bootstrap，10,000 次，seed=42。
- 场景标签只用于训练集诊断和分桶，不作为 testA/testB 的运行时路由。

## 结果

| 桶 | 图像 | 视频 | 基线 F1 | gamma F1 | Δpp | paired CI（pp） |
|---|---:|---:|---:|---:|---:|---|
| 全部 | 7100 | 8 | 0.778516 | 0.778316 | −0.020 | [−0.077, +0.021] |
| `low_light` | 3100 | 3 | 0.837850 | 0.837520 | −0.033 | [−0.175, +0.051] |
| `normal` | 4000 | 5 | 0.731254 | 0.731106 | −0.015 | [−0.046, 0.000] |

全局候选相比基线新增 2 个 TP，同时新增 15 个 FP，F1 反而下降。三个低照度视频的区间仅作诊断，不能外推为独立的 illumination 因果结论；训练集中的 weather 与 illumination 共线，见 `docs/eda.md` 和 `docs/scene_split_audit_20260902.md`。

## 裁决与证据

方案没有达到稳定正收益：全局点估计为负，所有分桶的 CI 下界均不大于 0。因此关闭该开关，不做 gamma 训练型增强，不修改 `configs/default.yaml` 的生产默认值。

可复核产物（均在 `outputs/`，不纳入 Git 大文件）：

- `outputs/lvo_lowlight_eval_screen_20260906/protocol.json`
- `outputs/lvo_lowlight_conf_scan_20260906/conf_checkpoint_scan.json`
- `outputs/lvo_lowlight_bucket_eval_20260906/lowlight_bucket_eval.json`
- `scripts/evaluate_lowlight_buckets.py`
