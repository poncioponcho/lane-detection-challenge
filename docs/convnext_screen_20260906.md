# ConvNeXt-Tiny 15ep screen（2026-09-06）

## 结论

AutoDL 上的 CLRNet + ConvNeXt-Tiny 15ep screen 已完成，训练和独立回放均通过。
在同一份 800 张验证集、同一 `conf=0.40` 和同一官方 Oracle 下，
ConvNeXt-Tiny 的最佳点 F1 为 **0.7838715842**，CLRNet-R50 15ep 对照为
**0.7839576239**，差异 **−0.0086pp**。两者统计上不可区分，未形成足以
覆盖风险与额外训练成本的稳定收益。

本次 screen **不进入生产、不生成新的 A 榜包，也不消耗 A 榜额度**。生产继续冻结为：
CLRNet-R50 36ep、`800×320 + cut_height=180`、显式
`model.head.cfg.test_parameters.conf_threshold=0.50`，现有 A 榜 incumbent 为
记录 `714962`、得分 `0.73444`。

## 实验协议

- AutoDL 项目 HEAD：`84d7db7df50f6e2ba966b5b1daf6239aac63ebd7`。
- 配置：`configs/unlanedet/clrnet_convnext_tiny_hardlane.py`。
- 输入：`800×320`、`cut_height=180`；batch size `12`；最大迭代 `7875`；seed `42`；
  `cudnn_benchmark=False`。
- 初始化：CULane ConvNeXt-Tiny 权重经命名空间转换后加载；转换器报告提取
  `186` 个 `backbone.*` tensor。
- 评估：固定 `data/processed/manifest_val_v1_seed42.jsonl`，800 张图，显式
  `conf_threshold=0.40`；最佳点用 `model_best.pth` 做独立 eval-only 回放。
- sidecar：schema version `1`，语义为
  `positive_class_softmax_probability`，`post_nms=true`，导出阈值 `0.40`。
- 最终裁决复核：冻结官方 Oracle，Python `3.12.13`、numpy `2.1.3`、
  scipy `1.15.3`、opencv `4.12.0.88`。

## 结果

| 模型 / checkpoint | F1 | TP | FP | FN | 备注 |
|---|---:|---:|---:|---:|---|
| ConvNeXt-Tiny `model_best` @ iter 5249 | 0.7838715842 | 2051 | 527 | 604 | 独立回放，最佳点 |
| ConvNeXt-Tiny `model_final` @ iter 7875 | 0.7794654874 | 2027 | 519 | 628 | 训练末点 |
| CLRNet-R50 15ep `model_0004724` | 0.7839576239 | 2072 | 559 | 583 | 同协议对照 |

ConvNeXt-Tiny 相对 R50 15ep 的全局差异为 `−0.0086039735pp`。它减少了
`32` 个 FP，但同时减少 `21` 个 TP；F1 没有改善。

以 8 个验证 clip 为 cluster、10,000 次、seed `42` 的 paired bootstrap 仅作
训练内诊断，ConvNeXt-Tiny−R50 的 95% 区间为 **[−1.6122, +1.3594]pp**。
该验证集存在同视频训练重叠，不能替代 video-disjoint LVO，也不能作为无偏泛化
或 A 榜预测依据。

## 运行与证据

AutoDL 运行目录与状态：

- `/hy-tmp/lane-outputs/runs/convnext_tiny_15ep`
- `/hy-tmp/lane-outputs/convnext_tiny_screen_20260906.status`：`complete`
- AutoDL 远端代码 HEAD：`84d7db7df50f6e2ba966b5b1daf6239aac63ebd7`
- `model_best.pth` SHA-256：`7fd8d4b5bf3d60219fa5f3126c25eba43bc70196d794529557acfa6adb457908`
- `model_final.pth` SHA-256：`3e724bd7cf5cb7ede20f925f4af5fc07b87e1344465e9a772eba5d32342ee8cb`
- adapted ConvNeXt 权重 SHA-256：`338b37519c43cd2e6ccb62f5b70b0557bff9f0a45ca91e18dc6d636b20b9c7fb`

本地回传证据：

- 压缩包：[outputs/convnext_screen_evidence_20260906.tar.gz](../outputs/convnext_screen_evidence_20260906.tar.gz)
- 压缩包 SHA-256：`edebc9851848faea1ac74cb5b624d96015ff8bb10dbe93c10ea416a9b9fc17f7`
- 解包目录：`outputs/convnext_screen_20260906_evidence/`
- ConvNeXt Oracle：`outputs/convnext_screen_20260906_evidence/oracle_convnext_best.json`
- R50 Oracle：`outputs/convnext_screen_20260906_evidence/oracle_r50_screen_selected_best.json`
- 远端最佳独立回放：`best_eval_independent_v3/`，覆盖 800/800 个 `.lines.txt`。

独立回放过程中曾有两次只在启动参数/环境层失败的目录，均未覆盖有效结果，
现场已保留在 AutoDL：

- `best_eval_independent.failed-missing-env-20260906/`：SSH 会话未继承
  `HARDLANE_PROJECT_ROOT`。
- `best_eval_independent_v2/`：LazyConfig 收到未加字符串引号的绝对路径。

## 后续裁决

1. ConvNeXt-Tiny 15ep screen 记为完成但 **NO-GO**，不替换 R50，不启动 ConvNeXt
   36ep 完整训练。
2. 不把本次标准 val 点估计写成 LVO 收益；若未来重新打开该方向，必须先做
   video-disjoint 训练/评估，并沿用同一 Oracle 与放行门。
3. 下一关键路径回到 9/10 定模型、最终重训、复现与冻结；不重复提交已有的
   `submit_testA_t05.zip`。
