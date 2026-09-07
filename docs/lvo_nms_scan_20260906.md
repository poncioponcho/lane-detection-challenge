# 2026-09-06 LVO NMS 小扫报告

## 结论

NMS 小扫不构成生产改进，两个变体均 NO-GO；生产候选继续冻结为
`t=0.50`，不消耗 A 榜额度。

## 协议

- 输入：同一批 CLRNet-R50 36ep checkpoint 的 8-fold leave-one-video-out
  eval-only 预测，7100 张图，每张恰好一次。
- 候选在 `conf_threshold=0.0` 下导出，sidecar 语义为
  `positive_class_softmax_probability`，`post_nms=true`。
- 变体：`nms_topk=8,nms_thres=50`；`nms_topk=12,nms_thres=30`。
- 由冻结官方 Oracle 评估，参考为原始 NMS（`nms_topk=12,nms_thres=50`）；
  CI 以 8 个 video cluster 成对 bootstrap，10000 次，seed=42。

## 结果

| 变体 | 阈值 | 输出线数 | F1 | 相对 `final@0.40` | 同阈值相对原始 NMS |
|---|---:|---:|---:|---:|---:|
| 原始 NMS | 0.40 | 21758 | 0.778516 | — | — |
| `topk8` | 0.40 | 21758 | 0.778516 | +0.000pp | +0.000pp |
| `thres30` | 0.40 | 21778 | 0.778785 | +0.027pp | +0.027pp |
| 原始 NMS | 0.50 | 20852 | 0.774350 | −0.417pp | — |
| `topk8` | 0.50 | 20852 | 0.774350 | −0.417pp | +0.000pp |
| `thres30` | 0.50 | 20862 | 0.774577 | −0.394pp | +0.023pp |
| 原始 NMS | 0.60 | 19910 | 0.765272 | −1.324pp | — |
| `topk8` | 0.60 | 19910 | 0.765272 | −1.324pp | +0.012pp |
| `thres30` | 0.60 | 19916 | 0.765394 | −1.312pp | +0.012pp |

`thres30` 相对原始 NMS 的同阈值 video-paired 95% CI 为：

- `0.40`：`[0.000, +0.056]pp`；
- `0.45`：`[0.000, +0.045]pp`；
- `0.50`：`[0.000, +0.048]pp`；
- `0.55`：`[0.000, +0.044]pp`；
- `0.60`：`[−0.005, +0.036]pp`。

区间下界没有严格大于 0，且点增益仅为约 0.01–0.03pp；因此不能把它
解释为稳定收益。`topk8` 在所有扫描点输出与基线完全一致，说明阈值以上
候选未触及该 cap。

## 证据边界

- 完整机器可读结果：`outputs/lvo_nms_conf_scan_20260906/conf_checkpoint_scan.json`
  （该目录按仓库规则不入 Git）。
- 两组候选分别位于
  `outputs/lvo_nms_scan_20260906/nms_topk8/` 和
  `outputs/lvo_nms_scan_20260906/nms_thres30/`，均通过 7100 文件覆盖及
  sidecar 合约校验。
- 这是 LVO 上的后处理筛选，不证明 testA 端收益；没有稳定 LVO 收益时不
  进行 A 榜探针提交。

## 后续

NMS 方向关闭，下一项进入廉价 video-disjoint 分辨率筛选；低照度仍只做
条件化小试，ConvNeXt-T 暂不启动。
