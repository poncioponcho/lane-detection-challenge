# 36ep 修正概率 sidecar 模块化消融（2026-09-06）

## 结论

本轮在修正后的正类 softmax 概率 sidecar 上完成 16 个后处理变体的
video-disjoint LVO 消融。所有变体均通过冻结官方 Oracle 评测，但没有一个同时满足
效应量、paired bootstrap 和 LOCO 放行条件；本轮不生成、不提交新的 testA 包。

生产候选继续冻结为 CLRNet-R50 36ep、800×320 + cut=180、显式 conf=0.50。
已有 A 榜 incumbent 为记录 714962、得分 0.73444 的
outputs/submit_testA_t05.zip；不重复提交同一包。

## 实验协议

- 数据：7100 张 LVO 图像、8 个 video cluster，使用固定 manifest。
- 候选池：同一批修正 C final 导出的 post-NMS 预测，导出阈值为 0.0；
  sidecar 语义为 positive_class_softmax_probability。
- 模块：真实概率阈值 5 个、下半幅亮度条件阈值 6 个、纵向跨度过滤 3 个、
  折线长度过滤 2 个。
- 评测：每个变体均由冻结官方 score.py 计算全局与逐 video TP/FP/FN；
 采用 10,000 次、seed=42 的 video-level paired bootstrap。
- 放行门槛：相对 confidence=0.40 基线的 ΔF1 ≥ +0.50pp、paired CI 下界严格大于 0，
 以及 8 个 leave-one-video-out（LOCO）差异全部为正。

## 结果

| 变体 | 模块 | 输出线数 | F1 | Δpp | paired CI（pp） | LOCO 最小/最大（pp） |
|---|---|---:|---:|---:|---|---|
| confidence_t0p40 | 概率阈值 | 21758 | 0.778516 | +0.000 | [+0.000,+0.000] | [+0.000,+0.000] |
| confidence_t0p45 | 概率阈值 | 21310 | 0.776653 | -0.186 | [-0.452,+0.074] | [-0.258,-0.100] |
| confidence_t0p50 | 概率阈值 | 20852 | 0.774350 | -0.417 | [-0.928,+0.103] | [-0.561,-0.246] |
| confidence_t0p55 | 概率阈值 | 20402 | 0.770970 | -0.755 | [-1.493,+0.036] | [-0.963,-0.506] |
| confidence_t0p60 | 概率阈值 | 19910 | 0.765272 | -1.324 | [-2.351,-0.204] | [-1.644,-0.972] |
| luma_L100_base040_hi0p50 | 亮度自适应 | 21591 | 0.778647 | +0.013 | [-0.141,+0.221] | [-0.076,+0.062] |
| luma_L100_base040_hi0p60 | 亮度自适应 | 21426 | 0.777218 | -0.130 | [-0.506,+0.304] | [-0.275,+0.033] |
| luma_L110_base040_hi0p50 | 亮度自适应 | 21688 | 0.779308 | +0.079 | [-0.006,+0.253] | [-0.003,+0.101] |
| luma_L110_base040_hi0p60 | 亮度自适应 | 21644 | 0.779618 | +0.110 | [-0.018,+0.356] | [-0.008,+0.145] |
| luma_L120_base040_hi0p50 | 亮度自适应 | 21721 | 0.779097 | +0.058 | [+0.000,+0.188] | [+0.000,+0.072] |
| luma_L120_base040_hi0p60 | 亮度自适应 | 21695 | 0.779276 | +0.076 | [+0.000,+0.237] | [+0.000,+0.094] |
| geometry_yspan_min40_t040 | y-span 过滤 | 21727 | 0.778519 | +0.000 | [-0.045,+0.050] | [-0.015,+0.025] |
| geometry_yspan_min60_t040 | y-span 过滤 | 21615 | 0.777676 | -0.084 | [-0.250,+0.052] | [-0.122,+0.014] |
| geometry_yspan_min80_t040 | y-span 过滤 | 21409 | 0.775979 | -0.254 | [-0.554,+0.065] | [-0.333,-0.112] |
| geometry_length_min150_t040 | 线长过滤 | 21542 | 0.774909 | -0.361 | [-0.728,-0.017] | [-0.455,-0.185] |
| geometry_length_min250_t040 | 线长过滤 | 19926 | 0.746016 | -3.250 | [-6.720,-0.221] | [-4.134,-1.477] |

## 模块裁决

- 概率阈值：LVO 上 0.40 是本轮参考；提高到 0.45–0.60 均退化，不能覆盖 A 榜
  上 0.50 的跨域证据，生产阈值保持显式 0.50。
- 亮度自适应：L=110、亮图阈值 0.60 的点估计最高（+0.110pp），但 CI 下界为
  -0.018pp、LOCO 最小值为 -0.008pp，属于弱正点，不放行、不提交、不训练。
- 几何过滤：y-span 40px 基本不变，60/80px 退化；线长过滤为负，全部关闭。
- 结合前序证据，score-aware ranker、NMS 变体、条件化 gamma、960×480 + cut=0、
  无分数 flip-union 也均为 NO-GO；不进行裸 union 或未经 LVO 放行的 A 榜探针。

## 可复核产物

- 脚本：scripts/evaluate_modular_ablation.py
- 完整 JSON：outputs/modular_ablation_20260906_v3/modular_ablation.json
- 表格报告：outputs/modular_ablation_20260906_v3/modular_ablation.md
- 输入 sidecar：outputs/lvo_clrnet_r50_36ep_c_export_20260906/final/prediction_scores.json
  （SHA-256：bc8f61ae36b4190f88d535d0b5a12a4fcc5f3b49a1267d4211d903babb935bc9）
- 结果完整性：16 个变体各覆盖 7100/7100 个 .lines.txt；stable_gain_variants=[]。
