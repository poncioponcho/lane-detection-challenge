# AutoDL GPU 租期压缩执行计划（2026-09-06）

## 目标

AutoDL 3090 仍有约 6 天租期。当前生产 incumbent 不变，但 GPU 立即用于验证
“分类目标加权能否压低跨域 FP”的唯一低风险训练假设；所有候选都必须可回退、可复核，
不得用 GPU 空档替代计划，也不得用标准 val 点估计直接提交。

## 固定基线与不变量

- 生产：CLRNet-R50 36ep、`800×320 + cut_height=180`、显式 `conf_threshold=0.50`。
- A 榜 incumbent：记录 `714962`，得分 `0.73444`；不重复提交对应压缩包。
- 训练初始化：同一 adapted CULane CLRNet-R50 权重，seed `42`，15ep screen。
- 唯一变量：`model.head.cfg.cls_loss_weight`，候选为 `3.0`、`4.0`；基线值为 `2.0`。
- 评测：同一 manifest、修正后的概率 sidecar、冻结官方 Oracle；screen 只作 triage。

## 执行链

1. 运行 `scripts/autodl/run_r50_loss_weight_screen.sh`，得到两组标准 val 独立回放证据。
2. 只将最有希望的一组送入 8-video video-disjoint LVO 15ep；两组都不改善则关闭本方向。
3. LVO 放行条件为 ΔF1 ≥ `+0.50pp`、paired CI 下界 > `0`、8 个 LOCO 差异全为正。
4. 只有 LVO 放行者才允许做 36ep；否则 GPU 切换到 R50 多种子复现和最终交付演练。
5. 新候选最多生成一个 testA 包，且只有在 LVO 放行后才考虑使用 A 榜额度；任何情况下
   不替换 incumbent，直到官方结果真实返回并记账。

## 六天窗口

| 窗口 | GPU 工作 | 交付物 |
|---|---|---|
| 9/6–9/7 | 两组 15ep screen | `screen_r50_clsweight{3,4}_15ep` 训练、回放和 SHA 证据 |
| 9/7–9/8 | 最佳组 8-fold LVO 15ep | Oracle 全局/逐 video 结果、paired bootstrap、LOCO |
| 9/8–9/9 | 通过者 36ep；否则 R50 seed 复现 | 新权重或复现基线，不覆盖既有 run |
| 9/9–9/10 | 必要的 36ep LVO/最终导出 | 仅放行候选进入 testA 打包 |
| 9/10 | T60 + 最新 A 榜分布重估 | 定模型裁决，默认仍优先 incumbent |
| 9/11–9/12 | 最终复现、导出、演练 | SHA 固化、B 榜沙盘、AutoDL 交接/停机 |

## 失败与止损

loss 非 finite、覆盖不全、HEAD/patch 不一致或磁盘剩余低于 9GB 时立即停止该 run，
保留现场并转入 fallback；不重复启动同一失败实验。ConvNeXt、分辨率、低照度、NMS、
score-aware 和已有模块化后处理方向不因租期压力重新打开。
