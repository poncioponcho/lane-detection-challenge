# 融合产物 Oracle 重放审计（2026-09-09）

## 结论

此前 `lvo36_fusion_*` 与 `mixed_fusion_*` 的报告出现 `TP=0/FN=0`，原因是重放时传入了错误的 GT 根目录；对应预测目录本身覆盖完整且含有效线条。使用冻结官方 Oracle、同一 800 图 manifest 和正确的
`data/raw/dataset/_extract/train_full/Lane/anno_txt` 根目录重放后，确认这些产物没有隐藏的高收益分支。

| 产物 | 修正后 F1 | TP / FP / FN |
|---|---:|---:|
| `lvo36_q4` | 0.8829380260 | 2308 / 265 / 347 |
| `lvo36_q5` | 0.8826794258 | 2306 / 264 / 349 |
| `lvo36_q6` | 0.8809980806 | 2295 / 260 / 360 |
| `mixed_q6` | 0.8842991012 | 2312 / 262 / 343 |
| `mixed_q7` | 0.8842105263 | 2310 / 260 / 345 |
| `mixed_q8` | 0.8836496071 | 2305 / 257 / 350 |

## 可复现口径

- manifest：`data/processed/manifest_val_v1_seed42.jsonl`（800 图）
- GT：`data/raw/dataset/_extract/train_full/Lane/anno_txt`
- Oracle：冻结 `src/eval/official_oracle/score.py`，通过 `src/eval/oracle_runner.py`
- 输出证据：`outputs/reports/*_corrected_20260909.json`

## 处置

该修正只恢复真实评估读数，不改变任何模型、配置、incumbent 或榜单提交。上述候选不作为冲前三的独立新能力；继续等待当前 6300 图 LVO 链和低权重 VAT/mask 实验的正式结果。
