# 2026-09-05 D→C LVO raw/conf 扫描记录

## 结论

D→C 流程已完成。8 个 leave-one-video-out fold 均通过训练、holdout 导出和预测覆盖校验；`midpoint` 与 `final` 各覆盖 7100/7100 张图。随后使用冻结官方 Oracle 对两个 checkpoint、8 个置信度阈值做了 16 个变体扫描。

扫描完整 JSON（含全局及逐 video TP/FP/FN）保留在本地被 `.gitignore` 排除的：

`outputs/lvo_conf_checkpoint_scan_20260905/conf_checkpoint_scan.json`

## 关键读数

- 参考：`final_conf_0p40`，F1=`0.752703109`，TP/FP/FN=`16255/2501/8180`。
- `final_conf_0p30`：F1=`0.759815559`，TP/FP/FN=`16643/2730/7792`，相对参考 `+0.7112pp`，video-paired 95% CI=`[+0.2363,+1.1391]pp`。
- `final_conf_0p35`：F1=`0.756899602`，TP/FP/FN=`16469/2613/7966`，相对参考 `+0.4196pp`，CI=`[+0.1384,+0.6684]pp`。
- `final_conf_0p45/0p50/0p55/0p60/0p65` 相对参考分别为 `−0.5143/−1.0766/−1.6653/−2.2955/−3.1230pp`，均未形成方向反转。
- midpoint 最优也是 conf=`0.30`，F1=`0.754492120`；低于 final conf=`0.30`，因此 checkpoint 候选暂选 final。

## 证据边界与自审

1. CI 的重采样单位是 8 个 video cluster，结果是调参证据，不追认尚未写入 DECISIONS 的新治理闸门。
2. LVO 有 GT，testA 没有 GT；不能从 testA 文件或本地可视化伪造 F1。
3. `.lines.txt` 行序没有被当作置信度；过滤只使用 evaluator 导出的 score sidecar。
4. LVO 与 A 榜方向存在冲突：A 榜 `714962=0.73444`（conf=`0.50`）高于 `714942=0.72794`（conf=`0.40`）。因此不直接改 `configs/default.yaml` 或生产默认阈值。

## 下一步执行方案

1. 使用现有生产 checkpoint（不训练）生成 testA conf=`0.30`、`0.35` 两个 eval-only 目录。
2. 对每个目录执行 900 文件覆盖检查、官方一位小数重契约、zip verify，并保留 checkpoint/config/manifest SHA。
3. 将已有 conf=`0.50` A 榜候选作为基线；每次真实提交最多改变一个变量，未得到榜单反馈前不选择新默认值。
4. 若低阈值提交没有改善，回退 conf=`0.50`；若改善，再只保留实际改善的阈值进入冻结候选。

## 已执行的 testA 候选

第 1 步已完成：使用生产 checkpoint `model_0016799.pth`（SHA-256=`e5919ae561d96d141c7c76a1287d99eeb0ee916d277005f1d3ea4fa036e824cd`）在 AutoDL 做了两次 eval-only 推理。远端实验代码 HEAD 为 `6129e3b8e9f0898fe7c898fbff711af879a3e986`，manifest SHA-256=`50efc9d6e3a125c5ce7574944eab6532e9235cf079e58b33c9d35d1f9fa8a2f3`。

- conf=`0.30`：900/900 文件，2909 条线，7 个空文件；zip=`557847` bytes，SHA-256=`1c4475e332b6e277427c280bae40c70d10b0cea5240c16673620ce56262747f0`，官方 verify PASS。
- conf=`0.35`：900/900 文件，2875 条线，7 个空文件；zip=`553927` bytes，SHA-256=`3742a589d875262b2d682531d8d9c4fb51bd603786bc1a0906fab4fa6d67c797`，官方 verify PASS。

两个候选均已回传至被 `.gitignore` 排除的 `outputs/testA_conf_candidates_20260905/`。testA 无 GT，当前不宣称本地 F1；下一步只能在用户确认后逐个真实提交，继续保留已有 conf=`0.50` 作为榜单基线，不自动改默认配置。
