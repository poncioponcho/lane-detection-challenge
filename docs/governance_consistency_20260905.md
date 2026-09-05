# 2026-09-05 治理一致性与证据边界校验

| 项 | 内容 |
|---|---|
| 文档版本 | **v1.1** |
| 状态 | **active**（9/5 校验记录；后续以 DECISIONS 新裁决更新） |
| 版本链 | v1.0（初次校验）→ **v1.1**（导入路径修复后的复核） |
| 上游依据 | `docs/DECISIONS.md` §14 / §17–§26；`configs/default.yaml` v6 |
| 更新日期 | 2026-09-05 |

## 结论

本次校验基于工作区落盘文件、LVO OOF 官方 Oracle 结果和 AutoDL 只读状态完成。当前生效目标已统一为：

| 键 | 当前值 | 唯一依据 |
|---|---:|---|
| `target.b_board_f1_work` | `0.77` | `configs/default.yaml` / DECISIONS §26 |
| `target.b_board_f1_stretch` | `0.79` | 同上 |
| `target.b_board_f1_pred` | `0.75` | 同上 |
| `target.gap_pp` | `4.0` | 同上 |
| `target.stage_ceiling` | `0.81` | 同上 |
| `budget.gpu_cny_cap` | `200` | 同上 |
| `team.size` | `1` | 同上 |
| `target.recheck_dates` | `2026-09-10` | 同上 |

`overview.md`、`docs/ARCHITECTURE.md`、`TASKS.md` 和 `docs/a_bang_snapshot.md` 的主动导语已同步到上述口径。DECISIONS 的早期 84.0/82.0/80.3 等数字和 A 榜历史快照正文保留为历史证据，并明确不再作为当前常量。

## 统计规矩的生效层级

| 层级 | 当前状态 | 解释 |
|---|---|---|
| DECISIONS §17.3 + `configs/default.yaml::validation` | 生效 | `split: by_clip`、`bootstrap: paired_clip`、点估计门槛、paired CI 下界和 LOCO；适用于原 v1 clip-level 验证链的训练内监控 |
| R1–R4 | 仅 memory 草案 | video-disjoint 泛化、video-cluster bootstrap、video 口径阈值重标、能力预测只吃诚实口径；尚未并入 DECISIONS，因此没有改写 YAML validation |
| LVO OOF | 已有证据，不等于新治理闸门 | 8-fold leave-one-video-out，7100 图，8 个 video；用于当前跨 video 泛化判断和方向排序 |

这一区分是刻意保留的：不能把尚未批准的 R1–R4 追认成正式裁决，也不能继续把泄漏 val 当作跨域泛化尺子。

## 证据核验

### LVO 与泄漏诊断

- `configs/splits/v1_seed42.yaml` 的 8 个 val clip 属于 5 个 video；训练 manifest 的 71 个 clip 覆盖这 5 个 video，因此 5/5 val video 在 train 中有其他 clip。
- 泄漏 val 的 baseline 官方 Oracle：F1=`0.8081202420`，训练过程诊断 best 为 `0.8089010346`；两者都降级为训练内监控。
- LVO OOF：TP/FP/FN=`18151/4097/6284`，F1=`0.7776278303`；8 个 video cluster bootstrap 95% CI=`[0.6886338548, 0.8374020769]`。
- OOF prediction tree 覆盖 7100 张图；271 张为空预测，最大输出 6 条线。文件只含几何 `.lines.txt`，不含候选置信度。

### A 榜口径

`0.73444` 来自真实提交记录 `714962`，对应 36ep CLRNet-R50 生产模型的 conf=`0.5` 变体；不是 val 推算值。对照记录为 `714942=0.72794`（conf=`0.4`），flip-union TTA 为 `0.72950`。

### 已有实验的正式读数

| 实验 | 官方 Oracle 点估计 | paired clip CI | 当前判定 |
|---|---:|---:|---|
| 雾+雨增强 | `−0.0205pp` | `[−2.406,+2.131]pp` | NO-GO；不再作为默认主线 |
| 分辨率方案 B | `+0.1881pp` | `[−1.831,+1.934]pp` | 未过旧 clip 闸门，不能证明有效 |
| 36ep baseline vs screen | `+2.4163pp` | `[−1.616,+8.541]pp` | 仅说明训练阶段变化，不能消除 video 泄漏 |

LVO 场景归因中，`v546797496` F1=`0.4676`，FP/img=`1.73`，而其他 fog video FP/img=`0.76`；去掉该 video 后 fog F1 从 `0.6561` 升至 `0.7411`。因此当前证据支持“压 FP/重复线”优先，不支持把 fog/rain 增强作为默认高杠杆方向。

## 当前方向排序

1. **零 GPU、几何可靠的压 FP 后处理**：在 video-disjoint OOF 上做重复线过滤、几何 NMS、保守输出截断等扫描；先核对输出顺序和语义，不能把 `.lines.txt` 行序当成 score。
2. **一次 checkpoint raw/conf-aware 推理扫描**：现有 OOF 不能扫 conf；需要重新推理取得候选分数，再做 conf/NMS/max-output 的组合扫描。结果先作为调参证据，待正式 video 口径批准后再作为闸门结论。
3. **36ep video-disjoint LVO**：用同一生产 checkpoint 做完整跨 video 尺子，成本约一轮 GPU；启动前必须按 runbook 归档明确旧 run 释放空间，禁止直接删除。
4. **训练型增强/分辨率**：只有在第 2、3 项给出方向后再开；fog/rain 不再默认优先。
5. **score-aware TTA**：既有 flip-union A 榜为 `0.72950`，低于 t05，不再做无分数 union；只有带分数匹配、融合、NMS 的版本值得重新评估。

几何预筛已落盘于 `outputs/lvo_geometry_scan_20260905_v3/`：重复线过滤 5–20px 全部 no-op；cap4 为 `−1.260pp`、cap5 为 `−0.206pp`，均不构成可保留的正向变体。

## 远端状态

截至 `2026-09-05T07:26Z`：AutoDL `pipeline.status=complete`，winner=`clrnet_r50`；独立的 36ep LVO 实验 `lvo36.status=running`，正在 `fold_03_v566817042` 训练，C 阶段守护进程仍等待 D 完成。磁盘约 100GB 中使用 92%、剩余约 8.2GB；当前无并行 CUDA 实验，D 完成前不启动其他训练。

## 校验命令结果

- YAML 解析、目标/预算/队伍常量一致性：PASS。
- split 泄漏计算：PASS；63 个 train clip / 8 个 val clip，5 个 val video 与 train video 集合相交。
- `git diff --check`：PASS。
- 定向契约测试（Oracle hash、manifest、split）：`24 passed`。
- 全套测试：`119 passed`（含 score sidecar 契约测试）；此前 `test_evaluator_still_scores_labeled_after_unlabeled_support` 的导入路径问题已修复为同时注入仓库根目录与 `src/`，并通过复跑确认。
