# 2026-09-05 治理一致性与证据边界校验

| 项 | 内容 |
|---|---|
| 文档版本 | **v1.7** |
| 状态 | **active**（9/6 A 榜反馈、sidecar 量纲修正、NMS/分辨率/低照度筛选已纳入；后续以 DECISIONS 新裁决更新） |
| 版本链 | v1.0（初次校验）→ v1.1（导入路径修复后的复核）→ v1.2（D→C 完成与阈值扫描）→ v1.3（testA 候选执行）→ v1.4（A 榜反馈）→ v1.5（sidecar 语义修正）→ v1.6（NMS 小扫 NO-GO）→ **v1.7**（分辨率/低照度 NO-GO） |
| 上游依据 | `docs/DECISIONS.md` §14 / §17–§30；`configs/default.yaml` v6 |
| 更新日期 | 2026-09-06 |

## 结论

本次校验基于工作区落盘文件、LVO OOF 官方 Oracle 结果、AutoDL 只读状态和 C 阶段回传产物完成。当前生效目标已统一为：

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

`overview.md`、`TASKS.md` 和 `docs/a_bang_snapshot.md` 的主动导语已同步到上述口径。DECISIONS 的早期 84.0/82.0/80.3 等数字和 A 榜历史快照正文保留为历史证据，并明确不再作为当前常量。NMS、`960×480 + cut=0` 分辨率组合和条件化低照度 gamma 筛选均已完成并关闭，不改变生产候选；生产继续为 `800×320 + cut=180`、`t=0.50`。

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
| 分辨率 `960×480 + cut=0` eval-only | `−50.938` 至 `−56.460pp`（阈值 `0.40–0.60`） | 所有 CI 下界 `<0` | NO-GO；该结论适用于整套预处理组合 |
| 条件化低照度 gamma（≤42，γ=`0.85`） | `−0.020pp`（全局） | `[−0.077,+0.021]pp`；low_light `[−0.175,+0.051]pp` | NO-GO；不进入生产或训练增强 |
| 36ep baseline vs screen | `+2.4163pp` | `[−1.616,+8.541]pp` | 仅说明训练阶段变化，不能消除 video 泄漏 |

LVO 场景归因中，`v546797496` F1=`0.4676`，FP/img=`1.73`，而其他 fog video FP/img=`0.76`；去掉该 video 后 fog F1 从 `0.6561` 升至 `0.7411`。因此当前证据支持“压 FP/重复线”优先，不支持把 fog/rain 增强作为默认高杠杆方向。

### D→C raw/conf 扫描（2026-09-05）

- D 状态为 `complete`；8/8 个 fold 的 `fold_state.json` 均为 `status=pass`。
- C 状态为 `complete`；`midpoint` 和 `final` 各覆盖 7100/7100 张图，8 个 fold 均通过；候选导出阈值为 `0.0`，每条导出线都有独立 metadata score。
- 冻结官方 Oracle 扫描 16 个变体全部 `status=pass`，输入为 8 个 video cluster、7100 张图、10000 次成对 bootstrap；完整 JSON 位于被 `.gitignore` 排除的 `outputs/lvo_conf_checkpoint_scan_20260905/conf_checkpoint_scan.json`。
- 参考为 `final_conf_0p40`，F1=`0.752703109`。`final_conf_0p30` 为 F1=`0.759815559`，Δ=`+0.7112pp`，video-paired CI=`[+0.2363,+1.1391]pp`；`final_conf_0p35` 为 F1=`0.756899602`，Δ=`+0.4196pp`，CI=`[+0.1384,+0.6684]pp`。
- 更高阈值均持续损失召回；但 A 榜真实记录 `714962=0.73444`（conf=`0.5`）高于 `714942=0.72794`（conf=`0.4`），所以 `0.30/0.35` 只进入 testA 候选验证，不直接替换当前榜单候选。

### testA 候选执行（2026-09-05）

- 使用同一生产 checkpoint 做 conf=`0.30/0.35` 两次 eval-only；两者均通过 900/900 文件覆盖、官方一位小数重契约和 zip verify。
- conf=`0.30`：2909 条线、7 个空文件、557847 bytes，SHA-256=`1c4475e332b6e277427c280bae40c70d10b0cea5240c16673620ce56262747f0`。
- conf=`0.35`：2875 条线、7 个空文件、553927 bytes，SHA-256=`3742a589d875262b2d682531d8d9c4fb51bd603786bc1a0906fab4fa6d67c797`。
- 候选 zip 和证据位于被 `.gitignore` 排除的 `outputs/testA_conf_candidates_20260905/`；不把无 GT 的 testA 结果解释为 F1，也不在未获榜单反馈前改默认阈值。

### conf=0.30 真实 A 榜反馈（2026-09-06）

- 提交记录 `715300` 返回官方得分 `0.72613`；提交文件 SHA-256 与上项一致，900/900 文件校验通过。
- 该分数低于 `714962 / t=0.50 / 0.73444` `0.831pp`，也低于 `714942 / t=0.40 / 0.72794`；`conf=0.30` 退出生产候选，默认阈值继续保持 `t=0.50`。
- `conf=0.35` 尚未提交，不对其 A 榜效果作推断。详细提交账本见 `docs/a_bang_submit_20260906_conf030.md`。

### sidecar 量纲 P0 修正（2026-09-06）

- pinned CLRNet 的 decode gate 使用 `softmax(logits[:, :2])[:, 1]`，但旧
  `predictions_to_pred()` 曾把原始正类 logit 写入 `metadata["conf"]`；旧 C
  sidecar 范围约为 `[-3.05, 2.16]`，所以旧 LVO `conf` 扫描不是概率扫描。
- 旧扫描与旧 C 产物保留为历史 raw-logit 实验，不能继续支撑阈值映射或“LVO/A
  榜冲突完全来自域差异”的解释；A 榜直接解码记录本身不受此 sidecar 问题影响。
- 当前代码已把 sidecar 分数改为正类 softmax 概率，加入 schema/version、范围、
  `post_nms` 和语义校验；扫描器会拒绝 legacy/raw-logit sidecar。修正后的 C
  使用新目录 `outputs/lvo_clrnet_r50_36ep_c_export_20260906/`，并已用于后续
  NMS、分辨率和低照度 eval-only 筛选。
- 直接解码与离线过滤的等价性检查入口为
  `scripts/check_decode_equivalence.py`；通过前不把离线阈值曲线当作生产证据。

## 当前方向排序

1. **证据与解码链已收口**：sidecar 已改为正类 softmax probability，直接解码↔离线过滤等价性检查通过；旧 raw-logit 曲线降级为历史记录。
2. **score-aware / NMS 已关闭**：cross-fit score-aware ranker、几何预筛、cap4/5、无分数 flip-union 与 `nms_thres/nms_topk` 小扫均未达到稳定收益，不进入生产。
3. **分辨率与低照度已关闭**：`960×480 + cut=0` 的整套组合在 `0.40–0.60` 全部退化；条件化 `gamma=0.85` 在全局及 low_light/normal 分桶均无稳定正收益。详见 `docs/lvo_resolution_scan_20260906.md` 和 `docs/lvo_lowlight_scan_20260906.md`。
4. **当前生产冻结**：CLRNet-R50 36ep、`800×320 + cut=180`、显式 `conf_threshold=0.50`；不为已判 NO-GO 变体消耗剩余 A 榜额度，继续维护本地 val ↔ A 榜相关性台账。
5. **唯一保留的高成本探针**：若 9/10 资源、磁盘和时间门均通过，再做 ConvNeXt-T 15ep screen；不直接启动完整主干训练。

几何预筛已落盘于 `outputs/lvo_geometry_scan_20260905_v3/`：重复线过滤 5–20px 全部 no-op；cap4 为 `−1.260pp`、cap5 为 `−0.206pp`，均不构成可保留的正向变体。

## 远端状态

截至 D/C 日志 `2026-09-05T14:15:55Z`：AutoDL `pipeline.status=complete`，winner=`clrnet_r50`；独立 36ep LVO 的 D/C 均为 `complete`。sidecar 已完成概率语义修正，直接解码等价性、score-aware、NMS、分辨率和低照度 eval-only 筛选均已完成；随后 testA conf=`0.30/0.35` 候选均已完成并回传，conf=`0.30` 已完成真实 A 榜验证。远端磁盘约 100GB 中使用 98%、剩余约 2.3GB；在资源门通过前不启动新的大规模训练。

## 校验命令结果

- YAML 解析、目标/预算/队伍常量一致性：PASS。
- split 泄漏计算：PASS；63 个 train clip / 8 个 val clip，5 个 val video 与 train video 集合相交。
- `git diff --check`：PASS。
- 定向契约测试（Oracle hash、manifest、split）：`24 passed`。
- 全套测试：`141 passed`（含 score sidecar、直接解码/离线过滤等价性和低照度配置契约测试）；此前 `test_evaluator_still_scores_labeled_after_unlabeled_support` 的导入路径问题已修复为同时注入仓库根目录与 `src/`，并通过复跑确认。
- C 输入覆盖校验：midpoint/final 各 7100/7100 prediction files，score image 各 7100/7100，PASS。
- raw/conf checkpoint 扫描：16 variants、全局与 8 video cluster Oracle、10000 次 paired bootstrap，`status=pass`。
- testA 候选校验：conf=`0.30/0.35` 各 900/900 文件，zip 清单精确匹配 manifest，官方 verify PASS。

## NMS 小扫补充（2026-09-06）

- `nms_topk=8,nms_thres=50` 与原始 `nms_topk=12,nms_thres=50` 在 0.40–0.60
  概率阈值下输出和 F1 完全一致。
- `nms_thres=30` 的同阈值最大点增益为 `+0.027pp`；8-video paired CI
  下界为 0，未达到“严格大于 0”的确认条件。生产 NMS 和 `t=0.50` 均不变。
- 结果由冻结官方 Oracle 生成，完整证据为
  `docs/lvo_nms_scan_20260906.md` 与
  `outputs/lvo_nms_conf_scan_20260906/conf_checkpoint_scan.json`（outputs
  不入 Git）。分辨率与定向低照度筛选也已完成且均为 NO-GO；关键路径转入
  9/10 定模型门槛，ConvNeXt-T 仅保留为资源门控下的最后 15ep screen。
