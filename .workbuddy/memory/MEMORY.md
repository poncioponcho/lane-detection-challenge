# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 只存跨会话必须延续的约定与铁律。过程细节看 `2026-*.md` 日志。
> 最近一次整理：2026-09-15（合并重复项、剔除已作废结论）。

## 0. 铁律（最高优先级）

- **仲裁源**：`docs/DECISIONS.md` + `configs/default.yaml`；跨文档引用一律用**章节锚点**，**禁用行号**（09-01 事故：行号随编辑漂移失效）。
- **成绩只认冻结 Oracle 全局单次调用**（`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`）；**per-clip 禁止平均**。本地诊断 metric 仅供训练期扫描。
- **提交纪律**：统一走 `prepare_submit.py`（manifest 枚举 + 1 位小数 canonicalize）→ verify → 官方 `check_submission.py`；**不自动交**，交前用户确认。incumbent 配方 = **54ep model_best(conf0.50) + trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`）。
- **🚫 提交一律"交包 + 备注给用户手动提交"，绝不反复开浏览器**（2026-09-13 实证）：沙箱内 Chromium 恒 `sandbox initialization failed`；用户扫的是 agent-browser **临时**窗口，close 即丢 → 反复扫码是严重体验事故，已被点名。
- **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧全禁。
- **候选放行三门禁**：全局 ΔF1 ≥ +1pp；paired video bootstrap CI 下界 > 0；≥5/8 video 正向。

## 1. 评估协议与锚点

- **切分铁律**：`v1_seed42` 是 clip-level，val 100% video 级泄漏 → **v1 val 上 <2pp 的消融不可判定**。新消融先问"是否 video-disjoint"。
- **权威基线**：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1 = 0.777628**（2026-09-15 再次逐位复现，尺子可信），video-cluster 95% CI 半宽 **7.4pp** → 本地几乎判不出 <7pp 差异。
- **⚠ 36ep LVO 导出是未过阈值原始树**（11.07/11.46 线每图），**不可**作诊断基准；15ep OOF 是 3.13 线每图。
- **⚠ full71 模型在 800 图 val 上的 0.88~0.89 是污染数**（val ⊂ 训练），**禁止**与 0.7776 / A 榜分比较。
- **A 榜已关闭**（规则：8/19–9/14，每日 3 发，9/14 已用尽）。**B 榜 9/16 00:00–9/17 17:00，6 发取 max，定最终名次**。A 榜实测门槛：R3 0.81604 / 榜首 0.82389（我方 0.73574，差 8pp）。

## 2. 环境坑（本机 + 实例）

- 本地 CPU venv `lane`（py3.13）；**冻结 Oracle 必须在持久路径** `envs/lane-oracle-py312`（py3.12 + numpy2.1.3/scipy1.15.3/cv2 4.12），`/private/tmp` 会被系统清理。
- **BSD grep 对中文 + `\|` 静默零命中** → 中文/多模式检索必须用内置 Grep 工具。
- **`pkill -f` / `pgrep -f` 会匹配到承载该字符串的自身 shell** → 命令自杀（exit 137）。用 `ps -eo pid,comm`。
- **工作区批量删除有守卫**：`shutil.rmtree` 删 >50 文件会被 `SAFE_DELETE_BULK_CONFIRM_REQUIRED` 拦下（exit 137）。实验脚本**覆盖写**即可，不要先删。
- **长命令会被沙箱 SIGTERM**：>2min 的本地任务用 `run_in_background`；远端用 `setsid nohup ... & disown`。`scp -r` 目录易被杀 → 先在远端 `tar czf` 再传单文件。
- `software-*` subagent 调 TaskList 必崩；长文档定向修订主 Agent 直接 Edit 更稳。
- git push 走 SSH 会卡 passphrase；可靠写法：
  `git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push -u origin <branch>`

## 3. 实例与代码同步（恒源云）

- **生产实例** `i2b0715374400501416`（3090-24G 包天，已续费过 9/15）。**SSH 连接串随实例重建而变，每次开机必须从控制台重取，禁止沿用旧值**；2026-09-15 可用 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`（BatchMode）。
- 代码同步 = **git bundle**（非 pull）：本地 `git bundle create /tmp/hl.bundle <实例HEAD>..HEAD` → scp 到 `/hy-tmp/bundles/hl.bundle.new` → 实例 `mv` 原子替换 → `git fetch ... && git merge --ff-only`。实例 `origin` 指向陈旧 bundle，**勿据此判断同步状态**。
- 实例发车三守卫（HEAD 一变就重跑）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + **`validate_run.KNOWN_MODELS`**（三处，漏一处就拿不到 evidence）。
- **`infer_testA.py` 读的 evidence 形状**：`selected_best_checkpoint` 必须含 `path` + `sha256`（+ bytes/metric_iteration/checkpoint_iteration），且权重必须与 `run_evidence.json` 同在一个 run-dir。缺 `sha256` 会 `KeyError`。
- 训练吞吐（3090）：36ep ≈ 2.5h（31932 iters）。

## 4. ★ 盈亏线经济学（`F1 = 2TP/(P+G)`，θ = F1/2）

全集池化下可微，令 θ = TP/(P+G) = F1/2（testA ≈ 0.3675，P+G = 5819.8）：

| 操作 | 条件 | 单条 ΔF1 |
|---|---|---|
| 删线 | 被删者**真线率 < θ** | `+2θ/(P+G)` = **+0.0126pp** |
| 加线 | 新增者**真线率 > θ** | `+2(1−θ)/(P+G)` = **+0.0217pp** |
| 修复（伪→真） | — | `+2/(P+G)` = **+0.0344pp** |

> ⚠️ 修正：曾误传"修复比加线高效 12 倍"。**正确是 1.58×**（把"净增 45% 真线率并集"误当成"新增纯真线"）。修复的真正优势在体量：OOF 上可修 2739 条 vs 共识门槛能捞 ~96 条。
> 通用式：删/加 n 条、真线率 r → `ΔF1 = 2/(P+G) · n · (θ − r)`（删）或 `2/(P+G) · n · (r − θ)`（加）。

## 5. 已闭环（勿重推）

- **分辨率 / cut_height 轴关闭（三次独立否证）**：960×384（+0.028pp）、1366×540（**−1.542pp**）、cut400（几何剖面判否）。判据：**三种裁剪下预测 top 中位数都是 564** → 模型自己找地平线，不被裁剪面框住。
- **全局仿射修正：无**（2026-09-15）。`dx = c1(x−683)+c2(y−360)` 回归：k_scale = −0.0029/px（400px 处仅 −1.15px）vs rms 残差 **31.4px**；符号均值 −0.77px。无系统性偏置。
- **平滑 / 稀疏化：0**（2026-09-15）。sub4 +0.013pp、ma5 +0.073pp（冻结 Oracle 全局）。预测本身已足够平滑，无抖动可修。
- **共识加线**：控制变量是**同意比例**不是票数。本地 100% 同意 → 真线率 0.5688；testA 用 ≥2/10（20%）→ **0.3414 < 0.3675** → 实测 **−0.145pp**。加树只会让"≥k 票"更松。B 榜门槛须 ≥50%。杠杆仅 ≈ **+0.28pp**（96 条）。
- **共识平均（`--average`）对已有线是零收益**：9/14 分解显示 ΔTP 全部来自新增线，被保留线净变化 = 0。
- 其它已封死：降 conf（低置信带真线率 23%）、单支撑 union（38.5% ≈ 盈亏）、去重（互 IoU>0.5 全 0 条）、融合/投票换线、flip-TTA（−0.49pp）、端点上外推（−2.43pp）、OS 过采样（testA −1.07pp）、VAT、ConvNeXt-T（与 R50 持平）、雾雨增强。
- **共识删 FP 死**：2664 条中 86.6% 被全部 4 模型支持 → FP 是**系统性偏差**，不是个体怪癖。
- **短残线过滤（span80）有效**：删 54 条、其中伪线 70.4% > 删线盈亏线 63.2% → **+0.127pp**。B 榜保留 `--min-span 80`。
- **测试集无 score sidecar** → 改阈值必须重跑推理。

## 6. trim（唯一正迁移轴）—— 2026-09-15 定稿

- 机理：CLRNet 把近端一律推到 y=719，而 GT 远端起始时 bottom 只到 396–615 → 过冲稀释 30px 描边 IoU。规则 `bottom = gt_bottom_for_top(top) + margin`（表来自 train GT）。
- **全局 margin 曲线（OOF）**：0 → 0.797224｜20 → 0.811045｜**40 → 0.819699**｜60 → 0.817685。**最优 40**，比 margin 0 高 **+2.25pp**，比 raw 高 +4.22pp。
- **testA 型几何上 margin 是空操作**：起点 ≈564 → `gt_bottom(564)=712`，+40 = 752 > 719 → 不切任何点。故 testA 只有 margin 0 有效（+0.069pp）。
- **⭐ 不对称（B 榜最划算的一注）**：margin 40 下界 **−0.069pp**（testA 型），上界 **+2.25pp**（train 型）。
- **判别量 = `f<450`（起点 y<450 的线占比），阈值 10%**：`f<450 ≈ 0` → spread 必为 0；`>10%` → 用 margin 40。**`f<400` 与 top p10 已证伪**（`v644768616_1_0_1085` 的 f<400=0 但 spread +25.81pp）。
- **逐 clip 选 margin 做不到**：71 clip 中 70 个个体最优 ≠ 全局最优；train 型内部有两派方向相反（要 0 的 vs 要 60 的），单 clip 差距可达 +50pp。oracle 上界仅 +1.90pp 且无 GT 特征可预测 → 放弃，用全局 40。
- **testA 与 train 几何不同**：testA 预测 top 549/564/624、span 155；train OOF top 244/434/564、span 210。本地全量 OOF 结论必须过「GT top≥530 的 12 clip 子集」复核才可迁移。

## 7. 近失配取证（2026-09-15，尺子可信）

15ep 诚实 OOF：P=22248 / G=24435 / TP=18151 / FP=4097 / FN=6284，精确 0.816、召回 0.743。

| IoU 带 | 条数 | span_iou 中位 | lat_iou 中位 | 说明 |
|---|---|---|---|---|
| 0.45–0.50 | 942 | 0.611 | 0.574 | 最便宜的一档 |
| 0.35–0.45 | 1231 | 0.499 | 0.529 | |
| 0.25–0.35 | 466 | 0.353 | 0.720 | span 受限 67% |
| 0.5–0.6（脆弱 TP） | **2958** | — | — | 占全部预测 16.3%，指标就压在这条线上 |

- 近失配合计 **2739 条**，全修上限 **+11.73pp**；span 受限 1397 / lateral 受限 1342，**几乎对半**。
- 横向残差 |dx| 中位 **8.29px**、符号均值 +0.76px → **逐线误差，无全局结构** → 后处理无从下手。
- testA 分解（junk 探针）：G=3155.8，TP 2138.9/FP 525.1/FN 1016.9，精确 0.803/召回 0.678 → **域税是召回税**，主因是"检到但 IoU 未过 0.5"。
- 模型在 testA **少检**：2.96 线/图 vs GT 3.51 线/图。

## 8. B 榜作战（2026-09-15 v2，`docs/runbook_testB.md`）

发车前先算 testB 的 `f<450` 定 margin M∈{0,40}。六注：
1. `54ep model_best conf0.50 + trim(M)`（地板）｜2. **SWA + trim(M)**（见下）｜3. train 型 → margin 对冲注；testA 型 → `conf0.55 + trim0`｜4. 共识 ≥50% 同意 + span80 + trim｜5. `cut400 conf0.35 + trim + span80`｜6. 自适应。
- **不交 hires**（−1.54pp）；**不降共识门槛到 ≥2/10**。
- **⭐ SWA 依据**：54ep 的 val 早越过峰值（iter 24863 → 0.89565；final 31968 → 0.89429），incumbent 用的就是 model_best(24863)。沿**同一轨迹**平均 {24863,29007,…,31967} → 保证同 basin，风险极低；输出 2650–2663 线 ≈ incumbent 原始 2688（对比跨种子 soup 2795–2828，多 5%）。已建 `swa3/swa4/swa7_54ep` 并跑完 testA 推理。
- **已验证可打包**（全部过 `prepare_submit` + 官方 `check_submission`）：`outputs/submit_testA_night_{conf55,conf60,soupA,soupB,soupC,swa3,swa4,swa7}_m0.zip`。
- **无 GT 的候选台账**（`scripts/profile_night_candidates_20260915.py`）：conf55 = 保留 2595/删 69/新增 0（**纯删除赌注**，需被删 69 条真线率 <36.75%，估 r≈0.30 → 约 +0.16pp）；conf60 = 删 129（更大赌注，倾向更差）；soupA = 删 39/新增 170；soupB = 删 51/新增 215。
- **所有新候选都无法离线打分**（testA 无 GT、A 榜已关）→ 排序依据是机制强度 + 方差，不是实测。
