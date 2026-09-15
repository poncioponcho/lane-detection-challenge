# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 只存跨会话必须延续的约定与铁律。过程细节看 `2026-*.md` 日志。
> 最近一次整理：2026-09-15 夜间（压缩合并，删已作废条目）。

## 0. 铁律

- **仲裁源**：`docs/DECISIONS.md` + `configs/default.yaml`；跨文档引用用**章节锚点**，**禁用行号**。
- **成绩只认冻结 Oracle 全局单次调用**（`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`）；**per-clip 禁止平均**。本地诊断 metric 仅供训练期扫描。
- **提交纪律**：`prepare_submit.py`（manifest 枚举 + 1 位小数 canonicalize）→ verify → 官方 `check_submission.py`；**不自动交**，交前用户确认。incumbent 配方 = **54ep model_best(conf0.50) + trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`，tag `incumbent-0.73574`）。
- **🚫 提交一律"交包 + 备注给用户手动提交"，绝不反复开浏览器**：沙箱内 Chromium 恒 `sandbox initialization failed`；用户扫的是 agent-browser 临时窗口，close 即丢。
- **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧全禁。
- **候选放行三门禁**：全局 ΔF1 ≥ +1pp；paired video bootstrap CI 下界 > 0；≥5/8 video 正向。
- **跨机器时间必须标时区**：实例跑 UTC，北京 = UTC+8。B 榜按**北京时间**口径（9/16 00:00–9/17 17:00）。

## 1. 评估协议与锚点

- **切分铁律**：`v1_seed42` 是 clip-level，val 100% video 级泄漏 → **v1 val 上 <2pp 的消融不可判定**。
- **权威基线**：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1 = 0.777628**（多次逐位复现），video-cluster 95% CI 半宽 **7.4pp** → 本地几乎判不出 <7pp 差异。
- **⚠ 两个禁用的假数**：36ep LVO 导出是未过阈值原始树（11 线/图），不可作基准；full71 模型在 800 图 val 上的 0.88~0.89 是污染数（val ⊂ 训练）。
- **A 榜已关闭**。B 榜 6 发取 max 定最终名次。A 榜门槛：R3 0.81604 / 榜首 0.82389（我方 0.73574，差 8pp）。

## 2. 环境坑（本机 + 实例）

- 冻结 Oracle 在持久路径 `envs/lane-oracle-py312`（py3.12 + numpy2.1.3/scipy1.15.3/cv2 4.12）；`/private/tmp` 会被系统清理。
- **BSD grep 对中文 + `\|` 静默零命中** → 中文检索必须用内置 Grep 工具。
- **`pkill -f`/`pgrep -f` 会匹配到自身 shell** → 命令自杀（exit 137）。用 `ps -eo pid,comm`。
- **工作区批量删除有守卫**：`shutil.rmtree` 删 >50 文件被 `SAFE_DELETE_BULK_CONFIRM_REQUIRED` 拦下（exit 137）。实验脚本**覆盖写**，勿先删。
- **长命令会被沙箱 SIGTERM**：>2min 本地任务用 `run_in_background`；远端 `setsid nohup ... & disown`。`scp -r` 目录易被杀 → 远端先 `tar czf` 再传单文件。
- `software-*` subagent 调 TaskList 必崩；长文档定向修订主 Agent 直接 Edit 更稳。
- git push：本沙箱直连 GitHub 出网被拦（502）；可靠写法 `git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push -u origin <branch>`。
- **未跟踪文件不被"工作树干净"守卫拦住**，且不在 bundle 备份里（已发生：实例上有 2 个来源不明的 `ms720/ms880` 配置残留）。

## 3. 实例与代码同步（恒源云）

- **生产实例** `i2b0715374400501416`（3090-24G 包天）。**SSH 连接串随实例重建而变，每次开机必须从控制台重取，禁止沿用旧值**（2026-09-15 用 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`，BatchMode）。
- 代码同步 = **git bundle**：本地 `git bundle create /tmp/hl.bundle <实例实际HEAD>..HEAD` → scp → 实例 `mv` 原子替换 → `git fetch && git merge --ff-only`。实例 `origin` 指向陈旧 bundle，**勿据此判断同步状态**。bundle 基点必须用实例实际 HEAD，否则 "Already up to date" 白跑。
- **发车三守卫**（HEAD 一变就重跑）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + **`validate_run.KNOWN_MODELS`**（三处，漏一处拿不到 evidence）。smoke 守卫跑真实 batch 过整条 train_process，能在发车前暴露增强构造失败。
- **`infer_testA.py` evidence 形状**：`selected_best_checkpoint` 须含 `path` + `sha256`（+ bytes/metric_iteration/checkpoint_iteration），权重须与 `run_evidence.json` 同 run-dir。缺 `sha256` 会 `KeyError`。
- 吞吐（3090）：36ep ≈ 2h（21312 iters，0.33 s/iter）；54ep ≈ 3.75h。

## 4. ★ 盈亏线经济学（F1 = 2TP/(P+G)，θ = F1/2）

testA ≈ θ 0.3675，P+G = 5819.8：

| 操作 | 条件 | 单条 ΔF1 |
|---|---|---|
| 删线 | 被删者真线率 < θ | +2θ/(P+G) = **+0.0126pp** |
| 加线 | 新增者真线率 > θ | +2(1−θ)/(P+G) = **+0.0217pp** |
| 修复（伪→真） | — | +2/(P+G) = **+0.0344pp** |

> 修正：曾误传"修复比加线高效 12 倍"，正确是 **1.58×**。修复的优势在体量（OOF 可修 2739 条 vs 共识能捞 ~96 条）。
> 通式：删/加 n 条、真线率 r → `ΔF1 = 2/(P+G)·n·(θ−r)`（删）或 `2/(P+G)·n·(r−θ)`（加）。
> 本项目 θ 即盈亏线：删线要求被删者真线率 < **36.75%**，加线要求 > 36.75%。

## 5. 已闭环（勿重推）

- **分辨率 / cut_height 轴关闭（三次独立否证）**：960×384（+0.028pp）、1366×540（**−1.542pp**）、cut400（−0.77pp）。判据：三种裁剪下预测 top 中位数都是 564 → 模型自己找地平线。
- **全局仿射修正无**：k_scale −0.0029/px（400px 处仅 −1.15px）vs rms 残差 31.4px，符号均值 −0.77px。
- **平滑 / 稀疏化 0**：sub4 +0.013pp、ma5 +0.073pp。预测已足够平滑。
- **共识加线**：控制变量是**同意比例**不是票数。本地 100% 同意 → 真线率 0.5688；testA ≥2/10（20%）→ 0.3414 < 0.3675 → 实测 −0.145pp。B 榜门槛须 ≥50%。杠杆仅 +0.28pp（96 条）。
- **共识平均（`--average`）对已有线零收益**：ΔTP 全部来自新增线。
- **共识删 FP 死**：2664 条中 86.6% 被全部 4 模型支持 → FP 是系统性偏差。
- **🔴 台账"dropped"筛子此前被误标定（2026-09-15 夜间修正）**：swa*/soup* 是 54ep 轨迹的**派生**模型，dropped 天然小（29~51）；**任何独立新训练的模型对 incumbent 天然有 dropped ∈ [104,175]**（实测 36ep 四对照：s42 175 / s101 139 / s202 126 / s303 104），这是模型间不可约分歧，与质量无关。**新判据：必须与同排期同协议的对照组比**，落在 [104,175] = 零效应；<104 才更贴近 incumbent；>175 才算有害。
- **推论**：36ep 新模型**单独交必 ≈ −2pp**（dropped 135 × 真线率 0.80 的删除代价远大于 novel 收益）；只有 **union（dropped=0）** 形态可用，单支撑 novel 真线率 ≈ 盈亏线（+0.07pp 量级）。→ **想靠新模型提分必须训到 54ep 同排期**。
- **空图数是"36ep vs 54ep"排期效应**：36ep 模型空图一律 22–26，54ep 只有 9，与种子/增强无关。→ segmask 的"空图 9→30"MISS 判据主要也是排期效应，不是模型变稀疏。
- **🚫 "多模型一致性当伪真值"失效（三次独立印证）**：9 树 ≥7/9 同意构造伪真值，incumbent 命中率 92.7% 而真实率仅 80.3% → 伪真值召回 1.154 > 1。**任何"用共识当 GT"的定价/筛选都不要用**；可读的只有**相对印证率**（gate6 0.558 > uni 0.406 > swa3/4/7 0.25 > soup 0.18~0.22）。
- 其它已封死：降 conf（低置信带真线率 23%）、单支撑 union（38.5% ≈ 盈亏）、去重（互 IoU>0.5 全 0 条）、flip-TTA（−0.49pp）、端点上外推（−2.43pp）、OS 过采样（−1.07pp）、CLRerNet（−0.45pp）、跨种子 soup（−0.38pp）、VAT、ConvNeXt-T（与 R50 持平）、雾雨增强、**segmask binary（MISS，模型变稀疏与冲召回意图相反）**。
- **短残线过滤（span80）有效**：删 54 条、伪线率 70.4% > 盈亏线 → **+0.127pp**。B 榜保留 `--min-span 80`。**只对"共识平均产生的残线桩"用，不要对独立模型自己的短线套用**。
- **🔴 `filter_short_lanes.py --base` 必须与 `--src` 的 rel 路径逐层对齐**（事故已加守卫）：base 缺文件或 `base_kept==0` 时脚本拒跑。跑完必看 `base_kept` ≈ base 线数。
- **`testA_54ep_raw` = 2688 线 ≠ 已交包 2664 线**（阈值不同），**不可当基线**；基线一律以「已交 zip 本身」为源。
- **测试集无 score sidecar** → 改阈值必须重跑推理。

## 6. trim（唯一正迁移轴）

- 机理：CLRNet 把近端一律推到 y=719，GT 远端起始时 bottom 只到 396–615 → 过冲稀释 30px 描边 IoU。规则 `bottom = gt_bottom_for_top(top) + margin`。
- **全局 margin 曲线（OOF）**：0 → 0.797224｜20 → 0.811045｜**40 → 0.819699**｜60 → 0.817685。最优 40，比 margin 0 高 **+2.25pp**，比 raw 高 +4.22pp。
- **testA 型几何上 margin 是空操作**：起点 ≈564 → gt_bottom(564)=712，+40 = 752 > 719 → 不切。testA 只有 margin 0 有效（+0.069pp）。
- **⭐ 不对称（B 榜最划算的一注）**：margin 40 下界 −0.069pp（testA 型），上界 +2.25pp（train 型）。
- **判别量 = `f<450`（起点 y<450 的线占比），阈值 10%**：≈0 → spread 必为 0（用 margin 0）；>10% → margin 40。**`f<400` 与 top p10 已证伪**。
- **逐 clip 选 margin 做不到**（70/71 clip 个体最优 ≠ 全局最优，oracle 上界仅 +1.90pp 且无 GT 特征可预测）→ 用全局 40。
- **testA 与 train 几何不同**：testA 预测 top 549/564/624、span 155；train OOF top 244/434/564、span 210。本地全量 OOF 结论必须过「GT top≥530 的 12 clip 子集」复核才可迁移。

## 7. 近失配取证（尺子可信）

15ep 诚实 OOF：P=22248 / G=24435 / TP=18151 / FP=4097 / FN=6284，精确 0.816、召回 0.743。

| IoU 带 | 条数 | span_iou 中位 | lat_iou 中位 |
|---|---|---|---|
| 0.45–0.50 | 942 | 0.611 | 0.574 |
| 0.35–0.45 | 1231 | 0.499 | 0.529 |
| 0.25–0.35 | 466 | 0.353 | 0.720 |
| 0.5–0.6（脆弱 TP） | **2958** | — | — |

- 近失配合计 2739 条，全修上限 **+11.73pp**；span 受限 1397 / lateral 受限 1342，几乎对半。
- 横向残差 |dx| 中位 8.29px、符号均值 +0.76px → **逐线误差，无全局结构** → 后处理无从下手。
- testA 分解（junk 探针）：G=3155.8，TP 2138.9/FP 525.1/FN 1016.9，精确 0.803/召回 0.678 → **域税是召回税**，主因"检到但 IoU 未过 0.5"。testA 少检：2.96 线/图 vs GT 3.51 线/图。

## 8. B 榜作战（`docs/runbook_testB.md`）

发车前先算 testB 的 `f<450` 定 margin M∈{0,40}。六注：
1. `54ep model_best conf0.50 + trim(M)`（地板）｜2. **`uni_swa4_g6`**（swa4 ∪ gate6；**严格支配 swa4，不再单发**）｜3. `cons_gate6`（≥60% 同意，纯增量 52）｜4. train 型 → margin 对冲；testA 型 → `conf0.55 + trim0`｜5. **`uni_occlude`**（incumbent 2664 全保留 + occlude 增量 119 = 2783，dropped 0）｜6. 自适应。
- **🔴 第 5 注已改**：原 runbook 写 `cut400 conf0.35`，但它在 A 榜**实测 −0.772pp**——B 榜取 max，已知负数不该占名额，换成 `uni_occlude`（dropped=0 的纯加注）。**若 runbook 与本文冲突，以本文为准。**
- **`scripts/profile_raw_candidate.py`（新工具）**：把**原始预测树**（非 zip）trim 后直接出台账 + 几何画像（`f<450`、span、top、空图）。做同排期对照全靠它。
- **不交 hires**（−1.54pp）；**不降共识门槛到 ≥2/10**；不投 soup。
- **⭐ SWA 依据**：54ep val 早越过峰值（iter 24863 → 0.89565；final → 0.89429），incumbent 用的就是 model_best(24863)。沿**同一轨迹**平均 {24863,29007,…,31967} → 同 basin，风险极低；输出 2650–2663 线 ≈ incumbent 2688（跨种子 soup 2795–2828，多 5%）。
- **⭐ union 优于单发 swa4**：union 原样保留 A 侧（继承 SWA 位移——containment 台账看不见，因为线身份没变只是位置变了），B 侧叠加增量；gate6 含全部 incumbent 线 → dropped=0。**凡"新模型略优于 incumbent"都可照此合成。**
- **实例上真实可用的独立支撑树只有 7 棵**：36ep_s42/s101/s202/s303、clrernet36、cut400、hires。旧版写的 seed43/44/clrernet_15ep **在实例上没有 run-dir**。swa*/soup* 是 base 派生，**不能算独立共识票**。
- **共识门槛标定（testA）**：10 树 k=6(60%) → 66 条/63 图；7 树 k=4(57%) → 65/60；7 树 k=5(71%) → 33/33 → **B 榜用 7 树 + k=4**。⚠️ 本地 `outputs/testA_support_trees/clrernet_36ep` 只有 158/900 文件（残缺），别当基线。
- **候选形态速查（相对 incumbent 2664 线）**：`uni_swa4_g6` = kept 2664/dropped 0/novel 69 ⭐；`cons_gate6` = kept 2664/dropped 0/novel 52；`conf55` = 纯删除 69（估 r≈0.30 → +0.16pp）；`conf60` = 删 129；`swa4` = 删 29/加 27；`soupA/B/C` = 删 39~51/加 170~215（最大赌注，不投）。
- **已验证可打包**（过 prepare_submit + 官方 check）：`outputs/submit_testA_night_{conf55,conf60,soupA,soupB,soupC,swa3,swa4,swa7}_m0.zip`。
- **9/16 执行 = 两条命令**：实例 `scripts/autodl/run_testB_infer.sh`（核对图 → 建 manifest → base54(0.50/0.55)+7 支撑+swa4 共 10 次推理 → 打印 `f<450` → 打 tgz）；本地 `scripts/build_testB_candidates.sh <tgz>`（定 margin → 建 5 注 → 逐个官方预检 → 打印路径+备注）。
- **🔴 `build_testB_manifest.py` 严禁 glob 全量 JPEGImages**（已修）：实例 JPEGImages 含 71 训练 clip + 9 testA + 17 `_hflip` → 原逻辑产出 ~9700 行"testB"清单且不报错。现行为 = **补集**（减 train/testA 已知 clip、排除 `_hflip`），未见 clip 数 ≠ `--expect-clips`（默认 10）就拒跑并列出 clip 名。
- **bundle 必须带 manifest 回传**：`testB_bundle.tgz` 含 `data_processed/manifest_testB.jsonl` + `.list.txt`，本地脚本自动安装并校验 ≥1000 行。
- **所有新候选都无法离线打分**（testA 无 GT、A 榜已关）→ 排序依据是机制强度 + 方差 + 相对印证率，不是实测。
- **⭐ 污染 val 不能当筛子**：segmask 实验 val +0.128pp 但台账反向（dropped 204、空图 9→30）且台账对。**新 full71 模型先算台账（2 分钟 CPU），再考虑花额度**。

## 9. 实验台账（进行中/已结题）

| 分支 | 变量 | 状态 |
|---|---|---|
| `exp/segmask-binary` | `seg_mask_mode=binary_union` | **MISS**（原判据成立但理由被修正，见 §5） |
| `exp/occlusion-aug` | train_process 插 `CoarseDropout`（放仿射之前，p=0.5） | **NULL 零效应**（dropped 135 落在 36ep 对照 [104,175] 正中；污染 val 0.88095 亦低于基线） |
| `all71_seed101_clrnet_r50_54ep` | 同排期独立种子 54ep | 2026-09-15 UTC 15:47 发车，ETA 北京 ~02:35；**目的 = 让 union 的 novel 线具备 54ep 质量**（唯一能提高加线真线率的路子） |
| 旁路 | `EXPERIMENT_OVERRIDE_KEYS` 已含 `seg_mask_mode` → 单变量配方实验优先走 `--experiment-override`，不必新建配置。`candidate_topk` 不在白名单（输出 2.96 线/图 vs topk=12，不可能绑定）。 |

**判据先写死的规矩**：新实验先算台账——若 `dropped` 明显超既有候选（swa 29~36 / soupB 51 / cut400 155）→ 判 MISS 不发额度；形状健康才发 1 发（≥+0.30pp 为有戏，带内不采信）。
