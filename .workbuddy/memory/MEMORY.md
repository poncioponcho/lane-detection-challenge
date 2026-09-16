# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 只存跨会话必须延续的约定与铁律。过程细节看 `2026-*.md` 日志。
> 最近一次压缩重写：2026-09-16 09:00（合并重复章节、剔除已作废条目）。

## 0. 铁律

- **仲裁源**：`docs/DECISIONS.md` + `configs/default.yaml`；跨文档引用用**章节锚点**，禁用行号。
- **成绩只认冻结 Oracle 全局单次调用**（`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`）；per-clip 禁止平均。
- **提交**：`prepare_submit.py`（manifest 枚举 + 1 位小数 canonicalize）→ verify → 官方 `check_submission.py`；**交包 + 备注给用户手动交，绝不自动交、绝不反复开浏览器**（沙箱 Chromium 恒失败）。
- **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧全禁。
- **候选放行三门禁**：全局 ΔF1 ≥ +1pp；paired video bootstrap CI 下界 > 0；≥5/8 video 正向。
- **跨机器时间标时区**：实例 UTC，北京 = UTC+8。B 榜窗口按北京时间 9/16 00:00–9/17 17:00。
- **清理目录一律 `mv` 到 /tmp，不要 `rm -rf`**（>50 文件触发守卫 exit 137）。
- incumbent 配方 = **54ep model_best(conf0.50) + trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`，sha256 `a7dcef5b…72af`）。基线一律以「已交 zip 本身」为源，`testA_54ep_raw` 是 2688 线不可当基线。

## 1. 评估协议与锚点

- **切分铁律**：`v1_seed42` 是 clip-level，val 100% video 级泄漏 → v1 val 上 <2pp 的消融不可判定。
- **权威尺子**：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1 = 0.777628**（多次逐位复现），video-cluster 95% CI 半宽 **7.4pp**。GT 已在本地 `data/gt_train/anno_txt` → OOF 筛选完全本地可跑。
- **两个禁用的假数**：36ep LVO 导出（未过阈值原始树）；full71 模型在 800 图 val 的 0.88~0.89（val ⊂ 训练，污染）。
- A 榜已关闭（我方 0.73574，榜首 0.82389，差 8pp）。B 榜 6 发取 max。
- OOF 是 **train 型几何**，结论只有 testB 也是 train 型时才迁移。

## 2. 环境坑（本机 + 实例）

- **BSD grep 对中文 + `\|` 静默零命中** → 中文检索必须用内置 Grep 工具。
- **`pkill -f`/`pgrep -f` 会匹配自身 shell** → 自杀（exit 137）。用 `ps -eo pid,comm`。
- 本机**没有 `timeout` 命令**。
- **长命令被沙箱 SIGTERM**：>2min 用 `run_in_background`；远端 `setsid nohup ... & disown`。`scp -r` 目录易被杀 → 远端先 `tar czf` 再传单文件。
- `software-*` subagent 调 TaskList 必崩；长文档定向修订主 Agent 直接 Edit 更稳。
- git push 走 `git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push`。
- **未跟踪文件不被"工作树干净"守卫拦住**，也不在 bundle 里。
- 冻结 Oracle 在持久路径 `envs/lane-oracle-py312`（py3.12 + numpy2.1.3/scipy1.15.3/cv2 4.12）；`/private/tmp` 会被系统清理。

## 3. 实例与代码同步（恒源云）

- 生产实例 `i2b0715374400501416`（3090-24G 包天）。**SSH 串随实例重建而变，每次开机从控制台重取**；2026-09-16 09:00 实测 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`（BatchMode）**仍可用**。
- **代码同步 = git bundle**：本地 `git bundle create /tmp/hl.bundle <实例HEAD>..HEAD` → scp → 实例 `mv` 原子替换 → `git fetch && git merge --ff-only`。实例 `origin` 指向陈旧 bundle，勿据此判断同步状态。
- **发车三守卫**（HEAD 一变就重跑，全在 `scripts/autodl/`，**不是** `scripts/`）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + `validate_run.KNOWN_MODELS`（三处）。
  - ⚠ `probe_weights.py` 是**开训练前**的权重兼容性/适配探测器（下载 release asset 并对齐 head shape），**不是**"已训好的 run-dir 能不能推理"的健康检查。两者别混。
  - 🚀 **已训好的 run-dir 要做 smoke**就用：`scripts/autodl/infer_testA.py --run-dir <R/run> --split testA --manifest <迷你manifest> --conf-threshold <c> --output-dir /tmp/smoke_<tag>`。它**只读** `run_evidence.json`、evidence 只写 `--output-dir`，且**强制校验 sha256(checkpoint)==run_evidence.sha256**（第 139–141 行）→ 顺带验权重完整性。迷你 manifest 放 /tmp（勿放项目树，会触发 `assert_tracked_worktree_clean`）。5 张图 × 11 树 ≈ 6 min。
  - **2026-09-16 23:55 实测**：这条 smoke 路线零污染（跑完 `find $R -maxdepth 2 -newermt "-20 minutes"` 必须为空）。
- `infer_testA.py` evidence 的 `selected_best_checkpoint` 须含 `path` + `sha256`。evidence **不记耗时**（无 started_at 类字段），要估时间只能实测。
- ⏱ **别把训练时长当推理时长**：36ep ≈ 2h、54ep ≈ 3.75h 是**训练**吞吐。**推理**快两个数量级——2026-09-16 实测 54ep 单次全量 900 张 = **42 s**（≈21 img/s，含 ~11 s CUDA/加载固定开销）→ **12 次推理约 8–12 min**，远短于 runbook 注释的"约 25 min"。据此 testB 图 **9/17 16:00 前到位即够**（推理 + 回传 + 本地建包 ≈ 30 min）。
- **🚫 "多模型一致性当伪真值"失效**：9 树 ≥7/9 伪真值下 incumbent 命中率 92.7% vs 真实率 80.3%（召回 1.154 > 1）。**任何"用共识当 GT"的定价/筛选都不要用**，可读的只有相对印证率。

## 4. 盈亏线经济学（θ = F1/2）

testA：θ = 0.3675，P+G = 5819.8，u = 2/(P+G) = 3.4365e-4。
删/加 n 条、真线率 r → `ΔF1 = u·n·(θ−r)`（删）或 `u·n·(r−θ)`（加）。单条：删 +0.0126pp｜加 +0.0217pp｜修复 +0.0344pp。
→ 删线要求被删者真线率 **< 36.75%**，加线要求 **> 36.75%**。

## 5. ★ 加线定价：看"多少棵树同意"，不看"哪个模型产的"

拟合 `r = 0.2816 + 0.3083 × corroboration`（两点锚定：probe_g4 0.3800 / soupB 0.3361）。
脚本 `scripts/calibrate_corroboration_20260915.py`；快速版 `scripts/price_uni_s101_54ep_20260916.py`（只对含 novel 线的 rel 建伪真值，12 s）。

| 候选 | n | corr | r_est | ΔF1 est | 处置 |
|---|---:|---:|---:|---:|---|
| gate6 novel | 52 | 0.519 | 0.442 | +0.132pp | 注 2（下界） |
| uni_swa4_g6 | 69 | 0.377 | 0.398 | +0.071pp | 注 3 |
| swa4 单发 | 27 | 0.222 | 0.350 | −0.016pp | 被 uni 支配 |
| uni_s101_54ep | 96 | 0.167 | 0.333 | −0.115pp | 弃 |
| conf55（删 69） | 69 | 0.333 | 0.384 | −0.039pp | 弃 |
| uni_occlude | 119 | 0.227 | 0.352 | −0.067pp | 弃 |
| soupB | 215 | 0.177 | 0.336 | −0.235pp | 锚点 |

- ⚠ 拟合对高 corr 系统性**低估**（incumbent 0.930 → 估 0.568 vs 真实 0.803）→ 高 corr 的估计是下界；0.18–0.33 区间更可信。
- **单支撑 union 增量线 corr 恒在 0.17–0.23 → 一律为负。**

## 6. 🔴 新模型（独立训练）已被证据关死

- **dropped 噪声带 [104,175] 与种子/排期/增强无关**（36ep 四对照 175/139/126/104；54ep s101 = 145；occlude 135）→ 模型间不可约分歧，与质量无关。
- **"训到 54ep 同排期能让 novel 线具备 54ep 质量"已证伪**（s101 54ep dropped 145 > 同种子 36ep 的 139）→ **不再投任何新训练的模型**。
- 独立新模型单交 ≈ −2pp；只有 union（dropped=0）形态不亏，但增量线仍负期望。残留唯一用法＝当第 8 棵支撑树，会动已标定配方且无法打分 → 默认不用。
- **空图数也是种子/模型依赖**（s101 54ep 21，s42 54ep 9）→ "空图是排期效应"作废。
- 判据：新实验先算台账（2 min CPU，不用 GPU）——dropped ∈ [104,175] = 零效应；**污染 val 不能当筛子**。

## 7. 已闭环（勿重推）

- **分辨率 / cut_height**：960×384 +0.028pp、1366×540 **−1.542pp**、cut400 **−0.772pp**。模型自己找地平线（三种裁剪 top 中位数都 564）。
- **平滑 / 稀疏化 ≈ 0**：sub4 +0.013、ma5 +0.073。
- **span80 短残线过滤**：只对共识平均产生的残线桩有效（54 条、伪线率 70.4% → +0.127pp）；**对独立模型自己的短线是负的（OOF −0.106pp）**。
- **共识门槛**：控制变量是**同意比例**不是票数。≥20% → −0.145pp；≥40% → +0.035pp；**>~70% 一条线都不加**（变"地板改名重交"）。
- **共识删 FP 死**：incumbent 2664 条中 86.6% 被全部 4 模型支持 → FP 是系统性偏差。
- 已封死：全局仿射修正、降 conf（低置信带真线率 23%）、去重、flip-TTA（−0.49）、端点上外推（−2.43）、OS 过采样（−1.07）、CLRerNet（−0.45）、跨种子 soup（−0.38）、VAT、ConvNeXt-T、雾雨增强、segmask binary（MISS）、occlusion 增强（NULL）。
- 测试集无 score sidecar → 改阈值必须重跑推理。
- `filter_short_lanes.py --base` 必须与 `--src` rel 路径逐层对齐（已加守卫），跑完必看 `base_kept` ≈ base 线数。

## 8. trim（唯一正迁移轴）

- 机理：CLRNet 把近端一律推到 y=719，GT 远端起始时 bottom 只到 396–615 → 过冲稀释 30px 描边 IoU。规则 `bottom = gt_bottom_for_top(top) + margin`。
- **全局 margin 曲线（OOF）**：0 → 0.797224｜20 → 0.811045｜**40 → 0.819699**｜60 → 0.817685（40 比 0 高 **+2.25pp**）。
- **testA 型几何上 margin 是空操作**（起点 ≈564 → gt_bottom(564)+40 = 752 > 719）→ 只有 margin 0 有效（+0.069pp）。
- **⭐ 不对称（最大杠杆）**：margin 40 下界 −0.069pp / 上界 +2.25pp。
- **判别量 = `f<450`（起点 y<450 的线占比），B 榜阈值 3%**（盈亏平衡 p×2.25=(1−p)×0.069）。**`f<400` 与 top p10 已证伪**。
- **逐 clip 选 margin 做不到**（70/71 clip 个体最优 ≠ 全局最优，oracle 上界仅 +1.90pp 且无 GT 特征可预测）。

## 9. B 榜作战（`docs/runbook_testB.md` / `docs/action_testB_20260916.md`）

- **🔴 唯一阻塞项：testB 图像未上传实例**。`JPEGImages` 恒 **97 项**（71 训练 + 9 testA + 17 `_hflip`）。需用户从赛事平台下载 10 clip / 1000 张 → `scp` → 解压进 JPEGImages（期望 107）。**未到位前不建 manifest、不推理。**（9/16 09:00 第 7 次确认仍是 97）
- **执行 = 两条命令**：实例 `scripts/autodl/run_testB_infer.sh`（11 次推理 ≈25 min → 打印 `f<450` → 打 tgz）；本地 `bash scripts/build_testB_candidates.sh <tgz>`（自动定 margin → 建 5 注 → 逐个官方预检）。
- **六注终版（9/16 04:00，按生产支撑树实测重锚）**：地板 / **cons 9 树 56%（+24 线，+0.036pp ⭐）** / uni(swa4, cons56%) / cons 9 树 44%（+42 线） / cons 9 树 67%（+5 线） / 自适应。
- **阶梯分数 44/56/67%，按可用树数四舍五入**（m=9→4/5/6；m=8→4/4/5；m=7→3/4/5；m=6→3/3/4）。加线 = 0 的注别交（已内置告警）。
- **共识加线总共就值 +0.03~0.04pp**（A 榜实测 probe_g4 40% 档 94 线 = +0.035pp）。旧的"cons_gate6 +0.132pp"是错的——它依赖的本地支撑树（t05/seed43/44/clrernet_15ep）**在实例上无 run-dir**。**定价前先确认候选依赖的每棵树在实例上都有 run-dir。**
- **实例可用独立支撑树 = 9 棵**：s42/s101/s202/s303 36ep、clrernet36、cut400、hires、occlude_36ep、s101_54ep。swa*/soup* 是 base 派生，**不算独立票**。本地 `clrernet_36ep` 残缺（158/900），别当基线。
- **union 优于单发**：A 侧原样保留（继承位移收益），B 侧叠增量 → dropped=0。
- **SWA 依据**：54ep s42 val 峰值 iter 24863 → 0.89565（final 0.89429），incumbent 用的就是 model_best；沿同轨迹平均 → 同 basin。
- **solution.zip 已冻结 v2**：`/hy-tmp/solution_freeze_20260916_v2.tgz`，**2,761,901,698 bytes**，sha256 `869251a8acb1a04398ef5318b701301673bfb5015d3e78e1a5e054535dde613d`，HEAD `8bf26d0`，权重 11 个。v1（`b97cffd2…`）已作废。**出包代码一变就要重冻。**
- bundle 必须 `tar czf x.tgz .`（无外层目录）。`build_testB_manifest.py` 严禁 glob 全量 JPEGImages（已修 = 补集 + `--expect-clips 10` 校验，≠10 个 clip 会拒绝运行，那是设计行为）。
- 所有新候选都无法离线打分 → 排序依据是机制强度 + 相对印证率。
- ⚠️ `_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/` 与 `/tmp/_dryrun_quarantine_20260916` 内容是 testA，**禁止提交**。跑完演练必查 `outputs/` 下 `ls | grep -i testB` 为空。

## 10. 端到端演练（防临门一脚才炸）

- `build_testB_candidates.sh` 在 9/16 前从未真跑过。演练抓到两类静默失败：① 支撑树缺失 → 共识输出 `files=0`，后续 filter→pack 全失败**却不中断**（白扔 3 发）；② 门槛用 ceil → 少一棵树时 4/6=67% → 共识一条线不加。现均已加守卫。
- **演练 = 一键**：`scripts/make_dryrun_sandbox.py [DIR]` 自动克隆 `build_testB_candidates.sh` 并把 OUT/zip/manifest/**list/supports.json 全部成对重定向到 `/tmp/_DRYRUN_OUT_*`，**0 处泄漏才返回 0**；然后 `build_dryrun_bundle_20260916.py` 拼假 bundle → `bash /tmp/_DRYRUN_candidates_<date>.sh <bundle>` → 跑完必双查 `data/processed/manifest_testB.jsonl` 与 `outputs/submit_testB_*.zip` 均不存在。
- 🔴 **演练 harness 自身的 bug（9/16 23:20 抓到并修）**：`build_dryrun_bundle_20260916.py` 原用 `tar czf x.tgz -C /tmp <dirname>` → 归档带外层目录 → 解压后树沉一级 → 报 "base tree missing" + "supports missing" 的**假失败**，会掩盖真实回归。已改 `-C <OUT> .`（与真实 bundle 同构，见 §9）。
- **9/16 23:20 全链路演练通过**（10 min）：5 注 PRECHECK **全 OK**；`f<450=0` → M=0；7 树下 k=3/4/5 → 加 **45 / 14 / 0** 条，**逐位复现 §2.1 阶梯表**（44/14/0）→ 阶梯表与脚本实现互相印证。
- ⚠️ **演练暴露的真实风险：支撑树掉到 7 棵时第 5 注必废**。K6=round(0.67m)：m=9→6/9=67%（加 5 条）、m=8→5/8=62%、**m=7→5/7=71% > 70% 死区 → 加 0 条**。cut400/s101_54ep 任一推理失败即触发。脚本已告警，但**看到 "adds NO lanes" 就别交，改用第 6 注或只交前 4 注**。
