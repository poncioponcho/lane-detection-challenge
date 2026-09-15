# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 只存跨会话必须延续的约定与铁律。过程细节看 `2026-*.md` 日志。
> 最近一次整理：2026-09-16 03:10（压缩重写 + 并入 54ep 独立种子实验结果）。

## 0. 铁律

- **仲裁源**：`docs/DECISIONS.md` + `configs/default.yaml`；跨文档引用用**章节锚点**，**禁用行号**。
- **成绩只认冻结 Oracle 全局单次调用**（`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`）；**per-clip 禁止平均**。本地诊断 metric 仅供训练期扫描。
- **提交纪律**：`prepare_submit.py`（manifest 枚举 + 1 位小数 canonicalize）→ verify → 官方 `check_submission.py`；**不自动交，交前用户确认**。incumbent 配方 = **54ep model_best(conf0.50) + trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`）。
- **🚫 提交一律"交包 + 备注给用户手动提交"，绝不反复开浏览器**：沙箱内 Chromium 恒 `sandbox initialization failed`。
- **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧全禁。
- **候选放行三门禁**：全局 ΔF1 ≥ +1pp；paired video bootstrap CI 下界 > 0；≥5/8 video 正向。
- **跨机器时间必须标时区**：实例跑 UTC，北京 = UTC+8。B 榜按北京时间口径（9/16 00:00–9/17 17:00）。
- **本工作区清理目录一律 `mv` 到 /tmp，不要 `rm -rf`**（>50 文件触发守卫 exit 137）。

## 1. 评估协议与锚点

- **切分铁律**：`v1_seed42` 是 clip-level，val 100% video 级泄漏 → **v1 val 上 <2pp 的消融不可判定**。
- **权威基线（可真打分的尺子）**：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1 = 0.777628**（多次逐位复现），video-cluster 95% CI 半宽 **7.4pp**。GT 已拉回本地 `data/gt_train/anno_txt` → **OOF 筛选完全本地可跑，不依赖实例**。
- **⚠ 两个禁用的假数**：36ep LVO 导出是未过阈值原始树（11 线/图）；full71 模型在 800 图 val 上的 0.88~0.89 是污染数（val ⊂ 训练）。
- A 榜已关闭。B 榜 6 发取 max。A 榜门槛：R3 0.81604 / 榜首 0.82389（我方 0.73574，差 8pp）。
- **OOF 是 train 型几何**，结论只有在 testB 也是 train 型时才迁移；须过「GT top≥530 的 12 clip 子集」复核。

## 2. 环境坑（本机 + 实例）

- 冻结 Oracle 在持久路径 `envs/lane-oracle-py312`（py3.12 + numpy2.1.3/scipy1.15.3/cv2 4.12）；`/private/tmp` 会被系统清理。
- **BSD grep 对中文 + `\|` 静默零命中** → 中文检索必须用内置 Grep 工具。
- **`pkill -f`/`pgrep -f` 会匹配到自身 shell** → 命令自杀（exit 137）。用 `ps -eo pid,comm`。
- 本机**没有 `timeout` 命令**（macOS）。
- **长命令会被沙箱 SIGTERM**：>2min 本地任务用 `run_in_background`；远端 `setsid nohup ... & disown`。`scp -r` 目录易被杀 → 远端先 `tar czf` 再传单文件。
- `software-*` subagent 调 TaskList 必崩；长文档定向修订主 Agent 直接 Edit 更稳。
- git push 走 `git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push`（本沙箱直连 GitHub 被拦 502）。
- **未跟踪文件不被"工作树干净"守卫拦住**，也不在 bundle 备份里。

## 3. 实例与代码同步（恒源云）

- **生产实例** `i2b0715374400501416`（3090-24G 包天）。**SSH 串随实例重建而变，每次开机从控制台重取，禁止沿用旧值**（2026-09-16 03:00 实测可用：`root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`，BatchMode）。
- **代码同步 = git bundle**：本地 `git bundle create /tmp/hl.bundle <实例实际HEAD>..HEAD` → scp → 实例 `mv` 原子替换 → `git fetch && git merge --ff-only`。实例 `origin` 指向陈旧 bundle，**勿据此判断同步状态**。
- **发车三守卫**（HEAD 一变就重跑）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + **`validate_run.KNOWN_MODELS`**（三处）。
- **`infer_testA.py` evidence 形状**：`selected_best_checkpoint` 须含 `path` + `sha256`，权重须与 `run_evidence.json` 同 run-dir。
- 吞吐（3090）：36ep ≈ 2h；54ep ≈ 3.75h。
- **🔴 testB 图像仍未上传实例**：`JPEGImages` 恒 97 项（71 训练 + 9 testA + 17 `_hflip`）。**这是 B 榜唯一阻塞项，需用户手动下载上传。**

## 4. 盈亏线经济学（F1 = 2TP/(P+G)，θ = F1/2）

testA ≈ θ 0.3675，P+G = 5819.8，u = 2/(P+G) = 3.4365e-4：

| 操作 | 条件 | 单条 ΔF1 |
|---|---|---|
| 删线 | 被删者真线率 < θ | +2θ/(P+G) = **+0.0126pp** |
| 加线 | 新增者真线率 > θ | +2(1−θ)/(P+G) = **+0.0217pp** |
| 修复（伪→真） | — | +2/(P+G) = **+0.0344pp**（是加线的 1.58×） |

> 通式：删/加 n 条、真线率 r → `ΔF1 = u·n·(θ−r)`（删）或 `u·n·(r−θ)`（加）。
> 本项目：删线要求被删者真线率 < **36.75%**，加线要求 > **36.75%**。

## 5. ★ 加线定价：看"多少棵树同意"，不看"哪个模型产的"

拟合式 `r = 0.2816 + 0.3083 × corroboration`（两点锚定 = A 榜实测反解的 probe_g4 0.3800 / soupB 0.3361）。
脚本 `scripts/calibrate_corroboration_20260915.py`；**新脚本 `scripts/price_uni_s101_54ep_20260916.py` 只对含 novel 线的 rel 建伪真值（90 图而非 900），12 s 出结果**。

| 候选 | n | corr | r_est | ΔF1 est | 处置 |
|---|---:|---:|---:|---:|---|
| gate6 novel | 52 | 0.519 | 0.442 | **+0.132pp** | ⭐ 注 2 |
| uni_swa4_g6 | 69 | 0.377 | 0.398 | +0.071pp | 注 3 |
| swa4 单发 | 27 | 0.222 | 0.350 | −0.016pp | 被 uni 支配 |
| **uni_s101_54ep** | 96 | **0.167** | 0.333 | **−0.115pp** | **弃** |
| conf55 删除 | 69 | 0.333 | 0.384 | −0.039pp | 弃 |
| uni_occlude | 119 | 0.227 | 0.352 | −0.067pp | 弃 |
| soupB | 215 | 0.177 | 0.336 | −0.235pp | 锚点 |

- ⚠ 拟合对高 corr 系统性**低估**（incumbent corr 0.930 → r_est 0.568 vs 真实精确率 0.803）→ gate6/uni_swa4_g6 的估计是**下界**；0.18–0.33 区间内的更可信。
- **单支撑 union 增量线 corr 恒在 0.17–0.23 → 一律为负。**

## 6. 🔴 新模型（独立训练）这条路已被证据关死

- **dropped 噪声带 [104,175] 与种子、排期、增强都无关**（实测 36ep 四对照 s42 175 / s101 139 / s202 126 / s303 104；54ep s101 = **145**；occlude 135）。这是模型间不可约分歧，与质量无关。
- **"训到 54ep 同排期能让 novel 线具备 54ep 质量"已证伪**（2026-09-16）：s101 54ep dropped 145 > 同种子 36ep 的 139。→ **不再投任何新训练的模型**。
- 推论：独立新模型单交 ≈ −2pp；只有 union（dropped=0）形态不亏，但其增量线仍是负期望。
- 残留唯一用法：当第 8 棵支撑树（同同意率下线更多）→ 会动已标定配方且无法打分，**默认不用**。
- **空图数也是种子/模型依赖**（s101 54ep 空图 21，s42 54ep 空图 9）→ 此前"空图是 36ep/54ep 排期效应"的说法作废。
- **🚫 "多模型一致性当伪真值"失效**：9 树 ≥7/9 构造的伪真值，incumbent 命中率 92.7% 而真实率 80.3% → 召回 1.154 > 1。**任何"用共识当 GT"的定价/筛选都不要用**；可读的只有相对印证率。

## 7. 其它已闭环（勿重推）

- **分辨率 / cut_height 轴关闭**：960×384（+0.028pp）、1366×540（**−1.542pp**）、**cut400（−0.772pp 实测）**。模型自己找地平线（三种裁剪 top 中位数都 564）。
- **平滑 / 稀疏化 0**：sub4 +0.013pp、ma5 +0.073pp。
- **短残线过滤 span80**：只对"共识平均产生的残线桩"有效（54 条、伪线率 70.4% → **+0.127pp**）；**对独立模型自己的短线是负的（OOF 实测 −0.106pp）**。
- 其它已封死：全局仿射修正、降 conf（低置信带真线率 23%）、去重、flip-TTA（−0.49pp）、端点上外推（−2.43pp）、OS 过采样（−1.07pp）、CLRerNet（−0.45pp）、跨种子 soup（−0.38pp）、VAT、ConvNeXt-T、雾雨增强、segmask binary、occlusion 增强（NULL）。
- **共识门槛**：控制变量是**同意比例**不是票数。≥20% → −0.145pp；≥40% → +0.035pp；≥57%（k=4/7）→ 估 +0.05~+0.15pp。**B 榜用 7 树 + k=4，不再降到 ≥2/10**。
- **共识删 FP 死**：incumbent 2664 条中 86.6% 被全部 4 模型支持 → FP 是系统性偏差。
- **`testA_54ep_raw` = 2688 线 ≠ 已交包 2664 线**，不可当基线；基线一律以「已交 zip 本身」为源。
- **测试集无 score sidecar** → 改阈值必须重跑推理。
- **`filter_short_lanes.py --base` 必须与 `--src` 的 rel 路径逐层对齐**（已加守卫），跑完必看 `base_kept` ≈ base 线数。

## 8. trim（唯一正迁移轴）

- 机理：CLRNet 把近端一律推到 y=719，GT 远端起始时 bottom 只到 396–615 → 过冲稀释 30px 描边 IoU。规则 `bottom = gt_bottom_for_top(top) + margin`。
- **全局 margin 曲线（OOF）**：0 → 0.797224｜20 → 0.811045｜**40 → 0.819699**｜60 → 0.817685。最优 40，比 margin 0 高 **+2.25pp**。
- **testA 型几何上 margin 是空操作**（起点 ≈564 → gt_bottom(564)+40 = 752 > 719）→ 只有 margin 0 有效（+0.069pp）。
- **⭐ 不对称（最大杠杆）**：margin 40 下界 −0.069pp（testA 型），上界 +2.25pp（train 型）。
- **判别量 = `f<450`（起点 y<450 的线占比），阈值 10%**：≈0 → 用 margin 0；>10% → 用 40。**`f<400` 与 top p10 已证伪**。
- **逐 clip 选 margin 做不到**（70/71 clip 个体最优 ≠ 全局最优，oracle 上界仅 +1.90pp 且无 GT 特征可预测）。
- testA 预测 top 549/564/624、span 155；train OOF top 244/434/564、span 210 → 几何不同。

## 9. B 榜作战（`docs/runbook_testB.md` / `docs/action_testB_20260916.md`）

- 发车前先算 testB 的 `f<450` 定 M∈{0,40}，**阈值 3%**（不是旧的 10%）：两条分支极不对称——
  train 型线 margin 40 比 0 好 **+2.25pp**，testA 型线 margin 0 只比 40 好 **+0.069pp**，
  盈亏平衡 `p×2.25=(1−p)×0.069` → **p=2.98%**。最坏只亏 0.07pp，最好赚 2.25pp。
- **🔴 六注终版（9/16 04:00，阶梯已按生产支撑树实测重锚）**：
  地板 / **cons 9 树 56%**（+24 线，+0.036pp ⭐）/ uni(swa4, cons56%) / cons 9 树 44%（+42 线）/ cons 9 树 67%（+5 线）/ 自适应。
- **🔴 旧的"cons_gate6 +0.132pp"是错的**：它的 52 条增量线用本地支撑树（t05/seed43/44/clrernet_15ep）建的，
  **这些 run-dir 在实例上不存在** → testB 上复现不出来。**定价前先确认候选依赖的每棵树在实例上都有 run-dir。**
  本地 testA 预测树 ≠ 实例可复现 run-dir，是两个不同的集合。
- **生产支撑树实测阶梯**（9 棵、base=54ep raw、span80 后）：44%→42 线｜**56%→24 线 ⭐**｜67%→5 线｜
  （7 棵时）43%→44｜57%→14｜**71%/86%→0/0**。
  → **同意率 >~70% 共识一条线都不加**（会变成"地板改名重交"）；**同同意率下树越多线越多** → 支撑树 7→9 棵（加 occlude_36ep + s101_54ep）。
- 阶梯分数 44/56/67%，按可用树数四舍五入（m=9→4/5/6；m=8→4/4/5；m=7→3/4/5；m=6→3/3/4）。
- **共识加线这个杠杆总共就值 +0.03~0.04pp**（交叉验证：A 榜实测 probe_g4 40% 档 94 线 = +0.035pp）。
- **9/16 执行 = 两条命令**：实例 `scripts/autodl/run_testB_infer.sh`；本地 `scripts/build_testB_candidates.sh <tgz>`。
- **bundle 必须是 `tar czf x.tgz .`（无外层目录）**，否则本地脚本找不到 `testB_*`。bundle 含 `manifest_testB.jsonl` + list。
- **🔴 `build_testB_manifest.py` 严禁 glob 全量 JPEGImages**（已修 = 补集 + `--expect-clips 10` 校验）。
- **实例上真实可用的独立支撑树原来只有 7 棵**：s42/s101/s202/s303 36ep、clrernet36、cut400、hires。
  **9/16 扩到 9 棵**（+ `all71_seed42_clrnet_r50_occlude_36ep` + `all71_seed101_clrnet_r50_54ep`，两者都有 run-dir）。
  swa*/soup* 是 base 派生，**不能算独立票**。本地 `clrernet_36ep` 只有 158/900（残缺），别当基线
  （9/16 已重跑 `testA_clrernet36_c50` 900/900 补齐）。
- **⭐ union 优于单发**：A 侧原样保留（继承位移收益），B 侧叠增量 → dropped=0。
- **⭐ SWA 依据**：54ep s42 val 峰值 iter 24863 → 0.89565（final 0.89429），incumbent 用的就是 model_best(24863)；沿同轨迹平均 → 同 basin。
- **solution.zip 已冻结**：`/hy-tmp/solution_freeze_20260916.tgz`，2,495,124,936 bytes，sha256 `b97cffd20fc765de589f617d43a9611359dd9032cb6bacea1e4dfd2042165bbf`。
- **所有新候选都无法离线打分**（testA 无 GT、A 榜已关）→ 排序依据是机制强度 + 相对印证率，不是实测。
- ⚠️ 本地 `_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/` 与 `/tmp/_dryrun_quarantine_20260916` 内容是 testA，**禁止提交**。

## 10. 实验台账

| 分支 | 变量 | 状态 |
|---|---|---|
| `exp/segmask-binary` | `seg_mask_mode=binary_union` | **MISS** |
| `exp/occlusion-aug` | `CoarseDropout` p=0.5 | **NULL 零效应** |
| `all71_seed101_clrnet_r50_54ep` | 同排期独立种子 54ep | **MISS**（dropped 145 ∈ [104,175]；污染 val F1 0.89280）→ 见 §6 |
| 旁路 | `EXPERIMENT_OVERRIDE_KEYS` 已含 `seg_mask_mode` → 单变量配方实验走 `--experiment-override`。`candidate_topk` 不在白名单。 |

**判据先写死的规矩**：新实验先算台账（2 分钟 CPU，不用 GPU）——dropped 落在 [104,175] = 零效应；形状健康才发 1 发（≥+0.30pp 为有戏）。**污染 val 不能当筛子。**

## 10. 端到端演练（防"临门一脚才发现炸"）

- `build_testB_candidates.sh` 在 9/16 之前**从未真跑过**。演练抓到两类静默失败：
  ① 支撑树缺失 → 共识构建输出 `files=0`，后续 filter→pack 全失败**却不中断**，白扔 3 发；
  ② 门槛取整用 ceil → 少一棵树时 4/6=67%（而非 57%）→ 共识**一条线不加**。
  现已全部加守卫（动态支撑树、按同意率重算、空树跳过、加线=0 响亮告警）。
- **演练方法**：`scripts/build_dryrun_bundle_20260916.py` 用 testA 树拼一个假 testB bundle
  → 复制一份 `build_testB_candidates.sh` 到 /tmp 并把 OUT / zip / manifest 全部重定向到
  `/tmp/_DRYRUN_OUT_*` → 跑。**这样 outputs/ 里不会留下任何名字像 testB 的 testA 产物。**
- 跑完必做：`outputs/` 下 `ls | grep -i testB` 应为空；伪造的 `data/processed/manifest_testB.jsonl` 必须删。
