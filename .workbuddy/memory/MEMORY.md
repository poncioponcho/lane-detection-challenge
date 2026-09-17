# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 压缩索引；**完整事实源是 `2026-*.md` 日志**，冲突时以日志 + `docs/` 为准。跨文档引用用章节锚点，禁用行号。
> 最近压缩：2026-09-17 08:50（赛程/额度重大修正 + 大幅度压紧，未删结论）。

## 0. 铁律
- **仲裁源**：`docs/DECISIONS.md` + `configs/default.yaml` + `docs/official_rules.md`。
- 🔴 **赛程被官方顺延一天**（9/17 08:20 实读线上页，两次抓取一致）：A 榜 8/19–9/14；**B 榜 = 9/17 00:00 – 9/18 17:00（北京时）**；报名截至 9/18。旧记 9/16–9/17 17:00 **作废**（9/8 快照过时；侧证：9/16 全天 testB 未发布）。已改 `docs/official_rules.md` §4。
- 🔴 **额度 = 每日 3 次、不累积 → 本窗口共 6 发**（9/17 三 + 9/18 三）。**交前必须现场核对平台剩余额度，以平台为准。**
- **成绩只认冻结 Oracle 全局单次调用**（`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`，py3.12+numpy2.1.3/scipy1.15.3/cv2 4.12）；per-clip 禁止平均。`/private/tmp` 会被系统清理。
- **提交**：`prepare_submit.py` → verify → 官方 `check_submission.py`；**交包 + 备注给用户手动交，绝不自动交、绝不反复开浏览器**（沙箱 Chromium 恒失败）。
- **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧全禁。
- **候选放行三门禁**：全局 ΔF1 ≥ +1pp；paired video bootstrap CI 下界 > 0；≥5/8 video 正向。
- **清理目录一律 `mv` 到 /tmp，不要 `rm -rf`**（>50 文件触发守卫 exit 137）。
- incumbent = **54ep model_best(conf0.50) + trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`，sha256 `a7dcef5b…72af`）。基线一律以「已交 zip 本身」为源；`testA_54ep_raw` 是 2688 线不可当基线。

## 1. 评估协议与锚点
- **切分铁律**：`v1_seed42` 是 clip-level，val 100% video 级泄漏 → v1 val 上 <2pp 的消融不可判定。
- **权威尺子**：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1=0.777628**（逐位复现），video-cluster 95% CI 半宽 **7.4pp**。GT 在本地 `data/gt_train/anno_txt` → OOF 筛选完全本地可跑。
- **禁用的假数**：36ep LVO 导出（未过阈值原始树）；full71 在 800 图 val 的 0.88~0.89（val ⊂ 训练）。
- A 榜已关闭（我方 0.73574，榜首 0.82389，差 8pp）。**OOF 是 train 型几何**，结论只有 testB 也是 train 型时才迁移。

## 2. 环境坑（本机 + 实例）
- **BSD grep 对中文 + `\|` 静默零命中** → 中文检索必须用内置 Grep 工具。
- 🔴 **AppleDouble `._*` vs Python glob**：实例 80/80 clip 目录都带 100 个 `._<frame>.jpg`（Mac zip 产物）。**`ls *.jpg` 不看点文件**（历次帧数检查都报"正常 100"），而 **`Path.glob("*.jpg")` 会匹配** → `derive_list` 把 1000 帧算成 2000。叠加 `src/data/manifest.py:18` 的 `VERIFIED_100_FRAME_SPLITS={train,testA}`（不含 testB）→ 完全隐形。已修（过滤 `._` + 「所有 clip 帧数一致」守卫，故意不硬断言 100）。**教训：数文件一律用 Python，别用 ls。**
- ⚠ 后果是**整包无效**而非扣分：§3.1「缺失/**多余**/路径错误 → 整包无效」。
- **git bundle 的 ref 名默认是 `HEAD`**（不是分支名）→ `git fetch <bundle> HEAD:refs/heads/tmp && git merge --ff-only refs/heads/tmp`。**实例没有 `origin`** → 只能走 bundle；判同步看 `rev-parse HEAD`。
- **`pkill -f`/`pgrep -f` 会匹配自身 shell** → 自杀（exit 137）。用 `ps -eo pid,comm`。本机**没有 `timeout`**。
- **长命令被沙箱 SIGTERM**：>2min 用 run_in_background；远端 `setsid nohup ... & disown`。`scp -r` 目录易被杀 → 远端先 `tar czf` 再传单文件。
- `software-*` subagent 调 TaskList 必崩；长文档定向修订主 Agent 直接 Edit 更稳。
- git push：`git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push`。**未跟踪文件不被"工作树干净"守卫拦住，也不在 bundle 里。**

## 3. 实例（恒源云）与代码同步
- 实例 `i2b0715374400501416`（3090-24G 包天）。**SSH 串随重建而变**；实测 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`（BatchMode）自 9/16 09:00 起连续 10 次可用（hostname 未变）。**租约须覆盖到 9/18 17:00 之后**（实例有 2.7G 冻结包）。
- **代码同步 = git bundle**（§2）。实例 HEAD 落后本地若干 docs/memory 提交，但 `scripts/autodl/` 关键文件是 scp 部署（实例侧为未提交改动 + 未跟踪 watcher）→ **不需要全量同步**。
- **发车三守卫**（HEAD 一变就重跑，全在 `scripts/autodl/`）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + `validate_run.KNOWN_MODELS`（三处）。⚠ `probe_weights.py` 是**开训练前**的权重兼容探测器，不是"已训好 run-dir 能否推理"的体检。
- 🚀 **已训好 run-dir 做 smoke**：`scripts/autodl/infer_testA.py --run-dir <R/run> --split testA --manifest <迷你manifest> --conf-threshold <c> --output-dir /tmp/smoke_<tag>`（**强制校验 sha256(checkpoint)==run_evidence.sha256**）。迷你 manifest 放 /tmp（放项目树触发 `assert_tracked_worktree_clean`）。evidence 的 `selected_best_checkpoint` 须含 `path`+`sha256`；evidence **不记耗时**。
- ⏱ **别把训练时长当推理时长**：54ep≈3.75h 是**训练**；**推理** 900 张 = **42 s**；testB 全链（11 树推理+回传+建包）≈ **20–30 min**。
- **🚫 "多模型一致性当伪真值"失效**：9 树 ≥7/9 伪真值下 incumbent 命中率 92.7% vs 真实 80.3%（召回 1.154>1）。**任何"用共识当 GT"的定价/筛选都不要用。**

## 4. 盈亏线经济学（θ = F1/2）
testA：θ=0.3675，u=2/(P+G)=3.4365e-4。`ΔF1 = u·n·(θ−r)`（删）/ `u·n·(r−θ)`（加）。单条：删 +0.0126pp｜加 +0.0217pp｜修复 +0.0344pp。→ 删线要求真线率 **<36.75%**，加线 **>36.75%**。

## 5. 印证率拟合定价 —— ⛔ 已被 §12 取代，勿单独引用
`r = 0.2816 + 0.3083 × corroboration`（`scripts/calibrate_corroboration_20260915.py`）。**系统性低估 0.09–0.18**（证据：incumbent 高 corr 端估 0.568 vs 真 0.803；[0.50,0.55) 估 0.384 vs Oracle 反推 0.557；§12 实测 0.44–0.51）→ **旧表所有负值不可信**，改用 §12 直接测法。

## 6. ⛔ 新模型（独立训练）已关死
**dropped 噪声带 [104,175] 与种子/排期/增强无关**（36ep 四对照 175/139/126/104；s101_54ep 145；occlude 135）→ 模型间不可约分歧。「训到 54ep 能让 novel 线具备 54ep 质量」**已证伪**（s101 54ep 145 > 同种子 36ep 139）→ **不再投任何新训练模型**。独立新模型单交 ≈ −2pp；只有 union（dropped=0）形态不亏。**空图数也种子依赖**（s101 21 vs s42 9）。判据：新实验先算台账（2 min CPU）——dropped ∈ [104,175] = 零效应；**污染 val 不能当筛子**。

## 7. 已闭环（勿重推）
- **分辨率/cut_height**：960×384 +0.028pp、1366×540 −1.542pp、cut400 −0.772pp。模型自己找地平线（三种裁剪 top 中位数都是 564）。**平滑/稀疏化 ≈ 0**（sub4 +0.013、ma5 +0.073）。
- **span80** 只对共识平均的残线桩有效（54 条、伪线率 70.4% → +0.127pp）；对独立模型自己的短线是负的（OOF −0.106pp）。
- **共识门槛**：控制变量是**同意比例**不是票数（≥20% → −0.145pp；≥40% → +0.035pp；**>~70% 一条线不加**）。**共识删 FP 死**（incumbent 2664 条中 86.6% 被全部 4 模型支持 → FP 是系统性偏差）。
- **不需要 GT 即可证伪的两项已排除（9/17，几秒 CPU）**：① 单树内部重复线（30px 描边 IoU ≥0.5）= **0 对**（900 文件全扫）→ intra-tree dedup 无东西可删；② 非法几何：坐标全在框内、无 NaN、无退化线。教训：检查脚本硬编码 1280 宽会**自己制造假警报**（真实画布 **1366×720**）。
- 已封死：全局仿射修正、去重、flip-TTA（−0.49）、端点上外推（−2.43）、OS 过采样（−1.07）、CLRerNet（−0.45）、跨种子 soup（−0.38）、VAT、ConvNeXt-T、雾雨增强、segmask binary（MISS）、occlusion 增强（NULL）、**conf 阈值轴（§8）**。

## 8. ⛔ 置信阈值轴已量完并关闭（9/17）
产物 `outputs/lvo_conf_lowsweep_20260917/`（冻结 Oracle，8-fold LVO，7100 张 OOF，G=24435）。36ep 峰值 **conf=0.40 → 0.778516 vs 0.50 的 0.774350 = +0.417pp**，但三门禁 1 PASS/2 FAIL（Δ<+1pp；CI 下界<0）。
⛔ **独立模型复验证伪**：15ep clsweight3 同协议同 GT → 0.40 vs 0.50 = **+0.001pp** → **模型特有噪声**。**地板固定 conf=0.50，此轴关闭。**
轴形：0.35–0.50 极平高原；**悬崖在 0.25 以下** → 硬下限 0.25。边际真线率：0.15 带 r≈0.20｜[0.25,0.35) r≈0.35｜[0.40,0.50) r≈0.49。✅ 环境漂移实测 = 0 → 旧表可并表。零成本选项 `BASE_TAG=base54_c35`（**已无实测理由偏好**）。测试集无 score sidecar → 改阈值必须重跑推理。

## 9. trim（唯一正迁移轴）
- **原图实测 1366×720**。「近端被推到 y=719」= 底边最后一行。机理：CLRNet 把近端一律推到 y=719，GT 远端起始时 bottom 只到 396–615 → 过冲稀释 30px 描边 IoU。规则 `bottom = gt_bottom_for_top(top) + margin`。
- **全局 margin 曲线（OOF）**：0→0.797224｜20→0.811045｜**40→0.819699**｜60→0.817685（40 比 0 高 **+2.25pp**）。⚠ 该曲线在 **conf≈0.345** 测的（基线 P=22248），**不是 0.50** → 名义上行空间未经 conf=0.50 验证。
- **testA 型几何上 margin 是空操作**（起点 ≈564 → gt_bottom(564)+40=752>719）→ 只有 margin 0 有效（+0.069pp）。**⭐ 不对称 = 最大杠杆**：margin 40 下界 −0.069pp / 上界 +2.25pp。
- **判别量 = `f<450`（起点 y<450 的线占比），阈值 3%**（二元模型盈亏 2.98%；比例模型 1.65% → 死区 1.65%~2.98%）。✅ **不降门槛**：shot6 已免费覆盖死区（M=0 时 shot6 强制 trim40，取 max 会选中它）。`f<400` 与 top p10 已证伪。
- **逐 clip / 逐 top 桶选 margin 均堵死**：70/71 clip 个体最优 ≠ 全局最优（oracle 上界仅 +1.90pp 且无 GT 特征可预测）；按 top 桶需 8-fold OOF 预测树，本地与实例都缺 → 别再回头找。

## 10. B 榜作战（`docs/runbook_testB.md` / `docs/action_testB_20260916.md`）
- **窗口 9/17 00:00 – 9/18 17:00（§0），6 发取 max。**
- **执行 = 两条命令**：实例 `scripts/autodl/run_testB_infer.sh`（11 树推理 → 打印 `f<450` → 打 `/hy-tmp/testB_bundle.tgz`）；本地 `bash scripts/build_testB_candidates.sh <tgz>`（自动定 margin → 建 7 注 → 逐个官方预检）。
- **实例常驻到位触发器**：`scripts/autodl/watch_testB_and_run.sh`（setsid，锁 `/hy-tmp/testB_watch.lock`，日志 `/hy-tmp/testB_watch.log`）。每 60 s 数 clips（基线 80 = 71 train + 9 testA），>80 且 45 s 稳定 → 自动跑推理。**DEADLINE 已随窗口改为 `202609181700`**（旧值 202609171630 会白扔 25h）。**日志只在 start/TRIGGER/变化时打印，久无新行 ≠ 已死**；判存活用 `ps -eo pid,etime,args | grep "[w]atch_testB"`。

- **注单（文件名 = 唯一权威，勿用中间编号；§2.2 与 §2.4 曾编号错位，9/17 已修）**：

  | shot | 配方 | testA 实测加线 | 估 Δpp |
  |---|---|---:|---:|
  | `shot1_base54` | 54ep conf0.50 + trim(M) | 地板 | 基准 |
  | `shot2_uni_swa4_cons` | union(swa4, cons 56% 树) | swa4 增量 | 略低于 shot3 |
  | `shot3_consensus` | cons 9 树 k=5(56%) + span80 | +24 | +0.036 |
  | `shot4_cons_k5`(M=0) / `shot4_margin0`(M=40) | cons k=4(44%) / 地板 trim0 | +42 / 地板 | +0.027 / 几何对冲 |
  | `shot5_cons_k6` | cons k=6(67%) + span80 | +5 | +0.012（**最弱，7 选 6 先丢它**） |
  | `shot6_cons_k5_m40` | 几何对冲：cons 56% 树 + trim40 | +45 | 0 ~ **+2.25** |
  | `shot7_uni_cons_s101` | union(cons 56% 树, s101_54ep) → §12 | 约 +96 | **+0.24~0.47** |

  - ⚠ `shot4_cons_k5` 目录名里的 `k5` 是遗留，实际用 **K4(44%)** 构建（脚本 L222）。**判别以脚本行为为准。**
  - **加线 = 0 的注别交**（地板改名重交，脚本会告警）。**M=40 时 shot6 打印 not needed，该发换 shot4。**
- **6 发建议顺序**：shot1（保底）→ shot3（EV 最高）→ shot6（唯一多 pp 上界）→ shot7 → shot2 → shot4；**shot5 丢**。
- **⭐ shot6 的价值论证**：`f<450>3%` 是**单发期望最优**，但取 max 时期望是错的目标函数；它真正对冲的是**阈值模型可能错了**。实现用 `cons_k5_f80 + trim40`，不是裸 base。⚠ **已知代价**：真·testA 型下退化为 shot4 的复制（演练实测同为 2709 线），别误判脚本坏了。
- **阶梯分数 44/56/67% 按可用树数四舍五入**（m=9→4/5/6；m=8→4/4/5；m=7→3/4/5；m=6→3/3/4）。⚠ **支撑掉到 7 棵时 shot5 必废**（K6=5/7=71%>70% 死区 → 加 0 条）；cut400/s101_54ep 任一推理失败即触发。
- **实例可用独立支撑树 = 9 棵**：s42/s101/s202/s303 36ep、clrernet36、cut400、hires、occlude_36ep、s101_54ep。swa*/soup* 是 base 派生，**不算独立票**。本地 `clrernet_36ep` 残缺（158/900）。
- **共识加线总共就值 +0.03~0.04pp**（A 榜实测 94 线 = +0.035pp）。旧「cons_gate6 +0.132pp」是错的——依赖的本地支撑树**在实例上无 run-dir**。**定价前先确认候选依赖的每棵树在实例上都有 run-dir。**
- **union 优于单发**：A 侧原样保留（继承位移收益），B 侧叠增量 → dropped=0。**SWA 依据**：54ep s42 val 峰值 iter 24863 → 0.89565，incumbent 用 model_best；沿同轨迹平均 → 同 basin。
- **solution.zip 已冻结 v4**：`/hy-tmp/solution_freeze_20260916_v4.tgz`（2.7G）。✅ 9/17 回规则原文：§3.2 是「**B 榜发布前冻结**」+「赛后**前三名**上传字节一致包」→ 冻结时点已过，**规则里没有"代码一变就重冻"** → 9/17 改建包脚本不需重冻。
- bundle 必须 `tar czf x.tgz .`（无外层目录）。`build_testB_manifest.py` 严禁 glob 全量 JPEGImages（已修 = 补集 + `--expect-clips 10`）。
- 所有新候选都无法离线打分 → 排序依据是机制强度 + 相对印证率。
- ⚠️ `_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/`、`/tmp/_dryrun_quarantine_20260916`、`/tmp/_DRYRUN_OUT_*` 里的 zip 内容是 **testA，禁止提交**。跑完演练必查 `outputs/submit_testB_*.zip` 不存在。

## 11. 端到端演练
- 9/16 前建包脚本从未真跑过；演练抓到两类静默失败（**均已加守卫**）：① 支撑树缺失 → 共识输出 `files=0`，后续 filter→pack 全失败**却不中断**（白扔额度）；② 门槛用 ceil → 少一棵树时 4/6=67% → 一条线不加。
- **一键**：`scripts/make_dryrun_sandbox.py [DIR]`（OUT/zip/manifest/list/supports.json 成对重定向到 `/tmp/_DRYRUN_OUT_*`，0 泄漏才返回 0）→ `build_dryrun_bundle_20260916.py` 拼假 bundle → `bash /tmp/_DRYRUN_candidates_<date>.sh <bundle>` → 双查 `data/processed/manifest_testB.jsonl` 与 `outputs/submit_testB_*.zip` 均不存在。
- 🔴 **harness 自身 bug（已修）**：`build_dryrun_bundle_20260916.py` 原用 `tar czf x.tgz -C /tmp <dirname>` → 带外层目录 → 假失败。已改 `-C <OUT> .`。9/16 23:20 全链路演练通过（10 min，`f<450=0` → M=0，7 树下 k=3/4/5 → 加 **45/14/0** 条）。

## 12. ⭐ 单支撑 union 实测为**正**（9/17，推翻 §5 的"一律为负"）
产物 `outputs/lvo_union_probe_20260917/`；工具 `scripts/oracle_score_tree.py`。方法：基线树原样保留，只叠另一个模型的增量线，冻结 Oracle 打分 → **完全绕开 §5 的拟合**。

| 实验 | 基线 F1 | 支撑 | 新增 | union F1 | **ΔF1** | CI(pp) | 反推 r |
|---|---:|---|---:|---:|---:|---|---:|
| 等强对 | 0.774350（36ep@.50） | 15ep clsweight3@.50 | 1091 | 0.780068 | **+0.572** | [−0.225,+1.388] | **0.509** |
| 强+弱 | 0.778516（final@.40） | midpoint@.40 | 1165 | 0.780987 | **+0.247** | [−0.304,+0.858] | **0.440** |

- **两条都正**，且**支撑越弱 r 越低**（0.440 < 0.509）→ 代回 §5：`uni_occlude`(119) → **+0.30~0.58pp**；`uni_s101_54ep`(96) → **+0.24~0.47pp**。
- ⛔ 两条 CI 下界都 <0（不满足三门禁）；A 榜**实测**加 94 条共识线只值 +0.035pp，而 **union 在 A 榜从未实际提交过** → 同域真值只覆盖共识；OOF 是 train 型几何。
- 📌 **价值最高、唯一还没在同域验证过的杠杆。** §8.5 预先注册的触发条件是「出现下一个可提交窗口**或**额度 > 3」——**9/17 窗口修正后两条都满足** → 已实现为 **`shot7_uni_cons_s101`**（`--a`=cons 56% 树、`--b`=testB_s101_54ep；支撑选 54ep 而非 36ep，因为支撑越强 r 越高）。脚本带守卫：加线 >300（远超实测 96–119 区间）时告警并请人工裁决。
