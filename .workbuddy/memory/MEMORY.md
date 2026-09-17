# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 压缩索引；**完整事实源 = `2026-*.md` 日志 + `docs/`**，冲突以它们为准。引用用章节锚点，禁用行号。最近压缩 2026-09-17 11:50。

## 0. 铁律
- 仲裁源：`docs/DECISIONS.md` + `configs/default.yaml` + `docs/official_rules.md`。
- 🔴 **B 榜 = 9/17 00:00 – 9/18 17:00 北京时**（9/17 08:20 实读线上页；旧记 9/16–9/17 作废）。**每日 3 次不累积 → 共 6 发**；**交前必须核对平台剩余额度**。
- 成绩只认**冻结 Oracle 全局单次调用**：`~/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python`（py3.12+numpy2.1.3/scipy1.15.3/cv2 4.12）；per-clip 禁平均。`/private/tmp` 会被清。
- 提交：`prepare_submit.py` → verify → 官方 `check_submission.py`；**交包 + 备注给用户手动交，绝不自动交、不反复开浏览器**（沙箱 Chromium 恒失败）。
- 测试集禁训（伪标签/域适配/BN 适配/跨帧全禁）。
- 放行三门禁：ΔF1 ≥ +1pp；paired video bootstrap CI 下界 >0；≥5/8 video 正向。
- 清理一律 `mv` 到 /tmp（>50 文件触发守卫 exit 137）。
- incumbent = **54ep model_best(conf0.50)+trim0 = 0.73574**（`submit_testA_54ep_trim0.zip`，sha `a7dcef5b…`）。基线只认「已交 zip 本身」（`testA_54ep_raw` 2688 线不可当基线）。

## 1. 评估协议
- `v1_seed42` 是 clip-level，val 100% video 级泄漏 → v1 val 上 <2pp 的消融不可判定。
- 权威尺子：8-fold LVO 拼 7100 张 OOF → CLRNet-R50 15ep **F1=0.777628**（逐位复现），video-cluster 95% CI 半宽 **7.4pp**。GT 在本地 `data/gt_train/anno_txt` → OOF 筛选完全本地可跑。
- 禁用假数：36ep LVO 原始树导出（未过阈值）；full71 在 800 图 val 的 0.88~0.89（val ⊂ 训练）。
- A 榜已关（我 0.73574，榜首 0.82389）。⚠ **「OOF/val 结论可迁移到 testB」已被 9/17 实测否决**（§4：margin 符号相反）→ **只有域内实测（B 榜真实分数）算数，其余一律当假设。**

## 2. 环境坑（本机 + 实例）
- 中文检索用内置 Grep（BSD grep 对中文 + `\|` 静默零命中）。**数文件一律用 Python**（`ls` 不看点文件）。
- 🔴 **AppleDouble `._*`**：实例 clip 带 100 个 `._<frame>.jpg`，`Path.glob("*.jpg")` 会匹配 → 帧数算成 2000；叠加 `manifest.py:18` 的 `VERIFIED_100_FRAME_SPLITS` → 完全隐形。已修。真实画布 **1366×720**（硬编码 1280 会造假警报）。
- git bundle ref 默认 `HEAD` → `git fetch <bundle> HEAD:refs/heads/tmp && git merge --ff-only refs/heads/tmp`；实例**无 origin**。
- **`pkill -f`/`pgrep -f` 会匹配自身 shell 自杀（exit 137）** → 用 `ps -eo pid,comm`。本机无 `timeout`。
- >2min 用 run_in_background；远端 `setsid nohup … & disown`。`scp -r` 目录易被杀 → 先 `tar czf`；大文件 `split -b 30m`（~400KB/s）。
- heredoc 里 `#` 注释是数据 → 注释放 heredoc 外。
- push：`git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push`。

## 3. 实例（恒源云）
- `i2b0715374400501416`（3090-24G 包天）。`root@i-1.gpushare.com:59725` + `~/.ssh/lane_id` 自 9/16 09:00 连续 11+ 次可用。**租约须覆盖到 9/18 17:00 之后**（实例有 2.7G 冻结包）。
- 发车三守卫：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型登记 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + `validate_run.KNOWN_MODELS`（三处）。
- ⏱ 训练 54ep≈3.75h；推理 900 张 42s；testB 全链（13 树推理+回传+建包）20–30min。
- 🚫 **"多模型一致性当伪真值"失效**：9 树 ≥7/9 伪真值下 incumbent 命中 92.7% vs 真实 80.3% → 别用共识当 GT。
- 🔴 **干净工作树守卫会静默废整轮**：`run_training.py:186` `assert_tracked_worktree_clean`；scp 的 `M` 文件 → 13 次推理全被拒而脚本仍 `exit 0` → 已跟踪文件必须「本地提交 → bundle → 实例 merge」；批量脚本逐产物计数并显式非 0 退出。
- 🔴 **坐标越界：官方是 clamp 不是拒绝**（`score.py:65-67`；`check_submission.py` 不校验坐标范围，只查 UTF-8/偶数/有限/去重≥2点/≤64线/≤2048点/文件集合精确匹配）。testB 13 棵树均有 3–21 条线落 `x∈(1365,1366]`（最多 1.0px）→ 已改 1.0px 容差+clamp（`d31eb99`）。A 榜 x 最大 1362.1 故从未暴露。
- ⚠ **`pack()` 把 prepare_submit 输出丢进 `/dev/null`** → 建包失败静默，必须查退出码。

## 4. trim —— 🔴 9/17 B 榜实测：margin 40 在 testB 上**有害**，本轴按域内数据重写
- 🔴🔴 **域内实测（唯一可信的依据）**：同一批 1910 条线、只差 margin → **margin40 = 0.65672 vs margin0 = 0.67527，margin40 低 1.855pp**。离线 OOF 预测的 **+2.25pp 符号是反的**。
- ⛔ **`f<450` 判据作废**：它建立在「train 型 +2.25pp / testA 型 −0.069pp」这个二元模型上（据此推出 3% 阈值）。testB 线起点虽高（`f<450=0.8492`、top 中位数 **424**），margin40 却**主动伤害 −1.86pp**——落在模型没预料到的第三区间。**错的是二分法本身，不是阈值调错。此量不得再用于决定 margin。**
- 🔧 **规则语义纠正（关键）**：`apply_bottom_trim` **不是"截短"**，而是把近端**改写为 `拟合GT底部(top) + margin`** → 故 **margin0 也不是空操作**。实测近端 y 中位数：原始 **719.0** → margin0 **656.5** → margin40 **691.1**；分数序与长度序同向：**越短越好**（654 > 656.5 > 691）。→ **B 榜一律用 margin 0。**
- OOF margin 曲线（**已证明不可迁移**，仅存档）：0→0.797224｜20→0.811045｜40→0.819699｜60→0.817685。
- 逐 clip / 逐 top 桶选 margin 均已堵死（70/71 clip 个体最优 ≠ 全局最优；按 top 桶需 8-fold OOF 预测树，本地与实例都缺）。

## 5. 盈亏线（θ = F1/2）
testA：θ=0.3675，u=2/(P+G)=3.4365e-4。`ΔF1 = u·n·(θ−r)`（删）/ `u·n·(r−θ)`（加）。单条：删 +0.0126pp｜加 +0.0217pp｜修复 +0.0344pp。→ 删线要求真线率 <36.75%，加线 >36.75%。

## 6. ⛔ 已关闭 / 已封死的轴（勿重推）
- **独立新模型**：dropped 噪声带 [104,175] 与种子/排期/增强无关（四对照 175/139/126/104；s101_54ep 145；occlude 135）→ 模型间不可约分歧，单交 ≈ −2pp；只有 union（dropped=0）形态不亏。先算台账（2min CPU）；污染 val 不能当筛子。
- **置信阈值轴**：36ep conf=0.40 → +0.417pp（三门禁 1 PASS/2 FAIL），15ep clsweight3 复验 → **+0.001pp** = 模型特有噪声 → **地板固定 conf=0.50**，轴关。0.35–0.50 极平高原，**悬崖在 0.25 以下 → 硬下限 0.25**。测试集无 score sidecar → 改阈值须重跑推理。`BASE_TAG=base54_c35` 保留但无实测理由偏好。
- **印证率拟合 `r = 0.2816+0.3083×corr`** 系统性低估 0.09–0.18 → 已作废，勿单独引用，改用 §7 直接测法。
- 分辨率/裁剪：960×384 +0.028pp、1366×540 −1.542pp、cut400 −0.772pp（模型自己找地平线，三裁剪 top 中位数都是 564）；平滑/稀疏化 ≈ 0。
- **span80** 只对共识平均的残线桩有效（+0.127pp），对独立模型自己的短线是负的（−0.106pp）。
- **共识门槛**控制变量 = 同意比例（≥20% −0.145pp；≥40% +0.035pp；**>~70% 一条线不加**）。**共识删 FP 死**（incumbent 2664 条中 86.6% 被全部 4 模型支持 → FP 是系统性偏差）。
- 已排除：单树内部重复线（30px 描边 IoU≥0.5）= 0 对 → intra-tree dedup 无可删；非法几何全无。
- 封死（实测均负/无效）：全局仿射、去重、flip-TTA、端点上外推、OS 过采样、CLRerNet、跨种子 soup、VAT、ConvNeXt-T、雾雨增强、segmask binary(MISS)、occlusion(NULL)。

## 7. ⭐ 单支撑 union 实测为**正**（9/17）
工具 `scripts/oracle_score_tree.py`（基线原样 + 只叠增量线），产物 `outputs/lvo_union_probe_20260917/`。
- 等强对 +1091 线 → **+0.572pp**（r=0.509，CI [−0.225,+1.388]）；强+弱 +1165 线 → **+0.247pp**（r=0.440）。
- 两条都正，支撑越弱 r 越低 → `uni_s101_54ep`(96) ≈ **+0.24~0.47pp**。
- ⛔ CI 下界 <0；**union 在 A 榜从未实际提交过** → 唯一还没在同域验证过的杠杆 → `shot7_uni_cons_s101`。脚本守卫：加线 >300 告警（本次 +264，未触发）。

## 8. B 榜作战（`docs/runbook_testB.md` / `docs/action_testB_20260916.md`）
- 窗口 9/17 00:00 – 9/18 17:00，**6 发取 max**。
- ✅ **数据到位（9/17 08:29）**：本地 `data/raw/dataset/测试集B.zip`（240,471,948 B，sha `69114a30…`）；实例 JPEGImages 含 10 个 testB clip（90=80+10）。官方清单 `<lane_root>/data/testB.txt`（1000 行，`--official-list` 必须用它，顺序≠目录推导序）。
- ✅ **六注已建成（9/17 09:25）**：`outputs/submit_testB_shot{1_base54,2_uni_swa4_cons,3_consensus,4_margin0,5_cons_k6,7_uni_cons_s101}.zip`，全部过官方预检 + 本地 verify，各 1000 文件 / 根 `submit/`。线数 1910/2150/2099/1910/2030/2174。
- 🔴🔴 **9/17 14:14 首批 3 个分数（目前唯一的域内真值）**：`shot1_base54`(margin40) **0.65672** ｜ `shot4_margin0`(margin0) **0.67527** ｜ `shot3_consensus`(cons 56%+span80) **0.71459**。
- ⭐⭐ **最大发现：共识新增线在 testB 上真线率极高**。zip 层面验证：shot4 的 **1910 条线全部原样出现在** shot3 里（1000 文件零例外），shot3 **只多 189 条** → **这 189 条独自贡献 +3.93pp**。反推 **r ≈ 0.59–0.93**（盈亏线 θ = 基线 F1/2 = **0.3376**）→ **r 是盈亏线的 2–2.7 倍**。
  机理：testB 明显更难（**同配方 A 榜 0.73574 vs testB 0.67527，低 6.05pp**）→ 地板 FN 更多，共识补回的正是**真缺失**。A 榜地板已够好，同机制只值 +0.035pp（r≈0.378 贴线）→ **「共识族是小杠杆」是 A 榜样本的产物，不可迁移。**
- 📐 **门槛标定（`docs/consensus_gate_calibration_20260914.md`，OOF 5 支撑，盈亏线 0.3921）**：边际真线率 20%→0.0055｜40%→0.0506｜60%→0.342｜80%→0.4313｜100%→0.5809；且**目标域整条曲线右移约一个档（+20pp 同意率）**（testA 实测 20% 档 r=0.341）。testB 的 56% 档实测 r≈0.6–0.9 → **testB 曲线比 testA 还要再右移**。
- 📊 **各门槛在 testB 的加线量**（相对地板 1910；产物 `build/gate_k*`）：k=2(22%) **+569**｜k=3(33%) **+382**｜k=4(44%) **+269**｜k=5(56%) +190｜k=6(67%) +120。
- 📌 **明日三发 = 门槛阶梯 44% / 33% / 22%**（`shotB_gate_k4/k3/k2`，span80 + **margin0**）：右移后最优门槛落在 22%–44% 之间，三发正好夹住。**取 max ⇒ 失败提交零代价**，故全押加线侧；已落袋的 0.71459 是保险。
- **提交历史**：9/17 已交 3 发（shot1/shot4/shot3，见上表分数）。**剩余 3 发在 9/18 17:00 前用完** → 交门槛阶梯 `shotB_gate_k4/k3/k2`。旧的「shot1→shot3→shot6→shot7→shot2→shot4」顺序**已作废**（shot6 因 M=40 从未生成）。
- **注单（文件名 = 唯一权威，勿用中间编号）**：`shot1_base54` 地板+trim(M)；`shot2_uni_swa4_cons` union(swa4, cons 56% 树)；`shot3_consensus` cons 9 树 k=5(56%)+span80，+24；`shot4_margin0` 地板 trim0；`shot5_cons_k6` cons k=6(67%)，+5（最弱，先丢）；`shot6_cons_k5_m40` 几何对冲（0~+2.25pp）；`shot7_uni_cons_s101` union(cons 56% 树, s101_54ep)，+96。**加线 = 0 的注别交。**
- ⭐ **shot6 价值论证**：`f<450>3%` 是单发期望最优，但取 max 时期望是错的目标函数；它真正对冲「阈值模型可能错了」。
- 阶梯分数 44/56/67% 按可用树数四舍五入（m=9→4/5/6；m=8→4/4/5；m=7→3/4/5；m=6→3/3/4）。⚠ 支撑掉到 7 棵时 shot5 必废（K6=5/7=71%>70% 死区）。
- **实例独立支撑树 9 棵**：s42/s101/s202/s303 36ep、clrernet36、cut400、hires、occlude_36ep、s101_54ep（swa*/soup* 是 base 派生，不算票）。定价前先确认每棵依赖树在实例有 run-dir。
- **两条命令**：实例 `scripts/autodl/run_testB_infer.sh`（13 树推理 → 校验 → 打 `f<450` → `/hy-tmp/testB_bundle.tgz`）；本地 `bash scripts/build_testB_candidates.sh <tgz>`（定 margin → 建 6 注 → 官方预检，约 13min）。
- **常驻触发器**：`scripts/autodl/watch_testB_and_run.sh`（setsid，锁 `/hy-tmp/testB_watch.lock`，日志 `/hy-tmp/testB_watch.log`，DEADLINE `202609181700`）。**日志只在 start/TRIGGER/数量变化时打印，久无新行 ≠ 已死**；判存活 `ps -eo pid,etime,args | grep "[w]atch_testB"`。
- **solution.zip 已冻结 v4**：`/hy-tmp/solution_freeze_20260916_v4.tgz`（2.7G）。规则 §3.2 =「B 榜发布前冻结」+「赛后前三名上传字节一致包」→ 改脚本不需重冻。bundle 必须 `tar czf x.tgz .`。
- ⚠️ `_DRYRUN_20260915_*`、`/tmp/_dryrun_quarantine_20260916`、`/tmp/_DRYRUN_OUT_*` 里的 zip 内容是 **testA，禁止提交**。
