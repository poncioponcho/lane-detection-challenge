# 项目长期记忆（恶劣场景下的车道线检测挑战赛）

> 本文件只存跨会话必须延续的项目约定与铁律。过程细节看每日日志。

## 文档治理（DECISIONS §14 立制，2026-09-01）

- **仲裁源**：`docs/DECISIONS.md` 是唯一争议仲裁源；常量唯一定义处 = DECISIONS + `configs/default.yaml`，其余文档只引用。
- **状态头**：每份治理文档头部必须有 版本/状态/版本链；改内容必须同步改版本头。
- **引用规则**：跨文档引用一律用**章节锚点**（"ARCHITECTURE §6.5"），**禁用行号**（2026-09-01 事故实证：行号随编辑漂移全部失效；连行数都会烂）。
- **裁决传导**：DECISIONS 拍板后必须在一个工作回合内传导到 TASKS/ARCHITECTURE/overview，否则视为裁决未完成。
- **任务编号双体系**：TASKS.md 用 W0–W6/T*.x（工作流层，对外汇报），ARCHITECTURE §6.2 用 T00–T72（任务层，实施细节）；引用必须带文档名前缀。

## 环境坑（本机实测）

- **BSD grep（bash 里的 grep）对中文+`\|` 交替模式会静默返回零命中**（连纯 ASCII 子模式都不匹配），必须用内置 Grep 工具（ripgrep）做中文内容检索。
- `software-*` 系列 subagent 调 TaskList 必崩（Tools 列表为空）；多 Agent 并行易触发 429 限流。长文档定向修订由主 Agent 直接 Edit 更稳。
- 本地 CPU 开发 venv：`/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane`（numpy 2.5.2 / opencv-python-headless 5.0.0 / scipy 1.18.1 / pytest 9.1.1，python 3.13.12）。沙箱内复核须显式 `--basetemp=/private/tmp/<dedicated>`；2026-09-02 §25 后现测 **106 passed**。GPU 权重/模型/推理/训练实验只在 AutoDL，绝不以本地静态检查冒充。独立 Oracle 临时环境 `/private/tmp/lane-oracle-py312`（Python 3.12.13 + 官方精确 pins）；最终裁决不得使用训练环境。
- **`.gitignore` 目录模式陷阱**：无前导斜杠的 `data/` 会匹配**任意层级**同名目录（误伤 `src/data/` 源码）；忽略根目录必须写 `/data/`（锚定）。

- **切分协议铁律（2026-09-04 诊断，最高优先级）**：`v1_seed42` 是 **clip-level** 切分，**val 100% video 级泄漏**——val 的 8 clip 来自 5 个 video，这 5 个 video 的其余 63 clip 全在 train 里（7100 图 = 8 video / 71 clip）。因此 v1 val 上出现天花板效应，paired bootstrap CI 半宽 ±1.9pp，**任何 <2pp 的消融在 v1 val 上都不可判定**（实测：像素量 ×1.8 仅 +0.188pp）。→ 新消融必须先问"评估集是否 video-disjoint"，否则等于烧 GPU 换噪声。修法：8-fold leave-one-video-out 拼 7100 张 OOF 全集。**已闭环（2026-09-05）**：CLRNet-R50 15ep LVO 全局 F1=0.777628、video-cluster 95% CI=[0.6886,0.8374]——bootstrap 独立单位=8 video（71 clip 口径会低估，CI 半宽 7.4pp 非泄漏诊断时预期的 0.7pp）；最弱域 v546797496(F1 0.468)。这是 val 0.808(泄漏) 高估 ≈3pp 的机制解释，产物 `outputs/lvo_clrnet_r50_15ep_20260904/`。诊断见 `outputs/reports/video_leakage_diagnosis_20260904.md`。

## 已闭环的技术事实（勿重新推导）

- **metric 复刻已完成且自检全绿**（2026-09-01，`tests/test_metric_selfcheck.py` 8/8）：cv2 `thickness=30` 有效线宽≈31px → **IoU=0.5 真实边界 ≈10.3px**（非理想模型的 10px）；A 榜反演标定 Q-A3 已关闭。
- F1 恒等式 `F1=2·TP/(P+G)`；检对:抑FP 边际价值 = 2.40×；放宽阈值充要条件 = 新线匹配率 > F1/2≈41.6%（DECISIONS §9）。
- 已拍板（DECISIONS §12 + §15.2）：报名完成（daniel1547）/ 可线下答辩 / **单人** / 预算 200 元；框架 UnLanedet；**baseline 主干 = 双路 15ep 筛选赢家（CLRNet-R50 vs ADNet-R34，CULane 预训练起步禁 from-scratch）**；ConvNeXt-T（CULane 80.21）为 9/10 升级备选。
- 单人裁剪（§13）：砍多主干对比 / K-fold / 过采样 / 时序（不做且不问）；TTA 与复原前置降级。人工 ≈95–105h。
- UnLanedet 权重调研（`docs/weight_scout_report.md`，2026-09-02 复核）：同框架 CULane 报告水位不变；**无 DLA-34、无 RVLD/α-SimADNet**。CLRNet generic/named-r50 两资产映射矛盾，必须 AutoDL shape/load 探针裁决；真实训练权重入口是 `train.init_checkpoint`，不是残留帮助文字里的 `MODEL.WEIGHTS`。
- 目标滚动重估（§15.1）首次已触发：工作 82.0 / 冲刺 84.0 / 预测 ≈80.3 / 缺口 ≈3.7pp；剩余节点 9/5、9/10。J4 网格扫描防过拟合三件套（§9）。
- **git 远程已闭环**：私有仓 `poncioponcho/lane-detection-challenge`（PRIVATE），origin/main 与 main 同步，每轮 commit 后 `git push`。GitHub MCP 无建仓权限（403），走本地 `gh` + SSH。
- **A 榜实测门槛（`docs/a_bang_snapshot.md` 每日 20:00 自动快照，9/14 失效）**：09-01 前三 ≈0.790/0.786/0.784；**09-04 榜首 0.80826（同队 +1.14pp 冲刺激活）**/前三线 0.78693/前五 0.77199/前十 0.75377。**§15.1 二次重估已于 2026-09-05 拍板（DECISIONS §26 + default.yaml v6）**：目标下修 工作 0.77（保前十）/ 冲刺 0.79（冲前5）/ 预测 0.75 / ceiling 0.81；**自队 A 榜锚点 = 0.73444（36ep 生产 conf=0.5，记录 714962；conf 0.4→0.5 曾 +0.65pp）**；A 榜额度 ≈3/日。评估统计规矩 R1–R4 草案在 09-05 日志，待另行裁决。
- **2026-09-02 开工前置已全部闭环**：T11/T12 metric 对齐+差分、T19 manifest、T18 oracle_runner、T24 71 段场景人工标签、T23 clip-level 切分均完成。`configs/splits/v1_seed42.yaml` 固化 63 train / 8 val 段（6300/800 图），标签 SHA `0ed561e4…f948`；训练前门禁解除。
- **真实三格式 T20/T21 已闭环（DECISIONS §21）**：JSON=`annotations.lane[]` 且标签跨目录；PNG 是 palette BGR instance 图，禁取单通道；`mean_lateral_error` 必须对 y 排序。裁决门=text↔JSON 点级精确 + PNG 10px union IoU≥0.75。全量 text↔JSON 7100/7100 全等、262 空 GT 同空、badlist 1/7100=0.0141%（底边极短二点线，保留不删）。
- **T22 EDA/切分二次固化（DECISIONS §22）**：7100 图/24435 线，空 GT 3.690%，越界/非有限/去重塌缩均 0；330 crop 触及 16.08% 线，clip 空图率 0–56%，weather 与 illumination 完全混杂。旧 scene-only val 空 GT=0；现以逐图车道数直方图作 scene 完全同分 tie-break，scene 目标不变，新 val 空 GT=35/800。split SHA `715eb8a0…fd4ab`，后续实验禁止再改 split。
- **T2.1/T2.2 只在 AutoDL 动态验收（DECISIONS §23）**：本地已落地 manifest-backed HardLaneDataset、CLRNet-R50/ADNet-R34 config、pinned UnLanedet patch、三权重 load/shape 探针、双模型空/非空 loss+backward/demo smoke 和 1 epoch runner；本地仅验证数据数量 train/val=6300/800、空 GT=227/35、短线保留、语法/patch/89 tests。待 AutoDL 返回真实 JSON/日志前任务保持进行中。
- **预测提交入口（DECISIONS §24）**：`HardLaneEvaluator` 的 5 位诊断预测禁止直接交给 `pack_submit`；统一走 `prepare_submit.py`，按 manifest 精确枚举并经 `export_lines(ndigits=1)` canonicalize，再 pack→verify，labeled rehearsal 可追加 Oracle。缺失默认失败、显式才作空；多余/stale/traversal/舍入塌缩均拒绝。非 identity 契约测试结果 `TP/FP/FN=1/1/1,F1=0.5`。
- **T40 已收口（DECISIONS §24）**：不开发自研 `engine/trainer.py/checkpoint.py`；使用 pinned UnLanedet `tools/train_net.py`、两套 LazyConfig、原生 AMP/PeriodicCheckpointer/BestCheckpointer 与 AutoDL 持久目录 `--resume`。动态证据仍等 AutoDL。
- **AutoDL 流水线铁律（DECISIONS §25）**：全局 `train.seed=42`，`cudnn_benchmark=False`，周期 checkpoint 保留 40 个。上游 best hook 不持久化历史，故不单信 resume 后的 `model_best.pth`；必须用 metrics 历史 best iteration 定位 checkpoint 并 eval-only 回放。唯一编排入口是 `run_pipeline.sh gate|screen|baseline`，36ep 必须 fresh 启动新 schedule；关机前下载 handoff tar + SHA。
- **非 identity 跨环境审计已闭环（DECISIONS §19，2026-09-02）**：本地 metric（cv2 5.0/scipy 1.18/numpy 2.5）vs 官方 pins 跨环境，1034 渲染用例 + 576 图整图差分逐比特/逐图零分歧（TP=1488/FP=812/FN=521/F1=0.690647，case SHA `6d3b1061…fbc96f`，复跑 `scripts/run_nonidentity_diff.py`）→ **本地诊断 metric 放行训练期扫描；成绩只认 Oracle 全局单次调用，per-clip 禁止平均**（逐图均值偏离全局 −0.99pp）。提交双防线落地：export 序列化后去重<2 拒绝 + verify 独立复核（逗号/严格 1 位小数/去重<2/边界 >1365·>719/≤64条/≤2048点/全链 smoke）；manifest image_path 四要素校验；`interp_lane` 去重已移除（数组/JSON 失败语义=Oracle）；画布常量收口 `common.types.CANVAS_W/H`。该批次 66 tests，§20 后 74，§21 后 76，§22 后当前 80；原致命 zip 已 FAIL。
