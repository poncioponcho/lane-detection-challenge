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
- 本地 venv：`/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane`（numpy 2.5.2 / opencv-python-headless 5.0.0 / scipy 1.18.1，python 3.13.12）。

## 已闭环的技术事实（勿重新推导）

- **metric 复刻已完成且自检全绿**（2026-09-01，`tests/test_metric_selfcheck.py` 8/8）：cv2 `thickness=30` 有效线宽≈31px → **IoU=0.5 真实边界 ≈10.3px**（非理想模型的 10px）；A 榜反演标定 Q-A3 已关闭。
- F1 恒等式 `F1=2·TP/(P+G)`；检对:抑FP 边际价值 = 2.40×；放宽阈值充要条件 = 新线匹配率 > F1/2≈41.6%（DECISIONS §9）。
- 已拍板（DECISIONS §12 + §15.2）：报名完成（daniel1547）/ 可线下答辩 / **单人** / 预算 200 元；框架 UnLanedet；**baseline 主干 = 双路 15ep 筛选赢家（CLRNet-R50 vs ADNet-R34，CULane 预训练起步禁 from-scratch）**；ConvNeXt-T（CULane 80.21）为 9/10 升级备选。
- 单人裁剪（§13）：砍多主干对比 / K-fold / 过采样 / 时序（不做且不问）；TTA 与复原前置降级。人工 ≈95–105h。
- UnLanedet 权重调研（`docs/weight_scout_report.md`，2026-09-01）：同框架 CULane 复现 ConvNeXt-T 80.21 > CLRNet-R50 79.30 > CLRerNet-R34 79.20 > CLRNet-R34 78.99 > ADNet 77.88；**无 DLA-34、无 RVLD/α-SimADNet**；权重均 GitHub Releases（tag=Weights）可得；注入 = detectron2 LazyConfig `MODEL.WEIGHTS` opts；CLRNet-R34 行链接文件名带 r50 疑错位须核验；配置迁移清单见报告问题 4/5。
- 目标滚动重估（§15.1）首次已触发：工作 82.0 / 冲刺 84.0 / 预测 ≈80.3 / 缺口 ≈3.7pp；剩余节点 9/5、9/10。J4 网格扫描防过拟合三件套（§9）。git 本地仓已建（远程备份待用户确认）。
