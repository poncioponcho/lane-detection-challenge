# 恶劣场景下的车道线检测挑战赛 — 任务分解 v2.8（单人版）

| 项 | 内容 |
|---|---|
| 文档版本 | **v2.8（单人版 + §24 预测提交链/T40 收口）** |
| 状态 | **active**（执行跟踪唯一入口；进度以本文勾选列为准） |
| 版本链 | v1 → … → v2.7（§23 AutoDL/UnLanedet 执行面）→ **v2.8**（§24 预测提交链/T40 收口） |
| 上游依据 | `docs/DECISIONS.md` §12–§15 / §17–§24｜ `docs/ARCHITECTURE.md` v2.0 §6.5（单人关键路径） |
| 常量取值 | 目标 **82.0（工作）/ 84.0（冲刺）** / 预算 200 元 / 单人，一律以 `configs/default.yaml`（v4）为准，本文只引用（§15.1 重估后） |
| 更新日期 | 2026-09-02（周三）｜距 A 榜截止 12 天｜距 B 榜截止 15 天 |

**勾选标记**：✅ 完成 ｜ 🔵 进行中 ｜ ⬜ 未开始 ｜ ❌ 已砍（单人裁剪，理由见 DECISIONS §13）｜ ⚠️ 降级（时间允许才做，不进关键路径）

---

## 0. 编号体系映射（消除歧义，DECISIONS §11 决定）

两套任务编号并存，职责不同：**本文 W/T*.x = 工作流层（对外汇报与进度跟踪）**；**ARCHITECTURE §6.2 的 T00–T81（非连续编号）= 任务层（开发实施与依赖分析）**。引用规则：对外汇报用 W 编号，代码/实施细节用 T 编号，跨文档引用一律带文档名前缀（如 "ARCHITECTURE T42"）。

| 工作流层（本文） | 对应任务层（ARCHITECTURE §6.2） | 内容 |
|---|---|---|
| W0 启动与环境 | T00 / T01 / T02 / T30 | 骨架、依赖锁定、conventions、云环境 |
| W1 评测与提交管线 | T10 – T19（其中 T19 与 W2 数据清单共责） | common 基础设施、Oracle/metric/manifest、bootstrap CI、报告、打包校验 |
| W2 数据与基线 | T19 – T24、T31、T40 – T43 | manifest、三格式解析、一致性、EDA、场景标注、按段切分、dataloader、UnLanedet 原生训练执行面、baseline 36ep、首评 |
| W3 恶劣场景专项 | T50、T53、T55（+T34 分桶诊断） | 后处理链、退化增强（雾+雨 1 组）、分辨率 2 档 |
| W4 调参与定稿 | T51、T60 – T63 | 阈值扫描、定模型、重训、复现、沙盘 |
| W5 冻结与终局 | T65、T70 – T72 | 冻结 SHA-256、B 榜首提、终提交 |
| W6 答辩与收尾 | T80 – T81 | 消融可视化、答辩材料与彩排（仅前三受邀） |

---

## 1. 由评分规则倒推的硬约束（不变，全队共识 = 单人自检清单）

1. **精度门槛**：线宽 30px 无抗锯齿，`IoU = (30−d)/(30+d)`，TP 要求 IoU>0.5 → 横向误差 **< ~10.3px**（T12 实测修正，理想模型 10px；见 DECISIONS §9.1）。
2. **车道线编号不参与评分**（匈牙利按几何匹配）→ 不需要 ID 分类头，但必须预测对**条数**。
3. **全集汇总** → 不能靠简单图刷分，长尾难样本同等权重。
4. **B 样条 k≤3 稠密化** → 输出折线密集采样（≥10–20 点，弯道更密），纵向跨度覆盖可见范围。
5. **按视频段切分验证集**，禁止按图随机切（同段连续帧相似 → F1 虚高 3–8pp）。
6. **A 榜是探针不是考试**：只做本地 metric 校准与链路验证，不做调参依据。
7. **（单人新增）检对比抑 FP 值钱 2.40×**（DECISIONS §9 J3）→ 主战场是召回侧（退化增强 + 分辨率），后处理只是收尾。
8. **（§17.1 新增）评测双层制**：一切最终裁决（A/B 闸门、冻结定稿、台账结论数）以冻结官方 `score.py`（Oracle）读数为准；本地 metric 仅诊断/扫描，须过差分测试套件方可参与过程判断。
9. **（§17.6/§18.3）manifest 有序清单制**：`image_id = <clip>/<frame>`；从官方清单逐行构造 + 无重复断言；存在性按 split 区分（labeled 强制图像+GT，testA/testB 只断言图像、gt_path=None）；每段 100 帧仅对已核实 train/testA 强制；缺预测按空预测计 FN；一切聚合与分桶从同一 manifest 派生。

---

## 2. 任务清单（单人版）

### W0 · 启动与环境（9/1 – 9/2）

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ✅ | T0.1 | 完成赛事报名 | 报名状态已参赛（账号 daniel1547） | P0 | 9/1 ✅ |
| 🔵 | T0.2 | **下载数据**：训练集 71 段/7100 张 + 三类标签、testA 9 段/900 张、`db_info.yaml`、`testA.txt`；落盘后上传 AutoDL 数据盘（DECISIONS §15.5/§23） | ✅ 落盘 + 全量核实完成（9/1 晚：71×100 / 9×100 / 三格式齐全 / anno_txt↔Json 条数互证 24435=24435 / 官方脚本到手，§17.9）；**AutoDL 云端上传待确认** | P0 | 9/1 晚–9/2 |
| 🔵 | T0.3 | 仓库骨架（`src/ configs/ outputs/ scripts/ docs/ tests/`）+ AutoDL GPU 环境 | 本地骨架已建 ✅；**git 本地 + GitHub 私有远程已推 ✅（`poncioponcho/lane-detection-challenge`，DECISIONS §15.6）**；AutoDL GPU 实例待取得 | P0 | 9/1–9/2 |
| ✅ | T0.4 | 数据 EDA：场景/条数/点数/空标注/几何健康/切分偏移 | `src/data/eda.py` + `docs/eda.md` + 8 类固定 overlay；全量 7100 图/24435 线，空 GT 3.69%，越界/非有限/塌缩线均 0；发现旧 val 空 GT=0 后以车道条数直方图作 scene 同分 tie-break，新 val 空 GT=35/800，scene 主目标不变（DECISIONS §22） | P0 | ✅ 9/2 |

### W1 · 评测与提交管线（9/2 – 9/5）— 已部分提前完成

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ✅ | T1.1 | **metric 对齐官方实现（DECISIONS §17.1）**：本地诊断层已逐字义对齐——匈牙利 `cost=1−iou`、参数均匀稠密化 `(N−1)*5+1` + 逐段 `cv2.line`、float64 输入、异常即抛（禁线性回退）；官方 score.py 冻结为 Oracle（`src/eval/official_oracle/`，SHA-256 `b2f4c9b2…2de0d2`） | T1.2 差分套件全绿；本地 metric 仅作诊断/扫描，最终裁决仍一律走 Oracle | P0 | ✅ 9/2 |
| ✅ | T1.2 | **metric 差分测试套件（§17.1）**：边界/线数/空缺文件/重复点/折返/越界/匈牙利/异常传播/哈希；真实 identity 全集；非 identity 跨环境整图差分 | 当前 `tests/` **94 passed**；identity 7100 图/24435 线 TP=24435/F1=1；跨 cv2 5.0↔4.12 非平凡渲染 1034 用例逐比特一致，整图 576 图/2009 线逐图 TP/FP/FN 零分歧（总计 1488/812/521，F1=0.690647） | P0 | ✅ 9/2 |
| ✅ | T1.3 | 预测交付唯一入口 `prepare_submit.py`：任意精度 raw → 一位小数 canonical → pack → verify → labeled 时可选 Oracle；精确文件集/安全路径/资源上限/严格边界/序列化后去重/全链几何 smoke | 5 位预测确实被重写为 1 位；舍入塌缩、缺失、多余、canonical 陈旧文件、traversal 均拒绝；非 identity 全链 Oracle=`TP/FP/FN=1/1/1,F1=0.5`；真实 train/testA 旧 smoke 保持 PASS（DECISIONS §24） | P0 | ✅ 9/2 |
| ✅ | T1.4 | 按视频段 hold-out 切分本地验证集（8 段）：scene 二元特征为主目标、逐图车道条数直方图为同分 tie-break；seed=42 下 100,000 候选 + 单交换；`weather:mixed` 强制留 train | `src/data/split_by_clip.py` + 10 项测试；`configs/splits/v1_seed42.yaml` 固化 63/8 段、6300/800 图、段 ID 零交集；scene 最大偏差仍 0.06338，val 空 GT 35 张（4.375%），标签/manifest/条数直方图 SHA-256 入配置 | P0 | ✅ 9/2 |
| ⬜ | T1.5 | A 榜首提（只烧 1 次额度验链路） | 平台返回有效分数，与本地 val 差值入台账 | P1 | 9/5 |
| ✅ | T1.6 | （新增）A 榜每日快照提醒（每日 20:00，9/14 前） | 自动提醒已建；榜单 JS 渲染需人工记录前 5 名 | P1 | ✅ 9/1 |
| ✅ | T1.7 | **oracle_runner 适配层（§18.2）**：冻结文件运行前 SHA-256 校验；强制 Python 3.12 + 官方精确 pins；只以子进程调用 score.py；全局全量单独调用 + 逐 clip 子集分别调用；结构化输出 TP/FP/FN/P/R/F1/n_images | `src/eval/{oracle_integrity,oracle_runner}.py` + 4 项 runner 测试；真实 train identity：全局 7100 图 TP=24435/FP=0/FN=0/F1=1.0，71 个 per_clip 均独立调用；样例 `outputs/reports/oracle_identity_train.json` | P0 | ✅ 9/2 |

### W2 · 数据与基线（9/2 – 9/8）※ 依赖 T0.2 数据下载

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ✅ | T2.0 | 三格式解析 + 一致性校验（ARCHITECTURE T20/T21）：支持真实 `annotations.lane[]`、跨目录布局、palette instance PNG、多方向点序；lossless text↔JSON 精确比对 + PNG 10px union IoU 门 | 7100/7100 text↔JSON 点数组全等（24435 线）；262 空 GT 三格式同空；PNG union IoU min/P1/median=0.7282/0.8860/0.9117；badlist 1/7100=0.0141% <0.1%，极短二点线保留不删；`tests/test_parse_labels.py` 8 项，审计 `docs/label_consistency_audit_20260902.md` | P0 | ✅ 9/2 |
| 🔵 | T2.1 | **AutoDL 专用** UnLanedet 落地 + 三 checkpoint shape/load 探针：固定 commit `0392184…4c`，同时核验 CLRNet 两个疑似错位文件与 ADNet-R34；本地禁下载/加载大权重 | 本地已备 setup/probe/config 执行面；待 AutoDL 输出两主干 compatible/missing/unexpected/shape-mismatch、实际 R50 文件名、adapted checkpoint SHA 与 demo 可视化后完成 | P0 | 9/3 |
| 🔵 | T2.2 | manifest 驱动 `HardLaneDataset` + 双套 config（1366×720；palette 原始索引；保留空图和 2/3 点线；稳定连续去重）；T40 已收口为 pinned `tools/train_net.py` + 原生 AMP/checkpointer/`--resume`，权重只经 `train.init_checkpoint` 注入；CLRNet decode `top_k` 解耦为 `test_parameters.nms_topk` | 本地已完成 CPU 契约/配置/脚本静态测试；待 **AutoDL** 跑通 CLRNet/ADNet 各空/非空 batch forward+loss+backward、demo、预处理 overlay、1 epoch train+val，并验证持久目录 `last_checkpoint` 续跑。字段映射固定：CLRNet target=8/classes=9/priors=192/candidate=12；ADNet target=8/anchors=300/candidate=12；最终输出仍扫 {7,8,10,12} | P0 | 9/3 白天 |
| ⬜ | T2.3 | baseline：**双路 15ep 廉价筛选（CLRNet-R50 vs ADNet-R34，CULane 权重 fine-tune）→ 赢家 36ep**（DECISIONS §15.2；筛选兼任管线 shakedown；AutoDL 4090 各 ≈6h + 36ep ≈13h；判定：\|ΔF1\|<1.5pp 取 CLRNet-R50） | **9/5 出分（M2）**：val F1@0.5 首个数据点（入门线 0.75）+ 主干裁决入台账 → **立即触发 §15.1 重估**；⚠️ fine-tune 预案（DECISIONS §16.3）：若出现过拟合形态（train 降 / val 降），第一调节项 = LR 降档（1/5–1/10）+ 缩短 schedule，排除后再查 bug；**随机性控制（§18.7）：双路筛选固定相同 seed/初始化（paired bootstrap 只覆盖样本不确定性，不覆盖训练随机性），进入最终定稿的改动再按 {101,202,303} 多 seed 复核** | P0 | 9/3 晚–9/5 |
| ❌ | T2.4 | ~~多主干横向对比（ADNet/CondLaneNet/UFLD/RESA）~~ | **已砍**（DECISIONS §13：20h 人工超单人容量；α-SimADNet/RVLD 原版接入分支已于 9/1 晚关闭——UnLanedet 未收录两者；主干由 T2.3 双路筛选定，见 DECISIONS §15.2） | ~~P1~~ | — |
| ✅ | T2.5 | **多维场景标签人工标注（§17.2/§18.4）**：71 段 × weather / illumination / artifact / geometry + confidence + 真实抽查帧号；每段 5 帧接触表，low-confidence 与稀有特征（持有段≤5）全部二审 | `data/processed/scene_labels.json`（已从 `/data/` 忽略规则中单独放行）；6 个稀有持有段完成密集二审；分布 clear/fog/mixed/rain=31/18/1/21，low-light/normal=31/40；唯一 mixed 段确认雾+积雪并留 train；审计见 `docs/scene_split_audit_20260902.md` | P0 | ✅ 9/2 |
| ✅ | T2.6 | **manifest 构建器（§17.6/§18.3）**：官方清单逐行保序构造；无重复/路径安全/存在性/split GT 语义/order/已核实 split 每段 100 帧全断言；`image_id=clip/frame` | `src/data/manifest.py` + 8 项契约测试；真实产物 `manifest_train.jsonl`=7100 条/71 段、`manifest_testA.jsonl`=900 条/9 段，testA `gt_path=null`，全部存在性校验通过 | P0 | ✅ 9/2 |

### W3 · 恶劣场景专项（9/6 – 9/11）

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ⬜ | T3.1 | 退化增强 **1 组高期望算子（雾 + 雨）**（ARCHITECTURE T53；84.0 的主来源） | **§17.3 双门槛**：点估计 ΔF1 ≥ **+1.0pp** 且 paired CI 下界 > 0，附 LOCO 敏感性；不过闸则回退 | P0 | 9/6–9/9 |
| ⚠️ | T3.2 | 图像复原前置（CLAHE/Zero-DCE/去雾）A/B | **降级**：不在单人关键路径；若做需评估 B 榜 41h 窗口吞吐 | ~~P1~~ | 时间允许 |
| ⬜ | T3.3 | 后处理链（ARCHITECTURE T50）：等距重采样、端点外推、断裂补全、曲率平滑，各自独立开关 | 单开关 A/B；**§17.3 双门槛**：点估计 ΔF1 ≥ **+0.5pp** 且 paired CI 下界 > 0 才保留 | P1 | 9/7–9/10 |
| ⬜ | T3.4 | 分桶诊断：低照度/雨雾/逆光/反光/阴影/弯道/路口 分桶 F1，Top-3 短板 | `docs/bucket_analysis.md` + ≥20 张失败案例 | P0 | 9/8–9/10 |
| ❌ | T3.5 | ~~短板桶过采样定向补强~~ | **已砍**（DECISIONS §13 = ARCHITECTURE T56：人工高、边际低；定向增强部分已并入 T3.1） | ~~P1~~ | — |

### W4 · 调参与定稿（9/8 – 9/14）

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ⬜ | T4.1 | 阈值扫描（ARCHITECTURE T51）：置信度/NMS/max_output_lanes/点数/外推长度，以 F1 为目标；**J2 停止条件：新增线匹配率 ≤ F1/2≈41.6% 即停**；max_output_lanes 网格 {7,8,10,12}（§17.4/§18.5） | **J4 三件套（DECISIONS §9/§15.3/§17.3）**：平台区取点（非单点峰值）+ 扫描全表入台账（含负结果）+ 双门槛（点估计 ≥+0.5pp 且 paired CI 下界 > 0，附 LOCO 敏感性）；**选择偏差纪律（§18.7）**：同一 val 选优再算 CI 存在选择偏差——网格扫描定位为调参证据、取平台区稳健点，选中后的普通 CI 不得描述为无偏确认 | P0 | 9/8–9/11 |
| ❌ | T4.2 | ~~按段 K=3 折交叉验证~~ | **已砍**（DECISIONS §13：单折 + 段级 bootstrap CI 替代，第二折记为风险写入台账） | ~~P1~~ | — |
| ⬜ | T4.3 | 本地 val ↔ A 榜相关性台账（每次提交：本地 F1 / 配置 hash / A 榜 F1） | ≥4 组有效样本，能判定系统偏差 | P0 | 9/8–9/14 |
| ⚠️ | T4.4 | TTA 探索（水平翻转对不对称车道可能有害） | **降级**（DECISIONS §13：1 GPU·h 但需人工接，不进关键路径） | ~~P2~~ | 时间允许 |
| ❌ | T4.5 | ~~时序/跨帧信息利用~~ | **已砍且不问**（DECISIONS §6/§17.8：帧编号步长 3 但源 FPS 未知，原「1s 间隔 × 60km/h = 16.7m」测算作废；结论理由 = 合规不确定 + 实现成本 + 缺少收益证据） | ~~P3~~ | — |

### W5 · 冻结与终局（9/14 – 9/17）

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ⬜ | T5.1 | 冻结 `solution.zip`（代码+权重+配置+README）+ 64 位 SHA-256 + 精确字节数 | `docs/freeze.md`，压缩包双份留存 | P0 | 9/14 前 |
| ⬜ | T5.2 | B 榜管线演练（testA 全流程：推理→打包→校验→提交，计时） | 全流程 ≤90 分钟 | P0 | 9/13–9/14 |
| ⬜ | T5.3 | 9/16 00:00 B 榜发布 → 推理 → 首提交 | 力争 9/16 06:00 前首个 B 榜分 | P0 | 9/16 |
| ⬜ | T5.4 | B 榜有限调优（约 6 次额度，单变量） | 每次单变量，榜单取历史最优 | P1 | 9/16–9/17 |
| ⬜ | T5.5 | 9/17 终提交（**15:00 前完成**，防拥堵） | 提交成功截图存档 | P0 | 9/17 |
| ⬜ | T5.6 | 若 TOP3：字节一致的 solution.zip + 可复现代码 | SHA-256 逐位比对一致 | P1 | 9/17–9/20 |

### W6 · 答辩与收尾（9/18 – 10/24）※ 仅 B 榜前三受邀

| 状态 | ID | 任务内容 | 验收标准 | 优先级 | 起止 |
|---|---|---|---|---|---|
| ⬜ | T6.1 | 技术报告 + 答辩 PPT | 15 页内，消融表 ≥5 组 | P2 | 9/18–10/20 |
| ⬜ | T6.2 | 答辩彩排（作品 70% / 答辩 30%，仅前三内部排序） | 计时演练 ≤3 次 | P2 | 10/18–10/23 |
| ⬜ | T6.3 | 参赛心得投稿（周边奖励） | 稿件发出 | P3 | 10/10 前 |

---

## 3. 单人关键路径与里程碑（ARCHITECTURE §6.5 口径）

```
【开工前置（用户裁决，DECISIONS §17/§18：完成前不进 T2.2/T31）】
✅ 官方 Oracle 冻结 + 三文件 SHA-256 守护（T17 / `tests/test_official_oracle_hash.py`）
→ ✅ T1.1/T1.2 本地 metric 对齐官方 + 差分测试套件全绿（9/2）
→ ✅ T2.6/T19 manifest 有序清单 → ✅ T1.7/T18 oracle_runner 结构化全局+逐 clip 输出（9/2）
→ ✅ T2.5/T24 多维场景标签人工标注 → ✅ T1.4/T23 按段切分（9/2）

【主路径】
✅ 数据落盘+核实（9/1 晚，§17.9）→ ✅ T2.0 解析+一致性 badlist（9/2，1/7100）→ ✅ T0.4 EDA+hold-out 二次固化（9/2）
→ T2.2 dataloader+双套 config（9/3 白天）
→ T2.3 双路 15ep 筛选（9/3 晚–9/4 晨，兼任 shakedown）→ 赢家 36ep（9/4）→ 9/5 出分 M2（触发 §15.1 重估）
→ T3.3 后处理 + T4.1 阈值扫描（0 GPU，CPU 并行）
→ T55 分辨率方案组合（§17.5）→ T3.1 退化增强（雾+雨 1 组）
→ T60 定模型（9/10，A 榜分布重估门槛）→ T61 重训 → T62 复现
→ T5.2 沙盘演练 → T5.1 冻结（9/14）→ T5.3 B 榜首提（9/16）→ T5.5 终提（9/17 15:00）
```

| 日期 | 里程碑 | 判据 |
|---|---|---|
| 9/1 ✅ | 报名完成；骨架+venv+metric 自检全绿；git 本地+远程备份；**数据落盘 + 全量核实（§17.9）；官方 Oracle 冻结（§17.1）** | T0.1 / T0.3 / Oracle ✅ |
| 9/1 晚 | ✅ 数据下载完成并核实（71×100 / 9×100 / 三格式齐全）；云端上传挂夜传待确认 | T0.2 |
| 9/2–9/3 | **M1（减载版 + §17/§18 开工前置）**：metric 对齐 + 差分套件 → manifest → oracle_runner → 场景标注 + 切分；解析 badlist / EDA 成文 / dataloader + 双套 config + 双权重核验；submit 打包器解耦至 9/4–9/5 | T1.1 / T1.2 / T1.7 / T2.6 / T2.5 / T1.4 / T2.0 / T0.4 / T2.1 / T2.2 |
| 9/3 晚–9/4 晨 | **双路 15ep 筛选**（CLRNet-R50 vs ADNet-R34，AutoDL 顺序 ≈12h，兼任 shakedown） | T2.3 |
| 9/4–9/5 | **赢家 36ep → M2 出分 → 立即触发 §15.1 重估** | T2.3 |
| 9/8 | 后处理 + 阈值扫描定稿（CPU 侧） | T3.3 / T4.1 |
| 9/10 | 分辨率 2 档结论 + 增强组结论 + **A 榜重估门槛** | T55 / T3.1 / T60 |
| **9/14** | **A 榜截止 + solution.zip 冻结 + 演练完成** | T5.1 / T5.2 |
| **9/16–9/17** | **B 榜窗口（41h）** | T5.3 – T5.5 |
| 10/24 | 决赛答辩（若前三） | T6.1 / T6.2 |

**延迟警戒线**：任何关键路径环节延期 ≥2 天，B 榜 41h 窗口不可承受，立即触发范围再裁剪（优先砍 ⚠️ 降级项，不动 P0）。

---

## 4. 资源与预算（数值以 `configs/default.yaml` 为准）

- **预算**：200 元（已拍板）。GPU 实验事实环境为 AutoDL；租用时长以首个 1 epoch 实测吞吐重算，旧 Kaggle 免费额度测算不再作为当前执行计划。
- **本地机**：Mac M4 / 24GB / 磁盘 81GB——只做 CPU 数据契约、评测、后处理、预测格式归一化与打包；不加载大权重、不做模型推理/训练。
- **AutoDL 纪律**：输出只写持久目录；每 epoch 原生 checkpoint + eval；租期结束/关机前核对 `last_checkpoint`、`model_best.pth`、日志和预测已落盘。模型动态证据不得用本地 CPU 结果替代。
- **依赖锁定**：`pip freeze` 存档（TOP3 复现要求）；本地 venv：`envs/lane`（numpy 2.5.2 / opencv 5.0.0 / scipy 1.18.1）。

---

## 5. 风险登记（关键项）

| # | 风险 | 状态 | 缓解 |
|---|---|---|---|
| R1 | 本地 metric 与官方不一致 | **已缓解**——当前 94 项测试；7100 图 identity + 1034 渲染用例 + 576 图非 identity 跨环境差分全绿 | Oracle 双层结构；最终裁决仍只走 Oracle；T1.5 首提作平台链路锚点 |
| R2 | 验证集按图随机切 → 虚高 3–8pp | **已缓解**——63/8 段显式固化，6300/800 图，段 ID 零交集 | `split_by_clip.py` + `v1_seed42.yaml` + 契约测试；后续实验只复用该 split |
| R3 | A 榜单点调参过拟合 | 开放 | 阈值只在本地 val 定 |
| R4 | B 榜 41h 窗口故障无余量 | 开放 | T5.2 演练计时 + 提前 2h |
| R9 | AutoDL GPU 暂不可得或租期中断 | 开放 | CPU 交付链先收口；到卡后先跑探针/动态门，输出落持久目录 |
| R10 | AutoDL 4090 吞吐测算失准 | 开放 | 首个 1 epoch 实测后重算筛选与 36ep 时长 |
| R11 | **单人带宽**：任何 illness/生活占用直接吃掉唯一人力 | 开放 | ⚠️ 降级项全部让路 P0；关键路径日清 |
| R12 | 文档常量漂移（2026-09-01 已发生并修复） | **已立制** | DECISIONS §14 四条制度 + `configs/default.yaml` 收口 |

---

## 6. 拍板状态（DECISIONS §12 已落，此处仅登记）

| # | 问题 | 状态 |
|---|---|---|
| Q5 | 报名 | ✅ 已完成 |
| Q6 | 答辩出席 | ✅ 可线下出席（仅前三受邀，不构成翻盘） |
| Q2 | 队伍规模 | ✅ **单人**（§13 裁剪已传导到本文） |
| Q1 | GPU 预算 | ✅ 200 元 |
| Q3 | 框架 | ✅ **UnLanedet**（已生效）；主干 = **双路 15ep 筛选赢家**（CLRNet-R50 vs ADNet-R34，DECISIONS §15.2；ConvNeXt-T 为 9/10 升级备选） |
| Q4 | 其他 GPU | 未提及视为无（如变化请显式声明） |

**A 榜门槛状态**：9/1 首份快照已取得（前三 0.79076 / 0.78566 / 0.78422，见 `docs/a_bang_snapshot.md`）；仍需每日跟踪，9/10 用届时最新分布重估门槛，不把首日快照当最终竞争水位。

---

## 7. 附：关键参考

- **评分水位**：α-SimADNet F1@0.5 = 83.2（HardLane-F100 全集 10600 张，仅量级参考）；CLRNet-DLA34 CULane F1@50 = 80.47（原版 CLRNet 仓库数字，**UnLanedet 无 DLA-34**）；UnLanedet 复现水位见 `docs/weight_scout_report.md`。
- **代码**：UnLanedet https://github.com/zkyntu/UnLanedet ｜ CLRNet https://github.com/jie311/CLRNet
- **评分细节**：赛题"三、评审规则"——画布 1366×720、线宽 30px、无抗锯齿、B 样条 k≤3、匈牙利、全集汇总。
- **metric 实现细节**：`docs/T11_T12_metric_done.md`（含 10.3px 边界实测修正）。
