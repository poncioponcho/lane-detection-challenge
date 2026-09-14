# 提交就绪审视（2026-09-15 上午）

> 结论先行：**今天不能提交任何东西。** 唯一窗口是 **9/16 00:00 – 9/17 17:00（B 榜，6 发取 max）**。
> 但下面 §1 有一个**可能改变全局的机会**，需要你上平台确认。
> 本次审视对全部 12 个候选包重跑了官方预检（不依赖历史记录），并**新发现 4 个问题、已全部修复**。

---

## 1. ⭐ 头条：先确认 A 榜入口是否还活着

官方规则写的是「A 榜阶段 8/19 – **9/14**」，今天是 9/15，按字面 A 榜已结束。
**但请你在平台上确认提交入口是否仍然可用。** 如果还能交：

- A 榜**不计入最终成绩**（规则 §4），所以那 3 发额度的唯一价值就是**取测量**。
- 我们目前**所有新候选都没有实测分数**（见 §4），只能靠机制强度排序。
- 今天打 3 发就能把这些不确定一次性消掉，例如：
  1. `submit_testA_night_cons_gate6_v2_m0.zip`（2733 之外的纯增量注，问「共识加线在本域到底正不正」）
  2. `submit_testA_night_swa4_54ep_m0.zip`（问「同轨迹权重平均有没有用」）
  3. `submit_testA_night_uni_swa4_g6_m0.zip`（组合注）
- 这 3 个答案直接决定 B 榜 6 发该发哪几个。**比今天多抢 0.2pp 有价值得多。**

若入口已关闭，则按 §5 的 B 榜计划走。

---

## 2. 已就绪清单（全部实测过）

### 2.1 候选包（12 个，全部通过官方 `check_submission`，均 900 文件）

| 包 | 线数 | 相对 incumbent |
|---|---|---|
| `submit_testA_54ep_trim0.zip`（**incumbent，自队最优 0.73574**） | 2664 | — |
| `submit_testA_night_uni_swa4_g6_m0.zip` | 2733 | kept 2664 / dropped **0** / novel 69 |
| `submit_testA_night_cons_gate6_v2_m0.zip` | 2717 | kept 2664 / dropped **0** / novel 52 |
| `submit_testA_night_dryrun_g6_m0.zip` | 2717 | 与 gate6 同构（通路预演产物） |
| `submit_testA_night_swa4_54ep_m0.zip` | 2662 | kept 2635 / dropped 29 / novel 27 |
| `submit_testA_night_swa7_54ep_m0.zip` | 2656 | kept 2628 / dropped 36 / novel 28 |
| `submit_testA_night_swa3_54ep_m0.zip` | 2649 | kept 2629 / dropped 35 / novel 20 |
| `submit_testA_night_conf55_m0.zip` | 2595 | novel 0 / dropped 69（纯删除） |
| `submit_testA_night_conf60_m0.zip` | 2535 | novel 0 / dropped 129 |
| `submit_testA_night_soupA_m0.zip` | 2795 | dropped 39 / **novel 170** |
| `submit_testA_night_soupB_m0.zip` | 2828 | dropped 51 / **novel 215** |
| `submit_testA_night_soupC_m0.zip` | 2816 | dropped 48 / **novel 200** |

### 2.2 执行脚本（两条命令，均已实际执行验证）

- 实例端 `scripts/autodl/run_testB_infer.sh` — 已在逻辑上完整预演（图像检查 → manifest → 10 次推理
  → `f<450` 判据 → 打包）。9 个推理 run-dir 的 evidence **全部合规**（status/model/ck 存在/sha256 齐备）。
- 本地端 `scripts/build_testB_candidates.sh` — **已在 testA 数据上端到端跑通全部 5 注**，每注都过官方预检。
- `--supports-json` 共识通路 — 已跑通并逐位复现 gate6。
- 7 树门槛标定 — 已实测（k=4 → 65 条 / k=5 → 33 条）。

---

## 3. 🔴 本次审视新发现并已修复的 4 个问题

### P0（会酿成大错）`build_testB_manifest.py` 会把全量数据当成 testB

`derive_list()` 原本 glob **整个** `JPEGImages`。实例上那是 71 训练 clip + 9 个 testA clip + 17 个
`_hflip` 目录。平台若不提供官方列表（走 fallback），它会产出 **约 9700 行的"testB"清单**，
把训练集和 testA 全算成 testB → 推理跑几小时、产出完全错误的包，而且**不报错**。
`_hflip` 目录还会被当成新 clip。

**已修**：testB 定义为「JPEGImages 中减去 train/testA manifest 已知 clip、再排除 `_hflip`」的补集；
并加硬校验——未见 clip 数 ≠ `--expect-clips`（默认 10）就**拒绝运行并列出名字**。
**已合成测试**：11 个未见 clip 时按预期拒绝并点名 `testAclipX`；排除后正确产出 1000 行 / 10 clip；
`_hflip` 从未入选。

### P0（会误提交）预演留下了假的 testB 制品

昨晚预演生成了 `data/processed/manifest_testB.jsonl`（内容是 **testA** 清单）与
5 个名为 `submit_testB_shot*`、内容却是 **testA** 预测的包。留着的话 9/16 有被当成真包误交的风险。

**已清**：假 manifest 与 5 个假包已删除；预演目录已重命名为
`outputs/_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/`（删除被安全守卫拦下，改名是安全结果）。

### P1（会让本地建包缺料）manifest 没随 bundle 回传

manifest 在实例上生成，但本地建包脚本也要读它。**已修**：bundle 现包含
`data_processed/manifest_testB.jsonl` 与 `.list.txt`，本地脚本自动安装并校验行数 ≥1000。

### P2（不严谨构造）shot5 对独立模型误用了 span80

`span80` 是针对**共识平均产生的残线桩**验证的（9/13：新增线里 29.4% 是平均伪影）。
cut400 是独立模型，它的短线是自己的输出、不是合并伪影 → 在没上过榜的配方上多加一个未经测量的
删除赌注（−43 条）。**已改**：shot5 = cut400 + trim，纯recipe，与 A 榜标定时测的形态一致。

---

## 4. 必须说清的风险与未知（不粉饰）

1. **所有新候选都没有实测分数。** testA 无 GT、A 榜已关。排序依据是
   **机制强度 + 方差 + 相对印证率**，不是分数。
2. **相对印证率本身也失败了它自己的校验**：用 9 树 ≥7/9 造伪真值，incumbent 命中 92.7% 而真实率仅 80.3%
   → 反推召回 **1.154 > 1**，说明"模型共识"里混着系统性假线（第三次独立印证）。
   绝对 r 值与 ΔF1 估计已作废，只有相对排序可用。
3. **testB 形态未知但大概率同 testA**（A 榜 9 段、B 榜 10 段，同出 19 段留出视频）。
   若如此：trim 只值 +0.07pp，全部赌注落在模型差异上，**期望提升量级 +0.1~0.3pp**。
4. **我方 0.73574 距 A 榜实测门槛 R3 0.81604 / 榜首 0.82389 差约 8pp**——这是建模差距，
   后处理补不回来。B 榜目标应是「稳住 0.73x 并吃到 0.2~0.3pp 的小杠杆」，不是冲前三。
5. **`build_testB_manifest.py` 只在合成数据上验证过**（真实 testB 目录不存在）。9/16 首次真跑时，
   若 clip 数不是 10 它会拒绝并打印实际数量——这是设计行为，按提示核对即可。
6. 实例已续费、GPU 空闲、磁盘 24G/50G、代码同步至 `7c7e9c7`（本轮修复将另提交）。

---

## 5. 9/16 执行计划（B 榜，6 发取 max）

**前置（需你）**：把 testB 图像从平台下载并传到实例（`/hy-tmp/datasets/HardLane/Lane/JPEGImages/`）。
浏览器自动化在本环境不可用，这一步必须由你完成，或由你提供直链。

```
# 1) 实例（约 20 分钟）
bash scripts/autodl/run_testB_infer.sh /hy-tmp/datasets/HardLane/Lane
# → 打印 f<450 并给出该用哪个 margin（40 或 0），产出 testB_bundle.tgz

# 2) 本地（约 15 分钟）
scp -i ~/.ssh/lane_id -P 59725 root@i-1.gpushare.com:/hy-tmp/testB_bundle.tgz ./
bash scripts/build_testB_candidates.sh ./testB_bundle.tgz
# → 5 个包 + 绝对路径 + ≤50 字备注
```

**注序（已按全部证据定稿）**：

| 注 | 构造 | 依据 | 风险 |
|---|---|---|---|
| 1 | `54ep conf0.50 + trim(M)` | 现役配方，地板 | 极低 |
| 2 | `cons_gate6`（≥60% 同意）+ span80 + trim(M) | 纯增量 52 条、dropped 0；新增印证率 0.558 最高 | 低 |
| 3 | `uni_swa4_g6` + trim(M) | 拥有 swa4 全部几何；新增印证率 0.406 | 低 |
| 4 | train 型 → margin 对冲注；testA 型 → `conf0.55` | trim 强不对称（−0.069 / +2.25pp） | 中 |
| 5 | `cut400 conf0.35 + trim(M)` | 唯一未上过榜的配方 | 中 |
| 6 | 自适应：前 5 注最优者叠新变量 | 取 max | — |

⛔ **不投**：soupA/B/C（新增印证率 0.18–0.22，全候选最低且体量最大）、conf0.60、hires、共识 ≥2/10。

**每次提交前我会给你：包的绝对路径 + ≤50 字备注文案，你手动提交。**
（本环境浏览器自动化不可靠，反复让你扫码是已被批评过的体验事故，不再走那条路。）
