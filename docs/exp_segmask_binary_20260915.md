# 实验记录：segmask binary_union（2026-09-15）

> 单变量实验。分支 `exp/segmask-binary`，tag `exp-segmask-binary-start`（commit `a9c8d3f`）。
> 冻结基线：tag `incumbent-0.73574`（`86adb29`）。

## 1. 假说

| | |
|---|---|
| **观察** | testA 精度 0.804 / 召回 0.678；榜首量级需精度 0.900 / 召回 0.760 → **召回是绑定短板**（FN 1015 条 ≈ FP 的 1.9 倍） |
| **假说** | 辅助分割监督从**实例级**（palette id 1..8）改成**二值车道/背景**，会让模型更强调"车道在哪里"而非"哪条是哪条"，从而在遮挡/弱光处多开火 |
| **变量** | `seg_mask_mode`: `overflow_background` → `binary_union`（唯一改动） |

## 2. 前置否证（诚实记录）

造配置时的原始理由是"默认模式会把超预算的 palette id 映射成背景、丢掉真实车道像素"。
**全量普查 7100 张掩膜后该理由基本不成立**：

- `num_classes = max_gt_lanes + 1 = 9`（`max_gt_lanes = 8`）
- 实测 max palette id 最高到 **10**，但**溢出像素只占 0.206%**（84468 / 40,975,105）
- 抽样 2280 张里只有 **17 张**（0.75%）存在任何溢出

→ 所以本实验真正在测的是**监督语义**（二值 vs 实例级），不是 id 预算损失。假说强度：中偏弱。

## 3. 发车记录

```
model      clrnet_r50_segmask_binary
run-name   all71_seed42_clrnet_r50_segmask_binary_36ep
epochs 36  seed 42  iters/epoch 592  eval-every-epochs 6  max-to-keep 6
manifest   /hy-tmp/lane-outputs/experiments_manifest_train_all71.jsonl
```

**三道守卫（HEAD 变更后必跑）全部通过**：

| 守卫 | 结果 |
|---|---|
| `probe_weights.py --no-download` | ok（10s） |
| `smoke_dataloader_and_loss.py` | ok（24s） |
| 三处登记 `CONFIGS` + `WEIGHT_BASE_MODEL` + `KNOWN_MODELS` | ✓（2/2/1 处命中） |

**实测吞吐**：iter 519 时 **0.337 s/iter**、max_mem 3592M、**ETA 1:56:40**（03:45 发车 → 约 **05:47 完成**）。
`seg_loss` 0.067 且正常下降 → 二值编码确实生效。

日志 `/hy-tmp/lane-outputs/exp_segmask_binary_20260915.log`，
状态 `/hy-tmp/lane-outputs/exp_segmask_binary_20260915.status`。

## 4. 判据（先写死，避免事后找理由）

拿不到诚实本地分（full71 模型 val 被污染），所以只有两条判据：

1. **A 榜若仍独立存活**（3 发/日）：训练完 → testA 推理 → **发 1 发**测真分。
   - 判为**有戏**：≥ incumbent 0.73574 + 0.30pp（超出 ±0.27pp 噪声带）
   - 判为**无戏**：≤ 0.73574
   - 落在带内：不采信，作为 B 榜探索位
2. **A 榜不可用**：直接作为 B 榜的一个探索位（6 发取 max，负也不亏）。

## 5. 复现与回滚

```bash
git checkout exp-segmask-binary          # 实验分支
git checkout incumbent-0.73574           # 回滚到冻结基线
```

- 回滚**不需要重建任何东西**：incumbent 的权重、包、配方卡都在（`docs/incumbent_recipe_card_20260915.md`）。
- 若实验被否：`git checkout 主干 && git branch -D exp/segmask-binary`（先确认没别的东西挂在上面）。

## 6. 版本管理现状（2026-09-15）

| 项 | 状态 |
|---|---|
| 本地 git | 分支 `exp/segmask-binary`；tags `incumbent-0.73574`、`exp-segmask-binary-start` |
| 全历史 bundle | 已存 **实例** `/hy-tmp/backups/lane_full_20260915.bundle`（1.3MB，`git bundle verify` 报 "complete history"）；实例上 tag 可见 |
| GitHub 推送 | **本沙箱不可用**——出网被拦（`CONNECT tunnel failed, response 502`），回落到 SSH 会卡 passphrase。**需要你手动推**：<br>`git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push origin --all --tags` |
| 已知盲区 | 本地有 **23 个未跟踪文件**（含 `scripts/*.py`、`docs/*.md`、以及 `AGENTS.md`/`SOUL.md` 等 agent 配置）。未跟踪文件**不会被"工作树干净"守卫拦住**，也不在任何备份里（bundle 只含已提交内容）。建议明确：哪些该入库、哪些该 ignore。 |
| 实例历史遗留 | 2 个未纳入 git 的配置 `clrnet_r50_hardlane_ms720.py` / `ms880.py`，来源不明、无实验记录 |

---

# 结果（2026-09-15 15:20）：**MISS**

训练 `complete`（03:45 → 05:51 UTC ≈ 2h06m，比 ETA 略长）。

## 1. 污染 val 说"小幅变好"

| run | best val F1 | final |
|---|---|---|
| 基线 36ep | 0.88302 | 0.88170 |
| **segmask 36ep** | **0.88430** | 0.88298 |
| 基线 54ep | 0.89565 | 0.89429 |

→ segmask 比同协议基线高 **+0.128pp**。但 val 是污染数，**不可采信**（cut400 那次 val +1.05pp 却对应 testA −0.772pp）。

## 2. containment 台账说"明确变差"——而且方向与设计意图相反

| 候选 | kept | dropped | novel | 空图 |
|---|---|---|---|---|
| incumbent | — | — | — | **9** |
| **segmask36** | 2460 | **204** | 98 | **30** |
| cut400（实测 −0.772pp） | 2509 | 155 | 181 | 10 |
| soupA（实测 −0.379pp） | 2625 | 39 | 170 | 17 |
| swa4 | 2635 | 29 | 27 | — |

- **dropped=204 是全部候选里最大的删除量**，空图从 9 张涨到 **30 张**。
- 即：模型**变稀疏了**。实验意图是"冲召回"，结果是**召回下降**——机制层面的直接否证，不需要分数。
- 定价（假设新增 98 条 r_a=0.40）：`r_d=0.45 → −0.47pp`；`r_d=0.55 → −1.17pp`；`r_d=0.65 → −1.87pp`。
  只有 `r_d < ~0.40` 才为正，而已知基座自带的线约 80% 为真 → **几乎必然为负**。

## 3. 判定与动作

- **判定：MISS。机制被否证**（辅助分割从实例级改二值 → 模型更保守/更稀疏，与召回目标相反）。
- **动作：不发 A 榜额度测它。** 预先写死的判据是"≤ incumbent 为无戏"，台账已给出清晰先验（大概率 −0.5 ~ −1.9pp），
  花一发给一个已知方向为负的候选是浪费；额度留给出 B 榜前的其他用途。
- 包已备好但**标记为不投**：`outputs/submit_testA_night_segmask36_m0.zip`（2558 线，过官方预检）。

## 4. ⭐ 方法论收获（比实验结果更值钱）

**污染 val 与 containment 台账指向相反，而台账是对的。**

- 对一个新的 full71 基座模型，**台账的 `dropped` 数是最便宜的、形状上诚实的早期筛选量**：
  它直接回答"这个模型相比现役是更激进还是更保守"，而污染 val 完全掩盖了这一点。
- 因此：**任何新基座模型，先算台账再考虑花额度**。台账成本 ≈ 2 分钟 CPU，判据是"dropped 是否大幅超过既有候选"。

## 5. 旁路发现（记录，不入库）

`run_training.EXPERIMENT_OVERRIDE_KEYS` **已经包含 `model.head.cfg.seg_mask_mode`** →
本次实验本可以走 `--experiment-override` 而不必新建配置文件。下次单变量配方实验优先用 override，更轻。
另：`candidate_topk` → `test_parameters.nms_topk`（推理期参数）**不在白名单**；
但我们的输出只有 2.96 线/图而 topk=12，**topk 几乎不可能是绑定约束**，不投。
