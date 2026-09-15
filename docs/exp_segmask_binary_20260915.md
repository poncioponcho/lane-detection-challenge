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
