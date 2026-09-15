# B 榜行动清单（2026-09-16 晨）— 先看第 0 节，它是唯一阻塞项

> 生成：2026-09-15 夜间（北京时间）。B 榜窗口 **9/16 00:00 – 9/17 17:00**，6 发取 **max**，定最终名次。
> 与 `docs/runbook_testB.md` 冲突时**以本文为准**（本文含 9/15 夜间的三处修订）。

---

## 0. 🔴 唯一阻塞项：testB 图像还没到实例上

实测（9/15 23:10）：实例 `/hy-tmp/datasets/HardLane/Lane/JPEGImages` 仍是 **97 项**
= 71 训练 clip + 9 testA clip + 17 个 `_hflip` 增强目录。**testB 的 10 个 clip 未上传。**

官方规则写"测试集 B：10 段 / 1000 张，末期发布"，所以图要**你从平台手动下载**。

**要做的两件事**（顺序不能反）：

```bash
# 1) 本地：把下载好的 testB 传到实例（保持 JPEGImages/<clip_id>/<frame>.jpg 层级）
scp -i ~/.ssh/lane_id -P <PORT> testB.zip root@<HOST>:/hy-tmp/

# 2) 实例：解压进 JPEGImages
ssh -i ~/.ssh/lane_id -p <PORT> root@<HOST> \
  "cd /hy-tmp/datasets/HardLane/Lane/JPEGImages && unzip -q /hy-tmp/testB.zip && ls | wc -l"
# 期望：97 + 10 = 107（若多了 _hflip 目录也算正常，manifest 脚本会排除）
```

- **SSH 连接串随实例重建而变**：每次开机从控制台「登录指令」重取，禁止沿用旧值
  （9/15 用的是 `root@i-1.gpushare.com:59725`）。
- 图没到位之前，**后面所有候选都交不出去**——不是代码问题，是没数据。

---

## 1. 图到位后：两条命令出全部 5 注

```bash
# 实例（10+1 次推理，约 25 min）→ 产出 testB_bundle.tgz
ssh -i ~/.ssh/lane_id -p <PORT> root@<HOST> 'bash -s' < scripts/autodl/run_testB_infer.sh

# 本地（自动定 margin → 建 5 注 → 逐个官方预检 → 打印绝对路径）
bash scripts/build_testB_candidates.sh <下载回来的 testB_bundle.tgz>
```

脚本会打印每注的**绝对路径 + 线数 + 预检结果 + ≤50 字备注**，交哪一注由你拍板。

---

## 2. 六注怎么排（9/15 夜间修订版）

| 注 | 配方 | 形态 | 依据 |
|---|---|---|---|
| 1 | `54ep model_best conf0.50 + trim(M)` | 地板 | incumbent，**0.73574**（A 榜实测） |
| 2 | **`uni_swa4_g6`** | kept 2664 / **dropped 0** / novel 69 | 严格支配 swa4；相对印证率 0.406（全场第二） |
| 3 | `cons_gate6`（7 树 ≥4 票 + span80） | kept 2664 / dropped 0 / novel 52 | 纯增量，印证率 0.558（全场第一） |
| 4 | testA 型 → `conf0.55 + trim0`；train 型 → margin 对冲 | 纯删除 69 条 | 估 +0.16pp |
| 5 | **`uni_occlude` / `uni_s101_54ep`** | kept 2664 / dropped 0 / novel 119 | **替换原 cut400** |
| 6 | 自适应 | — | 看前 5 注结果再定 |

**修订 1（重要）**：第 5 注原为 `cut400 conf0.35`，但它在 A 榜**实测 −0.772pp**。
B 榜是取 max，已知负数不该占名额 → 改成 dropped=0 的纯加注。
`build_testB_candidates.sh` 已内置优先级：`testB_s101_54ep` > `testB_occlude` > `cut400` 兜底。

**修订 2**：`run_testB_infer.sh` 的推理名单加了 `all71_seed42_clrnet_r50_occlude_36ep`（+1 次推理，+1.5 min）。

**修订 3**：`f<450` 决定 trim margin M∈{0,40}。testA 实测 `f<450 ≈ 0` → M=0，trim 基本是空操作；
若 testB 是 train 型（`f<450 > 10%`）→ M=40，**上界 +2.25pp**（OOF 实测）。脚本自动判定。

---

## 3. 夜间已完成的事（无需你处理）

- **遮挡增强实验结题 = 零效应 NULL**（不是有害）。跑了 4 个同排期对照后确认：
  occlude 的 `dropped=135`，而 36ep 对照组是 104–175（均值 136）→ 与换随机种子无法区分。
- **修正了一个会误导后续所有实验的方法论错误**：台账"dropped > 51 → MISS"的筛子此前被误标定——
  swa/soup 是 54ep 轨迹的**派生**模型，dropped 天然小；**独立新训练的模型天然有 dropped ∈ [104,175]**。
  推论：36ep 新模型**单独交必 ≈ −2pp**，只有 union（dropped=0）形态能用。
  详见 `.workbuddy/memory/2026-09-15.md` §19。
- **GPU 在跑** `all71_seed101_clrnet_r50_54ep`（UTC 15:47 发车，ETA 北京 ~02:35）。
  目的：拿到**同 54ep 排期**的独立模型，让 union 的增量线具备 54ep 质量。
- **新候选已打包并通过官方预检**：`outputs/submit_testA_night_uni_occlude_m0.zip`（2783 线，dropped=0）。
- **新工具** `scripts/profile_raw_candidate.py`：原始预测树直接出台账 + 几何画像。

---

## 4. 提交纪律（不变）

- 一律**交包 + 备注，由你手动提交**；我绝不自动交，也不反复开浏览器。
- 每注交前都过 `prepare_submit.py` + 官方 `check_submission.py`。
- ⚠️ 本地有个 `_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/` 目录，内容是 **testA**、名字像 testB，**禁止提交**。
