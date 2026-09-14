# B 榜作战 Runbook（2026-09-16 00:00 – 09-17 17:00，41h，6 发取 max）

> 所有命令都是**今天在实例上实测跑通过的**，直接照抄即可，不要临场改。
> 环境：实例 `i2b0715374400501416`，repo `/hy-tmp/lane-detection-challenge`，UnLanedet `/hy-tmp/UnLanedet`。
> ⚠️ **SSH host/port 随实例重建而变** —— 每次开机后从控制台「登录指令」重新取，禁止沿用旧值。

## 0. 发车前 5 分钟自检（缺一项就别发车）

```bash
# 0.1 实例活着 & GPU 空闲 & 磁盘够
ssh -p <PORT> root@<HOST> "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h /hy-tmp | tail -1"

# 0.2 实例 HEAD 必须等于本地 HEAD（代码同步 = git bundle，见 §1）
ssh -p <PORT> root@<HOST> "cd /hy-tmp/lane-detection-challenge && git rev-parse HEAD && git status --porcelain | grep -v '^??' | head"

# 0.3 三道 HEAD 绑定守卫（HEAD 一变就要按序重跑，否则 run_training.py 直接 exit）
#     注意：validate_run.py 有自己的 KNOWN_MODELS 名单，与 run_training.CONFIGS 是两份
ssh -p <PORT> root@<HOST> "python3 -c \"
import json
for p in ('/hy-tmp/lane-outputs/weight_probe.json','/hy-tmp/lane-outputs/smoke/dataloader_loss_smoke.json'):
    d=json.load(open(p)); print(p, d.get('status'), d.get('project_git_head'))\""
```

## 1. 代码同步（本地 → 实例，**用 bundle，不要 pull**）

```bash
# 本地
cd "/Users/seyonmacbook/WorkBuddy/恶劣场景下的车道线检测挑战赛"
git bundle create /tmp/hl.bundle <实例当前HEAD>..HEAD
scp -i ~/.ssh/lane_id -P <PORT> /tmp/hl.bundle root@<HOST>:/hy-tmp/bundles/hl.bundle.new
# 实例（mv 原子替换后再 fetch）
ssh -p <PORT> root@<HOST> "cd /hy-tmp/lane-detection-challenge && \
  mv /hy-tmp/bundles/hl.bundle.new /hy-tmp/bundles/hl.bundle && \
  git fetch /hy-tmp/bundles/hl.bundle HEAD && git merge --ff-only FETCH_HEAD && git rev-parse HEAD"
```

## 2. 拿到 testB 数据后，第一件事：建 manifest

```bash
# 实例上（或本地，manifest 只依赖 JPEGImages 目录）
python scripts/build_testB_manifest.py \
  --lane-root /hy-tmp/datasets/HardLane/Lane \
  --output data/processed/manifest_testB.jsonl
# 若平台给了官方列表：加 --official-list <testB.txt>
# 内置校验：100 帧/clip（10 clip × 100 = 1000 图）、路径安全、重复 id、文件存在
```

## 3. 推理：base + 10 棵支撑树（每棵 ≈1.5 min，共 ≈20 min）

```bash
ssh -p <PORT> root@<HOST> 'bash -s' <<'EOS'
set -u
export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/hy-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights
export HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs
export HARDLANE_PYTHON=/usr/local/miniconda3/envs/py39/bin/python
cd "$HARDLANE_PROJECT_ROOT"
# 每棵树的 run-dir + 输出名 + conf（testA 实测选点，testB 先照搬）
# ⚠️ 2026-09-15 校正：下面这份名单才是实例上真实存在的 run-dir。
#    旧版写的 seed43 / seed44 / clrernet_15ep 在实例上**没有** run-dir，照抄会直接失败。
#    可用的独立支撑树 = 7 棵（hires / cut400 / 36ep_s42 / s101 / s202 / s303 / clrernet36）。
while read -r run conf tag; do
  "$HARDLANE_PYTHON" scripts/autodl/infer_testA.py \
    --run-dir "/hy-tmp/lane-outputs/runs/$run" --split testB --conf-threshold "$conf" \
    --output-dir "/hy-tmp/lane-outputs/testB_$tag" --skip-if-complete
done <<'LIST'
all71_seed42_clrnet_r50_54ep            0.50 base54
all71_seed42_clrnet_r50_36ep            0.50 seed42
all71_seed101_clrnet_r50_36ep           0.50 seed101
all71_seed202_clrnet_r50_36ep           0.50 seed202
all71_seed303_clrnet_r50_36ep           0.50 seed303
all71_seed42_clrernet_r50_36ep          0.50 clrernet36
all71_seed42_clrnet_r50_cut400_36ep     0.35 cut400
all71_seed42_clrnet_r50_hires_36ep      0.40 hires
swa4_54ep_stage                         0.50 swa4
LIST
```

- `swa4_54ep_stage` 是**派生**自 base 的（同轨迹平均），**不要**把它算进共识的"独立票"
  —— 否则同意率是虚高的。它只作为第 2 注的独立候选模型用。
- 因此**独立支撑树 n = 7**。门槛按同意率定：50% → ≥4/7；60% → ≥5/7（`--min-support` 传 4 或 5）。
- t05（63 段血统，A 榜 0.73444）的 run-dir 在实例上未定位到 → **不进名单**，别临时去猜路径。

## 4. trim 决策（**无需 GT，纯几何**）

```bash
# 统计 testB base 预测的 top 分布
python - <<'PY'
import glob, statistics
tops=[]
for f in glob.glob("/hy-tmp/lane-outputs/testB_base54/testB/predictions/*/*.lines.txt"):
    for line in open(f).read().splitlines():
        if line.strip():
            ys=[float(t) for i,t in enumerate(line.split()) if i%2==1]
            tops.append(min(ys))
far=sum(1 for t in tops if t<450)/len(tops)
print("top p50=%.0f  top<450 占比=%.1f%%"%(statistics.median(tops), 100*far))
PY
```
**判据（2026-09-15 重写，见 `docs/night_optimization_20260915.md` §5）**：

先在 testB 预测上算 `f<450 = 起点 y < 450 的线占比`（ten clip 合计即可，也可逐 clip 看）：

| `f<450` | 判定 | 用哪个 margin | 理由 |
|---|---|---|---|
| **> 10%** | train 型几何 | **40** | 全局曲线 margin 0→0.7972 / 20→0.8110 / **40→0.8197** / 60→0.8177；40 比 0 高 **+2.25pp** |
| **≤ 10%** | testA 型几何 | **0** | 起点 ≈564 时 `gt_bottom_for_top(564)+40 = 752 > 719` → margin 40 是**空操作**，等于白丢 trim0 的 +0.069pp |
| 10% ~ 50% | 暧昧 | **两注对冲**（0 与 40 各交一注） | 判别量能定"有没有影响"，**定不了方向** |

⚠️ **不要用 `f<400` 或 top p10 当判别量**（2026-09-15 证伪）：
`v644768616_1_0_1085` 的 `f<400 = 0.000` 但 margin spread **+25.81pp**；top p10 = 419 同时出现在
spread 25.81pp 和 0.00pp 的 clip 上。`f<450` 是唯一能把"spread 必为 0"的那一类干净切出来的量。

⚠️ **逐 clip 选 margin 做不到**：71 clip 里 70 个的个体最优 ≠ 全局最优，且 train 型 clip 内部
存在**方向相反的两派**（`v546797496_*`/`v644768616_*` 要 0，`v566817042_1_0_{299,349,464,533,633,728}`/`v777679069_*` 要 60），
单 clip 内差距可达 **+50pp**。逐 clip oracle 上界仅 +1.90pp 且无法用无 GT 特征预测 → **放弃逐 clip 选择，用全局 40**。

## 5. 建共识并集 + 过滤 + 打包（**一注 ≈3 min**）

支撑树列表在脚本里是**硬编码 testA 路径**的，B 榜必须自己传一份（否则会去 testA 目录找），写成 JSON：

```bash
cat > /hy-tmp/testB_supports.json <<'JSON'
[
 {"name": "seed42_36ep",  "path": "/hy-tmp/lane-outputs/testB_seed42/testB/predictions"},
 {"name": "seed101_36ep", "path": "/hy-tmp/lane-outputs/testB_seed101/testB/predictions"},
 {"name": "seed202_36ep", "path": "/hy-tmp/lane-outputs/testB_seed202/testB/predictions"},
 {"name": "seed303_36ep", "path": "/hy-tmp/lane-outputs/testB_seed303/testB/predictions"},
 {"name": "clrernet_36ep","path": "/hy-tmp/lane-outputs/testB_clrernet36/testB/predictions"},
 {"name": "cut400_36ep",  "path": "/hy-tmp/lane-outputs/testB_cut400/testB/predictions"},
 {"name": "hires_36ep",   "path": "/hy-tmp/lane-outputs/testB_hires/testB/predictions"}
]
JSON

python scripts/build_testA_consensus_union_20260913.py \
  --base /hy-tmp/lane-outputs/testB_base54/testB/predictions \
  --dst  /hy-tmp/testB_consensus_g4 \
  --min-support 4 --average --supports-json /hy-tmp/testB_supports.json
# 只滤"新增的"短残线（--base 必给，否则会砍掉 base 自带的短线）
python scripts/filter_short_lanes.py \
  --src /hy-tmp/testB_consensus_g4 \
  --base /hy-tmp/lane-outputs/testB_base54/testB/predictions \
  --dst /hy-tmp/testB_consensus_g4_f80 --min-span 80
# canonicalize + pack + verify + 官方预检
python src/submit/prepare_submit.py \
  --raw-pred-dir /hy-tmp/testB_consensus_avg_k2_f80 --canonical-dir /tmp/canonB \
  --out-zip outputs/submit_testB_consensus_k2_f80.zip \
  --manifest data/processed/manifest_testB.jsonl \
  --report outputs/reports/prepare_submit_testB_consensus_k2_f80.json
python src/eval/official_oracle/check_submission.py \
  --zip_path outputs/submit_testB_consensus_k2_f80.zip --list_path /tmp/testB_list.txt
```
（`/tmp/testB_list.txt` 由 manifest 生成：`'/'+image_path` 每行一条。）
支撑树缺失时脚本会直接报 `missing support trees: [...]` 并退出，不会静默跑错。
`--supports-json` 通路已于 2026-09-13 实测（传不存在的路径会正确报缺失）。

**🔴 `--base` 路径必须与 `--src` 的 rel 结构逐层对齐**（2026-09-15 真事故）：
把 `outputs/testA_54ep_raw`（多一层 `testA/predictions/`）传给
`filter_short_lanes.py --base` 时，`base / rel` 全部落空 → 打印 `base_kept=0`，
于是**每一条线都被当成"新增"**，短残线过滤把 incumbent 自己的 136 条线也删了
（2730 → 2594，比 incumbent 的 2664 还少 70 条）。
脚本现已加硬守卫：`base_missing > 0` 或 `base_kept == 0` 都会 **拒绝运行**并提示去查多出的目录层级。
B 榜跑完过滤后**务必看一眼 `base_kept` 是否 ≈ base 的线数**，不是就直接停手。

**参考值（testA gate6 实测）**：`base_kept=2669 added_kept=48 added_dropped=13`
→ 过滤后 2717 线 → 打包 2717 线。gate6 = 10 棵支撑树 + `--min-support 6`（60% 同意）。

## 6. 六注怎么排（v2，2026-09-15 夜班重写）

证据：`docs/night_optimization_20260915.md`（近失配取证 + 平滑轴关闭 + trim margin 逐 clip 标定）。
9/14 实测：共识并集（≥2/10）**−0.145pp**、hires 1366×540 **−1.542pp**。

**发车前先跑 §4 的 `f<450` 判据定 margin（M ∈ {0, 40}）。**

**最终台账（2026-09-15，`scripts/profile_night_candidates_20260915.py`，相对 incumbent 2664 条线）**：

| 候选 | 线数 | kept | dropped | novel | 说明 |
|---|---|---|---|---|---|
| incumbent | 2664 | — | — | — | 地板 |
| **uni_swa4_g6** | **2733** | **2664** | **0** | **69** | ⭐ **严格支配 swa4**（几何取 swa4，再叠 gate6 增量） |
| cons_gate6 | 2717 | 2664 | 0 | 52 | 纯增量，下行零 |
| swa4 / swa7 / swa3 | 2662 / 2656 / 2649 | 2635 / 2628 / 2629 | 29 / 36 / 35 | 27 / 28 / 20 | SWA 几何（台账看不见"线被轻移"的收益） |
| conf55 | 2595 | 2595 | 69 | 0 | 纯删除 |
| soupA / B / C | 2795 / 2828 / 2816 | 2625 / 2613 / 2616 | 39 / 51 / 48 | 170 / 215 / 200 | ⛔ 不投 |

| 注 | 构造 | 依据 | 风险 |
|---|---|---|---|
| **1（主注）** | `54ep(model_best) conf0.50 + trim(M)` | 现役配方，地板 | 极低 |
| **2** | **`uni_swa4_g6` + trim(M)** | ⭐ 见 §6.2；拥有 swa4 全部几何 + 42 条经 60% 同意率筛选的线 | 低 |
| **3** | `cons_gate6`（≥60% 同意）+ span80 + trim(M) | 纯增量 52 条，下行零；估 +0.06~+0.15pp | 低 |
| **4** | 若 train 型 → **margin 对冲注**（0 与 40 取另一个）；若 testA 型 → `conf0.55 + trim0` | §4 不对称（下界 −0.069pp / 上界 +2.25pp）；conf0.55 纯删除 69 条、估 +0.16pp | 中 |
| **5** | `cut400 conf0.35` + trim(M) + span80 | 唯一未上过 A 榜的配方 | 中 |
| **6** | 自适应：前 5 注最优者叠一个新变量 | 取 max | — |

**已验证可打包（2026-09-15 全部通过 `prepare_submit` + 官方 `check_submission`）**：
`outputs/submit_testA_night_{conf55,conf60,cons_gate6_v2,dryrun_g6,uni_swa4_g6,soupA,soupB,soupC,swa3,swa4,swa7}_m0.zip`。
B 榜同构造只需把 `--src` 换成 testB 预测树重跑同一条命令。
`dryrun_g6` 与 `cons_gate6_v2` 字节级同构（同为 2717 线 / 80235 点），
证明 **§5 的 `--supports-json` 通路已被实际执行验证过**（不只是文字描述）。

### 6.1 ⭐ 为什么第 2 注给 SWA（同轨迹权重平均）

实例证据显示 **54ep 的 val 早就越过峰值**：

- 峰值 iter **24863** → F1 **0.89565**；final iter 31968 → F1 **0.89429**（已回落 0.136pp）
- 现役 incumbent 用的正是 `model_best.pth`（iter 24863），**没有**更晚的更优点

沿**同一条轨迹**平均 `{24863, 29007, 29599, 30191, 30783, 31375, 31967}`（脚本 `scripts/autodl/make_soup.py`）
与跨种子 soup 的关键区别：**保证在同一 basin 内**，崩掉风险极低（跨种子 soup 输出 2795–2828 线，多 5%，方差明显更大）。
实例上已构建 `swa3_54ep / swa4_54ep / swa7_54ep` 三个权重并跑完 testA 推理（2650 / 2663 / 2657 线）。

### 6.2 ⭐ 为什么第 2 注是 union 而不是 swa4 本身

`scripts/build_union_pair_20260915.py --a <swa4_trim> --b <gate6_trim> --dst ...`

- A 侧把 swa4 的 2662 条线**原样保留**（含 SWA 带来的轻微位置移动 —— 这部分收益
  containment 台账看不见，因为线的身份没变，只是位置变了）。
- B 侧再叠 gate6 相对 swa4 新增的 71 条。
- 由于 gate6 含**全部** incumbent 线，swa4 丢掉的那 29 条会被自动补回 → 最终 `dropped=0`。
- 结果 `kept=2664 / dropped=0 / novel=69`：**拥有 swa4 的全部几何，还多 42 条经 60% 同意率筛选的线**
  → **严格支配 swa4**，所以不该再单发 swa4。

testA 实测：`union: a_lanes=2662 added_from_b=71 b_lanes_already_in_a=2646 total=2733`。

**§4.5 预测几何剖面自检（B 榜当天必做，不花额度）**：把 testB 的 base 预测与已知基线剖面对比 ——
基线 testA 剖面为 `线数 2664 / 空图 9 / top p5-p50-p95 = 549-564-624 / span = 95-155-170 / bottom=719 占 95.8%`。
若某模型的 top 中位数偏离 564 很多（例如 <450），说明它学到的是 train 型几何 → **§4 的 trim 判据要按它的分布重算**，不能照搬。
（2026-09-14 实测：cut400 的 top 中位数仍是 564，**"裁剪面框住地平线"这个假设不成立** —— 模型是从图像内容自己找地平线的。分辨率/cut_height 轴到此可以正式关掉。）

**门槛按"同意率"设，不是按"票数"设**。tick 分布是**双峰**的（本地 5 棵支撑：1 票 69679 / 2 票 15501 / 3 票 769 / 4 票 320 / 5 票 136），加支撑树的**数量**只会让"≥k 票"更松。

**收益量级（务必放对预期）**：`ΔF1 ≈ n × 2.82e-5`（n = 新增条数，真线率≈0.45 时）。testA 上正确门槛只能捞 **96 条（≥5/10）/ 66 条（≥6/10）** → 约 **+0.28pp**。
→ **共识是 +0.2~0.3pp 的小杠杆；trim 是 +4pp 级的大杠杆。额度优先给 trim 的几何判定（不花 GPU、不需要 GT）。**

**⛔ 明确不投（2026-09-15 定价，见 `docs/night_optimization_20260915.md` §6.2）**：

- **跨种子 soup A/B/C**。containment 台账显示它们本质是"删 39–51 条 + **新增 170–215 条**"，
  与 9/14 已实测为负的共识加线**同一型**（目标域同类新增实测真线率 0.3414 < 盈亏线 0.3675）。
  代入定价式 → soupA **−0.15pp**、soupB **−0.19pp**。换了件衣服的同一注，不交。
- **conf0.60**：纯删除 129 条，含质量更高的 0.55–0.60 带（r 估 0.40+）→ 倾向负。
- **hires**（−1.54pp）、**共识门槛降到 ≥2/10**（−0.145pp）。

**候选形态速查（相对 incumbent 2664 条线）**：

| 候选 | kept | dropped | novel | 形态 | 定价 |
|---|---|---|---|---|---|
| conf55 | 2595 | 69 | 0 | 纯删除 | 估 **+0.16pp**（r 估 0.30） |
| swa4 / swa7 | 2635 / 2628 | 29 / 36 | 27 / 28 | **近乎纯重定位** | 低方差，机制最优 |
| soupA / B / C | 2625 / 2613 / 2616 | 39 / 51 / 48 | 170 / 215 / 200 | 加线为主 | **−0.15 ~ −0.19pp** |
- **每一注都必须能交出字节一致的 solution.zip**（§7），否则赢了也交不出包。
- 榜单取历史最优 → 六注之间是"免费期权"，**先交最稳的，把高方差的放后面**。

## 7. solution.zip 冻结（B 榜发布前必须完成）

- 内容：完整代码 + 权重 + 配置 + 依赖说明；报 **64 位 SHA-256 + 精确字节数**
- 无官方体积上限（<200MB 只约束预测包），权重必须内嵌
- **每注单独冻结一套**，记录 `SHA-256 / bytes / 权重来源 run-dir`
- 前三名要上传**字节完全一致**的包 → 冻结后不要再 `pip install`/改文件

## 8. 铁律（每一条都有前科）

1. **测试集禁训**：伪标签、测试域适配、BN 适配、跨帧 —— 全禁（规则明示）。
2. **成绩只认冻结 Oracle 全局单次调用**；`per-clip` 禁止平均（逐图均值偏离全局 −0.99pp）。
3. **提交一律走 `prepare_submit.py`**（manifest 枚举 → canonicalize 1 位小数 → pack → verify）+ 官方 `check_submission.py` 预检；**不自动交，交前用户确认**。
4. **一次提交 = 1 方程 2 未知数**（`F1 = 2TP/(P+G)`）。要分解 TP/FP/FN 只能用 junk 探针（`docs/a_board_decompose_probe_20260913.md` §2 公式，已两次逐位验证）。
5. **加线盈亏线 = F1/2；删线盈亏线 = 1 − F1/2**（与 P/G 无关，见 `docs/union_true_rate_20260913.md`）。任何"多检/少检"先过这一关。
6. **本地尺子的边界**：`v1_seed42` 的 val 是 clip-level，video 级 100% 泄漏；新消融必须先问"是否 video-disjoint"。本地 OOF 结论**必须**用「GT top≥530 的 12 clip 子集」复核后才可迁移到测试域（testA 与 train 几何不同）。
7. **实例发车三道守卫**（HEAD 一变就重跑）：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型名登记进 `run_training.WEIGHT_BASE_MODEL` **和** `validate_run.KNOWN_MODELS`。两个脚本必须 `cd /hy-tmp/UnLanedet` + `PYTHONPATH=/hy-tmp/UnLanedet`。
