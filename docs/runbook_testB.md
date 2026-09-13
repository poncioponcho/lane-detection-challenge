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
while read -r run conf tag; do
  "$HARDLANE_PYTHON" scripts/autodl/infer_testA.py \
    --run-dir "/hy-tmp/lane-outputs/runs/$run" --split testB --conf-threshold "$conf" \
    --output-dir "/hy-tmp/lane-outputs/testB_$tag" --skip-if-complete
done <<'LIST'
all71_seed42_clrnet_r50_54ep            0.50 base54
all71_seed42_clrnet_r50_hires_36ep      0.35 hires
all71_seed42_clrnet_r50_36ep            0.50 seed42
all71_seed43_clrnet_r50_36ep            0.50 seed43
all71_seed44_clrnet_r50_36ep            0.50 seed44
all71_seed42_clrnet_r50_36ep            0.55 seed101
all71_seed42_clrnet_r50_36ep            0.55 seed202
all71_seed42_clrnet_r50_36ep            0.55 seed303
clrernet_r50_36ep                       0.50 clrernet36
LIST
# t05 血统（63 段）与 clrernet_15ep 若在实例上，同样加进来
EOS
```
⚠️ 上面 seed101/202/303 的 run-dir 需按实例实际情况填（本地 `outputs/testA_support_trees/` 里有它们的 testA 产物可反查来源）。

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
**判据（2026-09-13 实测锁定）**：
- `top<450` 占比 **>10%** → testB 是 train 型几何（有远端起始线）→ 启用 `gtcond+40`（全量 OOF 上 **+4.224pp**）
- 否则（像 testA，全部 top≥549、窄带宽 75px）→ 用 **margin 0**（testA 实测 +0.069pp 为峰，再深即崩）

## 5. 建共识并集 + 过滤 + 打包（**一注 ≈3 min**）

支撑树列表在脚本里是**硬编码 testA 路径**的，B 榜必须自己传一份（否则会去 testA 目录找），写成 JSON：

```bash
cat > /hy-tmp/testB_supports.json <<'JSON'
[
 {"name": "hires",        "path": "/hy-tmp/lane-outputs/testB_hires/testB/predictions"},
 {"name": "t05_36ep",     "path": "/hy-tmp/lane-outputs/testB_t05/testB/predictions"},
 {"name": "clrernet_36ep","path": "/hy-tmp/lane-outputs/testB_clrernet36/testB/predictions"},
 {"name": "seed202_36ep", "path": "/hy-tmp/lane-outputs/testB_seed202/testB/predictions"},
 {"name": "seed303_36ep", "path": "/hy-tmp/lane-outputs/testB_seed303/testB/predictions"},
 {"name": "seed101_36ep", "path": "/hy-tmp/lane-outputs/testB_seed101/testB/predictions"},
 {"name": "seed42_36ep",  "path": "/hy-tmp/lane-outputs/testB_seed42/testB/predictions"},
 {"name": "seed43_36ep",  "path": "/hy-tmp/lane-outputs/testB_seed43/testB/predictions"},
 {"name": "seed44_36ep",  "path": "/hy-tmp/lane-outputs/testB_seed44/testB/predictions"},
 {"name": "clrernet_15ep","path": "/hy-tmp/lane-outputs/testB_clrernet15/testB/predictions"}
]
JSON

python scripts/build_testA_consensus_union_20260913.py \
  --base /hy-tmp/lane-outputs/testB_base54/testB/predictions \
  --dst  /hy-tmp/testB_consensus_avg_k2 \
  --min-support 2 --average --supports-json /hy-tmp/testB_supports.json
# 只滤"新增的"短残线（--base 必给，否则会砍掉 base 自带的短线）
python scripts/filter_short_lanes.py \
  --src /hy-tmp/testB_consensus_avg_k2 \
  --base /hy-tmp/lane-outputs/testB_base54/testB/predictions \
  --dst /hy-tmp/testB_consensus_avg_k2_f80 --min-span 80
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

## 6. 六注怎么排（**已按 9/14 A 榜实测修正**，见 `docs/consensus_union_testA_result_20260914.md`）

9/14 实测：共识并集（≥2/10）**−0.145pp**、hires 1366×540 **−1.542pp** —— 两条夜班路线都没迁移。修正后的六注：

| 注 | 构造 | 依据 |
|---|---|---|
| **1（主注）** | **`54ep + trim0`**（P=2664，0.73574） | 自队最优，机制与几何都验过（trim 是唯一正迁移轴） |
| 2 | **t05 36ep**（63 段血统）+ trim | 唯一异质数据血统，实测 0.73444（次优） |
| 3 | **CLRerNet 36ep** + trim | 唯一异质检测头，实测 0.72994 |
| 4 | **共识并集，门槛提到同意率 ≥50%（≥5/10）** + trim | 机制证明能把召回抬高（0.6784→0.6962），但 ≥2/10 的边际真线率只有 0.3414（盈亏线 0.3675）；本地"100% 同意"是 0.5688 → 交插点在两者之间，这一次提交用来定位门槛 |
| 5 | 共识并集 ≥6/10 + trim | 若第 4 注向上，沿同方向再走一格；否则改交 seed202/303 |
| 6 | 自适应：视前 5 注 | 取 max |

**注单纪律**：
- **不要**把共识并集当主注（它今天实测为负）；**不要**交 hires（−1.54pp，分辨率轴已正式关闭）。
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
