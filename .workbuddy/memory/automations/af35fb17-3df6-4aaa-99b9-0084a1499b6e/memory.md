# 自动化执行记忆 — 夜间优化（s101_54ep 转候选）

## 2026-09-16 03:00（首次记录）

- SSH 串沿用 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id` → **连通**（实例未重建）。
  注意：本机无 `timeout` 命令，别在 ssh 前加。
- 训练已完成（`run_evidence.json status=pass`，iter 31967/31968，污染 val F1 0.89280）。
  testA 推理**上次夜班已跑过**（UTC 18:58），`--skip-if-complete` 幂等跳过，未重跑。
- 台账：dropped=145 / novel=96 / 2615 线 / 空图 21 → **54ep 同排期没有降低 dropped**，
  落在 36ep 对照带 [104,175] 正中 → 判 MISS（原假说被证伪）。
- union 2760 线已打包 + 官方预检通过：`outputs/submit_testA_night_uni_s101_54ep_m0.zip`。
- 定价 corr 0.167 → r_est 0.333 < 盈亏线 → −0.115pp → **建议不占 B 榜注位**，包仅留档。
- **🔴 testB 图像仍未上传实例（JPEGImages 97 项）**——B 榜唯一阻塞项，每轮都要复查并上报。
- 新脚本 `scripts/price_uni_s101_54ep_20260916.py`：只对含 novel 线的 rel 建伪真值，12 s 出定价，
  比 `calibrate_corroboration_20260915.py` 快一个量级，以后给单个候选定价优先用它。

### 下次可直接复用的命令

```bash
ssh -i ~/.ssh/lane_id -p 59725 -o BatchMode=yes root@i-1.gpushare.com \
  "ls /hy-tmp/datasets/HardLane/Lane/JPEGImages | wc -l"   # 97 = testB 未到位

PY=/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python
$PY scripts/profile_raw_candidate.py --src <tree>/testA/predictions --name X --margin 0
$PY scripts/build_union_pair_20260915.py --a outputs/testA_night_20260915/build/incumbent_tree \
    --b outputs/testA_night_20260915/build_X_m0 --dst .../build_uni_X_m0
$PY scripts/price_uni_s101_54ep_20260916.py     # 改常量 UNION/INCUMBENT 即可复用
```
