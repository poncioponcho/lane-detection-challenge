# B 榜守望自动化 — 执行纪要

## 2026-09-16 13:00（第 1 次轮询）
- 结果：**BLOCKED** — 实例 `/hy-tmp/datasets/HardLane/Lane/JPEGImages` 仍为 97 项，testB 的 10 个 clip 未上传。
- SSH 串（13:00 实测可用）：`ssh -i ~/.ssh/lane_id -p 59725 root@i-1.gpushare.com`，hostname `I2b07f5374400501416`。
- 幂等守卫状态：无 testB zip、无 done marker、无演练残留 → 需要继续轮询。
- 未执行任何推理/建包/训练。

## 2026-09-16 16:00（第 2 次轮询）
- 结果：**BLOCKED**（同上，第 9 次确认 97）。SSH 串与 hostname 未变，实例未重建。
- 额外排查：`/hy-tmp` 无任何 testB 命名文件；近 24h 新增压缩包只有 `testA_s101_54ep.tgz`、`solution_freeze_20260916_v4.tgz`
  → 结论仍是 testB **从未被下载过**，不是"传了没解压"。
- 幂等守卫通过，全流程未跑。剩余窗口约 25h。

## 2026-09-16 19:07（第 3 次轮询）
- 结果：**BLOCKED**（第 10 次确认 97）。SSH 串与 hostname 未变，实例未重建；近 8h 无新上传压缩包。
- 幂等守卫通过，全流程未跑。剩余窗口约 22h。

## 2026-09-16 22:09（第 4 次轮询）
- 结果：**BLOCKED**（第 11 次确认 97）。SSH 串 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id` 仍可用，hostname 未变（连续第 4 次同实例）。
- `/hy-tmp` 最新条目 = 9/16 02:54 `solution_freeze_20260916_v4.tgz`，19h 无新上传 → testB 从未下载（非"传了没解压"）。
- 幂等守卫通过，全流程未跑。剩余窗口约 19h。

## 下次执行要点
- 先查 `outputs/submit_testB_*.zip` 与 `outputs/state_testB_done.marker`，有则直接结束。
- SSH 连不上 → 报告「需用户从控制台重取登录指令」后结束，不要猜端口。
- JPEGImages > 97 才继续：`run_testB_infer.sh` → 拉 tgz → `build_testB_candidates.sh`。

## 2026-09-17 01:12（第 5 次轮询）
- 结果：**BLOCKED**（第 12 次确认 97）。SSH 串 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id` 仍可用，hostname 未变（连续第 5 次同实例）。
- `/hy-tmp` 最新条目仍是 9/16 02:54 `solution_freeze_20260916_v4.tgz`，22h 无新上传 → testB 从未下载。
- 幂等守卫通过，全流程未跑。剩余窗口约 15h48m（9/17 17:00 关闭）。
- ⏰ 已标注临界点：testB 需在 9/17 16:00 前到位（推理+回传+建包 ≈ 30 min）。

## 2026-09-17 01:43（第 6 次轮询）
- 结果：**BLOCKED**（第 13 次确认 97 / 排除 `_hflip` 后 80）。SSH 串 `root@i-1.gpushare.com:59725` + `~/.ssh/lane_id` 仍可用，hostname `I2b07f5374400501416` 未变（第 6 次同实例）。
- 幂等守卫通过（无 testB zip、无 marker、无演练残留）。
- 本轮产出（不等图也能做的最高价值项）：把「等图延迟」从最坏 3h 压到 ~1h。
  * 实例部署常驻触发器 `scripts/autodl/watch_testB_and_run.sh`（pid 25862），图到位即自动推理；三重自愈（3 次重试 + 300 s 复跑到 16:30 + 锁防双开）。
  * 本守望自动化 rrule 改为 `INTERVAL=1`，`validFrom 06:00` / `validUntil 17:00` → 白天每小时，夜里不空转。只保留这一个守望自动化，避免并发建包。
  * 已核实实例 HEAD `a480366` 含 AppleDouble 修复（`build_testB_manifest.py:86`），无需 bundle 同步。
- 提交：`307e17e`（watcher）、`d5566ab`（memory）。
- 下次执行要点：先查 `/hy-tmp/testB_bundle.tgz` 是否已存在（触发器可能已跑完）——在则直接拉包，跳过推理。

## 2026-09-17 02:04（第 7 次轮询）
- 结果：**BLOCKED**（第 14 次确认 97）。SSH 与 hostname 未变；触发器 pid 25862 存活。
- 幂等守卫通过。
- 本轮主要产出（与等图无关，可复用的知识）：**置信阈值整条曲线量完**，见 MEMORY §7b与 `docs/action_testB_20260916.md` §7。
  * 峰值 conf=0.40（F1 0.778516，+0.417pp vs 0.50），但三门禁只过 1 条 → **地板保持 conf=0.50**。
  * 纠正 memory 长期错误：那条「低置信带 23%」是 conf≈0.15 的带，不是 0.40–0.50 带（后者 r≈0.49）。
  * **硬下限：阈值绝不低于 0.25**（0.15 = −1.087pp 显著劣）。
  * 冻结 Oracle 与 09-06 那次**零漂移**（0.778516 逐位一致）。
- 部署：实例 `run_testB_infer.sh` 加 `base54_c35` pass；本地 `build_testB_candidates.sh` 支持 `BASE_TAG=` 覆盖（默认 base54，行为不变）。
- 提交：`6a071a7`（扫描结果）、`2df70d2`（BASE_TAG）、`99b5ba2`（memory）。
- 下次执行要点不变：先看 `/hy-tmp/testB_bundle.tgz` 是否已在（触发器可能已跑完），在则跳过推理直接拉包。

## 2026-09-17 03:35（第 8 次轮询）
- 结果：**BLOCKED**（第 15 次确认 97）。触发器 pid 25862 存活；幂等守卫通过。
- 本轮主要产出：**低阈值轴被独立模型证伪，正式关闭**。
  * 36ep LVO：conf 0.40 vs 0.50 = +0.417pp；**15ep LVO clsweight3：+0.001pp**（15ep 曲线 0.40=0.771559 / 0.45=0.771739 / 0.50=0.771547 / 0.60=0.766906）。差远大于各自 CI → 模型特有噪声。
  * 地板**固定 conf=0.50**，BASE_TAG / base54_c35 pass 保留但注释已更正为“无实测理由偏好”。
  * 副产品：发现 trim 的 +2.25pp 曲线是在 conf≈0.345 测的，**未在 conf=0.50 验证**（不影响 shot6 对冲逻辑）。
- 新建用户级技能 `~/.workbuddy/skills/oof-lever-replication-gate/`（先写死判据 → 找零成本测量方式 → 换第二个模型复验）。
- 提交：`24f1763`、`d2b97b9`。
- 下次执行要点不变：先看 `/hy-tmp/testB_bundle.tgz` 是否在；**不要再重推 conf 阈值这条轴**。

## 2026-09-17 08:04（第 9 次轮询）
- 结果：**BLOCKED**（第 16 次确认 80 / 总数 97）。SSH 串与 hostname 未变（第 7 次同实例）。幂等守卫通过。
- 触发器 pid 25862 存活 6h。⚠️ 记一条防误判：`testB_watch.log` 只在 start/TRIGGER/数量变化时落行，**日志久无更新 ≠ 进程死**，存活判定要用 `ps -eo pid,etime,args | grep watch_testB`。
- 本轮顺带完成 MEMORY.md 压缩重写（23.3KB→13.5KB，注入被截断必须先瘦身），修掉两个 §10 的重复编号。
- 剩余窗口 8h56m；临界点 16:00 前图必须到位。
- 下次执行要点不变：先看 `/hy-tmp/testB_bundle.tgz`；不要重推 conf 阈值轴（已证伪关闭）。
