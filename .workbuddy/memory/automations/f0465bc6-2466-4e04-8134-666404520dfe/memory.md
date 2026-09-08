# risk-on 960x384 LVO screen 巡检

## 约定
- 只读巡检；无关键节点变化则静默，只更新 `last_reported_utc` / `last_disk_avail_gb`。
- 关键节点定义：任一折 status 新变 pass/failed；8 折全 pass；runner 消失但 status=running；磁盘 <10GB。
- 状态文件：`outputs/reports/riskon_watch_state.json`；采样日志：`outputs/monitor_riskon_960x384_20260907.jsonl`。

## 执行历史
- 2026-09-07T13:25:56Z — 首次记录。**有关键节点**：fold_02_v546797496 → pass（F1 0.479376，Δ +1.178pp，首个正向折）。进度 3/8，fold_03 训练中（399/7875）。磁盘 57.65GB，runner 存活。三折 ΣΔ=+0.247pp，剩余 5 折需均 +1.55pp 才过主门；未触发全负止损。已更新状态文件。

- 2026-09-07T14:29:19Z — 无关键节点，静默。fold_00..03 仍 pass，fold_04 训练中 5319/6990（eta 8:02），runner 存活，磁盘 58.72GB。仅更新 last_reported_utc / last_disk_avail_gb。
- 2026-09-07T14:08:43Z — fold_03_v566817042 → pass（F1 0.742156，Δ +0.883pp）；四折 ΣΔ=+1.13pp、均值 +0.283pp，剩余 4 折需均 +1.718pp；新增 twin-pair 噪声探针（fold_01/fold_03 基线差 0.59pp 但 Δ 差 1.726pp → 折级噪声 ±1pp）。硬止损规则：6 折均值 <0.6pp 即不可达。

## 注意事项
- 远端 poll fold_state.json 时若该折刚完成训练正在 eval，需隔 ~2-3 分钟复采才能拿到 pass 终态（eval 约 2 分钟）。
- 长时间 `sleep` + ssh 组合在自动化里会被 SIGTERM，改成分次短命令。
- 2026-09-07T15:32:00Z — **有关键节点，已汇报**。fold_04_v576104564 → pass（F1 0.856813，**Δ -0.314pp**）；fold_05_v644768616 → pass（F1 0.789587，**Δ -1.149pp**）。进度 6/8，fold_06 训练中 3479/8370（eta 0:23:42），runner 存活，磁盘 59.79GB。
  - **止损触发**：6 折 ΣΔ=-0.333pp、均值 -0.056pp（远低于预注册规则阈值 +0.6pp）；计数加权全局 dF1=-0.052pp（CAND 0.772437 vs BASE 0.772959，dTP=-48/dFP=-49）。
  - **门条件 3 已数学不可能**：仅 2/6 折为正，即使 fold_06/07 全胜也只有 4/8 < 5/8。
  - 结论：risk-on 960x384 + cut180 判定为负，不放行 36ep。仍跑完 fold_06/07 只为凑齐 8 单元 bootstrap。已写入状态文件 `stop_loss_arithmetic.after_6_folds`。
- 2026-09-07T16:42Z — **终态，已汇报**。8/8 全 pass，`lvo_training_complete.json` 已生成。fold_06 +0.644pp / fold_07 +0.483pp；ΣΔ +0.795pp、均值 +0.099pp、正向 4/8；计数加权 dF1 **+0.045pp**；bootstrap CI [-0.450,+0.624]pp、p=0.746。**三项门全 FAIL → 最终判负**：不训练 36ep、不打包、不提交 A 榜，incumbent 未动。已写入 `after_8_folds`。
  - 巡检任务至此**目的达成**，可考虑停用本自动化（实验已终止、runner 已完成）。若继续跑，只会重复静默。
  - 经验：折刚跑完时 fold_state 可能仍为 running（eval 后约 1-2 min 才落 pass），短等 100s 复采即可拿到终态，无需跨轮次。
- 2026-09-07T17:45Z — 无关键节点，静默。8/8 仍 pass，complete json 存在，runner 已正常退出，磁盘 60.56GB。仅更新 last_reported_utc / last_disk_avail_gb。
  - 工具坑：`pgrep -f lvo_video_runner.py` 在 ssh 远端会匹配到承载该字符串的 shell 自身 → 假阳性 RUNNER_ALIVE。改用 `ps -eo pid,etime,args | grep ... | grep -v grep` 才准。
  - 本巡检目的已达成（实验终态判负），后续轮次预期恒为静默，建议停用。
- 2026-09-07T18:47:37Z — 无关键节点，静默。8/8 仍 pass、`lvo_training_complete.json` 存在、runner 已退出、磁盘 60.56GB（与上一轮完全一致，远端无写入活动）。仅更新 last_reported_utc / last_disk_avail_gb，并在 jsonl 追加采样行。
  - 注意：`fold_state.json` 顶层无 `f1` 键（采样打印为 None），F1 存于嵌套字段；判定关键节点时以 `status` 为准即可，F1 从 reported_fold_status 取。
  - 连续多轮恒静默，本自动化已无信息增量，建议停用以免空耗。
- 2026-09-07T19:49:53Z — 无关键节点，静默。8/8 pass、`lvo_training_complete.json` 存在、runner 已退出、磁盘 60.57GB。仅更新 last_reported_utc / last_disk_avail_gb 并追加 jsonl。
  - 已连续 4 轮（17:45 / 18:47 / 19:49）恒静默，远端零写入活动（磁盘字节数与前轮逐位一致）。**本自动化信息增量已归零，强烈建议停用**——实验终态判负已于 2026-09-07T16:42Z 定案，继续轮询只消耗 ssh 与 token。
- 2026-09-07T20:51:35Z — 无关键节点，静默。8/8 pass、`lvo_training_complete.json` 存在（远端时间戳仍为 Sep 7 16:38，未变动）、runner 已退出、磁盘 60.57GB。仅更新 last_reported_utc / last_disk_avail_gb 并追加 jsonl。
  - 已连续 5 轮（17:45 / 18:47 / 19:49 / 20:51）恒静默，远端零写入活动。**本自动化信息增量归零，建议停用。**
- 2026-09-07T21:53:16Z — 无关键节点，静默。8/8 pass、`lvo_training_complete.json` 存在（时间戳仍 Sep 7 16:38）、runner 已退出、磁盘 60.57GB（与前 3 轮逐位一致）。仅更新 last_reported_utc / last_disk_avail_gb 并追加 jsonl。
  - 已连续 6 轮恒静默。**第 4 次建议停用本自动化**：实验终态判负已定案，远端无任何写入，每轮只消耗 ssh + token。若第 7 轮仍静默，主 Agent 应主动提示用户删除该自动化而非继续轮询。
- 2026-09-07T22:56:14Z — 无关键节点，静默。8/8 pass（fold_state.json mtime 全部停在 09-07 12:03–16:38）、`lvo_training_complete.json` 存在（仍 Sep 7 16:38）、runner 已退出（`ps -eo` 确认，无假阳性）、磁盘 60.57GB（63509588 KB，与前 4 轮逐位一致）。仅更新 last_reported_utc / last_disk_avail_gb 并追加 jsonl。
  - **已连续 7 轮恒静默，达到预设阈值 → 本轮已在回复中主动提示用户删除该自动化**。实验终态判负定案，远端零写入，信息增量确认为零。
  - 工具坑补充：上轮记录过 pgrep 假阳性；本轮改为远端内直接 `grep -o '"status"[^,]*' fold_state.json | head -1` 取 status，避开在远端调用本地 python 绝对路径（该路径远端不存在，会静默返回 NA）。这条更省事也更准。
- 2026-09-07T23:47Z（用户拍板）— **本自动化已置 PAUSED，停止轮询**。连续 7 轮恒静默后用户采纳「暂时停用」建议。
  - 停用条件已满足：实验终态判负定案（09-07T16:42Z）、远端零写入、8/8 pass 无新事件可产生。
  - 附带信息：该自动化本身带 `validUntil=2026-09-08T09:00`，不暂停也会在今日 09:00 自然过期。
  - 巡检产物原地保留不删（watch_state.json / jsonl / 远端实验目录）。
  - 若日后要重启：只有在新实验真正起 runner 时才有意义，届时应同步更新 experiment_root_remote 与 reported_fold_status。
