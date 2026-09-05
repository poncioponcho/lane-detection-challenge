# 自动化执行记录（lane AutoDL baseline 收官检查点）

## 2026-09-04 08:05（首次执行，status=complete，任务终结）
- 远端 `/hy-tmp/lane-outputs/pipeline.status` = **complete**，log 尾部 `baseline exit rc=0 2026-09-03T17:37:23+00:00`。
- 筛选赢家：**clrnet_r50**（sidecar `winner` 字段）。
- metrics.json 36 个 eval 点 argmax：**best iteration = 16799，val F1 = 0.8089010345500683**（TP=2072/FP=396/FN=583，800 val 图）；次高 18374 = 0.80790（差 0.1pp，噪声带内）。
- `selected_best_eval/eval_evidence.json` 确认 selected checkpoint = `model_0016799.pth`（sha256 e5919ae5…4cd）。
- 产物已下载到本地 `outputs/`：`handoff_baseline.tar.gz`（271,755,187 bytes, 4073 files）+ `handoff_baseline.tar.gz.json`（sidecar）。本地 `shasum -a 256` 复算 = `b79217ff010428cb0b515767d33e2cb7b8370d352a663a1eb710d1e954c20ce4`，与 sidecar 逐字符一致。
- 全程只读探测 + 下载，未改远端、未杀进程。status=complete → 未创建后续自动化。
- 遗留提醒（已写入当日工作日志）：下一步须用独立 Oracle 环境 `/private/tmp/lane-oracle-py312` 对 model_0016799.pth 做 eval-only 回放，不得单信 model_best.pth；A 榜提交必须用户显式拍板，绝不自动提交。
