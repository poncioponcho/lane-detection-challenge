# 夜间任务执行指令（AutoDL baseline 流水线值守）

> 适用对象：夜间值守的 AI 代码助手（Codex/自动化 agent）。
> 目标：在不打扰用户的前提下，让恒源云实例上的 baseline 实验流水线（gate → screen → baseline）一直跑到 `complete`，拿到 `handoff_baseline.tar.gz` 并回传本地、生成报告；出现可定位的失败时做最小修复后继续。

## 1. 总原则

1. **最小改动**：每次修复只改必要文件，修复必须先在本地验证语法/逻辑，再 bundle 同步。
2. **不打扰用户**：正常运行只记录不汇报；阶段性完成（gate/screen/baseline）或需要决策时才汇报。
3. **证据优先**：所有判断基于 `pipeline.status` / `pipeline.log` / `train.log` / `metrics.json`，禁止凭记忆臆断。
4. **禁止删除**：失败的 run 目录一律改名归档（`.stale-*` / `.failed-*` 后缀），绝不 `rm`。
5. **不要主动 git commit / push**，除非正在修复一个已确认的新故障（原因见 §7 的跨 commit 恢复守卫）。

## 2. 当前任务快照（2026-09-03 晚）

- 实例：`ssh -i ~/.ssh/lane_id -p 34529 -o BatchMode=yes root@i-1.gpushare.com`
- 关键路径：
  - 项目仓库：`/hy-tmp/lane-detection-challenge`（git，与本地 main 同步）
  - UnLanedet：`/hy-tmp/UnLanedet`（pinned `0392184`，已打 `patches/unlanedet_hardlane.patch`）
  - 数据：`/hy-tmp/datasets/HardLane/Lane`（JPEGImages / anno_txt / Annotations）
  - 输出：`/hy-tmp/lane-outputs`（`pipeline.status`、`pipeline.log`、`runs/`、`weight_probe.json`、`smoke/`）
- 本地 HEAD 起点：`06f1da8`（2026-09-03 晚，含当天全部 8 个修复）。
- 已通过：weight probe、两模型 smoke、gate clrnet 训练+终评（F1 0.7384/0.7359）、gate adnet 训练（total 15.33 全 finite）。
- 当前阶段：screen（clrnet 15ep → adnet 15ep → 选优）→ 之后 baseline 36ep。

### 失败链全景（当天已修，供排查参考）

| # | 症状 | 根因 | 修复 commit |
|---|------|------|------------|
| 1 | probe/modelzoo CWD 相对路径 FileNotFoundError | 从 UnLanedet 根运行 | cc89204 |
| 2 | LazyConfig 覆写裸路径 SyntaxError | literal_eval 无回退 | dc49dea |
| 3 | imgaug 增强路径 np.bool 崩溃 | numpy>=1.24 移除别名 | 79a1206 |
| 4 | gate iter0 AddmmBackward0 NaN（首轮） | 增强输出 NaN | 28e649b |
| 5 | setup 每次重编译 5-15min | 无跳过守卫 | ea6f21c/22b59e3 |
| 6 | gate iter0 NaN（fp16 tan 退化） | theta 邻 0/1 | 9e15f19 |
| 7 | ClampBackward1 inplace 版本计数 8≠4 | 原地索引写 | daa4bcd+a99d99f（patch 由 git diff 重建） |
| 8 | gate iter0 AddmmBackward0 NaN（再发） | 上游遗留 set_detect_anomaly(True) | b6dd060 |
| 9 | gate iter~5 CUDA device-side assert | mask palette id 9/10 越界 seg 类别数 9 | 7b00b9a+ffa9cea |
| 10 | gate 终评 No module named 'common' | 评估进程缺 sys.path 注入 | d72f6af |
| 11 | 跨 commit 恢复被拒 | run_training 一致性守卫（设计内） | 归档 run 目录重跑即可 |
| 12 | evaluate_selected 裸路径 SyntaxError | f-string 拼接未加引号 | 06f1da8 |

## 3. 夜间任务清单与优先级

| 优先级 | 任务 | 说明 |
|-------|------|------|
| P0 | 保持流水线推进 | status=failed 时：先诊断（§7），可修则最小修复+重启；不可修则按 §8 记录并等待（不要盲目反复重启同一失败） |
| P1 | 周期监控与状态记录 | 按节奏采样 status/log/GPU，追加写入监控记录（§6） |
| P2 | 完成产物回传 | `pipeline.status == complete` 时：下载 `handoff_baseline.tar.gz` 到本地 `outputs/`，校验非空 |
| P3 | 结果报告 | 按 §9 生成 `outputs/overnight_report_<date>.md`（含各阶段 F1、修复清单、耗时） |
| P4 | 收尾 | 若 baseline 已完成且已汇报，删除对应的 cron 提醒任务（避免空转） |

## 4. 执行时间窗口与轮询节奏

- 监控轮询：每 20-30 分钟一次（训练期）；状态切换关键点（gate 完成、screen 完成、baseline 完成）前后可加密到 10 分钟。
- GPU 空闲判定：`nvidia-smi` 连续 2 次采样 util≈0% 且 status=running → 疑似挂死，进入 §7.3。
- 预期耗时基准（3090，batch 8，~0.26s/iter）：
  - gate：每模型 525 iter ≈ 2.5 min 训练 + ~2-3 min 评估；两模型含 probe/smoke 合计 ~15-20 min。
  - screen：每模型 15ep = 7875 iter ≈ 35-40 min 训练 + 15 次周期评估 ≈ +15-25 min；两模型合计 ~2h。
  - baseline（胜者 36ep）：18864 iter ≈ 85-95 min + 周期评估；合计 ~2-2.5h。
  - 全程顺利约 4.5-5.5h。当晚 22:00 前后已进入 screen，预计次日 03:00-04:00 前完成。

## 5. 资源使用限制

- 实例 GPU 完全让给训练：夜间不要在实例上跑任何额外 Python/CUDA 任务（诊断单跑除外，用完即删 `/hy-tmp/diag_*`）。
- 本地机器：只做轻量检查与文档编辑；不要下载大数据集。
- `pipeline.log` 会持续增长：只在需要时 `tail`/`grep`，不要 `cat` 全文件。
- 磁盘：`runs/` 下每个训练 run 约 1-2GB（多 checkpoint）；发现实例磁盘 >85% 时归档最老的 `.stale-*` 目录（改名到 `/hy-tmp/archive/`，仍不删除）。
- 每次修复的 git bundle 是增量的（`prev..main`），体量 KB 级，直接 scp。

## 6. 实时监控机制

单次采样命令（一条 SSH 拿全）：

```bash
ssh -i ~/.ssh/lane_id -p 34529 -o BatchMode=yes -o ConnectTimeout=15 root@i-1.gpushare.com \
  'cat /hy-tmp/lane-outputs/pipeline.status; echo ---; tail -3 /hy-tmp/lane-outputs/pipeline.log; \
   echo ---; ls /hy-tmp/lane-outputs/runs/ | grep -v stale | tr "\n" " "; echo; \
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

判断矩阵：

| status | GPU | 动作 |
|--------|-----|------|
| running | >50% | 正常，记录 iter/loss 后等待 |
| running | ≈0%（两次连续） | 查最后 20 行 log 判断卡点（数据加载/评估 IO），10 分钟后再采样，仍无进展按 §7.3 |
| failed:N | - | 按 §7 诊断决策树处理 |
| complete | - | P2 回传 → P3 报告 → P4 收尾 |

监控记录追加到本地 `outputs/overnight_monitor_<date>.jsonl`，每行：`{"ts_utc": "...", "status": "...", "gpu": "...", "last_iter": ..., "note": "..."}`。

## 7. 异常处理流程（决策树）

### 7.1 训练/评估崩溃（failed:N）

1. `grep -n "baseline launch" pipeline.log` 定位本次 launch 行号，只读该段（`sed -n "N,$p"`）。
2. 按错误关键字分流：
   - `SyntaxError` + 路径 → 查找新的裸路径 override 拼接点，复用 `run_training.override`（同 06f1da8）。
   - `ModuleNotFoundError` → 看缺的模块：`common` → sys.path 注入模式（同 d72f6af）；其他包 → 检查依赖是否被省略。
   - `device-side assert` → 多为数据索引/类别越界，用 `CUDA_LAUNCH_BLOCKING=1 timeout 300 <同款 train_net 命令> train.max_iter=30` 复现拿确切位置（跑完删 `/hy-tmp/diag_*`）。
   - `NaN`（AddmmBackward0/anomaly）→ 确认三道既有防线仍在：theta cat 重建补丁、`set_detect_anomaly(False)`、数据侧 NaN 重试守卫；若都在，说明是新数值点，优先查 `train.log` 逐迭代 loss 定位首个异常迭代。
   - `refusing cross-commit resume` → 归档该 run 目录（`mv runs/<name> runs/<name>.stale-crosscommit`），直接重启流水线，不要改 launches.jsonl。
   - `corrupt patch` → 不要手改 hunk 计数；在临时仓库对 pinned 文件用 `git diff` 重新生成整份补丁（同 cb9f47c 流程），本地 `git apply --check` 通过后再同步。
3. 修复的硬性流程：本地改 → `py_compile`/AST 检查 → commit → `git bundle create /tmp/f.bundle <prev>..main` → scp → 实例 `git fetch + merge --ff-only` → **确认无进行中训练**（有则等它结束或接受作废）→ 重启 `setsid nohup bash /hy-tmp/lane-outputs/pipeline_runner.sh >/dev/null 2>&1 < /dev/null &`。
4. 同一错误若第 3 次修复尝试仍失败：停止尝试，把完整证据写入 §9 报告并标记 blocked，保留现场。

### 7.2 需要动 UnLanedet 时

- pinned HEAD 必须是 `0392184`；改动一律走 `patches/unlanedet_hardlane.patch`。
- `scripts/autodl/run_training.py` 的校验器要求 4 个子串同时存在：`nms_topk`、`.astype(bool)`、`predictions[..., 4].clamp(0.01, 0.99)`、condlane head 的 `set_detect_anomaly(False)`；动 clr_head/CondlaneNet head 前先对照。
- 补丁应用逻辑在 `setup_unlanedet.sh`（先于跳过守卫），工作区脏会导致 setup 报错——需要重打补丁时先 `git -C /hy-tmp/UnLanedet checkout -- .`。

### 7.3 疑似挂死

- 采 `ps aux | grep train_net`、`nvidia-smi`、最后 30 行 train.log。
- 若 dataloader 卡死（data_time 持续飙升）：再等 10 分钟（可能是磁盘 IO 抖动），无恢复则 kill 训练进程（仅 kill 训练相关 PID，不动其他），pipeline 会以 failed 退出，随后按 7.1 处理——auto-resume 会从最近 checkpoint 续训。

## 8. 任务完成后的结果记录与报告生成

`pipeline.status == complete` 时依次执行：

1. 回传产物：
   ```bash
   scp -i ~/.ssh/lane_id -P 34529 root@i-1.gpushare.com:/hy-tmp/lane-outputs/handoff_baseline.tar.gz outputs/
   tar -tzf outputs/handoff_baseline.tar.gz | head   # 校验非空
   ```
2. 从实例收集关键指标（`runs/baseline_<winner>_36ep/run_evidence.json`、`screen_decision.json`、`metrics.json` 的 F1 历史）。
3. 生成 `outputs/overnight_report_<date>.md`，必须包含：
   - 各阶段结果表（gate/screen/baseline 的 best_f1、final_f1、winner 及理由）；
   - 夜间发生的故障与修复对照表（延续 §2 格式）；
   - 时间线（各阶段起止 UTC/本地时间）；
   - 遗留风险与建议（如：screen 阶段某模型 loss 形态异常等观察）。
4. 向用户汇报一次（简要，附报告路径与最终 F1）。
5. 若已汇报过完成，删除对应 cron 任务（按 cron 系统的操作方式执行）。

## 9. 稳定性与安全性规范

- SSH 仅用 BatchMode + 专用密钥 `~/.ssh/lane_id`；任何命令不得交互式输入。
- 只读优先：诊断命令一律先只读（cat/grep/ls/nvidia-smi）；写操作仅限本 runbook 列出的动作。
- 禁止：`rm -rf` 任何数据/产物目录、修改冻结 manifest（`data/processed/manifest_*`）、修改 `configs/` 中已冻结的超参、push 到 GitHub、动实例上与本任务无关的目录（`/hy-tmp` 之外）。
- pip/环境：禁止夜间重装环境；setup 脚本自带跳过守卫，环境校验失败时按 7.1 记录，不要手动 pip。
- 凭据与密钥不落报告；日志中出现的 token/路径原样保留即可（实例为封闭环境）。
- 所有修复 commit 遵循仓库现有格式（中文 conventional commits，正文说明根因与证据）。

## 10. 快速重启命令（已验证）

```bash
ssh -i ~/.ssh/lane_id -p 34529 -o BatchMode=yes root@i-1.gpushare.com \
  'setsid nohup bash /hy-tmp/lane-outputs/pipeline_runner.sh >/dev/null 2>&1 < /dev/null & sleep 3; cat /hy-tmp/lane-outputs/pipeline.status'
```
