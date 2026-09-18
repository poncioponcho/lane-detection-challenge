# 环境坑 & 实例操作笔记（原 MEMORY.md §2 / §3，2026-09-17 下沉）

> 完整事实源仍是 `2026-*.md` 日志。本文件是这两节的副本，MEMORY.md 只留指针。
> 冲突以日志 + `docs/` 为准，禁用行号引用。

## A. 本机环境坑
- 中文检索用内置 Grep（BSD grep 对中文 + `\|` 静默零命中）。**数文件一律用 Python**（`ls` 不看点文件）。
- >2min 用 run_in_background；远端 `setsid nohup … & disown`。本机无 `timeout`。
- heredoc 里 `#` 注释是数据 → 注释放 heredoc 外。
- push：`git -c credential.helper='!/opt/homebrew/bin/gh auth git-credential' -c url.'https://github.com/'.insteadOf='git@github.com:' push`。
- 清理一律 `mv` 到 /tmp（>50 文件触发守卫 exit 137）。

## B. 实例（恒源云）坑
- 🔴 **AppleDouble `._*`**：实例 clip 带 100 个 `._<frame>.jpg`，`Path.glob("*.jpg")` 会匹配 → 帧数算成 2000；叠加 `manifest.py:18` 的 `VERIFIED_100_FRAME_SPLITS` → 完全隐形。已修。真实画布 **1366×720**（硬编码 1280 会造假警报）。
- git bundle ref 默认 `HEAD` → `git fetch <bundle> HEAD:refs/heads/tmp && git merge --ff-only refs/heads/tmp`；实例**无 origin**。
- **`pkill -f`/`pgrep -f` 会匹配自身 shell 自杀（exit 137）** → 用 `ps -eo pid,comm`。
- `scp -r` 目录易被杀 → 先 `tar czf`；大文件 `split -b 30m`（~400KB/s）。
- 实例 `i2b0715374400501416`（3090-24G 包天）。`root@i-1.gpushare.com:59725` + `~/.ssh/lane_id`。
- 发车三守卫：`probe_weights.py` → `smoke_dataloader_and_loss.py` → 新模型登记 `run_training.CONFIGS` + `WEIGHT_BASE_MODEL` + `validate_run.KNOWN_MODELS`（三处）。
- ⏱ 训练 54ep≈3.75h；推理 900 张 42s；testB 全链（13 树推理+回传+建包）20–30min。
- 🔴 **干净工作树守卫会静默废整轮**：`run_training.py:186` `assert_tracked_worktree_clean`；scp 的 `M` 文件 → 13 次推理全被拒而脚本仍 `exit 0` → 已跟踪文件必须「本地提交 → bundle → 实例 merge」；批量脚本逐产物计数并显式非 0 退出。
- ⚠ **`pack()` 把 prepare_submit 输出丢进 `/dev/null`** → 建包失败静默，必须查退出码。

## C. 方法类坑（这条属于结论，保留在 MEMORY 也可，见 §6）
- 🚫 **"多模型一致性当伪真值"失效**：9 树 ≥7/9 伪真值下 incumbent 命中 92.7% vs 真实 80.3% → 别用共识当 GT。
- 🔴 **坐标越界：官方是 clamp 不是拒绝**（`score.py:65-67`；`check_submission.py` 只查 UTF-8/偶数/有限/去重≥2点/≤64线/≤2048点/文件集合精确匹配，不查范围）。testB 13 棵树均有 3–21 条线落 `x∈(1365,1366]`（最多 1.0px）→ 已改 1.0px 容差 + clamp（`d31eb99`）。A 榜 x 最大 1362.1 故从未暴露。
