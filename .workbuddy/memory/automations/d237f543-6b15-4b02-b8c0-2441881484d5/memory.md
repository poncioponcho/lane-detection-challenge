# 自动化执行记忆：A 榜每日快照采集

## 任务 ID
d237f543-6b15-4b02-b8c0-2441881484d5

## 运行窗口
2026-08-19 ~ 2026-09-14（B 榜开启后失效；每日一次）

## 首次执行（2026-09-01 20:01 GMT+8）
- WebFetch 比赛页：仅返回静态赛题说明（含「参赛团队 83」），**排行榜分数为 SPA 动态渲染，WebFetch 拿不到**。
- 改用 agent-browser：打开页面 → 点「排行榜」标签(e14) → snapshot 读取成功。
- 抓取结果 Top5：0.79076 / 0.78566 / 0.78422 / 0.78344 / 0.77199。
- 队伍总数：页头注册数 83；排行榜实际提交记录 28 条（"总28条，共3页"）。
- 证据截图：`outputs/a_bang_20260901.png`。
- 数据落盘：`docs/a_bang_snapshot.md`（新建，含用途说明注释 + 首条记录，来源=自动）。
- 波动检查：首条无前日对比，未触发 >2pp 高亮。
- 结论：agent-browser 路线在本环境下可行，无需人工补录。

## 第 2 次执行（2026-09-02 20:01 GMT+8）

- WebFetch 再次确认只回静态壳（但页头「参赛团队 93」可用）；agent-browser 路线复用成功，排行榜 tab 仍是 `ref=e14`。
- Top5 抓取成功；注册 93 队 / 有提交 31 队。名次位 Δ 最大 +0.607pp，未触发 >2pp 高亮。
- 证据截图 `outputs/a_bang_20260902.png`；数据 + Δ 表 + 队伍级附录 + 门槛校准结论已落 `docs/a_bang_snapshot.md`。
- 关键结论：前三门槛实测 0.78566，推测区间 83–87 连续两日被否证（系统性高估 4–8pp）。
- 发现本自动化 prompt 内嵌的目标常量已 stale，与 `configs/default.yaml::target` 不一致，已按 default.yaml 为准并在回复中提示用户改 prompt。

## 经验 / 坑（后续运行复用）
- 该站排行榜必须切到「排行榜」标签才渲染数据；直接 WebFetch 主 URL 拿不到分数（SPA 空壳）。
- agent-browser 已预装在 `/opt/homebrew/bin/agent-browser`，node 正常，可直接用。
- 流程：open → wait networkidle → snapshot 较长（约152行），排行榜表头在"排名/团队名称/分数/提交次数/最佳成绩提交时间"，分数在团队名后一行。
- 数据来源判定：能读到具体分数=自动；读不到=人工待补（严禁伪造，改来源列并提醒用户）。
- 后续运行需读取本文件上一行做 ±2pp 波动对比；队伍总数优先取页头"参赛团队"注册数。
- **必须同时记录队伍 ID / 提交次数 / 最佳成绩时间**（09-01 漏记导致跨日无法区分「同队改进」与「新队插入」）。最佳成绩时间戳是判断队伍身份延续的关键线索。
- `wait --load networkidle` 在该 SPA 上可能挂住，用 `wait --load load` + 点 tab 后 `sleep 3` 更稳。
- 目标常量以 `configs/default.yaml::target` 为唯一仲裁源，**不要用本自动化 prompt 里写死的数字**（prompt 早于 §15.1 降级链，已 stale）。
- 截图用 `agent-browser screenshot outputs/a_bang_YYYYMMDD.png` 指定路径；结束务必 `agent-browser close`。
