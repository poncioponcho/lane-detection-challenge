# 场景标注与 clip-level 切分审计（2026-09-02）

## 1. 标注方法与质控

标注对象为训练集 71 个官方 clip 目录。首轮每段固定抽取 0/25/50/75/100% 五帧，接触表索引保留真实帧 ID。标签 schema 与 DECISIONS §17.2/§18.4 一致：weather、illumination 单选；artifact、geometry 多选；另存 confidence 与 spot_frames。

首轮唯一低置信且最终成为 singleton 特征的段为 `v566817042_1_0_728`。二审查看 00000–00297 间 11 个等距帧，确认道路两侧持续积雪/霜雪，同时存在重雾，故最终标 `weather=mixed`、`confidence=high`。稀有标签操作定义为二元特征持有段数 ≤5；因此 `artifact:glare` 的 5 个持有段也各追加 5 个非首轮帧二审，均确认雾中警灯或夜间对向车灯的持续/间歇光晕。所有最终 low-confidence 项（0 个）和全部稀有特征持有段（6 个去重）均已完成二审。没有可靠正例的 backlight、shadow、单独 snow 等值保持零样本。

最终主要分布：weather clear/fog/mixed/rain = 31/18/1/21；illumination low_light/normal = 31/40；artifact glare/none = 5/66；geometry 的 curve/crossroad/none 特征计数为 22/36/18（curve 与 crossroad 可共存，因此不相加为 71）。

## 2. 切分算法

`src/data/split_by_clip.py` 把每段标签转为二元特征；artifact/geometry 空集显式记为 none。仅 1 段持有的特征进入保护集、不得进 val，且不纳入可优化目标。其余 scene 特征以 `sum((val_rate - all_rate)^2)` 为首要目标，固定 seed=42 搜索 100,000 个候选，再做确定性单交换局部下降。

T22 EDA 随后发现 scene-only 最优解之间原来按 clip ID 偶然决胜，导致第一版 val 空 GT 为 0/800。现加入逐图车道条数 0–7 直方图偏差作为**完全相同 scene 分数候选之间的次级 tie-break**，不牺牲天气、光照、伪影、几何或 singleton 保护。修正后 val 空 GT=35/800=4.375%，接近全量 3.690%；车道直方图最大占比偏差 2.697pp。

最终 val 8 段（括号内为 weather / illumination / 关键附加标签）：

1. `v239132710_1_0_5898`（clear / low_light / crossroad）
2. `v239132710_1_0_7018`（clear / low_light / crossroad）
3. `v444733262_1_0_1109`（clear / low_light / crossroad）
4. `v444733262_1_0_6630`（clear / low_light / curve）
5. `v546797496_1_0_100`（fog / normal / none）
6. `v546797496_1_0_646`（fog / normal / glare）
7. `v644768616_1_0_1085`（rain / normal / curve）
8. `v777679069_1_0_2721`（rain / normal / crossroad）

结果为 63 train / 8 val 段、6300/800 图，段 ID 无交集。weather 的 val 计数为 clear/fog/rain=4/2/2；illumination 为 low_light/normal=4/4；glare=1；curve=2；crossroad=4；geometry:none=2。singleton `weather:mixed` 留 train，val 状态为 N/A。scene 平方偏差目标值 0.023761406466970843，最大单特征占比偏差 0.06338028169014087；次级车道直方图平方偏差 0.0013255858212656217，最大偏差 0.026971830985915494。train/val 空 GT 为 227/35 张，平均车道数为 3.457/3.319。

## 3. 可复现性锚点

- scene labels SHA-256：`0ed561e42167a0fa20c3ec5c35df646d59dc5c790e7272188dafedd84c5ff948`
- source manifest SHA-256：`a57067e263f97abe56bd385a419c33d415b32ac93f4b3392e26ef55d511a8c59`
- lane-count histograms SHA-256：`d05d855adabbd89739a9750c1d2c6ea7e64b987895a0e767dedc8df5a6afed06`
- split config SHA-256：`715eb8a0f695f4f26ef2d889176dc47f1a5d0aeb91aa25f26405062f3c1fd4ab`
- 固化配置：`configs/splits/v1_seed42.yaml`
- 派生清单：`data/processed/manifest_train_v1_seed42.jsonl`（6300）与 `data/processed/manifest_val_v1_seed42.jsonl`（800）

配置使用 JSON-compatible YAML（JSON 是 YAML 1.2 的子集），因此不引入 PyYAML 依赖。配置内同时保存算法名、目标定义、搜索次数、输入哈希、singleton 保护列表和逐特征 all/train/val 计数。
