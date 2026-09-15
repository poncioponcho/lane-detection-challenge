# 实验记录：遮挡增强 occlusion augmentation（2026-09-15）

> 单变量实验。分支 `exp/occlusion-aug`，tag `exp-occlude-start`（commit `05bd421`）。
> 冻结基线 tag `incumbent-0.73574`；同日前一个实验见 `docs/exp_segmask_binary_20260915.md`（**MISS**）。

## 1. 为什么选这个变量（不是拍脑袋）

前一个实验（segmask binary_union）刚被否掉，教训是**变量的机制必须对准实测的失效模式**。这一次严格按证据选：

| 证据 | 来源 |
|---|---|
| testA 场景性质 = **晨昏低照度 + 两侧高楼 + 大量停放车辆与公交遮挡车道** | 项目自己的逐 clip 场景诊断（clip 7907/4134 实证） |
| 绑定短板 = **召回**（0.678 vs 榜首量级 0.760；FN 1015 条 ≈ FP 的 1.9 倍） | junk 探针分解 |
| 现有增强管线已有：翻转 / 亮度 / 色相饱和度 / 运动模糊 / 中值模糊 / 仿射(平移+旋转+缩放) | `configs/unlanedet/clrnet_r50_hardlane.py` train_process |
| **缺口：没有任何遮挡类增强** | 同上 |

→ 假说：**合成遮挡物能让模型学会"推断被挡住的车道"**，直接补召回。

**排除的替代方案**：
- `xyt_loss_weight` 0.2→0.8（重加权定位损失）：这些是 CLRNet 在 CULane 上调过的论文默认值，改动先验弱；而且定位实验是**台账筛不出来的**（线集不变、只是位置变），等于盲发。
- 多尺度 TTA：**推理期**，但"加线"已被实测为 ≈0（≥40% 共识 +0.035pp），且"平均已有线"9/14 实测为 0 净收益。
- 更长训练：唯一有实测正趋势的轴，但外推仅 +0.1~0.2pp 且要 4.9h。

## 2. 变量（唯一改动）

在 train_process 的模糊块之后、仿射之前插入一条 imgaug 变换：

```python
dict(
    name="CoarseDropout",
    parameters=dict(p=(0.05, 0.15), size_percent=(0.08, 0.2)),
    p=0.5,
),
```

- **为什么是 `CoarseDropout`（像素级）而不是几何遮挡**：它只抹掉像素、**不动关键点**，所以车道标注仍然有效。
  几何变换需要在 `GenerateLaneLine` 里配套改写车道点处理，配置文件表达不了。
- **放在仿射之前**：这样遮挡块会随仿射一起被平移/旋转/缩放，更像真实场景里的遮挡物。
- 变换分发是 `getattr(iaa, aug['name'])(**parameters)`，所以这**纯粹是配置改动**，不碰框架代码。
- 已确认目标变换存在：`imgaug 0.4.0`，`iaa.CoarseDropout` 可用。

配置与基线 **diff 只有 docstring + 4 行**。三处登记齐备（`CONFIGS` / `WEIGHT_BASE_MODEL` / `KNOWN_MODELS`）。

## 3. 发车记录

```
model      clrnet_r50_occlude
run-name   all71_seed42_clrnet_r50_occlude_36ep
epochs 36  seed 42  iters/epoch 592  eval-every-epochs 6  max-to-keep 6
manifest   /hy-tmp/lane-outputs/experiments_manifest_train_all71.jsonl
发车       UTC 07:29（北京 15:29）
```

**三道守卫全过**：probe 11s、smoke 27s、三处登记 ✓。

⭐ **smoke 守卫在这里价值最大**：它跑的是一个**真实 batch 经过整条 train_process 管线**，
所以如果 `CoarseDropout` 构造失败（拼错名字、参数不合法），会在**发车前**就报出来，
而不是白烧 2 小时。这是本项目"守卫先行"纪律的一次实际收益。

## 4. 判据（先写死）

**先算 containment 台账，再决定是否花额度。**

- **台账否决线**：若 `dropped` 明显超过既有候选（swa 29~36、soupB 51、cut400 155），
  说明模型变保守了 → 与"冲召回"的意图相反 → **判 MISS，不发额度**（照 segmask 的流程）。
- **若台账形状健康**（dropped 小、novel 有量）：发 **1 发 A 榜**测真分。
  - **有戏**：≥ incumbent 0.73574 + 0.30pp（超出 ±0.27pp 噪声带）
  - **无戏**：≤ 0.73574
  - 带内：不采信，只作 B 榜探索位
- **注意**：遮挡增强的目标是"多检到被挡的线"，所以期望形态是 **novel 上升、dropped 保持小**。
  若 `novel` 没涨，说明遮挡增强没起到作用（模型没学会穿遮挡推断）。

## 5. 复现与回滚

```bash
git checkout exp/occlusion-aug
git checkout incumbent-0.73574      # 回滚，无需重建任何东西
```

## 6. 今日探索的两条通用教训（累计）

1. **污染 val 不能用作筛选**：segmask 的 val +0.128pp，而台账（dropped 204、空图 9→30）给出反向且正确的判断。
   cut400 的先例更狠：val +1.05pp → testA **−0.772pp**。
   → **新 full71 基座模型先算台账（2 分钟 CPU），再考虑花额度。**
2. **变量要按证据选，不能按"哪个参数还没调过"选**：segmask 的失败根因是我按"参数未测"选变量，
   而它对准的是一个**被自己普查否掉的机制**（溢出像素仅 0.206%）。今天这个遮挡实验是反过来的做法。
