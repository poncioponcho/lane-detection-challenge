# Incumbent 版本冻结卡（2026-09-15 11:35）

> 用途：把"当前最好版本"钉成一个可复原的快照。**冻结之后任何实验都不影响它**——
> 无论后续怎么重来，这个配方和它的包永远可以重新产出。
> 配套：git tag `incumbent-0.73574`。

## 1. 配方（一步都不能少）

| 项 | 值 |
|---|---|
| 模型 | CLRNet + ResNet-50（UnLanedet 框架） |
| 输入 | 800×320，`cut_height=180`（`configs/unlanedet/clrnet_r50_hardlane.py`） |
| 训练 | all71（71 clip / 7100 图），seed 42，54 epoch |
| 数据初始化 | `adapted_clrnet_r50_hardlane.pth` |
| 选点 | `model_best.pth`，metric_iteration **24863**（val F1 0.89565，**已越过峰值**） |
| 推理 | `conf_threshold = 0.50` |
| 后处理 | **bottom-trim，rule=`gt_cond`，margin 0** |
| 打包 | `prepare_submit.py`（1 位小数 canonicalize）→ verify → 官方 `check_submission.py` |

## 2. 指纹（可逐字节校验）

| 对象 | SHA-256 |
|---|---|
| 提交包 `outputs/submit_testA_54ep_trim0.zip` | `a7dcef5be7b57b12b31e8fdfd3b14c789899ffc547ea1a73dc47dde23fd472af` |
| 权重 `model_best.pth`（iter 24863） | `6749afe476c54743126e753d6a3948763951a8d91ae85d7825d3eb6005b7f97c` |
| 底模 `adapted_clrnet_r50_hardlane.pth` | `87acbe37ccaf133a1e5241863a60c6cab62b6ffa03285609db9cee23c9f91df5` |

## 3. 实测分数（全部在同一 testA 900 图上）

| 对象 | 分数 | 备注 |
|---|---|---|
| **本配方（incumbent）** | **0.73574** | 自队最优 |
| 54ep raw（无 trim） | ≈0.73505 | trim 贡献 +0.069pp |
| 9/14 共识并集 ≥2/10 + span80 | 0.73429 | −0.145pp |
| 9/14 hires 1366×540 | 0.72032 | −1.542pp |
| 9/15 ≥40% 共识门槛 | 0.73609 | +0.035pp |
| 9/15 soupB | 0.73195 | −0.379pp |
| 9/15 cut400 conf0.35 | 0.72802 | −0.772pp |
| A 榜前三门槛 / 榜首 | 0.81604 / 0.82389 | 差 8.0 / 8.8pp |

## 4. 复原步骤

```bash
git checkout incumbent-0.73574          # 代码
# 权重（二选一，两者等价）：
#   /hy-tmp/lane-outputs/runs/all71_seed42_clrnet_r50_54ep/model_best.pth  （sha256 见上）
#   或 outputs/weights_backup/incumbent_54ep_model_best.pth                 （本地备份）
bash scripts/autodl/run_testB_infer.sh <lane_root>   # 推理（含 base54 这一项）
bash scripts/build_testB_candidates.sh <bundle.tgz>  # 自动含 shot1 = 本配方 + trim(M)
```

`build_testB_candidates.sh` 的 **shot1 就是本配方**，所以冻结之后 B 榜的"地板"自动带上。

## 5. 已知未冻结的旁支（记录在案，不作为 incumbent）

- 实例上有 **2 个未纳入 git 的配置**：`clrnet_r50_hardlane_ms720.py`、`clrnet_r50_hardlane_ms880.py`
  （多尺度实验的残留，来源不明、无实验记录）。**不属 incumbent，也不要用**——先查清来源再说。
- 全部 `submit_testA_night_*.zip`（12 个）为 2026-09-15 夜班候选，均已过官方预检，但**未实测或实测为负**。
- `outputs/_DRYRUN_20260915_testA_content_DO_NOT_SUBMIT/` 是预演残留，**内容为 testA、名字像 testB，禁止提交**。
