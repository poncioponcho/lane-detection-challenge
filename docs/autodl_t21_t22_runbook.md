# AutoDL 训练与验收手册

第 1–5 节只在 AutoDL CUDA 实例执行。本地已经完成纯 CPU 数据契约和静态测试，但不构成模型验收；第 6 节的格式归一化/Oracle 可在 CPU 环境复核。

## 1. 路径与环境

选用带可用 PyTorch/CUDA 的 AutoDL 镜像，并按实际挂载位置填写绝对路径：

```bash
export HARDLANE_PROJECT_ROOT=/root/autodl-tmp/lane-detection-challenge
export HARDLANE_DATA_ROOT=/root/autodl-tmp/datasets/HardLane/Lane
export UNLANEDET_ROOT=/root/autodl-tmp/UnLanedet
export HARDLANE_WEIGHTS_ROOT=/root/autodl-tmp/weights/unlanedet
export HARDLANE_OUTPUT_ROOT=/root/autodl-tmp/lane-outputs
export HARDLANE_PYTHON=python
```

`HARDLANE_DATA_ROOT` 必须直接包含 `JPEGImages/`、`anno_txt/`、`Annotations/`。项目根必须包含两个冻结 manifest。

## 2. 固定框架与编译 CUDA 算子

```bash
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/setup_unlanedet.sh"
```

脚本固定 UnLanedet commit `03921844220adb2e65c840de2d9759478d5c3d4c`，应用 CLRNet `np.bool` 与 `candidate_topk` 补丁，安装依赖并编译 NMS/DCN。若 checkout 有未提交改动，脚本会拒绝覆盖。

## 3. 三份权重映射与适配

```bash
python "$HARDLANE_PROJECT_ROOT/scripts/autodl/probe_weights.py"
```

首次会下载两个 CLRNet 与一个 ADNet 发布资产，严格核对已知字节数，再分别与 R34/R50/ADNet 实例比较。成功产物：

- `$HARDLANE_OUTPUT_ROOT/weight_probe.json`
- `$HARDLANE_WEIGHTS_ROOT/adapted_clrnet_r50_hardlane.pth`
- `$HARDLANE_WEIGHTS_ROOT/adapted_adnet_r34_hardlane.pth`

报告必须为 `status=pass`，并包含资产 SHA、R50 实际映射、compatible/missing/unexpected/shape-mismatch 和重初始化参数。复跑可加 `--no-download`，确保只使用现有文件。

## 4. 数据、loss/backward 与 demo 冒烟

```bash
python "$HARDLANE_PROJECT_ROOT/scripts/autodl/smoke_dataloader_and_loss.py"
```

成功门：train/val 数量 6300/800、空 GT 227/35；CLRNet 和 ADNet 各一个空图与非空图 batch 的全部 loss/gradient 有限；target capacity=8、candidate top-k=12；生成两类图像：未经训练的 checkpoint demo（只验链路）和 GT 预处理坐标 overlay（验坐标 round-trip）。证据在 `$HARDLANE_OUTPUT_ROOT/smoke/`。

## 5. 分阶段流水线（推荐唯一入口）

```bash
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_pipeline.sh" gate
```

`gate` 依次执行 setup、权重映射探针、空/非空 batch loss+backward smoke，然后两模型各训 525 iter，再各续跑到 526 iter。续跑成功必须从 iteration 525 开始，不能重回 0。每个 run 完成后会：

- 核对 checkpoint 顶层/trainer iteration 和 optimizer state；
- 核对 800 张预测文件、2655 条 GT 及 TP/FP/FN/F1 恒等式；
- 从全历史 `metrics.json` 定位 best metric iteration，再按上游 final-eval 的 N→N−1 规则映射 checkpoint 内部 iteration；
- 对该权重独立 eval-only 回放，F1 必须与历史 best 精确相等。

门禁通过后跑双路 15 epoch 筛选：

```bash
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_pipeline.sh" screen
```

只有两路 `selected_best_eval/eval_evidence.json` 都为 pass，筛选器才会按历史 best F1 决策；`|Delta F1| < 0.015` 选 CLRNet，否则选更高者。最后跑赢家 36 epoch：

```bash
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_pipeline.sh" baseline
```

36 epoch 的首次启动必须从 adapted CULane checkpoint 开启新 cosine schedule，不续接已经跑完的 15 epoch schedule；若自身中断，则在同一 run 目录从 `last_checkpoint` 恢复。

三条命令均可安全重跑：已通过阶段会跳过，有 `last_checkpoint` 的中断 run 会续跑；若 checkpoint 刚到 max_iter 而 final eval 在断电前未落盘，runner 会用该最后权重做 eval-only 恢复，并将带 `_recovered_final_eval` 标记的实测行追加到指标历史。非空但无 checkpoint 的目录会拒绝覆盖。为保证可复现，训练要求项目 tracked files 干净，且同一 run 禁止跨 commit 续跑。两套 config 和 runner 都锁定 `train.seed=42`、`cudnn_benchmark=False`和 `max_to_keep=40`。

> 上游 `BestCheckpointer` 不保存 `best_metric/best_iter`，进程恢复后可能用更差权重覆盖 `model_best.pth`。因此本项目不把该文件名当真值；依靠保留的周期 checkpoint + 内部 iteration + 独立回放裁决。

只有第 3–5 节证据全部返回，T2.1/T2.2 才能从进行中改为完成。

### 关机前交接包

每个 pipeline 阶段都会自动生成 `$HARDLANE_OUTPUT_ROOT/handoff_<stage>.tar.gz` 及其 `.json` SHA 报告。`baseline` 包含权重探针/smoke/门禁/筛选/决策证据、日志、回放预测，以及赢家 36ep 真实历史最优 checkpoint。关机前必须将 tar.gz 和 `.json` 一起下载，不要只看 `model_best.pth`。

## 6. 原始预测转提交格式（不可跳过）

`HardLaneEvaluator` 的 `val/predictions/` 为诊断保留 5 位小数，不能直接交给 `pack_submit.py`。必须统一走 `prepare_submit.py`，让每条线在 1 位小数序列化后重新通过塌缩与边界检查。

带 GT 的 val 演练在装有冻结 Oracle 的 CPU 环境执行：

```bash
export HARDLANE_RUN_DIR="$HARDLANE_OUTPUT_ROOT/one_epoch/clrnet_r50"
export HARDLANE_ORACLE_PYTHON=/absolute/path/to/frozen-oracle-python

python "$HARDLANE_PROJECT_ROOT/src/submit/prepare_submit.py" \
  --raw-pred-dir "$HARDLANE_RUN_DIR/val/predictions" \
  --canonical-dir "$HARDLANE_RUN_DIR/val/canonical_predictions" \
  --out-zip "$HARDLANE_RUN_DIR/val/submit_val.zip" \
  --manifest "$HARDLANE_PROJECT_ROOT/data/processed/manifest_val_v1_seed42.jsonl" \
  --gt-dir "$HARDLANE_DATA_ROOT/anno_txt" \
  --official-python "$HARDLANE_ORACLE_PYTHON" \
  --oracle-output "$HARDLANE_RUN_DIR/val/oracle.json" \
  --report "$HARDLANE_RUN_DIR/val/prepare_submit.json"
```

testA/testB 没有 GT，只运行 canonicalize→pack→verify，不传 Oracle 三个参数：

```bash
python "$HARDLANE_PROJECT_ROOT/src/submit/prepare_submit.py" \
  --raw-pred-dir /absolute/path/to/raw-test-predictions \
  --canonical-dir /absolute/path/to/canonical-test-predictions \
  --out-zip /absolute/path/to/submit.zip \
  --manifest "$HARDLANE_PROJECT_ROOT/data/processed/manifest_testA.jsonl" \
  --report /absolute/path/to/prepare_submit.json
```

默认任何缺预测、额外预测或 canonical 目录陈旧额外文件都会失败；正式提交禁止加 `--missing-as-empty`。结构化报告必须为 `status=pass`，zip 内坐标必须严格为 1 位小数。
