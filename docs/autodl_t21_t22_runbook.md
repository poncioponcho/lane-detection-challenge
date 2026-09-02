# T2.1/T2.2 AutoDL 验收手册

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

## 5. 各跑 1 epoch train + val

```bash
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_one_epoch.sh" clrnet_r50
bash "$HARDLANE_PROJECT_ROOT/scripts/autodl/run_one_epoch.sh" adnet_r34
```

检查两个 run 目录均有 `train.log`、checkpoint、`last_checkpoint`、`val/diagnostic_metric.json` 与 `val/predictions/`。用原生 `--resume` 做一次最小续跑，确认 iteration 从 checkpoint 后继续。训练期 metric 是项目诊断层；最终裁决仍须将预测交给冻结官方 Oracle。

以 CLRNet 为例，从 525 iter 的 checkpoint 续跑 1 iter（ADNet 换对应 config/run_dir）：

```bash
cd "$UNLANEDET_ROOT"
python tools/train_net.py --resume \
  --config-file "$HARDLANE_PROJECT_ROOT/configs/unlanedet/clrnet_r50_hardlane.py" \
  --num-gpus 1 \
  "train.max_iter=526" \
  "train.eval_period=526" \
  "train.checkpointer.period=526" \
  "train.output_dir=$HARDLANE_OUTPUT_ROOT/one_epoch/clrnet_r50" \
  "dataloader.evaluator.output_basedir=$HARDLANE_OUTPUT_ROOT/one_epoch/clrnet_r50/val"
```

日志必须明确从 iteration 525 恢复，而非重新加载 `train.init_checkpoint` 后从 0 开始。

只有第 3–5 节证据全部返回，T2.1/T2.2 才能从进行中改为完成。

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
