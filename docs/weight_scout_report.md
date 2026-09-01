# UnLanedet 预训练权重可得性调研报告（weight-scout）

调研日期 2026-09-01｜委托方 DECISIONS §15.2｜状态 final

调研对象：zkyntu/UnLanedet（https://github.com/zkyntu/UnLanedet）
调研方式：只读网络调研（GitHub README / Model Zoo doc/model_zpp.md / config 源文件 / engine 源码）
用途：核实「恶劣场景下的车道线检测挑战赛」（HardLane-F100，9000 张，1366×720）baseline 从公开预训练权重 fine-tune 起步的权重可得性。

---

## 一、5 个调研问题逐条回答

### 问题 1：UnLanedet 是否提供 ADNet 预训练权重？

**是。** ADNet（ResNet34 主干）共有三套公开权重，均托管在 GitHub Releases（tag 为 `Weights`），直链可下：

| 数据集 | 指标 | 权重 URL |
|---|---|---|
| CULane | F1=77.88 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_culane.pth |
| TuSimple | Acc=96.65 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_tusimple.pth |
| VIL100 | F1=89.43 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_final_vil100.pth |

出处：Model Zoo 文档 `doc/model_zpp.md`（README "Model Zoo and Baselines" 一节链接指向该文件）。
注意：CULane F1=77.88 为复现版数字，model zoo 自述 "The performance of the model is not fully aligned with the original paper due to the time limit"，低于 ADNet 论文原报 ~80.5。

### 问题 2：CLRNet（ResNet-34 / DLA-34 主干）预训练权重可得性？

**部分可得。** CULane 下发布的 CLRNet 权重只有三种主干：**ResNet34 / ResNet50 / ConvNeXt-Tiny**：

| 数据集 | 主干 | 指标 | 权重 URL |
|---|---|---|---|
| CULane | ResNet34 | F1=78.99 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_r50_culane_model_best.pth |
| CULane | ResNet50 | F1=79.30 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_culane.pth |
| CULane | ConvNeXt-Tiny | F1=80.21 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_convnext_culane.pth |
| TuSimple | ResNet34 | Acc=96.64 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_tusimple.pth |

**DLA-34：UnLanedet 无权重、无 config。** README 2025-05-30 更新宣称 "We support DLA34 and ConvNexT backbone"，但 model zoo 未发布任何 DLA34 权重，`config/clrnet/` 下也无 dla34 配置文件。"CLRNet-DLA34 CULane F1@50=80.47" 为原版 CLRNet 官方仓库（https://github.com/vietanhdev/CLRNet）数字，不在 UnLanedet 生态内（本调研未核实原版仓库，该数字来源为委托方提供）。

**风险提示**：model zoo 表格中 ResNet34 行的下载链接文件名为 `clrnet_r50_culane_model_best.pth`（带 r50），与 ResNet50 行的 `clrnet_model_best_culane.pth` 疑似链接错位。下载后必须通过加载验证实际 backbone 结构，勿盲信文件名。

### 问题 3：config 中加载预训练权重的字段用法

UnLanedet 基于 detectron2 风格 LazyConfig，**没有 mmcv 式的 `load_from` 字段**。两级机制：

**(a) backbone ImageNet 预训练** —— config 内直接设 `pretrained=True`（config/clrnet/resnet34_culane.py、config/adnet/resnet34_culane.py 均如此）：

```python
model = L(CLRNet)(
    backbone = L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,                      # <-- 加载 torchvision ImageNet 权重
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck = L(FPN)(...),
    head = L(CLRHead)(...),
)
```

**(b) 完整模型权重（fine-tune 起点）** —— detectron2 风格，两条路径：
- `tools/train_net.py --config-file X --resume`：checkpointer 机制从 output_dir 中的 last_checkpoint 恢复/加载；
- 命令行 opts 覆盖：`unlanedet/engine/defaults.py` 的 default_argument_parser 明确支持（其 epilog 示例原文）：

```
$ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
```
（LazyConfig 用 `path.key=value` 形式覆盖。）

实操：把 CULane 权重路径通过 opts 传入（或放入 output_dir 配合 --resume）即可作为 fine-tune 初始化；backbone 部分权重与 config 匹配、head 最后一层因类别数/max_lanes 变化 shape 不同时按非 strict 加载。

### 问题 4：ADNet 在仓库中的配置文件路径与默认输入分辨率

配置文件（`config/adnet/` 下）：
- `config/adnet/resnet34_culane.py`（CULane）
- `config/adnet/resnet34_tusimple.py`（TuSimple）
- `config/adnet/resnet34_vil.py`（VIL100）

CULane 版关键参数（源文件原文摘录）：

```python
img_w = 800          # 训练输入宽
img_h = 320          # 训练输入高
ori_img_w = 1640     # 原图宽（CULane）
ori_img_h = 590      # 原图高（CULane）
cut_height = 270     # 顶部裁剪高度
sample_y = range(589, 230, -1)
```

结构：resnet34 + SA_FPN（in_channels=[128,256,512]）+ SPGHead（anchors_num=300, num_points=72, max_lanes=5）。

### 问题 5：CULane 预训练权重 fine-tune 到 HardLane-F100（1366×720）的机制判断

**顺畅。** ADNet 与 CLRNet 的 backbone+FPN 均为全卷积/anchor-based 结构，输入尺寸只是 config 参数：`img_w / img_h / ori_img_w / ori_img_h / cut_height / sample_y`。适配 1366×720 纯改 config 即可，backbone 权重完全匹配，无需 from-scratch。

需同步调整：
- `ori_img_w=1366, ori_img_h=720`，按视野重设 `cut_height` 与 `sample_y` 采样范围；
- 车道类别数 `num_classes` / `max_lanes` 按 HardLane-F100 标注调整（head 最后一层 shape 变化，加载权重非 strict 即可）；
- ADNet 的 SPGHead 构造参数 `img_width / img_height` 需与训练输入一致。

**ADNet 有公开权重（问题 1 已证），因此不触发换主干。** 若追求性能上限，CLRNet-R50（CULane F1=79.30）为备选。

---

## 二、UnLanedet CULane 复现分数表（完整出处）

出处：`doc/model_zpp.md`（https://github.com/zkyntu/UnLanedet/blob/main/doc/model_zpp.md），"CULane baselines" 一节。训练环境：AutoDL 单卡 3090/4090D 24GB。**注意：表中自述 "All models are trained from scratch"**，即这些权重均为从零训练（backbone 仅 ImageNet 初始化），不影响其作为 fine-tune 起点的可用性。

| Model | Venue | Backbone | F1 | 权重 |
|---|---|---|---|---|
| UFLD | ECCV | ResNet18 | 63.14 | 无（"-"） |
| CLRNet | CVPR | ResNet34 | 78.99 | clrnet_r50_culane_model_best.pth |
| CLRNet | CVPR | ResNet50 | 79.30 | clrnet_model_best_culane.pth |
| CLRNet | CVPR | ConvNeXt-Tiny | 80.21 | clrnet_convnext_culane.pth |
| CondLaneNet | ICCV | ResNet50 | 79.69 | 无（"-"） |
| CLRerNet | WACV | ResNet34 | 79.20 | clrernet_model_best_culane.pth |
| CLRerNet | WACV | ConvNeXT-Tiny | 79.89 | clrernet_convnext_culane.pth |
| ADNet | ICCV | ResNet34 | 77.88 | adnet_model_best_culane.pth |

权重 URL 前缀统一为 `https://github.com/zkyntu/UnLanedet/releases/download/Weights/`。
附：TuSimple 侧 ADNet-R34 Acc=96.65、CLRNet-R34 Acc=96.64；VIL100 侧 ADNet-R34 F1=89.43（同文档）。

---

## 三、总结论

**ADNet 权重可得=是；CLRNet 权重可得=是（R34/R50/ConvNeXt-Tiny，无 DLA-34）；建议主干=ADNet-R34（有 CULane 权重，无需换），性能上限备选 CLRNet-R50（CULane F1=79.30）。**

---

## 附：调研来源 URL 汇总

- 仓库主页：https://github.com/zkyntu/UnLanedet
- Model Zoo（权重与分数表）：https://github.com/zkyntu/UnLanedet/blob/main/doc/model_zpp.md
- 训练/加载方式文档：https://github.com/zkyntu/UnLanedet/blob/main/scripts/TRAIN.md
- ADNet CULane config：https://github.com/zkyntu/UnLanedet/blob/main/config/adnet/resnet34_culane.py
- CLRNet CULane config：https://github.com/zkyntu/UnLanedet/blob/main/config/clrnet/resnet34_culane.py
- 命令行参数/权重加载定义：https://github.com/zkyntu/UnLanedet/blob/main/unlanedet/engine/defaults.py
- 训练入口脚本：https://github.com/zkyntu/UnLanedet/blob/main/tools/train_net.py（经 scripts/train.sh 调用）
