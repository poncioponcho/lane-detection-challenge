# UnLanedet 预训练权重可得性调研报告（weight-scout）

调研日期 2026-09-01｜2026-09-02 执行面复核｜委托方 DECISIONS §15.2｜状态 pending AutoDL load probe

调研对象：zkyntu/UnLanedet（https://github.com/zkyntu/UnLanedet）
调研方式：只读网络调研（GitHub README / Model Zoo doc/model_zpp.md / config 源文件 / engine 源码）
用途：核实「恶劣场景下的车道线检测挑战赛」（HardLane-F100；当前已发布 train+testA 共 8000 张，testB 规模以官方清单为准；1366×720）baseline 从公开预训练权重 fine-tune 起步的权重可得性。

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
| CULane | ResNet34 | F1=78.99 | model-zoo 指向 `clrnet_r50_culane_model_best.pth`，**待 AutoDL load 探针裁决** |
| CULane | ResNet50 | F1=79.30 | model-zoo 指向 `clrnet_model_best_culane.pth`，**待 AutoDL load 探针裁决** |
| CULane | ConvNeXt-Tiny | F1=80.21 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_convnext_culane.pth |
| TuSimple | ResNet34 | Acc=96.64 | https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_tusimple.pth |

**DLA-34：UnLanedet 无权重、无 config。** README 2025-05-30 更新宣称 "We support DLA34 and ConvNexT backbone"，但 model zoo 未发布任何 DLA34 权重，`config/clrnet/` 下也无 dla34 配置文件。"CLRNet-DLA34 CULane F1@50=80.47" 为原版 CLRNet 官方仓库（https://github.com/vietanhdev/CLRNet）数字，不在 UnLanedet 生态内（本调研未核实原版仓库，该数字来源为委托方提供）。

**风险提示（2026-09-02 加强）**：model zoo 表格与训练日志/文件名互相矛盾。发布资产大小分别为 generic=263,994,164 bytes、named-r50=292,961,772 bytes；训练日志又明确 generic 日志对应 R34、r50 日志对应 R50。故上表链接映射不是事实，必须在 AutoDL 同时下载两份、对 R34/R50 实例化模型比较参数 shape 兼容率后裁决。`scripts/autodl/probe_weights.py` 已固化该门禁。

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

**(b) 完整模型权重（fine-tune 起点）** —— 2026-09-02 对 pinned commit `03921844220adb2e65c840de2d9759478d5c3d4c` 的真实调用链复核后修正：

- 新训练初始化：`tools/train_net.py` 实际调用 `checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)`；唯一有效字段是 **`train.init_checkpoint`**。
- 断点恢复：`--resume` + `train.output_dir` 内的 `last_checkpoint`。
- `MODEL.WEIGHTS` 只存在于残留的 detectron2 风格帮助文字，不被 LazyConfig 训练主路径读取，禁止再用。

正确覆盖示例：

```
python tools/train_net.py --config-file cfg.py train.init_checkpoint=/abs/path/adapted.pth
```

原始权重若有 shape 不匹配，PyTorch/fvcore 的非 strict 并不会自动吞掉同名异形张量。因此先由 AutoDL 探针生成过滤后的 adapted checkpoint，并把重新初始化参数清单写入证据 JSON，再交给 `train.init_checkpoint`。

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

**静态上可迁移，动态结论待 AutoDL。** ADNet 与 CLRNet 的 backbone+FPN 均为全卷积/anchor-based 结构，输入尺寸由 config 参数控制；但权重实际兼容率、CUDA 自定义算子、空 GT loss 与 demo 必须在 AutoDL 验收，不能由本地静态阅读代替。

需同步调整：
- `ori_img_w=1366, ori_img_h=720`，按视野重设 `cut_height` 与 `sample_y` 采样范围；
- CLRNet `max_lanes=8` 是 GT target 容量、`num_classes=9` 是辅助分割类别数、`num_priors=192` 保持模型容量；其源码 decode 原硬编码 `top_k=cfg.max_lanes`，须应用项目 patch 后由 `test_parameters.nms_topk=12` 控制；
- ADNet `max_lanes=8` 是 GT target 容量，`anchors_num/start_points_num=300` 保持模型容量，train/test `nms_topk=12` 控制候选；
- ADNet 的 SPGHead 构造参数 `img_width / img_height` 需与训练输入一致。

主干仍由 AutoDL 上 CLRNet-R50 vs ADNet-R34 的 15ep 筛选决定；本报告不提前改判。

---

## 二、UnLanedet CULane 复现分数表（完整出处）

出处：`doc/model_zpp.md`（https://github.com/zkyntu/UnLanedet/blob/main/doc/model_zpp.md），"CULane baselines" 一节。训练环境：AutoDL 单卡 3090/4090D 24GB。**注意：表中自述 "All models are trained from scratch"**，即这些权重均为从零训练（backbone 仅 ImageNet 初始化），不影响其作为 fine-tune 起点的可用性。

| Model | Venue | Backbone | F1 | model-zoo 展示文件名（CLR 两项待 AutoDL 裁决） |
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

**ADNet/CLRNet 发布资产可得；CLRNet 两文件的真实 backbone 映射尚未由 load 证据确认。工程候选保持 CLRNet-R50 与 ADNet-R34，必须在 AutoDL 完成权重探针、空/非空 loss、demo 和 1 epoch 后再进入 15ep 筛选。**

---

## 附：调研来源 URL 汇总

- 仓库主页：https://github.com/zkyntu/UnLanedet
- Model Zoo（权重与分数表）：https://github.com/zkyntu/UnLanedet/blob/main/doc/model_zpp.md
- 训练/加载方式文档：https://github.com/zkyntu/UnLanedet/blob/main/scripts/TRAIN.md
- ADNet CULane config：https://github.com/zkyntu/UnLanedet/blob/main/config/adnet/resnet34_culane.py
- CLRNet CULane config：https://github.com/zkyntu/UnLanedet/blob/main/config/clrnet/resnet34_culane.py
- 命令行参数/权重加载定义：https://github.com/zkyntu/UnLanedet/blob/main/unlanedet/engine/defaults.py
- 训练入口脚本：https://github.com/zkyntu/UnLanedet/blob/main/tools/train_net.py（经 scripts/train.sh 调用）
