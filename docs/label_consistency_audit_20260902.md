# 三格式标签一致性全量审计（2026-09-02）

## 结论

训练集 7100 张全量通过 T20/T21 门槛：badlist 仅 1 张，比例 `1/7100 = 0.0141%`，低于 `<0.1%` 验收线。

- `.lines.txt` ↔ JSON：7100/7100 图的 lane 数、顺序、每个 `(x,y)` 点均逐元素完全相同，共 24435 条线。
- 空 GT：262 张；三格式一致为空，未被解析器丢弃。
- PNG：官方文件是 palette instance PNG，OpenCV 会展开成 BGR；实例颜色不能只取单一通道。正式一致性门使用 10px 宽 `.lines.txt` 重绘与 PNG 非零前景的 union IoU，避免把遮挡/交汇下的有损中心线逆提取误判为源标签冲突。
- union IoU：min=0.728155、P1=0.886039、median=0.911731；阈值固定为 0.75，仅 `v576104564_1_0_17161/00120` 进入 badlist。

唯一 badlist 样本只有一条接近图像底边的极短二点线：`(422.8,711.9)→(428.7,704.9)`。JSON 与 lines 完全相同；PNG 前景 178 像素，差异来自短线端帽/取整对 IoU 的放大，不是需要删除或修标签的数据损坏。保留 badlist 供训练诊断，不从数据集中剔除。

## 修复的真实执行面缺陷

1. JSON 实际 schema 为 `annotations.lane[]`；旧 parser 只在合成 variants 上通过，真实 JSON 无法解析。
2. PNG 为调色板彩色实例图；旧代码固定取 `mask[...,0]`，会丢失红/绿通道实例，部分图甚至静默返回 0 条线。
3. 官方点序常为 bottom-to-top（y 递减）；旧 `mean_lateral_error` 直接调用要求 y 递增的 `np.interp`，同一条线可产生 90px 以上假误差。现已稳定按 y 排序，方向无关。
4. `parse_labels` 旧版只找图片同目录 sibling；真实标签分别位于 `anno_txt/`、`Json/`、`Annotations/`。现按官方目录结构解析，同时保留 sibling 模式用于合成测试。

## 复现

```bash
PYTHONPATH=src <lane-python> -m data.check_label_consistency \
  --manifest data/processed/manifest_train.jsonl \
  --lane-root data/raw/dataset/_extract/train_full/Lane \
  --report outputs/reports/label_consistency_train_20260902.json \
  --badlist data/processed/label_consistency_badlist.txt \
  --lat-err-tol 6.0 --count-tol 0 --mask-iou-tol 0.75
```

结构化报告与 badlist 属可再生产物，继续由 `/outputs/`、`/data/processed/*` 忽略；本审计文档与解析/测试代码入库。当前完整测试套件：76 passed（显式 `--basetemp=/private/tmp/...`）。
