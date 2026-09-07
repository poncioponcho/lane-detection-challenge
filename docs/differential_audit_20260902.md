# 非 identity 差分与下游防线审计（2026-09-02）

## 结论

`pred == GT` 的 identity 检查只能证明解析一致、链路可运行；因为所有自匹配 IoU 恒为 1，它不能独立证明渲染、0.5 阈值或匈牙利指派一致。为补足判别力，本轮使用真实 GT 派生的非平凡预测做了跨环境复核。

- 本地环境：OpenCV 5.0.0 / SciPy 1.18.1 / NumPy 2.5.2。
- 官方环境：OpenCV 4.12.0.88 / SciPy 1.15.3 / NumPy 2.1.3。
- 渲染层：1034 个 shift/亚像素/越界/折返/共线用例，mask 像素数、交集与 IoU 逐比特相同。
- 整图层：576 图、2009 条真实 GT 派生线，含 15% 漏检、FP 注入、0–35px 位移和一位小数序列化；逐图 TP/FP/FN 零分歧。两边全局均为 TP=1488、FP=812、FN=521、F1=0.690647；最近阈值边距 9.276e-05 的图仍同判。

因此，本地诊断 metric 可用于训练期扫描；最终裁决仍必须走冻结 Oracle。

## 下游缺口与修复

1. P0：校验器曾放行“连续去重后不足 2 点”，而官方会中断整份评分。现已在两层修复：导出器对一位小数序列化结果检查；最终校验器按官方语义再次检查并执行插值—绘制 smoke。
2. P1：校验器不再宽容逗号；边界严格为 x≤1365、y≤719；增加每图≤64 条、每条≤2048 点。
3. P1：manifest 现在校验 `image_path`、`image_id`、`pred_rel_path`、`gt_path` 与 clip/frame 的完整一致性。
4. P1：Oracle JSON 明示全局单次调用才是成绩，per-clip 只作诊断且禁止平均。576 图实测逐图 F1 均值比官方全局 F1 低 0.99pp。
5. P2：`interp_lane` 不再额外去重；去重只属于官方文本解析层，JSON/数组直喂的异常语义与 Oracle 一致。
6. P2：画布尺寸收口到 `common.types.CANVAS_W/H`，匹配参数引用 rasterize 的 `DEFAULT_*`，消除重复常量。

## 验收证据

- 本批次全套：66 passed；§20 后 74，§21 真实三格式测试加入后当前为 76 passed（均显式 `--basetemp=/private/tmp/...`，规避沙箱临时目录权限噪声）。
- 可复现脚本：`scripts/run_nonidentity_diff.py`；固定 seed=424242、请求 600 图（过滤空 GT 后实际 576 图），用例 SHA-256 `6d3b1061b059d2932f68bb3330617b9112b08cc4991767a2b302bcd643fbc96f`。
- 原致命 zip：`verify_submit` 从 PASS 改为 FAIL；控制组仍通过官方 parser。
- 真实训练标注：7100 文件 / 24435 线，全链 smoke PASS。
- testA：官方 manifest 900 路径，空预测打包与完整性校验 PASS。
- 机器报告：`outputs/reports/nonidentity_diff_20260902.json`；其他证据：`differential_audit_20260902.md`、`train_gt_submit_smoke.md`、`testA_empty_submit_smoke.md`。

复现命令（Oracle 临时环境存在时）：

```bash
<lane-venv>/bin/python scripts/run_nonidentity_diff.py \
  --local-python <lane-venv>/bin/python \
  --oracle-python /private/tmp/lane-oracle-py312/bin/python \
  --sample-images 600 --seed 424242 \
  --output outputs/reports/nonidentity_diff_20260902.json
```
