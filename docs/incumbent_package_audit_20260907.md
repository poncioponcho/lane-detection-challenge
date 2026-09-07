# incumbent testA 包复核（2026-09-07）

## 结论

当前生产 incumbent 包复核通过，未重新打包、未上传、未消耗 A 榜额度。

| 项 | 结果 |
|---|---|
| 文件 | `outputs/submit_testA_t05.zip` |
| 对应 A 榜记录 | `714962` |
| 官方得分 | `0.73444` |
| 文件数 | `900/900` |
| 包大小 | `537328` bytes |
| SHA-256 | `34e9b9346a5b06567a12cc7c7f75439133843dc1ae6ad7d5a164075f3ed7ccea` |
| CRC | PASS |
| 官方结构校验 | PASS |
| 仓库提交校验 | PASS |

## 复核项

1. 使用冻结的官方 `src/eval/official_oracle/check_submission.py`，对
   `data/raw/dataset/_extract/testA_full/Lane/data/testA.txt` 逐项检查，返回
   “校验通过：900 个结果文件的路径和文本格式均合法”。
2. 使用仓库 `src/submit/verify_submit.py`，在官方 Python 3.12 依赖环境
   `/private/tmp/lane-oracle-py312-fixed/bin/python` 中，以
   `data/processed/manifest_testA.jsonl` 的 900 个 `pred_rel_path` 作为精确文件集，
   返回 `verify PASS`，无缺失、无多余、无几何或序列化错误。
3. 使用 `unzip -tqq` 完成压缩包 CRC 检查，返回 `unzip_crc=PASS`。

## 交付纪律

- `outputs/testA_conf_candidates_20260905/conf_0p30/submit_testA.zip` 对应已提交记录
  `715300 / 0.72613`，不与本 incumbent 混用。
- 本次复核不触发官网提交；后续若进入最终提交，只能使用经过重新核验并明确登记的
  最终冻结包。
