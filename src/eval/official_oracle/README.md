# 官方评测 Oracle（冻结文件，禁止改动）

来源：比赛官方「提交样例_A.zip」（讯飞挑战平台，2026-09-01 下载，数据落盘核对见 DECISIONS §17.1）。

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `score.py` | `b2f4c9b21de1083de8a420c106262e799947fac37c938994f7d160b8c62de0d2` | 官方评分算法：**所有最终 A/B 裁决的唯一入口** |
| `check_submission.py` | `c7897bdbd81d19b56ee0b6b9a803c0645428403ffa13c133e1ea41a422c02043` | 官方提交预检查（verify_submit 的对齐基准） |
| `requirements_official.txt` | `eb92e5ce6ff4b97db3de410bc068837f3038f83561f9bde581b5055530f67c05` | 官方评测环境 pins：numpy 2.1.3 / scipy 1.15.3 / opencv 4.12.0.88 |

规则（DECISIONS §17.1）：

1. 本目录文件**逐字节冻结、不可变**。禁止「修改后更新哈希」；官方若发布新版评测脚本，**新增版本目录**（`official_oracle_v2/`）并保留旧版，新旧并存可差分，且须在 DECISIONS.md 留裁决记录。`tests/test_official_oracle_hash.py` 对三份文件做 SHA-256 断言守护（2026-09-02 落地）。
2. 运行环境独立于训练环境：Python 3.12 + `requirements_official.txt`（已验证可建，见 §17.1）。训练与评测通过 `.lines.txt` + manifest 解耦，训练环境**不**随之降级。
3. 本地快速 metric（`src/eval/rasterize.py` / `matching.py`）只用于诊断与扫描；只有通过差分测试套件后才允许参与过程性判断，最终裁决一律以本目录 `score.py` 的读数为准。
