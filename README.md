# HardLane Challenge — Lane Detection in Adverse Scenarios

> 科大讯飞 2026 AI 开发者大赛 · 恶劣场景下的车道线检测挑战赛  
> **A 榜 F1 = 0.73574**（基线 54-epoch CLRNet-R50）  
> **B 榜 F1 = 0.76053**（共识融合 + 门槛阶梯 k=1）  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 项目概述

车道线检测的难点不仅在模型架构，更在于**恶劣天气（雨/雾/夜间/逆光）**下的召回率暴跌和**评测指标的精确复刻**。本项目在 16 天单人赛期内交付了一套完整的工程体系，包含：

- **可插拔分层架构**（数据 → 模型 → 评测 → 提交）
- **双层评测体系**（官方 Oracle 冻结 + 本地诊断层 + 差分测试套件）
- **共识融合定价理论**（加线边际收益公式 + 盈亏线 θ = F1/2）
- **21 个测试文件 / 163 个测试用例**（Oracle 哈希 / 指标差分 / 提交校验 / 几何往返）
- **29 个核心 Python 模块** + **17 份配置** + **46 份文档** + **51 个实验脚本**

---

## 🏗️ 系统架构

```mermaid
flowchart TB
    subgraph DATA["① Data Layer — src/data/"]
        D1[manifest.py — ordered task list]
        D2[parse_labels.py — 3-format parser]
        D3[split_by_clip.py — video-level hold-out]
    end

    subgraph INTEG["② UnLanedet Bridge — src/integrations/"]
        M1[unlanedet_hardlane.py — manifest Dataset]
        M2[unlanedet_clrernet.py — CLRNet adapter]
        M3[unlanedet_vat.py — VAT training support]
    end

    subgraph EVAL["③ Evaluation Layer — src/eval/"]
        E0[official_oracle/score.py — frozen ⛔]
        E1[oracle_runner.py — subprocess adapter]
        E2[rasterize.py + matching.py — local metric]
        E3[diff_test.py — differential conformance]
    end

    subgraph SUBMIT["④ Delivery Pipeline — src/submit/"]
        V1[prepare_submit.py — 1-decimal canonicalize]
        V2[pack_submit.py — submit.zip builder]
        V3[verify_submit.py — 5-class error interceptor]
    end

    DATA --> INTEG --> EVAL --> SUBMIT
    EVAL -.-> Oracle Gate -.-> SUBMIT
```

**核心设计约束**：

| 约束 | 实现 |
|------|------|
| 坐标系不可变 | 全项目 1366×720 原图绝对像素；`Lane` dataclass 强类型（[src/common/types.py](src/common/types.py)） |
| 评测中立 | 本地 metric 严禁 import 模型层；Oracle 冻结后 SHA-256 断言守护（[src/eval/oracle_integrity.py](src/eval/oracle_integrity.py)） |
| 提交安全网 | prepare → pack → verify 三段流水线，拦截重复点/越界/资源超限等 5 类错误（[src/submit/verify_submit.py](src/submit/verify_submit.py)） |
| 常量唯一事实源 | 目标/预算/闸门参数只在 [configs/default.yaml](configs/default.yaml) 定义，代码侧禁止硬编码 |

---

## 🔑 技术亮点

### 1. 双层评测架构（Oracle 冻结 + 本地诊断）

官方 `score.py` 逐字节冻结于 `src/eval/official_oracle/`（SHA-256 存证），通过 [oracle_runner.py](src/eval/oracle_runner.py) 适配层在独立 py3.12 环境中以子进程调用。本地 metric（`rasterize.py` + `matching.py`）定位为诊断/扫描层，必须先通过 [diff_test.py](src/eval/diff_test.py) 差分测试套件才能参与过程判断。

```python
# src/eval/oracle_runner.py —— 生产入口
def run_official_eval(pred_dir, gt_dir, manifest, *, official_python, ...):
    hashes = verify_oracle_files()           # 运行前哈希断言
    env = verify_official_environment(...)   # 强制 py3.12 + 官方 pins
    result = _run_once(official_python, ...) # 子进程调用官方脚本
    return OfficialEvalResult(global_=..., per_clip=...)
```

**实测差异**（DECISIONS §17.1）：匈牙利 cost 构造（官方 `1−iou` vs 本地硬屏蔽）、稠密化方式（参数均匀 vs 弧长）、float32 vs float64 — 三处方向均为本地系统性偏高。Oracle 作为唯一裁决入口消除了自证循环。

### 2. 共识融合与加线定价理论

多个独立模型的预测可以通过**共识投票**融合：在同一位置至少 k 个模型都预测到了的车道线才被保留（或新增）。通过 F1 恒等式 `F1 = 2·TP/(P+G)` 可以推导出定价公式：

> **盈亏线 θ = F1/2** — 新增线的真线率必须超过此值才划算  
> **边际收益公式** — `ΔF1 = 2·n·(r − θ) / (P + G)`，其中 r = 真线率，n = 加线量

**关键推论**：只要 r > θ，ΔF1 与 n 正相关 — 加线越多 EV 越高。这在直觉上不明显但数学上严格成立。

**B 榜实测**：基准共识线 F1 = 0.71459 → θ = 0.3573。各门槛档位的边际真线率均远高于此：

| 门槛 | 加线量 | 真线率 r | ΔF1 |
|:---:|:---:|:---:|:---:|
| 56% | +189 | 0.82–0.90 | 基准 |
| 33% | +381 | 0.705 | +2.56pp |
| 22% | +568 | 0.577 | +3.65pp |
| 11% | +880 | 0.431 | **+4.59pp** |

共识阶梯是整个比赛中性价比最高的改动（零训练成本、纯粹推理后处理）。核心洞察：**共识新增线在 testB 上真线率远高于 testA**（0.82–0.90 vs 0.378），因为 testB 更难，基础模型的 FN 更多，共识补回的是**真缺失**而非噪声。

### 3. 提交安全网（三段流水线）

```
raw predictions ──▶ prepare_submit.py (1-decimal canonicalize)
                  ──▶ pack_submit.py (submit.zip, root = submit/)
                  ──▶ verify_submit.py (inject-error-resistant)
```

verify 层（[src/submit/verify_submit.py](src/submit/verify_submit.py)）硬拦截：

| 检查项 | 判定 |
|--------|------|
| 顶层目录必须是 `submit/` | 缺或多 → 拒绝 |
| 文件集合与清单精确匹配 | 漏/多/大小写 → 拒绝 |
| UTF-8 编码 / 偶数 token / ≥4 值 | 格式错误 → 拒绝 |
| 1 位小数精度 / 有限值 / 无 NaN/Inf | 精度退化 → 拒绝 |
| 坐标在 [0,1365]×[0,719] / 连续点去重后 ≥2 | 非法几何 → 拒绝 |
| 每图 ≤64 线 / 每条 ≤2048 点 | 资源超限 → 拒绝 |
| 全链 smoke test（官方 parse→interp→draw） | 推理异常 → 拒绝 |
| 压缩后 <200MB | 体积超限 → 拒绝 |

### 4. Oracle 冻结机制

```python
# tests/test_official_oracle_hash.py —— 单元测试守护
ORACLE_SHA256 = {
    "score.py":            "b2f4c9b21de1083de8a420c106262e799947fac37c938994f7d160b8c62de0d2",
    "check_submission.py": "<sha256>",
}
def test_oracle_not_tampered():
    for name, expected in ORACLE_SHA256.items():
        actual = sha256_file(ORACLE_DIR / name)
        assert actual == expected, f"Oracle tampered: {name}"
```

冻结策略：**禁止修改已冻结文件**；官方发布新版时新增版本目录 `official_oracle_v2/`，保留旧版。避免了「改评测脚本适配自家模型」的自证循环。

### 5. 可配置/可开关的架构设计

每个设计决策都是一个独立的配置开关（[configs/default.yaml](configs/default.yaml) + [configs/unlanedet/](configs/unlanedet/)）：

```python
# configs/unlanedet/clrnet_r50_hardlane.py — LazyConfig 风格
model = dict(
    name="CLRNet",
    backbone=dict(name="ResNet", layers=[3, 4, 6, 3], out_indices=(1, 2, 3)),
    # ... 所有超参数均可切换
)
data = dict(
    input_size=(800, 320),     # 可切 960×480 / 1366×720 ...
    cut_height=180,            # 可切 0 / 400 ...
    max_lanes=8,               # 训练容量
    nms_topk=12,               # 推理候选保留
)
```

实验纪律：每次只改一个变量；`configs/default.yaml::model.backbone_screen` 定义双路筛选候选，`fallback_chain` 定义主干降级路径。

---

## 📁 目录结构

```
├── src/                          # 核心库代码（29 模块）
│   ├── common/                   # 共享类型 + 几何辅助 + IO + 校验和
│   │   ├── types.py              #   Lane dataclass + 几何 helpers
│   │   ├── io_utils.py           #   .lines.txt / .json / .png 读写
│   │   └── checksum.py           #   SHA-256 + 精确字节数
│   ├── data/                     # 数据层（manifest 驱动）
│   │   ├── manifest.py           #   有序任务清单 + 断言
│   │   ├── parse_labels.py       #   三格式统一解析
│   │   ├── split_by_clip.py      #   video-level hold-out 切分
│   │   └── eda.py                #   7100 张三格式一致性 + 分布体检
│   ├── eval/                     # 评测层（双层：Oracle + 诊断）
│   │   ├── oracle_runner.py       #   生产入口（子进程调用官方脚本）
│   │   ├── oracle_integrity.py    #   哈希断言 + 环境校验
│   │   ├── rasterize.py          #   B 样条稠密化 + cv2 无抗锯齿描边
│   │   ├── matching.py           #   Hungarian IoU > 0.5 指派
│   │   ├── official_metric.py    #   本地 compute_f1 包装
│   │   ├── diff_test.py          #   本地 vs Oracle 差分测试套件
│   │   └── official_oracle/ ⛔    #   冻结官方 score.py / check_submission.py
│   ├── integrations/             # UnLanedet → HardLane 桥接
│   │   ├── unlanedet_hardlane.py  #   manifest Dataset + evaluator
│   │   ├── unlanedet_clrernet.py  #   CLRNet 特定配置
│   │   └── unlanedet_vat.py       #   VAT 训练脚本
│   └── submit/                   # 交付层（三段流水线）
│       ├── prepare_submit.py     #   任意精度 → 1 位小数 canonical
│       ├── pack_submit.py        #   submit.zip 打包
│       └── verify_submit.py      #   5 类错误硬拦截
│
├── scripts/                      # 一次性实验脚本（51 个）
│   ├── build_testB_consensus_union_20260913.py   # 共识融合构建
│   ├── exp_consensus_gate_calibration_20260914.py # 门槛标定
│   ├── build_testB_candidates.sh                 # B 档候选一键构建
│   └── autodl/                   # AutoDL GPU 环境训练/评测
│
├── configs/                      # LazyConfig 模型配置（17 个）
│   ├── default.yaml              #   ⭐ 全局常量唯一事实源
│   ├── splits/v1_seed42.yaml     #   8 段 val hold-out 定义
│   └── unlanedet/                #   CLRNet-R50 / ADNet-R34 / ConvNeXt-T
│
├── tests/                        # 单元测试（21 文件 / 163 用例）
│
├── data/processed/               # manifest 索引 + 场景标签（Git 放行）
│
├── patches/                      # UnLanedet 上游 patch
│
└── docs/                         # 完整设计文档 + 实验报告（46 份）
    ├── ARCHITECTURE.md           #   系统架构 v2.9（~1000 行）
    ├── DECISIONS.md              #   关键决策记录（~1200 行）
    ├── PRD.md                    #   产品需求 v2
    ├── eda.md                    #   数据探索报告
    └── consensus_gate_calibration_20260914.md
```

---

## 📊 实验结果

### 基线与后处理探索

| 改动 | 类型 | ΔF1 | 判定 |
|------|------|-----|------|
| CLRNet-R50 36ep baseline | 训练 | — | **incumbent 0.73574** |
| conf=0.40 阈值扫描 | 后处理 | +0.417pp → +0.001pp (LVO) | ❌ 模型特有噪声 |
| 960×384 分辨率调整 | 预处理 | +0.028pp | ❌ 不稳定 |
| ConvNeXt-Tiny 15ep screen | 主干切换 | −0.009pp | ❌ NO-GO |
| cls_loss_weight=3.0 FP 定向 | 训练 | −0.607pp | ❌ Red |

### 共识融合阶梯（B 榜实测）⭐

| 门槛 (min-support) | 加线量 | B 榜 F1 | ΔF1 相对基准 | 边际真线率 |
|:---:|:---:|:---:|:---:|:---:|
| 56% (基准) | +189 | 0.71459 | — | 0.82–0.90 |
| 33% | +381 | 0.74021 | +2.56pp | 0.705 |
| 22% | +568 | 0.75461 | +3.65pp | 0.577 |
| **11% (最优)** | **+880** | **0.76053** | **+4.59pp** | 0.431 |

**阶梯的可定价性**（`scripts/oracle_score_tree.py` 实测）：在盈亏线 θ = 0.3573 之上，档位越高（门槛越低 → 加线越多）收益越大。最终选择 11% 档，虽然真线率最低但仍远高于 θ，且加线量最大。

### 已排除的负方向（实测均负/无效）

> margin40 trim (−1.855pp on testB) · flip-TTA · 端点外推 · 全局仿射 · 多种子 soup · VAT · ConvNeXt-T · 雾雨增强 · segmask binary · occlusion · 跨种子 union（单交 ≈ −2pp，只有 full union 不亏）

---

## 🚀 快速开始

### 环境要求

| 层 | 版本 | 理由 |
|---|---|---|
| Python (训练) | 3.10.13 | AutoDL 4090 环境锁定 |
| Python (Oracle) | **3.12** | 官方 score.py 环境（独立 venv） |
| PyTorch | 2.1.2 + CUDA 11.8 | 训练 |
| numpy | **1.26.4** (训练) / **2.1.3** (Oracle) | 训练侧 ABI 兼容；Oracle 跟随官方 |
| scipy | 1.11.4 (训练) / **1.15.3** (Oracle) | spline 实现；**评测端与后处理端必须同版本** |
| opencv-python | 4.9.0.80 (训练) / **4.12.0.88** (Oracle) | `cv2.line(thickness=30, lineType=8)` 无抗锯齿是 IoU 的地基 |

### 运行单元测试

```bash
# 确保项目根目录在 PYTHONPATH 中
export PYTHONPATH=$PWD/src:$PWD

# 21 文件 / 163 用例
pytest tests/ -v
```

### 评测

```bash
# Oracle（裁决层，唯一生产入口）
export ORACLE_PYTHON=~/venvs/lane-oracle-py312/bin/python
export PYTHONPATH=$PWD/src

python -m src.eval.oracle_runner \
    --pred-dir outputs/preds/exp_id/testB/ \
    --gt-dir data/gt_testB/anno_txt/ \
    --manifest data/processed/manifest_testB.jsonl \
    --official-python $ORACLE_PYTHON \
    --output outputs/reports/exp_id/oracle.json
```

### 提交流水线

```bash
export PYTHONPATH=$PWD/src

# 三段式：prepare → pack → verify
python -m src.submit.prepare_submit raw_preds/testB/ staged/testB/
python -m src.submit.pack_submit raw_preds/testB/ outputs/submit.zip
python -m src.submit.verify_submit outputs/submit.zip
# ✅ / ❌ 每条验证项有明确报告
```

---

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 框架 | [UnLanedet](https://github.com/OpenDriveLab/UnLaneDet)（pinned commit + [custom patch](patches/unlanedet_hardlane.patch)） |
| 主干 | CLRNet-R50（CULane 预训练 fine-tune） |
| 训练引擎 | AutoDL 4090 + UnLanedet `SimpleTrainer` + AMP |
| 评测 | 官方 score.py（冻结） + 自制 `oracle_runner` 子进程适配 |
| 数据管理 | manifest-backed torch Dataset + 有序 JSONL |
| 统计 | Hungarian IoU matching + 全局 F1 汇总（禁止每段平均） |

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层系统架构 v2.9 — 完整接口定义、mermaid 图、P0 需求覆盖表 |
| [DECISIONS.md](docs/DECISIONS.md) | 全部关键决策记录 — 目标裁决、算力测算、闸门重定义、框架选型 |
| [PRD.md](docs/PRD.md) | 产品需求 v2 |
| [eda.md](docs/eda.md) | 7100 张三格式一致性校验 + 场景分布体检 |
| [official_rules.md](docs/official_rules.md) | 赛题规则摘录 |
| [consensus_gate_calibration_20260914.md](docs/consensus_gate_calibration_20260914.md) | 共识门槛标定全过程（OOF 域） |
| [measurement_results_20260915.md](docs/measurement_results_20260915.md) | LVO OOF 实测结果汇总 |

---

## 💡 经验教训

1. **离线 OOF 的结论不可盲目迁移到测试集** — margin40 在 train 域 +2.25pp，testB 域 −1.855pp，符号完全相反。原因是 OOF 的 train-like 分布 ≠ B 榜的 harder distribution。
2. **F1 = 2·TP/(P+G) → 盈亏线 θ = F1/2** 是定价一切加线/删线操作的通用公式，**与模型架构、共识门槛、加线量 n 均无关**。它直接给出阈值扫描、共识融合、删 FP 操作的经济判据。
3. **只要 r > θ，加线越多越好** — 这个看似违反直觉的单调性来自 θ 不随 n 变化的事实。它决定了共识阶梯的最优解在最激进档（11%）而非预期的中间档（44%）。
4. **本地 metric 必须有独立的差分测试套件** — 否则「自制尺子配合自制数据」的自证循环不可避免。diff_test.py 用 2 段 200 图 / 846 条 GT 逐数比对，发现并修正了 3 处系统性偏差。
5. **冻结官方脚本 = 最有性价比的一行代码** — SHA-256 断言 + 独立环境 + 子进程调用，彻底消除「评测脚本适配自家模型」的质疑。tests/test_official_oracle_hash.py 让任何提交到 Oracle 的改动都变成红灯。
6. **subprocess + frozen script 的适配层模式可复用** — 不修改第三方代码、不导入到主进程、不改写其算法，在 AI/竞赛/企业环境审计场景都适用。

---

## 📝 License

本仓库代码部分基于 [MIT License](LICENSE) 开放。官方评测脚本（`src/eval/official_oracle/`）归原作者所有，冻结使用。
