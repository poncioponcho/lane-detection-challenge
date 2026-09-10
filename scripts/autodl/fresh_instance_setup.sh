#!/usr/bin/env bash
# 全新实例一键装机（实例过期重开场景，09-10）：环境 → 数据集 → 源权重 → 代码 → adapted 初始化。
# 用法: bash fresh_instance_setup.sh <ssh_port>
# 前置: 本地 repo（0ebb713+）与 data/ 完整；/tmp/clrnet_r50_culane_model_best.pth 已预下载（292,961,772 字节）。
# 幂等：各步可重跑（已存在则跳过或覆盖）。
set -uo pipefail
PORT="${1:?usage: fresh_instance_setup.sh <ssh_port>}"
SSH="ssh -i $HOME/.ssh/lane_id -p $PORT -o BatchMode=yes -o ConnectTimeout=20 root@i-1.gpushare.com"
SCP="scp -i $HOME/.ssh/lane_id -P $PORT -o BatchMode=yes -o ConnectTimeout=20 root@i-1.gpushare.com"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY39="/usr/local/miniconda3/envs/py39/bin/python"
SRC_CKPT=/tmp/clrnet_r50_culane_model_best.pth

[ -f "$SRC_CKPT" ] || { echo "缺 $SRC_CKPT —— 先本地预下载（见 goal 日志）"; exit 1; }
LOCAL_SHA=$(shasum -a 256 "$SRC_CKPT" | awk '{print $1}')
LOCAL_BYTES=$(stat -f%z "$SRC_CKPT")

step() { echo "== [$(date -u +%H:%M:%S)] $* =="; }

step "1/7 体检"
$SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /hy-tmp | tail -1" || { echo "SSH 不通，核对端口"; exit 2; }

step "2/7 py39 环境 + torch 2.1.2+cu118"
$SSH 'if [ ! -x /usr/local/miniconda3/envs/py39/bin/python ]; then conda create -n py39 python=3.9 -y; fi; /usr/local/miniconda3/envs/py39/bin/python -m pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118' || exit 2

step "3/7 数据集上传（tar 文件→md5 双端核对→解包）"
$SSH "mkdir -p /hy-tmp/datasets/HardLane /hy-tmp/datasets/_staging /hy-tmp/weights /hy-tmp/lane-outputs"
cd "$REPO/data/raw/dataset/_extract"
echo "  3a. train_full/Lane 本地打包"
tar -czf /tmp/train_lane.tar.gz -C train_full Lane
T_MD5=$(md5 -q /tmp/train_lane.tar.gz); echo "  local md5: $T_MD5"
echo "  3b. 上传（4.7G，取决于上行带宽）"
$SCP /tmp/train_lane.tar.gz root@i-1.gpushare.com:/hy-tmp/datasets/_staging/train_lane.tar.gz
R_MD5=$($SSH "md5sum /hy-tmp/datasets/_staging/train_lane.tar.gz" | awk '{print $1}')
echo "  remote md5: $R_MD5"
[ "$T_MD5" = "$R_MD5" ] || { echo "MD5 不一致，中止"; exit 3; }
echo "  3c. testA JPEGImages 打包上传"
tar -czf /tmp/testa_jpg.tar.gz -C testA_full/Lane JPEGImages
$SCP /tmp/testa_jpg.tar.gz root@i-1.gpushare.com:/hy-tmp/datasets/_staging/testa_jpg.tar.gz
echo "  3d. 实例侧解包合并"
$SSH 'cd /hy-tmp/datasets/_staging && tar -xzf train_lane.tar.gz -C /hy-tmp/datasets/HardLane/ && tar -xzf testa_jpg.tar.gz -C /hy-tmp/datasets/HardLane/Lane/ && rm -f train_lane.tar.gz testa_jpg.tar.gz && find /hy-tmp/datasets/HardLane/Lane/JPEGImages -name "*.jpg" | wc -l && ls /hy-tmp/datasets/HardLane/Lane'

step "4/7 源权重上传（CULane R50，292,961,772B）"
$SCP "$SRC_CKPT" root@i-1.gpushare.com:/hy-tmp/weights/clrnet_r50_culane_model_best.pth
$SSH "md5sum /hy-tmp/weights/clrnet_r50_culane_model_best.pth; stat -c%s /hy-tmp/weights/clrnet_r50_culane_model_best.pth"
echo "  local sha256: $LOCAL_SHA bytes: $LOCAL_BYTES"

step "5/7 repo bundle + manifests"
cd "$REPO"
git bundle create /tmp/hardlane_full.bundle risk-on-res960x384-screen
$SCP /tmp/hardlane_full.bundle root@i-1.gpushare.com:/tmp/hardlane_full.bundle
$SSH 'if [ ! -d /hy-tmp/lane-detection-challenge/.git ]; then git init -q /hy-tmp/lane-detection-challenge && cd /hy-tmp/lane-detection-challenge && git fetch -q /tmp/hardlane_full.bundle "risk-on-res960x384-screen:risk-on-res960x384-screen" && git checkout -q risk-on-res960x384-screen; else cd /hy-tmp/lane-detection-challenge && git fetch -q /tmp/hardlane_full.bundle && git merge --ff-only FETCH_HEAD; fi; git -C /hy-tmp/lane-detection-challenge rev-parse --short HEAD'
$SCP "$REPO/data/processed/manifest_train.jsonl" root@i-1.gpushare.com:/hy-tmp/lane-outputs/experiments_manifest_train_all71.jsonl
$SSH "mkdir -p /hy-tmp/lane-detection-challenge/data/processed" && \
$SCP "$REPO/data/processed/manifest_train.jsonl" "$REPO/data/processed/manifest_val_v1_seed42.jsonl" "$REPO/data/processed/manifest_testA.jsonl" root@i-1.gpushare.com:/hy-tmp/lane-detection-challenge/data/processed/
$SSH "wc -l /hy-tmp/lane-outputs/experiments_manifest_train_all71.jsonl /hy-tmp/lane-detection-challenge/data/processed/manifest_testA.jsonl"

step "6/7 UnLanedet pinned 环境 + adapted 权重构建"
$SSH "cd /hy-tmp/lane-detection-challenge && export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane UNLANEDET_ROOT=/hy-tmp/UnLanedet HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs HARDLANE_PYTHON=$PY39 PYTHON_BIN=$PY39 && bash scripts/autodl/setup_unlanedet.sh && $PY39 scripts/autodl/probe_weights.py --no-download --output /hy-tmp/lane-outputs/weight_probe.json" || exit 2
$SSH "ls -la /hy-tmp/weights/adapted_clrnet_r50_hardlane.pth 2>/dev/null || ls /hy-tmp/weights/"

step "7/7 A2 发车（外链守卫自动等待）"
$SCP "$REPO/scripts/autodl/run_a2_full71_queue.sh" root@i-1.gpushare.com:/hy-tmp/lane-outputs/run_a2_full71_queue.sh
$SSH "chmod +x /hy-tmp/lane-outputs/run_a2_full71_queue.sh && cd /hy-tmp/lane-outputs && nohup bash run_a2_full71_queue.sh > a2_full71_queue.nohup.log 2>&1 & sleep 6; cat a2_full71_queue.status; pgrep -fc run_a2_full71_queue"
echo "== 完成。A2 4×36ep ≈9h；监控: tail -f /hy-tmp/lane-outputs/a2_full71_queue.log =="
