#!/usr/bin/env bash
# 全新实例一键装机（实例过期重开场景，09-10）：环境 → 数据集 → 源权重 → 代码 → adapted 初始化。
# 用法: bash fresh_instance_setup.sh <ssh_port>
# 前置: 本地 repo（0ebb713+）与 data/ 完整；/tmp/clrnet_r50_culane_model_best.pth 已预下载（292,961,772 字节）。
# 幂等：各步可重跑（已存在则跳过或覆盖）。
set -uo pipefail
PORT="${1:?usage: fresh_instance_setup.sh <ssh_port>}"
SSH="ssh -i $HOME/.ssh/lane_id -p $PORT -o BatchMode=yes -o ConnectTimeout=20 root@i-1.gpushare.com"
SCP="scp -i $HOME/.ssh/lane_id -P $PORT -o BatchMode=yes -o ConnectTimeout=20"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY39="/usr/local/miniconda3/envs/py39/bin/python"
SRC_CKPT=/tmp/clrnet_r50_culane_model_best.pth

[ -f "$SRC_CKPT" ] || { echo "缺 $SRC_CKPT —— 先本地预下载（见 goal 日志）"; exit 1; }
LOCAL_SHA=$(shasum -a 256 "$SRC_CKPT" | awk '{print $1}')
LOCAL_BYTES=$(stat -f%z "$SRC_CKPT")

step() { echo "== [$(date -u +%H:%M:%S)] $* =="; }

step "1/7 体检"
$SSH "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /hy-tmp | tail -1" || { echo "SSH 不通，核对端口"; exit 2; }

step "2/7 py39 环境 + torch 2.1.2+cu118（镜像无 conda PATH，一律绝对路径）"
$SSH 'if [ ! -x /usr/local/miniconda3/envs/py39/bin/python ]; then /usr/local/miniconda3/bin/conda create -n py39 python=3.9 -y; fi; /usr/local/miniconda3/envs/py39/bin/python -m pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118' || exit 2

step "3/7 数据集上传（tar 文件→md5 双端核对→解包）"
$SSH "mkdir -p /hy-tmp/datasets/HardLane /hy-tmp/datasets/_staging /hy-tmp/weights /hy-tmp/lane-outputs"
# split_upload <local> <remote> — 200m 分块、3 路并行、逐块 md5+重试、已传块跳过（断点续传）、远端重组后整文件终验
split_upload() {
    local src=$1 dst=$2
    local base lmd5=$(md5 -q "$src")
    echo "  split_upload $(basename "$src") ($(du -h "$src" | cut -f1)) md5=$lmd5"
    # 整文件已在远端且 md5 一致 → 直接跳过（重组后重跑场景）
    local whole=$($SSH "md5sum $dst 2>/dev/null" | awk '{print $1}')
    if [ "$whole" = "$lmd5" ]; then
        echo "  $dst already verified on remote (whole-file skip)"
        return 0
    fi
    local staging=/hy-tmp/datasets/_staging
    rm -f /tmp/schunk_*
    split -b 200m "$src" /tmp/schunk_
    local chunks=(/tmp/schunk_*)
    echo "  ${#chunks[@]} chunks; resume-check remote existing..."
    # 断点续传：远端已有且 md5 一致的块直接跳过
    local todo=()
    local c bn l md5r attempt
    for c in "${chunks[@]}"; do
        bn=$(basename "$c")
        md5r=$($SSH "md5sum $staging/$bn 2>/dev/null" | awk '{print $1}')
        l=$(md5 -q "$c")
        if [ "$l" = "$md5r" ]; then
            echo "    chunk $bn already on remote (skip)"
        else
            todo+=("$c")
        fi
    done
    echo "  to upload: ${#todo[@]} chunks"
    # 3 路并行 worker
    upload_worker() {
        local w=$1 i=0
        for c in "${todo[@]}"; do
            i=$((i+1)); [ $((i % 3)) -eq $((w % 3)) ] || continue
            local bn=$(basename "$c") l md5r attempt
            for attempt in 1 2 3 4 5 6; do
                $SCP -q "$c" "root@i-1.gpushare.com:$staging/$bn" && break
                echo "    chunk $bn attempt $attempt failed; retry in 20s"; sleep 20
            done
            l=$(md5 -q "$c"); md5r=$($SSH "md5sum $staging/$bn" | awk '{print $1}')
            [ "$l" = "$md5r" ] || { echo "    chunk $bn md5 mismatch"; return 1; }
            echo "    chunk $bn OK (worker $w)"
        done
    }
    upload_worker 0 & local p1=$!
    upload_worker 1 & local p2=$!
    upload_worker 2 & local p3=$!
    wait $p1 $p2 $p3 || { echo "  a worker failed"; return 1; }
    $SSH "cat $staging/schunk_* > $dst && rm -f $staging/schunk_*"
    md5r=$($SSH "md5sum $dst" | awk '{print $1}')
    [ "$lmd5" = "$md5r" ] || { echo "  whole-file md5 mismatch"; return 1; }
    echo "  split_upload $(basename "$src") DONE"
    rm -f /tmp/schunk_*
}

cd "$REPO/data/raw/dataset/_extract"
echo "  3a. train_full/Lane 本地打包（不压缩，JPEG/PNG 零收益）"
tar -cf /tmp/train_lane.tar -C train_full Lane
step "3b/7 上传 train（2.1G，分块续传）"
split_upload /tmp/train_lane.tar /hy-tmp/datasets/_staging/train_lane.tar || { echo "train 上传失败"; exit 3; }
echo "  3c. testA JPEGImages 打包上传"
tar -cf /tmp/testa_jpg.tar -C testA_full/Lane JPEGImages
split_upload /tmp/testa_jpg.tar /hy-tmp/datasets/_staging/testa_jpg.tar || { echo "testA 上传失败"; exit 3; }
echo "  3d. 实例侧解包合并"
$SSH 'cd /hy-tmp/datasets/_staging && tar -xf train_lane.tar -C /hy-tmp/datasets/HardLane/ && tar -xf testa_jpg.tar -C /hy-tmp/datasets/HardLane/Lane/ && rm -f train_lane.tar testa_jpg.tar && find /hy-tmp/datasets/HardLane/Lane/JPEGImages -name "*.jpg" | wc -l && ls /hy-tmp/datasets/HardLane/Lane'

step "4/7 源权重上传（CULane R50，292,961,772B，分块）"
split_upload "$SRC_CKPT" /hy-tmp/weights/clrnet_r50_culane_model_best.pth || exit 3
$SSH "stat -c%s /hy-tmp/weights/clrnet_r50_culane_model_best.pth"
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
$SSH "export PATH=/usr/local/cuda/bin:\$PATH; cd /hy-tmp/lane-detection-challenge && export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane UNLANEDET_ROOT=/hy-tmp/UnLanedet HARDLANE_WEIGHTS_ROOT=/hy-tmp/weights HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs HARDLANE_PYTHON=$PY39 PYTHON_BIN=$PY39 && bash scripts/autodl/setup_unlanedet.sh && $PY39 scripts/autodl/probe_weights.py --no-download --output /hy-tmp/lane-outputs/weight_probe.json" || exit 2
$SSH "ls -la /hy-tmp/weights/adapted_clrnet_r50_hardlane.pth 2>/dev/null || ls /hy-tmp/weights/"

step "7/7 A2 发车（外链守卫自动等待）"
$SCP "$REPO/scripts/autodl/run_a2_full71_queue.sh" root@i-1.gpushare.com:/hy-tmp/lane-outputs/run_a2_full71_queue.sh
$SSH "chmod +x /hy-tmp/lane-outputs/run_a2_full71_queue.sh && cd /hy-tmp/lane-outputs && nohup bash run_a2_full71_queue.sh > a2_full71_queue.nohup.log 2>&1 & sleep 6; cat a2_full71_queue.status; pgrep -fc run_a2_full71_queue"
echo "== 完成。A2 4×36ep ≈9h；监控: tail -f /hy-tmp/lane-outputs/a2_full71_queue.log =="
