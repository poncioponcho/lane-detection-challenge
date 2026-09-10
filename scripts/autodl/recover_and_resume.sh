#!/usr/bin/env bash
# 实例恢复一键化：体检 → 部署+发车 A2 → 拉 OOF 供 OS 三门裁决。
# 用法: bash recover_and_resume.sh <new_ssh_port>
# 依赖: ~/.ssh/lane_id；恒源云重启后端口映射会变，端口以控制台"SSH连接"为准。
set -uo pipefail
PORT="${1:?usage: recover_and_resume.sh <new_ssh_port>}"
SSH="ssh -i $HOME/.ssh/lane_id -p $PORT -o BatchMode=yes -o ConnectTimeout=20 root@i-1.gpushare.com"
SCP="scp -i $HOME/.ssh/lane_id -P $PORT -o BatchMode=yes -o ConnectTimeout=20"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_OUT="$REPO_ROOT/outputs"
STAMP=$(date -u +%Y%m%d)

echo "== 1. 体检 =="
$SSH "uptime | tail -1; df -h /hy-tmp | tail -1; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; pgrep -af 'train_net|lvo_video|overnight_optimizer' | head -4" || { echo "SSH 仍不通，确认控制台端口"; exit 2; }

echo "== 2. 核验 codex 链折损（os_v4 fold 状态） =="
$SSH 'for f in /hy-tmp/lane-outputs/lvo_os_v4_15ep_20260909/experiment/runs/fold_*/fold_state.json; do printf "%s: " $(basename $(dirname $f)); grep -o "\"status\": \"[a-z]*\"" $f | head -1; done 2>/dev/null; ls -d /hy-tmp/lane-outputs/lvo_* 2>/dev/null'

echo "== 3. 部署+发车 A2 队列 =="
$SCP "$REPO_ROOT/scripts/autodl/run_a2_full71_queue.sh" root@i-1.gpushare.com:/tmp/run_a2_full71_queue.sh && \
$SSH "mv /tmp/run_a2_full71_queue.sh /hy-tmp/lane-outputs/run_a2_full71_queue.sh && chmod +x /hy-tmp/lane-outputs/run_a2_full71_queue.sh && cd /hy-tmp/lane-outputs && nohup bash run_a2_full71_queue.sh > a2_full71_queue.nohup.log 2>&1 & sleep 5; cat /hy-tmp/lane-outputs/a2_full71_queue.status; pgrep -fc run_a2_full71_queue"

echo "== 4. 拉 OOF（OS 三门裁决原料，双端 md5） =="
$SSH "cd /hy-tmp/lane-outputs && tar -czf /tmp/oof_for_gate_${STAMP}.tar.gz \$(ls -d lvo_os_v4*_15ep*/experiment/runs/fold_*/holdout_eval lvo_plain_v1_15ep_20260909/experiment/runs/fold_*/holdout_eval 2>/dev/null | sed 's|/hy-tmp/lane-outputs/||') 2>/dev/null; ls -la /tmp/oof_for_gate_${STAMP}.tar.gz; md5sum /tmp/oof_for_gate_${STAMP}.tar.gz" || { echo "OOF 打包失败——检查实验目录名后手工调整"; exit 3; }
$SSH "cat /tmp/oof_for_gate_${STAMP}.tar.gz" | cat > "/tmp/oof_for_gate_${STAMP}.tar.gz"
md5 -q "/tmp/oof_for_gate_${STAMP}.tar.gz"
mkdir -p "$LOCAL_OUT/gate_oof_${STAMP}" && tar -xzf "/tmp/oof_for_gate_${STAMP}.tar.gz" -C "$LOCAL_OUT/gate_oof_${STAMP}" && echo "解包到 $LOCAL_OUT/gate_oof_${STAMP}"

echo "== 完成。下一步（本地）：冻结 Oracle 全局评分 + riskon_oracle_gate_report.py 三门报告 =="
