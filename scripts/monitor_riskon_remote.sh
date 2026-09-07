#!/usr/bin/env bash
# Read-only remote monitor for risk-on 960x384 LVO screen.
# Only reads remote state and appends to a local jsonl. Never touches remote
# processes, files, or the running experiment.
set +e

KEY="/Users/seyonmacbook/.ssh/lane_id"
REMOTE="root@i-1.gpushare.com"
PORT=34529
EX='/hy-tmp/lane-outputs/risk_on_res960x384_lvo_20260907'
OUT="/Users/seyonmacbook/WorkBuddy/恶劣场景下的车道线检测挑战赛/outputs/monitor_riskon_960x384_20260907.jsonl"
INTERVAL="${MONITOR_INTERVAL:-900}"

REMOTE_CMD="EX=$EX
status=\$(cat \$EX/status 2>/dev/null || echo unknown)
gpu=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
disk=\$(df -P /hy-tmp 2>/dev/null | awk 'NR==2{printf \"%.1f\", \$4/1048576}')
alive=\$(pgrep -f '[l]vo_video_runner.py' >/dev/null && echo yes || echo no)
folds_done=\$(ls -d \$EX/runs/fold_* 2>/dev/null | wc -l)
complete=\$(test -f \$EX/lvo_training_complete.json && echo yes || echo no)
cur=none; iter=0; eta=na; ckpt=0
for d in \$(ls -d \$EX/runs/fold_* 2>/dev/null | sort); do
  st=\$(cat \$d/fold_state.json 2>/dev/null | tr -d ' \n')
  case \"\$st\" in *'\"status\":\"running\"'*) cur=\$(basename \$d);; esac
done
if [ \"\$cur\" != none ]; then
  iter=\$(grep -oE 'iter: [0-9]+' \$EX/runs/\$cur/train.log 2>/dev/null | tail -1 | awk '{print \$2}')
  eta=\$(grep -oE 'eta: [0-9:]+' \$EX/runs/\$cur/train.log 2>/dev/null | tail -1 | awk '{print \$2}')
  ckpt=\$(ls \$EX/runs/\$cur/*.pth 2>/dev/null | wc -l)
fi
test -n \"\$iter\" || iter=0
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \"\$status\" \"\$alive\" \"\$cur\" \"\$iter\" \"\$eta\" \"\$ckpt\" \"\$folds_done\" \"\$disk\" \"\$gpu\""

while true; do
  ts_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sample="$(ssh -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=20 \
            -o StrictHostKeyChecking=no "$REMOTE" "$REMOTE_CMD" 2>/dev/null)"
  if [ -n "$sample" ]; then
    IFS=$'\t' read -r status alive cur iter eta ckpt folds_done disk gpu <<< "$sample"
    printf '{"ts_utc":"%s","status":"%s","runner_alive":"%s","current_fold":"%s","iter":%s,"eta":"%s","ckpt_in_fold":%s,"fold_dirs":%s,"disk_avail_gb":%s,"gpu":"%s"}\n' \
      "$ts_utc" "$status" "$alive" "$cur" "${iter:-0}" "$eta" "${ckpt:-0}" \
      "${folds_done:-0}" "${disk:-na}" "$gpu" >> "$OUT"
  else
    printf '{"ts_utc":"%s","status":"ssh_unavailable"}\n' "$ts_utc" >> "$OUT"
  fi
  if [ "$status" = "complete" ] || [ "$complete_seen" = 1 ]; then
    break
  fi
  sleep "$INTERVAL"
done
