#!/usr/bin/env bash
set -Eeuo pipefail

experiment_root="${1:?usage: monitor_lvo.sh EXPERIMENT_ROOT RUNNER_PID}"
runner_pid="${2:?usage: monitor_lvo.sh EXPERIMENT_ROOT RUNNER_PID}"
monitor_path="$experiment_root/monitor.jsonl"

sample() {
  local status="running"
  local gpu="unknown"
  local last_iter=0
  local current_fold="none"
  local note
  if [[ -f "$experiment_root/lvo_training_complete.json" ]]; then
    status="complete"
  elif [[ -f "$experiment_root/runner.log" ]] &&
       grep -q 'fold_failed\|Traceback (most recent call last)' "$experiment_root/runner.log"; then
    status="failed"
  elif ! kill -0 "$runner_pid" 2>/dev/null; then
    status="stopped"
  fi
  gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr -d '\n' || true)"
  [[ -n "$gpu" ]] || gpu="unavailable"
  while IFS= read -r state_path; do
    if grep -q '"status": "running"' "$state_path"; then
      current_fold="$(basename "$(dirname "$state_path")")"
    fi
  done < <(find "$experiment_root/runs" -mindepth 2 -maxdepth 2 -name fold_state.json -type f -print 2>/dev/null | sort)
  while IFS= read -r log_path; do
    local_iter="$(tail -30 "$log_path" | sed -n 's/.*iter: \([0-9][0-9]*\).*/\1/p' | tail -1 || true)"
    if [[ -n "$local_iter" ]]; then
      last_iter="$local_iter"
    fi
  done < <(find "$experiment_root/runs" -mindepth 2 -maxdepth 2 -name train.log -type f -print 2>/dev/null | sort)
  note="runner_pid=$runner_pid; current_fold=$current_fold"
  printf '{"ts_utc":"%s","status":"%s","gpu":"%s","last_iter":%s,"note":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$gpu" "$last_iter" "$note" >> "$monitor_path"
}

while kill -0 "$runner_pid" 2>/dev/null; do
  sample
  sleep 1200
done
sample
