#!/usr/bin/env bash
set -Eeuo pipefail

through="${1:-gate}"
case "$through" in
  gate|screen|baseline) ;;
  *) echo "usage: $0 {gate|screen|baseline}" >&2; exit 2 ;;
esac

for name in HARDLANE_PROJECT_ROOT HARDLANE_DATA_ROOT UNLANEDET_ROOT \
  HARDLANE_WEIGHTS_ROOT HARDLANE_OUTPUT_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "ERROR: $name must be set to an absolute AutoDL path" >&2
    exit 2
  fi
done

python_bin="${HARDLANE_PYTHON:-python}"
project="$HARDLANE_PROJECT_ROOT"
output="$HARDLANE_OUTPUT_ROOT"

if ! git -C "$project" diff --quiet || ! git -C "$project" diff --cached --quiet; then
  echo "ERROR: tracked project files are dirty; commit before AutoDL execution" >&2
  exit 2
fi
current_head="$(git -C "$project" rev-parse HEAD)"

json_passes_for_head() {
  "$python_bin" -c \
    'import json,sys; v=json.load(open(sys.argv[1])); raise SystemExit(v.get("status") != "pass" or v.get("project_git_head") != sys.argv[2])' \
    "$1" "$current_head" 2>/dev/null
}

bash "$project/scripts/autodl/setup_unlanedet.sh"

# UnLanedet's pinned modelzoo configs (e.g. config/clrnet/resnet34_culane.py via
# modelzoo.get_config) load "config/common/train.py" through a CWD-relative path,
# so every downstream step (weight probe, dataloader/loss smoke, gate/screen/
# baseline training, eval) must run from the UnLanedet repo root. All challenge-
# side paths below are env-var absolute (git -C / $project / $output), so this
# chdir cannot break them; it only fixes the pinned repo's relative lookups.
cd "$UNLANEDET_ROOT"

weight_report="$output/weight_probe.json"
if [[ -f "$weight_report" ]] && json_passes_for_head "$weight_report" && \
   [[ -f "$HARDLANE_WEIGHTS_ROOT/adapted_clrnet_r50_hardlane.pth" ]] && \
   [[ -f "$HARDLANE_WEIGHTS_ROOT/adapted_adnet_r34_hardlane.pth" ]]; then
  echo "weight probe already passed: $weight_report"
else
  "$python_bin" "$project/scripts/autodl/probe_weights.py" --output "$weight_report"
fi

smoke_report="$output/smoke/dataloader_loss_smoke.json"
if [[ -f "$smoke_report" ]] && json_passes_for_head "$smoke_report"; then
  echo "dataloader/loss smoke already passed: $smoke_report"
else
  "$python_bin" "$project/scripts/autodl/smoke_dataloader_and_loss.py" \
    --output "$smoke_report"
fi

for model in clrnet_r50 adnet_r34; do
  "$python_bin" "$project/scripts/autodl/run_training.py" \
    --model "$model" --epochs 1 --run-name "gate_${model}_1ep" \
    --auto-resume --skip-if-complete
  "$python_bin" "$project/scripts/autodl/run_training.py" \
    --model "$model" --epochs 1 --max-iter 526 --run-name "gate_${model}_1ep" \
    --resume --skip-if-complete
  "$python_bin" "$project/scripts/autodl/evaluate_selected.py" \
    --run-dir "$output/runs/gate_${model}_1ep" --skip-if-complete
done

if [[ "$through" == "gate" ]]; then
  "$python_bin" "$project/scripts/autodl/collect_results.py" \
    --through gate --archive "$output/handoff_gate.tar.gz" --force
  echo "AutoDL gate complete; evidence is under $output"
  exit 0
fi

for model in clrnet_r50 adnet_r34; do
  "$python_bin" "$project/scripts/autodl/run_training.py" \
    --model "$model" --epochs 15 --run-name "screen_${model}_15ep" \
    --auto-resume --skip-if-complete
  "$python_bin" "$project/scripts/autodl/evaluate_selected.py" \
    --run-dir "$output/runs/screen_${model}_15ep" --skip-if-complete
done

decision="$output/screen_decision.json"
winner="$($python_bin "$project/scripts/autodl/select_screen_winner.py" \
  --clr-evidence "$output/runs/screen_clrnet_r50_15ep/run_evidence.json" \
  --ad-evidence "$output/runs/screen_adnet_r34_15ep/run_evidence.json" \
  --output "$decision" --print-winner)"
echo "15-epoch screen winner: $winner ($decision)"

if [[ "$through" == "screen" ]]; then
  "$python_bin" "$project/scripts/autodl/collect_results.py" \
    --through screen --archive "$output/handoff_screen.tar.gz" --force
  exit 0
fi

# This is deliberately a fresh 36-epoch cosine schedule from the adapted
# CULane checkpoint. Resuming the exhausted 15-epoch screen would be invalid.
"$python_bin" "$project/scripts/autodl/run_training.py" \
  --model "$winner" --epochs 36 --run-name "baseline_${winner}_36ep" \
  --auto-resume --skip-if-complete
"$python_bin" "$project/scripts/autodl/evaluate_selected.py" \
  --run-dir "$output/runs/baseline_${winner}_36ep" --skip-if-complete
"$python_bin" "$project/scripts/autodl/collect_results.py" \
  --through baseline --archive "$output/handoff_baseline.tar.gz" --force

echo "AutoDL baseline pipeline complete: $output/runs/baseline_${winner}_36ep"
