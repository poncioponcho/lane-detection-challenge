#!/usr/bin/env bash
# Night-time finalizer for the risk-on 960x384 LVO screen.
#
# Idempotent: safe to run repeatedly. Exits 0 with a progress line when the
# eight folds are not all in yet, so a scheduler can simply retry later.
#
# Stage A (fast, decisive): pull the eight fold_state.json files, run the
#   diagnostic-metric gate verdict, write report + verdict json.
# Stage B (best effort): download the 7100-row OOF tree and confirm with one
#   frozen-Oracle global call. Failure here must never block Stage A.
set +e

PROJECT="/Users/seyonmacbook/WorkBuddy/恶劣场景下的车道线检测挑战赛"
KEY="/Users/seyonmacbook/.ssh/lane_id"
REMOTE="root@i-1.gpushare.com"
PORT=34529
EX="/hy-tmp/lane-outputs/risk_on_res960x384_lvo_20260907"
BASE="/hy-tmp/lane-outputs/lvo_clrnet_r50_15ep_20260904"
LOCAL="$PROJECT/outputs/riskon_res960x384_lvo_20260907"
LANE_PY="/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane/bin/python"
ORACLE_PY="/Users/seyonmacbook/.workbuddy/binaries/python/envs/lane-oracle-py312/bin/python"

SSH="ssh -i $KEY -p $PORT -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=no $REMOTE"
SCP="scp -i $KEY -P $PORT -o BatchMode=yes -o ConnectTimeout=25 -o StrictHostKeyChecking=no"

cd "$PROJECT" || exit 1
mkdir -p "$LOCAL"

DUMP='python3 -c "
import json,glob,sys
root=sys.argv[2]
rows=[]
for p in sorted(glob.glob(root+\"/runs/fold_*/fold_state.json\")):
    d=json.load(open(p))
    rows.append({\"fold_name\":d[\"fold_name\"],\"status\":d.get(\"status\"),
                 \"diagnostic\":d.get(\"holdout_eval\",{}).get(\"diagnostic\",{})})
print(json.dumps({\"folds\":rows}))
"'

# ---------------------------------------------------------------- Stage A
$SSH "$DUMP _ $EX" > "$LOCAL/folds_cand.json" 2>/dev/null
if [ ! -s "$LOCAL/folds_cand.json" ]; then
  echo "SSH_UNAVAILABLE"
  exit 0
fi

if [ ! -s "$LOCAL/folds_base.json" ]; then
  $SSH "$DUMP _ $BASE" > "$LOCAL/folds_base.json" 2>/dev/null
fi

PASS=$($LANE_PY -c "
import json,sys
d=json.load(open('$LOCAL/folds_cand.json'))
print(sum(1 for f in d['folds'] if f['status']=='pass'), len(d['folds']))
")
NPASS=${PASS%% *}; NTOT=${PASS##* }
echo "folds_pass=$PASS"

if [ "$NPASS" != "8" ] || [ "$NTOT" != "8" ]; then
  $SSH 'EX='"$EX"'; cur=$(grep -ohE "fold_[0-9]+_[a-z0-9]+" $EX/status 2>/dev/null | tail -1); \
        for d in $EX/runs/fold_*; do st=$(tr -d " \n" < $d/fold_state.json 2>/dev/null | grep -o "\"status\":\"[a-z]*\"" | head -1); \
        case "$st" in *running*) cur=$(basename $d);; esac; done; \
        iter=$(grep -oE "iter: [0-9]+" $EX/runs/$cur/train.log 2>/dev/null | tail -1); \
        eta=$(grep -oE "eta: [0-9:]+" $EX/runs/$cur/train.log 2>/dev/null | tail -1); \
        echo "current=$cur $iter $eta disk=$(df -P /hy-tmp | awk "NR==2{printf \"%.1fGB\", \$4/1048576}")"' 2>/dev/null
  exit 0
fi

$LANE_PY scripts/riskon_gate_report.py \
  --cand "$LOCAL/folds_cand.json" \
  --base "$LOCAL/folds_base.json" \
  --output-json "$LOCAL/gate_verdict.json" \
  --output-md "$PROJECT/outputs/reports/riskon_gate_20260908.md" \
  --label "risk-on 960x384 + cut_height=180, CLRNet-R50, 15ep" \
  --baseline-label "15ep R50 800x320 baseline"

# ---------------------------------------------------------------- Stage B
if [ -x "$ORACLE_PY" ]; then
  if [ ! -d "$LOCAL/oof/predictions" ]; then
    $SSH "cd $EX && tar czf /hy-tmp/riskon_oof.tgz oof" 2>/dev/null
    $SCP "$REMOTE:/hy-tmp/riskon_oof.tgz" "$LOCAL/oof.tgz" 2>/dev/null
    if [ -s "$LOCAL/oof.tgz" ]; then
      mkdir -p "$LOCAL/oof"
      tar xzf "$LOCAL/oof.tgz" -C "$LOCAL/oof" --strip-components=1 2>/dev/null
      rm -f "$LOCAL/oof.tgz"
    fi
  fi
  N=$(find "$LOCAL/oof/predictions" -name "*.lines.txt" 2>/dev/null | wc -l | tr -d " ")
  echo "oof_files=$N"
  if [ "$N" = "7100" ]; then
    PYTHONPATH=src "$LANE_PY" scripts/evaluate_lvo_video_oof.py \
      --manifest data/processed/manifest_train.jsonl \
      --pred-dir "$LOCAL/oof/predictions" \
      --gt-dir data/raw/dataset/_extract/train_full/Lane \
      --official-python "$ORACLE_PY" \
      --oof-evidence "$LOCAL/oof/oof_evidence.json" \
      --output-dir "$LOCAL/evaluation" \
      --model clrnet_r50 --input 960x384 --cut-height 180 --epochs 15 \
      --bootstrap 10000 --seed 42
  else
    echo "OOF_INCOMPLETE files=$N"
  fi
else
  echo "ORACLE_ENV_MISSING"
fi
