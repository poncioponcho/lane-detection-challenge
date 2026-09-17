#!/usr/bin/env bash
# Wait for the testB images to land on the instance, then run the whole
# inference pipeline unattended.
#
# Why this exists: the local watch automation only polls every few hours. If the
# user uploads testB right after a poll, the next poll can fall past the B-board
# deadline. This watcher lives on the instance and fires within ~1 min of the
# upload, so the only remaining latency is the inference itself.
#
#   setsid nohup bash scripts/autodl/watch_testB_and_run.sh >/dev/null 2>&1 &
#   disown
#
# Idempotency: refuses to start twice (lock file), and refuses to overwrite a
# bundle that is newer than the last observed upload.
set -u

export HARDLANE_PROJECT_ROOT=/hy-tmp/lane-detection-challenge
export HARDLANE_DATA_ROOT=/hy-tmp/datasets/HardLane/Lane
export HARDLANE_OUTPUT_ROOT=/hy-tmp/lane-outputs

IMG=$HARDLANE_DATA_ROOT/JPEGImages
LOG=/hy-tmp/testB_watch.log
LOCK=/hy-tmp/testB_watch.lock
BUNDLE=/hy-tmp/testB_bundle.tgz
# Beijing time; stop waiting after this.
# 2026-09-17 08:30 correction: the official page states the B-board window is
# 9/17 00:00 - 9/18 17:00 (41 h), NOT 9/16 - 9/17 17:00. The old value
# (202609171630) would have made this watcher give up with 25 h of window left.
DEADLINE=202609181700

count_clips () { ls "$IMG" 2>/dev/null | grep -v _hflip | wc -l | tr -d ' '; }

if [ -e "$LOCK" ]; then
  echo "$(date -u +%FT%TZ) already running (pid $(cat "$LOCK"))" >> "$LOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

echo "$(date -u +%FT%TZ) watcher start pid=$$ clips=$(count_clips)" >> "$LOG"

PREV=0
while true; do
  N=$(count_clips)
  NOW=$(TZ=Asia/Shanghai date +%Y%m%d%H%M)

  if [ "$NOW" -gt "$DEADLINE" ]; then
    echo "$(date -u +%FT%TZ) deadline passed (BJ $NOW) -- giving up" >> "$LOG"
    exit 1
  fi

  if [ "$N" -gt 80 ]; then
    # 80 = 71 train + 9 testA. Anything above means testB clips showed up.
    # Re-sample after a pause: a zip still being extracted must not trigger us.
    sleep 45
    N2=$(count_clips)
    if [ "$N2" -ne "$N" ]; then
      echo "$(date -u +%FT%TZ) clips still changing ($N -> $N2) -- wait for extraction" >> "$LOG"
      continue
    fi
    echo "$(date -u +%FT%TZ) TRIGGER clips=$N" >> "$LOG"
    cd "$HARDLANE_PROJECT_ROOT" || exit 1
    for ATTEMPT in 1 2 3; do
      bash scripts/autodl/run_testB_infer.sh >> "$LOG" 2>&1
      RC=$?
      echo "$(date -u +%FT%TZ) run_testB_infer.sh attempt=$ATTEMPT rc=$RC" >> "$LOG"
      if [ "$RC" -eq 0 ] && [ -s "$BUNDLE" ]; then
        break
      fi
      # A single failure must not burn the window: the most likely cause is a
      # half-extracted upload, which settles on its own within a couple of minutes.
      sleep 120
    done
    ls -la "$BUNDLE" >> "$LOG" 2>&1
    if [ "$RC" -eq 0 ] && [ -s "$BUNDLE" ]; then
      exit 0
    fi
    # Never give up while there is still window left: the user may still be
    # extracting, or may re-upload after a first bad attempt. Inference is
    # --skip-if-complete, so a retry only redoes the failed part.
    echo "$(date -u +%FT%TZ) will re-attempt in 300 s" >> "$LOG"
    sleep 300
    continue
  fi

  if [ "$N" -ne "$PREV" ]; then
    echo "$(date -u +%FT%TZ) clips=$N (waiting for >80)" >> "$LOG"
    PREV=$N
  fi
  sleep 60
done
