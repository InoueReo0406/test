#!/usr/bin/env bash
set -u

ROOT="/Users/inouereo/git_research/turorial_sam3"
LOG="$ROOT/sam3_access_watch.log"
CHECK_OUT="/tmp/sam3_check.out"
INTERVAL="${SAM3_WATCH_INTERVAL:-300}"
MAX_CHECKS="${SAM3_MAX_CHECKS:-0}"
CHECK_COUNT=0

cd "$ROOT" || exit 1
source .venv/bin/activate || exit 1

echo "=== watch start $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

while true; do
  CHECK_COUNT=$((CHECK_COUNT + 1))
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  python - <<'PY' >"$CHECK_OUT" 2>&1
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("facebook/sam3", "config.json")
    print("GRANTED")
    raise SystemExit(0)
except Exception as e:
    print(type(e).__name__, str(e).splitlines()[0])
    raise SystemExit(1)
PY
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "[$ts] access=granted" >> "$LOG"
    cat "$CHECK_OUT" >> "$LOG"
    PYTHONUNBUFFERED=1 python -u run_sam3_iou.py \
      --model-id facebook/sam3 \
      --text-prompt person \
      --image-size 224 \
      --score-threshold 0.2 \
      --mask-threshold 0.5 \
      --max-attempts 3 \
      --min-iou 0.0 >> "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] inference_exit=$?" >> "$LOG"
    echo "=== watch end $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
    break
  fi

  echo "[$ts] access=pending" >> "$LOG"
  cat "$CHECK_OUT" >> "$LOG"
  if [ "$MAX_CHECKS" -gt 0 ] && [ "$CHECK_COUNT" -ge "$MAX_CHECKS" ]; then
    echo "=== watch stop max_checks reached $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
    break
  fi
  sleep "$INTERVAL"
done
