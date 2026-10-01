#!/usr/bin/env bash
# AlphaGateau on the puzzle suite, supervised: restart the server and retry on death.
#
# ⚠️ Retries exist because the server has died twice mid-sweep. It is left plain
# `from_fen` on purpose -- the batched replacement was verified identical and 6.4x
# faster and still got reverted, because it leaked ~0.5 MB a position under load.
#
#   tail -f logs/ag-puzzles.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
REPO=$PWD
AG_DIR=${AG_DIR:-$REPO/../alphagateau}
LOG=logs/ag-puzzles.log
PY="uv run --no-project --python .venv/bin/python"
LIMIT=${1:-20000}

start_server() {
  systemctl --user stop ag-puz.scope 2>/dev/null
  pkill -f serve_ag 2>/dev/null; sleep 3
  ( cd "$AG_DIR" && systemd-run --user --scope -p MemoryMax=10G \
      --unit=ag-puz --quiet env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 .venv/bin/python "$REPO/scripts/alphagateau/serve_ag.py" \
      --ckpt models/chess_2024-08-20:00h13/000499.ckpt --port 8085 --pad 64 \
      --warm-n 0 > "$REPO"/logs/serve-ag-puz.log 2>&1 & )
  for i in $(seq 1 40); do
    grep -q "ready on" logs/serve-ag-puz.log 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

for attempt in 1 2 3 4 5 6; do
  printf '\n=== %s  attempt %d, %s puzzles\n' "$(date '+%F %T')" "$attempt" "$LIMIT" | tee -a "$LOG"
  start_server || { echo "server did not come up" | tee -a "$LOG"; continue; }
  tail -1 logs/serve-ag-puz.log | tee -a "$LOG"
  systemd-run --user --scope -p MemoryMax=5G --unit=puz-run --quiet \
    $PY -m scripts.puzzle_h2h --engine http://127.0.0.1:8085 --limit "$LIMIT" \
    --sims 0 --chunk 64 --out logs/puzzle-audit-ag.json >> "$LOG" 2>&1
  if [ -s logs/puzzle-audit-ag.json ]; then
    printf '\n=== %s  DONE\n' "$(date '+%F %T')" | tee -a "$LOG"
    grep -E "move_pass@1 .*solve" "$LOG" | tail -2 | tee -a "$LOG"
    exit 0
  fi
  echo "attempt $attempt failed, retrying" | tee -a "$LOG"
done
echo "gave up after 6 attempts" | tee -a "$LOG"
