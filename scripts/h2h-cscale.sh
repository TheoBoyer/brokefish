#!/usr/bin/env bash
# One network against itself at two interior Gumbel scales: A at --c-scale $CS,
# B at the default (mctx's 0.1). 200 games at n = 128, the arbiter's openings.
# Written 2026-09-09 after the blunder dissection found the paper's 1.0 removes a
# quarter of the fixable hanging moves at n = 128; this is the net effect in play.
#   scripts/h2h-cscale.sh <ckpt> <c_scale> [games] [n]
set -u
cd "$(dirname "$0")/.."
CKPT=${1:?checkpoint}; CS=${2:?c_scale}; GAMES=${3:-200}; N=${4:-128}
TAG="cscale$CS-vs-0.1-$(basename "$CKPT" .pt)"
LOG=logs/h2h-$TAG.log
PY="uv run --no-project --python .venv/bin/python"
say() { printf '\n=== %s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
cleanup() { pkill -f "serve_brokefish.py --ckpt $CKPT --port 809[12]" 2>/dev/null; }
trap cleanup EXIT
say "h2h $TAG: A c_scale=$CS vs B default, $GAMES games at n=$N"
: > logs/serve-a-$TAG.log; : > logs/serve-b-$TAG.log
$PY scripts/serve_brokefish.py --ckpt "$CKPT" --port 8091 --c-scale "$CS" > logs/serve-a-$TAG.log 2>&1 &
$PY scripts/serve_brokefish.py --ckpt "$CKPT" --port 8092 > logs/serve-b-$TAG.log 2>&1 &
for i in $(seq 1 90); do
  grep -q "ready on" logs/serve-a-$TAG.log 2>/dev/null \
    && grep -q "ready on" logs/serve-b-$TAG.log 2>/dev/null && break
  sleep 5
done
grep -q "ready on" logs/serve-a-$TAG.log && grep -q "ready on" logs/serve-b-$TAG.log || {
  say "servers did not come up"; tail -5 logs/serve-a-$TAG.log logs/serve-b-$TAG.log | tee -a "$LOG"; exit 1; }
$PY -m scripts.arbiter \
  --a http://127.0.0.1:8091 --b http://127.0.0.1:8092 \
  --games "$GAMES" --n "$N" --width 64 --opening-plies 8 --max-plies 300 \
  --seed 0 --out logs/h2h-$TAG.json --pgn logs/h2h-$TAG.pgn >> "$LOG" 2>&1
say "done"
