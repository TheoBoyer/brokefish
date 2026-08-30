#!/usr/bin/env bash
# Two brokefish checkpoints against each other over the arbiter, on the **same protocol
# as `h2h-vs-ag.sh`**: n = 128 both sides, 200 games over 100 openings, 8 opening plies,
# 300-ply cap, seed 0, `scripts/arbiter.py` as referee with `python-chess` the only
# authority on legality and result.
#
#   scripts/h2h-bf.sh <ckptA> <ckptB> <tag> [games] [n]
#
# ⚠️ **The protocol matches on purpose.** Our league rates on n = 16/64/256 and is a
# Bradley-Terry fit; this is 200 games at n = 128 with no fit at all. Running it at the
# AlphaGateau protocol makes a triangle: if A and B each played AG at n = 128, their
# scores there and their score here are directly comparable, and a violation of
# transitivity is itself a finding. `journal/2026-08-19-the-league-does-not-transfer.md`
# is the entry that needed this and did not have it.
#
# ⚠️ No `serve_ag` here, so no jit-boundary concern and no segmentation -- see that
# script's header for why segmenting is retired generally.
set -u
cd ~/brokefish
A=${1:?checkpoint A}; B=${2:?checkpoint B}; TAG=${3:?tag}; GAMES=${4:-200}; N=${5:-128}
LOG=logs/h2h-$TAG.log
PY="uv run --no-project --python .venv/bin/python"

say() { printf '\n=== %s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

cleanup() { pkill -f "serve_brokefish.py --ckpt $A" 2>/dev/null
            pkill -f "serve_brokefish.py --ckpt $B" 2>/dev/null; }
trap cleanup EXIT

say "h2h $TAG: A=$A  vs  B=$B, $GAMES games at n=$N"
: > logs/serve-a-$TAG.log; : > logs/serve-b-$TAG.log
$PY scripts/serve_brokefish.py --ckpt "$A" --port 8091 > logs/serve-a-$TAG.log 2>&1 &
$PY scripts/serve_brokefish.py --ckpt "$B" --port 8092 > logs/serve-b-$TAG.log 2>&1 &
for i in $(seq 1 90); do
  grep -q "ready on" logs/serve-a-$TAG.log 2>/dev/null \
    && grep -q "ready on" logs/serve-b-$TAG.log 2>/dev/null && break
  sleep 5
done
grep -q "ready on" logs/serve-a-$TAG.log && grep -q "ready on" logs/serve-b-$TAG.log || {
  say "servers did not come up"; tail -5 logs/serve-a-$TAG.log logs/serve-b-$TAG.log | tee -a "$LOG"; exit 1; }
tail -1 logs/serve-a-$TAG.log | tee -a "$LOG"
tail -1 logs/serve-b-$TAG.log | tee -a "$LOG"

$PY -m scripts.arbiter \
  --a http://127.0.0.1:8091 --b http://127.0.0.1:8092 \
  --games "$GAMES" --n "$N" --width 64 --opening-plies 8 --max-plies 300 \
  --seed 0 --out logs/h2h-$TAG.json --pgn logs/h2h-$TAG.pgn >> "$LOG" 2>&1
say "done"
