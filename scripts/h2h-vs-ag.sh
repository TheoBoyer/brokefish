#!/usr/bin/env bash
# One brokefish checkpoint against AlphaGateau's released checkpoint, over the arbiter.
#
#   scripts/h2h-vs-ag.sh <ckpt> <tag> [games] [n]
#
# ⚠️ **`XLA_PYTHON_CLIENT_MEM_FRACTION` is the whole reason this exists as a script.**
# `chain-t24h-adamw-int8.sh` inlined the match at 0.35, copied from
# `run-ag-puzzles.sh` -- but that sweep runs `--warm-n 0`, the *policy* path, which
# allocates no mctx tree. At n = 128 mctx allocates `[pad, n+1, 4672]` fp32 per batch,
# which is exactly the 154 288 128 bytes the failed run died asking for
# (64 x 129 x 4672 x 4). 0.35 of 8 GiB is 2.8 GiB and AG's own total allocation is
# 2.67 GiB, so the tree had nowhere to go. 0.55 leaves brokefish ~3.5 GiB, and
# brokefish needs well under that: at B = 64, n = 128 its tree is 856 x 129 x 64 = 7 MiB
# on top of a 13 MiB fp16 net.
#
# ⚠️ **Segmenting is no longer needed -- this warning was stale for eleven days.**
# It used to read "run it in segments of <=50 games", because `serve_ag.py` reached its
# 8 GiB cap after 99 games and was oom-killed mid-match. That leak was found and fixed
# **the same day** (`journal/2026-08-19-the-alphagateau-server-leak.md`): the server
# called `mctx.gumbel_muzero_policy` from inside the request handler with no `jax.jit`
# above it, so every request re-compiled an n_sim-deep search and every executable
# stayed resident -- 81.2 MiB/request. The cached `_search` at `serve_ag.py:107` fixed
# it to **0.2 MiB/request**, 400x, and RSS plateaus after the first request.
# Re-verified 2026-08-30 on a live match: 1233.92 -> 1234.05 MiB over 50 s of serving.
# A 200-game match runs in one call.
#
# Protocol is `logs/gate2-h2h-alphagateau.json`'s, unchanged, so the numbers compare
# directly: n = 128 both sides, 200 games over 100 openings, 8 opening plies, 300-ply
# cap. `t12h-gumbel-004009` scored 0.400 (20 W / 120 D / 60 L, -70 Elo) on it.
set -u
cd ~/brokefish
CKPT=${1:?checkpoint}; TAG=${2:?tag}; GAMES=${3:-200}; N=${4:-128}; SEED=${5:-0}; SKIP=${6:-0}
LOG=logs/h2h-$TAG.log
PY="uv run --no-project --python .venv/bin/python"
AG=models/chess_2024-08-20:00h13/000499.ckpt

say() { printf '\n=== %s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

start_servers() {
  systemctl --user stop ag-h2h.scope bf-h2h.scope 2>/dev/null
  # ⚠️ a scope that died `oom-kill` stays loaded in the FAILED state, and a
  # second systemd-run with the same --unit then refuses with "already loaded".
  # `stop` does not clear that; only `reset-failed` does. This cost the first
  # attempt its retries on 2026-08-19.
  systemctl --user reset-failed ag-h2h.scope bf-h2h.scope 2>/dev/null
  pkill -f serve_ag 2>/dev/null; pkill -f serve_brokefish 2>/dev/null; sleep 4
  : > logs/serve-ag-$TAG.log; : > logs/serve-bf-$TAG.log
  ( cd ~/alphagateau && systemd-run --user --scope -p MemoryMax=10G \
      --unit=ag-h2h --quiet env XLA_PYTHON_CLIENT_PREALLOCATE=false \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 .venv/bin/python serve_ag.py \
      --ckpt "$AG" --port 8085 --pad 64 --warm-n "$N" \
      > ~/brokefish/logs/serve-ag-$TAG.log 2>&1 & )
  ( systemd-run --user --scope -p MemoryMax=6G --unit=bf-h2h --quiet \
      $PY scripts/serve_brokefish.py --ckpt "$CKPT" --port 8081 \
      > logs/serve-bf-$TAG.log 2>&1 & )
  for i in $(seq 1 90); do
    grep -q "ready on" logs/serve-ag-$TAG.log 2>/dev/null \
      && grep -q "ready on" logs/serve-bf-$TAG.log 2>/dev/null && return 0
    grep -qi "error\|Traceback" logs/serve-ag-$TAG.log 2>/dev/null && return 1
    sleep 5
  done
  return 1
}

say "h2h $TAG: $CKPT vs alphagateau $AG, $GAMES games at n=$N"
for attempt in 1 2 3; do
  say "attempt $attempt"
  if start_servers; then
    tail -1 logs/serve-ag-$TAG.log | tee -a "$LOG"
    tail -1 logs/serve-bf-$TAG.log | tee -a "$LOG"
    $PY -m scripts.arbiter \
      --a http://127.0.0.1:8081 --b http://127.0.0.1:8085 \
      --games "$GAMES" --n "$N" --width 64 --opening-plies 8 --max-plies 300 \
      --seed "$SEED" --opening-skip "$SKIP" --out logs/h2h-$TAG.json --pgn logs/h2h-$TAG.pgn >> "$LOG" 2>&1
    [ -s logs/h2h-$TAG.json ] && break
  else
    { echo "servers did not come up:"; tail -5 logs/serve-ag-$TAG.log; } | tee -a "$LOG"
  fi
  sleep 10
done
pkill -f serve_ag 2>/dev/null; pkill -f serve_brokefish 2>/dev/null
say "done"
