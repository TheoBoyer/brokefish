#!/usr/bin/env bash
# PUCT against Gumbel m=16, **same network on both sides**, across the budget axis.
#
# ⚠️ Why this exists. Every run in this line trained with `--gumbel --gumbel-m 16
# --sims 128 --terminal-collapse`, and the AG bridge plays that configuration. But
# `eval/league.py` builds its players with `eval_config(n, B)`, which leaves
# `gumbel=False` and `terminal_collapse=False` at their `SearchConfig` defaults -- so
# **every Elo number this project has produced rates these networks under PUCT**, a
# search they were never trained for. Nobody chose that; it is a default nobody looked
# at. This measures what it costs.
#
# Same weights on both sides, so the only variable is the search. A score of 0.5000 at
# every budget means the league's choice is free and the ratings stand as they are.
# Anything else is a systematic term sitting under every comparison in `docs/ledger/`.
#
# ⚠️ Cheap, unlike the AG match: both engines are ours at ~0.4 s per 64-position ply
# rather than AlphaGateau's 24 s, so a 200-game match is minutes, not hours.
#
#   tail -f logs/puct-vs-gumbel.log
set -u
cd ~/brokefish
LOG=logs/puct-vs-gumbel.log
PY="uv run --no-project --python .venv/bin/python"
CKPT=runs/t12h-muon9-int8/checkpoints/t12h-muon9-int8-005010.pt
say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "servers: A = gumbel m=16 (training), B = PUCT (what the league rates)"
pkill -f "scripts.serve_brokefish" 2>/dev/null; sleep 3
nohup $PY -m scripts.serve_brokefish --ckpt $CKPT --port 8091 --name bf-gumbel \
  > logs/srv-gumbel.log 2>&1 &
nohup $PY -m scripts.serve_brokefish --ckpt $CKPT --port 8092 --name bf-puct --puct \
  > logs/srv-puct.log 2>&1 &
for i in $(seq 1 40); do
  grep -q "ready on" logs/srv-gumbel.log 2>/dev/null && \
  grep -q "ready on" logs/srv-puct.log 2>/dev/null && break
  sleep 5
done
tail -1 logs/srv-gumbel.log | tee -a "$LOG"; tail -1 logs/srv-puct.log | tee -a "$LOG"

# ⚠️ n = 16 is included on purpose even though no run trains there: the Gumbel paper's
# advantage is largest at small budgets, and `t12h-gumbel`'s own league grid fell
# +212/+124/+77 over n = 16/64/256. If the two searches separate anywhere it is there.
for N in 16 64 128 256; do
  say "n = $N, 200 games, gumbel(A) vs PUCT(B), same weights"
  $PY -m scripts.arbiter --a http://127.0.0.1:8091 --b http://127.0.0.1:8092 \
    --games 200 --n $N --width 64 --max-plies 300 --every 60 --seed 0 \
    --out logs/puct-vs-gumbel-n$N.json >> "$LOG" 2>&1
  say "n = $N done (exit $?)"
done
say "sweep finished"
grep -hE "^A scores" "$LOG" | tee -a "$LOG"
