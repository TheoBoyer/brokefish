#!/usr/bin/env bash
# Why do the puzzle probe and the match disagree? Three checks, in order of what they
# would rule out. Runs after the AG match so nothing contends for the card.
#
#   1  the three final checkpoints scored FROM DISK at n=0 (raw policy)
#      -> if these differ from the in-run probe, the saved checkpoint is not the
#         trained network and that is the whole answer.
#   2  AlphaGateau on the same 20000 at n=0, over the bridge
#      -> the first number that puts both engines on one scale on the same task.
#
# ⚠️ **No search on either side, and that is deliberate.** `move_pass@1` grades one
# move choice; putting a tree in front of it conflates the network with the search, and
# under `gumbel_m = 16` (ours) or `max_num_considered_actions = 16` (mctx's default,
# which theirs takes) it would partly measure the root cap rather than either engine.
# An earlier draft of this file had an n=128 arm labelled "the training search"; it was
# removed on 2026-08-16, and the label was wrong anyway -- `score_puzzles` goes through
# `eval_config`, which leaves `gumbel=False`, so it would have run PUCT.
#
#   tail -f logs/puzzle-audit.log
set -u
cd ~/brokefish
LOG=logs/puzzle-audit.log
PY="uv run --no-project --python .venv/bin/python"
say() { printf '\n\n=== %s  %s\n\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

G=runs/t12h-gumbel/checkpoints/t12h-gumbel-004009.pt
M9=runs/t12h-muon9/checkpoints/t12h-muon9-004409.pt
MI=runs/t12h-muon9-int8/checkpoints/t12h-muon9-int8-005010.pt

say "1/2  from disk, n=0 raw policy, 20000 puzzles  (compare to the in-run probe:"
say "     gumbel 0.3934 / muon9 0.4516 / muon9-int8 0.4545)"
$PY -m scripts.puzzle_h2h --ckpt $G $M9 $MI --limit 20000 --sims 0 \
  --out logs/puzzle-audit-n0.json >> "$LOG" 2>&1
say "1/2 done (exit $?)"

# Restart AG's server so it picks up the n=0 path added after the match started.
pkill -f serve_ag 2>/dev/null; sleep 3
( cd ~/alphagateau && XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.50 nohup .venv/bin/python serve_ag.py \
  --ckpt models/chess_2024-08-20:00h13/000499.ckpt --port 8083 --pad 64 \
  > ~/brokefish/logs/serve-ag-8083.log 2>&1 & )
for i in $(seq 1 40); do grep -q "ready on" logs/serve-ag-8083.log && break; sleep 8; done
tail -1 logs/serve-ag-8083.log | tee -a "$LOG"

say "2/2  AlphaGateau 000499 on the SAME 20000 puzzles, raw policy (n=0)"
# ⚠️ Raw policy on both sides, and that is the point. `move_pass@1` grades one move
# choice, so a search adds 369 ms a position on their side and imports
# `max_num_considered_actions = 16` -- a search probe measures the root cap as much as
# the network. At n = 0 both engines are compared on the same object, the policy head,
# and the whole suite fits in the time 500 puzzles would have taken with a search.
# ⚠️ The AG server must be restarted first: the n=0 path was added to serve_ag.py
# after it was launched for the match.
$PY -m scripts.puzzle_h2h --engine http://127.0.0.1:8083 --limit 20000 --sims 0 \
  --chunk 64 --out logs/puzzle-audit-ag.json >> "$LOG" 2>&1
say "2/2 done (exit $?)"

say "audit finished"

bash scripts/chain-puct-vs-gumbel.sh

