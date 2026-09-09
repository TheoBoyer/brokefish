#!/usr/bin/env bash
# Two evaluation jobs that were owed, 2026-09-09. Sequential, one card.
#
#   1/2  the direct match t24h-adamw-int8 vs t12h-int8 under Gumbel. The ledger's
#        +237 ± 79 for this pair (journal 2026-08-19) was measured on a PUCT league
#        while both networks trained under Gumbel (journal 2026-08-22); this is the
#        same pair on the instrument they trained for. (A joint league was the first
#        plan; t12h-int8 has no checkpoint history left, see the note at stage 1.)
#   2/2  the direct match t12h-reuse2 never got: 200 games at n=128 against
#        t24h-adamw-int8. Its 2026-08-20 log stops at "attempt 1".
#
# tail -f logs/chain-rerate-and-reuse2.log
set -u
cd "$(dirname "$0")/.."
LOG=logs/chain-rerate-and-reuse2.log
PY="uv run --no-project --python .venv/bin/python"
say() { printf '\n=== %s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# 2026-09-09 12:18: stage 1 as first written cannot run. `t12h-int8`'s step
# snapshots were deleted, only the rolling `t12h-int8.pt` remains, and the league
# refuses a run with no history. The direct match below is the fit-free reading of
# the same pair, which is the instrument the ledger now prefers anyway. The 2026-08-19
# league had these two at 77-14-17 head to head under PUCT.
say "1/2 h2h t24h-adamw-int8 vs t12h-int8 (final checkpoints), 200 games at n=128, Gumbel"
scripts/h2h-bf.sh runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt \
  runs/t12h-int8/checkpoints/t12h-int8.pt t24h-adamw-int8-vs-t12h-int8 200 128 >> "$LOG" 2>&1
say "1/2 done (exit $?)"

say "2/2 h2h t12h-reuse2 vs t24h-adamw-int8, 200 games at n=128"
scripts/h2h-bf.sh runs/t12h-reuse2/checkpoints/t12h-reuse2.pt \
  runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt t12h-reuse2-vs-t24h 200 128 >> "$LOG" 2>&1
say "2/2 done (exit $?)"
say "chain finished"
