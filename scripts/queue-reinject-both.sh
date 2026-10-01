#!/usr/bin/env bash
# Wait for the `t12h-reinject` chain to finish, reclaim its replay buffer, then start
# the `both` arm. Launch with nohup; it costs one sleeping shell.
#
# ⚠️ **It refuses rather than guesses.** Three gates, and any of them stops it:
#   1. the `ln1` chain must have written "chain finished" with exit 0 -- a crashed or
#      killed run must not silently be followed by a second 12 h;
#   2. its replay `.dat` must be gone or removable, because the `both` run needs ~3.95 G
#      (3.24 G preallocated buffer + ~0.7 G of checkpoints) and there is not room for
#      two of those on this disk;
#   3. at least 5 G must be free after the reclaim.
#
# ⚠️ **Reclaiming the buffer means `t12h-reinject` can no longer be resumed.** That is
# the same disposal the ledger already applies -- `t12h-wdl` kept only its 320 K index
# -- and it is safe *only* once the chain is complete, because the league reads
# checkpoints and never the buffer.
#
# To cancel: kill this script. It does nothing until the first chain exits.
#
#   tail -f logs/queue-reinject-both.log
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
LOG=logs/queue-reinject-both.log
PREV=runs/t12h-reinject/chain.log
DAT=runs/t12h-reinject/replay/t12h-reinject.dat

say() { printf '=== %s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

say "queued. waiting for the t12h-reinject chain to finish."
while pgrep -f "chain-t12h-reinject.sh" > /dev/null 2>&1; do sleep 60; done
say "chain-t12h-reinject.sh is gone."

# Gate 1 -- it must have finished, not died.
if ! grep -q "chain finished" "$PREV" 2>/dev/null; then
    say "ABORT: '$PREV' has no 'chain finished' line. Not starting a second 12 h on top"
    say "       of an incomplete run. Last lines:"
    tail -5 "$PREV" | tee -a "$LOG"
    exit 1
fi
if ! grep -qE "3/3 done \(exit 0\)" "$PREV" 2>/dev/null; then
    say "ABORT: the ln1 chain's final stage did not exit 0."
    grep -E "done \(exit" "$PREV" | tail -3 | tee -a "$LOG"
    exit 1
fi
say "ln1 chain completed cleanly."

# Gate 2 -- reclaim the buffer. The index stays; it is what describes the run.
if [ -f "$DAT" ]; then
    say "reclaiming $(du -h "$DAT" | cut -f1) from $DAT"
    rm -f "$DAT" || { say "ABORT: could not remove $DAT"; exit 1; }
else
    say "$DAT already gone."
fi

# Gate 3 -- enough room for buffer + checkpoints, with margin.
FREE_G=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
say "free after reclaim: ${FREE_G} G"
if [ "$FREE_G" -lt 5 ]; then
    say "ABORT: ${FREE_G} G free, and the run needs ~3.95 G plus room for its league."
    exit 1
fi

say "starting chain-t12h-reinject-both.sh"
exec bash scripts/chain-t12h-reinject-both.sh
