#!/usr/bin/env bash
# The slow, opt-in tests, once a week. They are the only tests that drive a search at
# n = 800 or a batch of 1024, and the only ones that run the deep perft; nothing else
# exercises them, and on 2026-09-09 a depth bound in `test_search_cuda.py` turned out
# to have been wrong since the first-play-urgency change of 08-02, five weeks earlier
# (`docs/journal/2026-09-09-core-algorithm-review-fixes.md`).
#
# About 30 minutes of card. Skips itself if a training run or a league holds the GPU,
# so it never competes with a chain. Log: logs/slow-tests/<date>.log, tail -f-able.
#
# Install (user crontab, Sunday 04:00):
#   (crontab -l 2>/dev/null; echo "0 4 * * 0 $PWD/scripts/weekly-slow-tests.sh") | crontab -
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/slow-tests
log="logs/slow-tests/$(date +%F).log"
if pgrep -f "brokefish.train.loop|brokefish.eval.league|scripts/h2h" >/dev/null; then
    echo "$(date -Is) skipped: the card is busy" >> "$log"
    exit 0
fi
{
    echo "=== $(date -Is) $(git rev-parse --short HEAD)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-project --python .venv/bin/python \
        -m pytest -q --slow -rs --tb=short \
        tests/test_search.py tests/test_search_cuda.py tests/test_train.py \
        tests/test_env.py tests/test_oracle.py
    echo "=== exit $? at $(date -Is)"
} >> "$log" 2>&1
