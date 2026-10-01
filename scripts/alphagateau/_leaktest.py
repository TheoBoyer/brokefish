"""Leak bisect for the AlphaGateau server.

⚠️ `ru_maxrss` is a high-water mark and never falls, so it cannot separate real growth
from a transient peak. Current RSS comes from /proc/self/status instead.

Three counters, because they separate the two candidate mechanisms:
  RSS          the symptom systemd kills on
  live arrays  jax.live_arrays() -- growth here means buffers are RETAINED
  cache        _derive._cache_size() -- growth here means the jit RE-TRACES

    AG_FASTFEN=1 python _leaktest.py 20 128     # today's path
    AG_FASTFEN=0 python _leaktest.py 20 128     # the 2026-08-13 path, which did not OOM
    AG_FASTFEN=1 python _leaktest.py 20 0       # no mctx tree, isolates the constructor
"""
import gc
import os
import random
import sys

import chess
import jax

sys.path.insert(0, os.environ.get("AG_DIR", os.getcwd()))  # their repo: mcts, models
from serve_ag import AGEngine  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 128


def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1048576.0
    return -1.0


def live():
    try:
        a = jax.live_arrays()
        return len(a), sum(x.size * x.dtype.itemsize for x in a) / 1048576.0
    except Exception:
        return -1, -1.0


def cache():
    try:
        import _fastfen
        return _fastfen._derive._cache_size()
    except Exception:
        return -1


rng = random.Random(0)
fens = []
while len(fens) < 64:
    b = chess.Board()
    for _ in range(rng.randint(6, 30)):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng.choice(ms))
    if not b.is_game_over():
        fens.append(b.fen())

eng = AGEngine("models/chess_2024-08-20:00h13/000499.ckpt", pad=64)
eng.moves(fens, SIM)
gc.collect()
r0 = rss()
n0, m0 = live()
print(f"MODE fastfen={os.environ.get('AG_FASTFEN', '1')} n_sim={SIM} | warm "
      f"RSS {r0:.3f} GiB  live {n0} arrays / {m0:.1f} MiB  cache {cache()}", flush=True)

for i in range(1, N + 1):
    eng.moves(fens, SIM)
    if i % 4 == 0:
        gc.collect()
        n, m = live()
        print(f"  req {i:3d}  RSS {rss():.3f} ({rss() - r0:+.3f})  "
              f"live {n} arrays ({n - n0:+d}) / {m:.1f} MiB ({m - m0:+.1f})  "
              f"cache {cache()}", flush=True)

gc.collect()
n, m = live()
print(f"GROWTH over {N} reqs: RSS {(rss() - r0) * 1024 / N:.1f} MiB/req | "
      f"live arrays {n - n0:+d} | live bytes {m - m0:+.1f} MiB | cache {cache()}",
      flush=True)
