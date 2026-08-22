#!/usr/bin/env bash
# The AlphaGateau match in segments, because the AG server leaks host RAM.
#
#   scripts/h2h-vs-ag-segmented.sh <ckpt> <tag> [segments] [games-per-segment] [n]
#
# ⚠️ Measured 2026-08-19: `serve_ag.py` under this workload grew to its 8 GiB cap in
# 99 games / 1 h and was oom-killed, taking the second colour pool with it. Each
# segment restarts both servers, so the leak is bounded by the segment length. Each
# Every segment uses **seed 0** with `--opening-skip`, so the four slices are disjoint
# and their union is exactly the seed-0 opening set -- the same 100 openings as
# `logs/gate2-h2h-alphagateau.json`. The match is therefore *paired* with that -70 Elo
# baseline rather than being a fresh sample of openings.
#
# Every segment is colour-balanced on its own (the arbiter plays each opening twice,
# once with each engine as White), so a partial run is still an unbiased estimate.
set -u
cd ~/brokefish
CKPT=${1:?checkpoint}; TAG=${2:?tag}; SEGS=${3:-4}; PER=${4:-50}; N=${5:-128}
for i in $(seq 1 "$SEGS"); do
  printf '\n######## segment %d/%d, seed 0, openings %d..%d, %d games\n' "$i" "$SEGS" $(( (i-1)*PER/2 )) $(( i*PER/2 - 1 )) "$PER"
  bash scripts/h2h-vs-ag.sh "$CKPT" "$TAG-s$i" "$PER" "$N" 0 $(( (i-1) * PER / 2 ))
done
printf '\n######## pooling\n'
uv run --no-project --python .venv/bin/python - "$TAG" "$SEGS" <<'PY'
import json,sys,math
tag,segs=sys.argv[1],int(sys.argv[2])
W=D=L=0
for i in range(1,segs+1):
    try: d=json.load(open(f'logs/h2h-{tag}-s{i}.json'))
    except OSError: print(f'  segment {i}: missing'); continue
    W+=d['wins']; D+=d['draws']; L+=d['losses']
    print(f"  segment {i}: {d['wins']}-{d['draws']}-{d['losses']}  score {d['score']:.3f}")
n=W+D+L
if n:
    s=(W+0.5*D)/n
    elo=-400*math.log10(1/s-1) if 0<s<1 else float('nan')
    print(f"\n  POOLED: {W}-{D}-{L} over {n} games, score {s:.4f}, {elo:+.1f} Elo")
    json.dump({'wins':W,'draws':D,'losses':L,'games':n,'score':s,'elo':elo},
              open(f'logs/h2h-{tag}-pooled.json','w'), indent=1)
PY
