# 2026-08-13 (thirteenth entry) — AlphaGateau's cadence does not transfer, and three corrections

Two days spent on the largest untested lever in the project — their **~150× more
optimiser steps per position generated** — and on the self-play budget beside it.
Both came back negative. The most useful thing produced was not a run: it was
reading `Akulen/AlphaGateau@master` line by line, which corrected several things
this journal had recorded from the paper's prose.

## E-n64 — the self-play budget is flat between 64 and 128

`t12h-n64` is `t12h-gumbel` with one thing changed: n = 64 instead of 128 during
self-play. Same 12 h, same cosine, same optimiser, same fp8 and collapse, same
Gumbel `m = 16`. Both ended annealed, so there is no schedule confound in either
direction.

| | `t12h-n64` | `t12h-gumbel` |
|---|---:|---:|
| generations | 3319 | 2059 |
| gradient steps | **6752** | 4185 |
| games | **278 679** | 174 216 |
| puzzle pass@1 | 0.4058 | 0.3934 |

**1.61× the steps and 1.60× the games in the same wall clock**, and the league says
it is worth nothing:

| eval budget | difference | z |
|---|---:|---:|
| n = 16 | +38.7 ± 75 | +1.02 |
| n = 64 | +9.5 ± 75 | +0.25 |
| n = 256 | −19.5 ± 86 | −0.45 |

Exactly break-even. Halving the self-play budget buys 1.61× the data and loses
precisely that much quality. ⚠️ The preregistered success condition was "beat
`t12h-gumbel`", so this is a **failed** gate and a useful negative: between 64 and
128 we are on a flat part of the trade-off, and `n` is a free choice to be made on
engineering grounds rather than on Elo.

⚠️ **The puzzle probe called this one wrong.** It had `t12h-n64` ahead throughout
(0.4058 against 0.3934) and the board says dead level. It called Gumbel-vs-PUCT
right and this wrong — one for two. Recorded because
[the Gumbel entry](2026-08-11-gumbel.md) leaned on it as a leading indicator.

## The AlphaGateau cadence, twice, both losses

`2026-08-09-alphagateau-read.md` measured our optimisation regime against theirs and
found batch 4096 at 0.815 samples/position against their batch 256 at 7.6. Every
lever this project has measured has been worth 1.1–2×; the reuse ratio is a
different axis and had never been moved.

⚠️ **It needs no MCTS rework.** `--samples-per-position` and `--batch` are already
flags. The rework buys running the two phases *concurrently*, which is efficiency,
not capability. That assumption had been shaping the roadmap and cost a day.

`t12h-agc`: 12 h at **304.00 steps/generation** — their 3904-over-131 072 to three
digits — batch 256, window ~1 M records, n = 128 Gumbel. 368 448 steps and 12.7 M
positions, **~19 % of their entire training**. Against `t12h-gumbel` at matched
runtime, direct play, both annealed:

| eval budget | `t12h-gumbel` W–D–L | `t12h-agc` |
|---|---:|---:|
| n = 16 | 31–2–3 | −361 |
| n = 64 | 32–3–1 | −452 |
| n = 256 | **36–0–0** | worse than −600 |

**108 games, score 0.060, about −478 Elo.** At n = 256 it did not take a game or a
draw.

`t12h-agrepro` then removed the three optimiser deviations that the code read
exposed — plain Adam, no clipping, flat lr, value weighted 0.5. It recovered
**about +0.02 on puzzle pass@1** and plateaued at 0.306 against `t12h-gumbel`'s
0.393. Stopped at step 315 552 rather than finishing, once it was clear the
optimiser was not the cause.

### The mechanism, which is the muon problem again

```
weight_norm   t12h-gumbel 131   |   t12h-agc 440   |   t12h-agrepro 1309 and climbing
```

With decoupled decay, growth per step ≈ `lr·c` and decay is `lr·wd·‖W‖`, so
`‖W‖* = c/wd` — independent of lr. `t12h-agc` at `wd = 0.01` equilibrated near 440;
`t12h-agrepro` at `wd = 0` has **no equilibrium at all** and grew monotonically. In a
pre-norm LayerNorm network the function is largely insensitive to weight scale while
the relative step `lr/‖W‖` is not, so at 300 k steps that run trained at roughly a
tenth of the control's effective learning rate while its nominal lr said 1e-3.

⚠️ **This is not overfitting**, and the arithmetic says so: 76.6 M samples over
10.4 M generated positions is **7.38 samples per position**, and held-out puzzle
score plateaued rather than declining. It is effective-lr collapse.

⚠️ And the uncomfortable consequence: the fix would be **more** decay, not less
(`wd ≈ 0.034` targets ‖W‖ ≈ 131), which is the opposite of AlphaGateau's own
setting. Their BatchNorm GNN evidently tolerates `wd = 0` for 1.95 M steps; this
transformer does not. **Untested** — `wd ≈ 0.03` at their cadence is the one-flag
experiment that would settle it, and it was not run.

## ⚠️ What their code says, against what this journal recorded

`train.py:180` is the entire optimiser:

```python
optimizer = optax.adam(learning_rate=config['learning_rate'])   # 0.001
```

Plain Adam. **No weight decay, no gradient clipping** — they compute `max_grad` in
the train step and only log it — **no schedule, no warmup**. `mcts.py:70` passes
`qtransform_completed_by_mix_value`, `gumbel_scale=1.0` and **no**
`max_num_considered_actions`, so mctx's default of 16. All three of those we already
matched.

Two things their code and their paper disagree about:

- ⚠️ Their eq. (10) is `−π^T log(π̃) + (v − ṽ)²`, unweighted. `train.py:275` calls
  `optax.l2_loss`, which is **`0.5·(x − y)²`**. The runs that produced their numbers
  weighted value at **half** the policy, so reproducing the paper's formula would not
  reproduce the experiment. `--value-weight` exists now; the default stays 1.0.
- ⚠️ Their committed `training_batch_size` is `2**7` per device over 8 devices — a
  global batch of **1024 over 976 updates**. The paper says **256 over 3904**. Both
  are one epoch over the 999 424-frame window, chunked differently. The released
  checkpoint's own config carries **256**, so the paper's setting is what ran.

### Six implementation differences, none of them validated

Audited end to end, excluding architecture:

1. **Truncated tails.** They mask the value target on every ply after the last
   termination in a 512-ply stream — on the order of 10–20 % of frames train
   policy-only. We cap a game at 512 plies and **label it a draw**, fabricating a
   target on 0.1 % of frames.
2. **Their window eviction is random, not FIFO.** `shuffle_window=True` keeps the
   window permuted and `concatenate([new, old])[:1M]` therefore drops a *uniformly
   random* subset of old frames. Same mean lifetime, long tail. Ours is a strict ring.
3. **Exact epoch vs sampling with replacement.** They permute and iterate; we draw
   `rng.integers` with replacement. Same expected reuse, higher variance.
4. **Window in frames vs games.** Theirs is exactly 1 000 000. Ours is
   `--window-games`, so the frame count floats with game length — measured 1 056 270
   early and 814 573 later, a 23 % swing.
5. **Their reuse ramps in over ~7 iterations** while the window fills; ours steps
   straight to 7.6.
6. **Edge cap.** We truncate to the E = 96 highest priors; they carry the full 4672
   action space.

⚠️ **pgx's chess rules were not checked against ours.** If their environment differs
on a rule, everything downstream differs and nothing above would show it.

## Two bugs the work exposed, both ours

⚠️ **§5.5's buffer floor was `cfg.batch`**, which scales *with* the batch and so
shrinks exactly when a high-reuse recipe needs it to grow. At AlphaGateau's settings
training began at 256 records and took **304 steps over a 465-record buffer** by
generation 3, loss collapsing to 1.68 and KL to 0.17 — memorising a few hundred
positions while every counter reported health. `--min-records` fixed it; default 0,
so nothing measured before moves.

⚠️ **`Search.reset` does not run terminal detection**, so `game_done` stays False and
invariant 8 ("a finished game was searched") cannot fire on a position handed in from
outside. The search builds a zero-edge root and it surfaces as invariant 6, three
layers from the cause. Found by the engine bridge below.

## Corrections to earlier entries

⚠️ **[2026-08-11](2026-08-11-gumbel.md) claims cross-league Elo *levels* are stable.**
It cites `t24h-fp8@8416:n256` at 1348.5 and 1354.5 on two fits, "6 Elo apart", and
uses that to call 1425.7 the best network *ever* rather than in that league. The same
checkpoint rates **1406.7** in the n=64 league — a **58 Elo** shift. Differences
within a league replicate; absolute levels move with pool composition. That 6-Elo
agreement was luck stated as a property. The "best measured" claim survives — 1459.1
is still the top of its own league — but not the reasoning given for it.

⚠️ **[2026-08-09](2026-08-09-alphagateau-read.md)'s "~150× more optimiser steps per
position"** holds for the paper's batch 256. At the repo's committed batch 1024 it is
**37×**. Both settings are named in that entry; the headline number is not qualified.

## What was built instead

An **engine HTTP contract**, so any two bots can be confronted regardless of stack:

```
POST /move  {"fens": [...], "n": 128}  ->  {"moves": ["e2e4", ...]}
GET  /health
```

The arbiter owns every board, `python-chess` is the only authority on legality and
termination, and each engine only answers *"given these positions, your moves"*.
Validated by `t12h-gumbel` playing itself over the wire: **200 games, score exactly
0.5000, 78-44-78** — the mirror-pairing identity holding move for move.

AlphaGateau's side imports their `load_model` and `recurrent_fn` unmodified and makes
their exact `gumbel_muzero_policy` call, so their search cannot be mis-reproduced.
Action→UCI is done by **stepping and diffing the placement** rather than decoding the
index: pgx stores the board from the mover's point of view and flips it every ply, and
exactly one legal move can produce a given placement, so 0 or 2 matches raises.

Their released checkpoint carries its own provenance: **iteration 499, 65 536 000
frames, 327.5 wall-clock hours** (self-play 178.4, train 100.5, eval 48.6) on 8 GPUs
= **~2620 GPU-hours**, `env chess v2`, `pgx 2.1.0-rc0`. Both figures this journal
recorded from the paper, confirmed first-hand.

⚠️ Measured cost of their side: **369 ms per position at batch 64** on the 4060,
against our 0.3 ms. A head-to-head is bounded by them by a factor of ~1000.
