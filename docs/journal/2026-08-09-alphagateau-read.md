# 2026-08-09 (tenth entry) — reading the paper Gate 2 is defined against

Gate 2 has said "beat AlphaGateau (~2100 Elo)" since 2026-07-28. Until today nobody
in this project had read the paper. `papers/` held AZ, AGZ and KataGo; every
AlphaGateau number in the repository traced to one line of
[the prior-art entry](2026-07-28-prior-art.md), written from the abstract.

It is four pages (NeurIPS 2024 workshop, arXiv:2410.23753, Rigaux & Kashima), the
code is public, and reading both changes the target rather than refining it.
`papers/alphagateau.txt` and the source at `github.com/akulen/AlphaGateau`.

## ⚠️ Their Elo is self-anchored, with the pool mean pinned at 1000

`elo.py:96`:

```python
[average * len(players)]  # We set the average elo to 1000
```

The rating is a weighted-least-squares fit over **their own pool of checkpoints**,
translated so the pool's mean is 1000. There is no Stockfish, no CCRL, no external
reference anywhere in the repository — grepped.

**So 2105 is on their arbitrary scale exactly as `+1010` is on ours.** "Beat
AlphaGateau (~2100 Elo)" compares two self-anchored scales that share nothing, and is
not a well-posed claim. It has been the project's headline goal for twelve days.

What *is* meaningful in their results is the gap under one protocol: **AlphaGateau
2105 ± 42 against AlphaZero 667 ± 38**, the same training loop with the CNN swapped
for their GNN. That is an architecture comparison, not an absolute strength claim, and
the paper is careful to present it that way — the misreading is ours.

## What replaces the gate

**Their checkpoints are in the repository.** The archive is 400 MB and most of it is
saved parameters. So a **direct head-to-head against the released network** is
possible, needs no rating conversion, and is the only unambiguous comparison
available. It is also cheaper than `roadmap.md` D4's external-engine ladder, which
was blocked on choosing engines. The cost is a bridge from JAX/pgx to our engine, not
a research problem.

## Two recorded facts were wrong

⚠️ **"AlphaGateau, at exactly 1M parameters, plateaus around 2100."** The paper says
the opposite: *"AlphaGateau and AlphaZero have not reached a performance plateau after
500 iterations."* That plateau was one of three arguments for revising the network
from 1M to 6.38M parameters; the other two (Neumann & Gros' scaling laws, the
lc0 T1-distilled and CF-6M sizes) are untouched, so the decision stands on two legs
instead of three. The parameter count itself I could **not** verify from the text —
the paper gives embedding dimension 128 and 5-6 layers and no total.

⚠️ **"128k games."** Their unit is 256 *streams* of 512 plies, stepped through
`auto_reset(env.step, env.init)` (`mcts.py:132`), so each stream contains many
complete games. An iteration is **131 072 positions**, and 500 iterations is
**65.5M positions**. The games figure is a stream count, and any comparison denominated
in games is meaningless without their mean game length, which is not published.

## The comparison that survives, in positions

| | positions | wall clock | hardware |
|---|---:|---:|---|
| AlphaGateau | 65.5 M | 13 d 16 h | 8 × RTX A5000 |
| `t24h-muon` (projected) | ~42 M | 24 h | 1 × RTX 4060 laptop |

**~8.7× the positions per hour on roughly a thirteenth of the compute.** An earlier
version of this comparison, denominated in games, gave 38× and was wrong.

## Their optimisation regime is nothing like ours

| | batch | steps / iteration | reuse | steps per new position |
|---|---:|---:|---:|---:|
| AlphaGateau | 256 | 3 904 | **7.6** | 0.0298 |
| brokefish | 4 096 | 2.03 per generation | **0.815** | 0.000198 |

**~150× more optimiser steps per position generated** — 16× from the smaller batch,
9.4× from the reuse. Window 1M frames, one epoch per iteration (`Ntrain = 1`), Adam at
1e-3, Gumbel MuZero through MCTX at **128 simulations**.

⚠️ The repo's committed default is `training_batch_size = 2**7` *per device* over 8
devices, i.e. a global batch of 1024; the paper's §5.3 experiment used 256 (3 904
mini-batches over the 1M window). The default is not the published setting.

⚠️ Their frames include post-termination padding inside a stream, so the effective
reuse over live positions is lower than 7.6 by an unmeasured factor.

**This bears directly on `docs/roadmap.md`'s cadence.** `samples_per_position = 0.815`
is AZ's ratio, and the one paper in the literature that beats AZ by 1438 Elo under its
own protocol uses roughly ten times more. That does not make 0.815 wrong — different
architecture, different search, different batch — but it removes "aim for ~1" as
something the field agrees on.

## What it does not change

The runs are unaffected. Every Elo number this project has produced is a self-anchored
measurement on a scale defined by `evaluation.md` §5.1, and remains exactly as valid
as it was yesterday. What changes is what may be said about *another* project's
number, and Gate 2 is the only thing that rested on that.
