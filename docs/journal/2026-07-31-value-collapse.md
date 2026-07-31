# The collapse of run `c2-8h`, and the constant that caused it

The first long training run — 7h50m, 177,669 games, 2,827 optimiser steps — produced
a policy loss that fell by a factor of three and a network that plays `g2-g3` from
the opening position and `g2-g3` again with a black queen hanging on d5.

The falling loss was the collapse, not the learning. This is what it was.

Written 2026-07-31, after C2's first long run.

---

## 1. What the run looked like

Policy KL against the search target fell 0.598 → 0.1995 and never stopped falling.
Every other measurement moved the wrong way at the same time.

| generations | decisive games | root edges visited | max `pi` | root entropy |
|---|---|---|---|---|
| 1–300 | 5.13 % | 4.35 | 0.545 | 1.088 |
| 601–900 | 3.04 % | 3.66 | 0.599 | 0.936 |
| 1201–1500 | 2.18 % | 3.49 | 0.609 | 0.902 |
| 1801–1954 | **1.76 %** | **3.42** | **0.611** | **0.892** |

Draws reached 99.28 % of stored positions. Only 0.28 % of games hit the 256-ply cap,
so this is not a truncation artifact.

## 2. The value head learned its own mean

Same network, positions chosen to be as far apart as a chess position can be:

| position | value |
|---|---|
| K+Q vs lone king (winning) | −0.0186 |
| K vs king+queen (lost) | −0.0120 |
| start | −0.0161 |
| black queen hanging | −0.0034 |

Total spread across everything tried: 0.03. The training loss confirms it from the
other side — 0.0096 against a floor of `E[z^2] = 0.0072`, so the head scores *worse*
than emitting the constant zero.

## 3. What the search was doing

Measured on the final checkpoint, from the opening position, Dirichlet noise off:

```
n =  64   visited 3/20   KL(pi||P) = 0.0855
n = 128   visited 3/20   KL(pi||P) = 0.0854
n = 256   visited 3/20   KL(pi||P) = 0.0853
n = 512   visited 3/20   KL(pi||P) = 0.0853
n = 800   visited 3/20   KL(pi||P) = 0.0855
```

⚠️ **A 12.5× increase in the simulation budget visits the same three moves and
returns the same target.** That is not a weak search, it is a search that cannot
move. `n` was not the variable.

## 4. The cause: a constant that did not ride the remap

`search.md` §3.5 stores `Q` in `[0, 1]` rather than AGZ's `[-1, 1]`, deliberately:
PUCT *adds* the value and exploration terms, so `Q`'s range fixes what `pb_c_init`
is worth, and `[0, 1]` is what keeps the published 1.25 meaning what it means. That
section also states the rule that matters here — the two conventions are equivalent
under `q01 = (v + 1) / 2` **only if the constants are rescaled with them**.

The value was rescaled. The first-play urgency was not.

| | `[-1, 1]` (AGZ) | `[0, 1]` (here) |
|---|---|---|
| draw | 0 | 0.5 |
| **FPU** | **0** | **0.5** — shipped as a literal `0` |
| `pb_c_init` | 2.5 | 1.25 |

AGZ scores an untried move at `Q = 0`, which in `[-1, 1]` is a **draw**: a neutral
prior on a move nobody has looked at. Carried across unchanged into `[0, 1]`, that 0
is a **certain loss**.

**The arithmetic.** At `n = 800`, against a value head near the draw value:

| edge | `pb_c * prior` | Q | score |
|---|---|---|---|
| top (N = 790, P = 0.65) | 0.030 | 0.500 | **0.530** |
| 4th (N = 0, P = 0.005) | 0.183 | **0.000** | **0.183** |

An untried move needs `prior > 0.0145` to be reachable at all, and that threshold
does not fall as the budget rises — `pb_c` carries `sqrt(N_v) / (N_e + 1)`, so more
simulations raise the incumbent's score as fast as the challenger's.

⚠️ **This is an absorbing state.** A move whose prior drops below the threshold can
never be visited, so it can never appear in a training target, so its prior can only
fall further. Ordinary under-exploration stagnates; this ratchets.

## 5. The loop

1. 99.3 % of `z` are 0 → the value head learns the constant → **Q carries no signal**.
2. With Q flat, PUCT visits roughly in proportion to the prior — but visits are
   integers and the argmax compounds, so `pi` is a **sharpened** `P`, not a better one.
3. Training the policy toward `pi` sharpens `P`.
4. Priors fall below 0.0145; §6.6 locks those moves out permanently.
5. The search narrows, decisive games become rarer, and step 1 gets worse.

The policy loss falls throughout, because the network is fitting a target it
generates itself.

## 6. The fix, and what it restores

`FPU_DRAW = 0.5` in `brokefish/search/torch_impl.py`, `kFpuDraw` in
`csrc/search.cuh`. `tests/test_search_cuda.py` holds the two together tree for tree.

Same collapsed checkpoint, and an untrained network for comparison:

| network | position | n | visited | KL(pi&#124;&#124;P) |
|---|---|---|---|---|
| untrained | start | 64 | **20/20** | **0.444** |
| untrained | start | 800 | 20/20 | 0.422 |
| untrained | queen hanging | 800 | **31/31** | **0.567** |
| `c2-8h` | start | 64 | 3/20 | 0.086 |
| `c2-8h` | start | 800 | 20/20 | **0.0013** |

Two things to read here.

**The search is a policy improvement operator again.** On an untrained network it
now returns a target 0.42–0.57 nats away from the prior, where the collapsed network
managed 0.086. That gap is the entire signal AlphaZero trains on.

⚠️ **The collapsed checkpoint is not recoverable.** Restore full exploration and it
still returns `KL = 0.0013` — it explores all twenty moves and prefers none, because
its value head is dead and its prior is spent. `checkpoints/c2-8h.pt` is a record of
the failure, not a starting point.

## 7. What this does not fix

The FPU made the collapse irreversible. It did not create it. With 99.3 % draws the
value target carries almost no information, and no selection rule repairs that. The
open question is where a value signal comes from before the network can force a
result — shorter games, a bootstrapped or n-step target instead of the pure final
outcome, or something else.

⚠️ Two further points on the run itself, neither caused by the above:

- `--checkpoint-every` overwrites a single file, so `c2-8h` has **no skill-vs-step
  curve**. The collapse hypothesis predicts skill peaked early and declined; a single
  endpoint cannot test it. Retained checkpoints are a prerequisite for the cost-Elo
  curve that is the project's deliverable.
- The last ~30 minutes of the run shared the GPU with the diagnostics above
  (11.4 → 12.5 s/generation). Nothing in this document rests on that window.

## 8. Two corrections to `search.md`

Both are in §3.5 and §6.6 now.

- §6.6 said an unvisited edge scores `0`, "which is AlphaZero's first-play urgency
  and is a loss from the mover's point of view". The description was accurate; it is
  the value that was wrong, and the phrase "is a loss" was written as a statement of
  fact rather than read as the alarm it was.
- §3.5 said that keeping `Q` in `[-1, 1]` and reusing 1.25 runs "twice the intended
  exploration". It is **half**. Substituting `q01 = (v + 1) / 2` and dropping the
  per-node constant gives `score = 2 * ((pb_c / 2) * P + q01) - 1`, so `[-1, 1]` at
  `c` is `[0, 1]` at `c / 2`.

## 9. Where this belongs on the betting sheet

[`fidelity.md`](2026-07-30-fidelity.md) is the register of places where this
repository might be wrong about the specifications it is copying, and `CLAUDE.md`
already records that the released `pseudocode.py` contradicts AGZ in three silent
ways. This is a fourth, and unlike the other three it was found by a training run
rather than by reading — which is the expensive way.

The general lesson is narrower than "check the pseudocode". It is that **an affine
change of variable has to be applied to every constant in that variable's space**,
and that the place to look for the one that was missed is wherever a magic number
sits next to a rescaled quantity.
