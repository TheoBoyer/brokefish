# 2026-08-04 — the terminal collapse: a cost win, not a strength win

`search.md` §6.6a. Landed in `ff6290b`, **off by default**.

## What was wrong

§6.1a's root sweep, landed two days earlier, guarantees that a mate at the root is
*present* in the edge set. It never guaranteed it was **chosen**, and nobody had
looked.

Measured on `t9h-n128-sweep`'s finished replay buffer — 88,322 positions from the
newest 600 games, decoded to FEN and checked against python-chess:

| | |
|---|---|
| positions with a mate in one | **785 (0.89 %)** |
| the mating move was in the record's support | **100.0 %** |
| median `pi` on it | **0.302** (mean 0.401, p10 0.039) |
| targets that were a point mass (`pi > 0.99`) | **0 of 785** |
| the mate was the argmax | **56.1 %** (53.6 % past `tau_plies`, where the argmax *is* the move played) |
| mean visits on the mating edge | 51.3 of 128 |

So the sweep worked perfectly and the search then declined to use it, **44 % of the
time**.

The mechanism is not subtle and is monotone in two independent variables. Those roots
are already won — `root_value` mean **0.886** — so a proved `Q = 1.0` leads its rivals
by ~0.11, while a fresh edge's exploration term is worth ~0.42, and 128 simulations
over 42.4 edges is 3.0 visits each. Exploration wins.

| root's edge count | `pi(mate)` | argmax is the mate |
|---|---|---|
| < 30 | 0.631 | 79.7 % |
| 30–40 | 0.514 | 70.4 % |
| 40–50 | 0.348 | 50.9 % |
| 50–60 | 0.244 | 35.9 % |
| ≥ 60 | **0.203** | **31.9 %** |

| `root_value` | `pi(mate)` | argmax is the mate |
|---|---|---|
| < 0.85 | 0.277 | 52.9 % |
| 0.85–0.90 | 0.325 | 46.3 % |
| 0.90–0.95 | 0.550 | 70.4 % |
| ≥ 0.95 | **0.792** | **97.7 %** |

Both say the same thing twice: the wider the root and the more comfortably won the
position, the less a proved mate is worth against PUCT's exploration term.

## The fix

A node with a proved-winning edge scores `-N` over those edges and `-inf` elsewhere,
so visits round-robin over them; at the root the stored target becomes uniform over
the winners. `edge_N` is untouched, so invariant 6 still reads the visits the
simulations actually made. Details and the parity trap are in `search.md` §6.6a.

## The experiment

7 h at `n = 128`, **41 of 42 config fields identical** to `t9h-n128-sweep` (verified
against the config stored in its resumable checkpoint; the 42nd is
`terminal_collapse`). Same `--total-steps 2600`, so the cosine schedule is identical
and a comparison at matched steps is clean. The startup weight-propagation check
printed the same numbers on both runs, so the initial weights were identical too.

**`collapsed_roots == terminal_checkmate` in 1011 of 1011 generations.** That identity
is the proof: a collapsed root always ends the game, so the counts can only agree if
every proved mate is converted. 100 %, against the control's 56 %.

| over the first 1011 generations | control | collapse |
|---|---|---|
| records generated | 10,352,640 | 10,352,640 (identical — the cadence rides records) |
| games finished | 81,168 | **90,134 (+11.0 %)** |
| checkmates | 43,315 | 45,646 (+5.4 %) |
| mean plies, gens 900–1000 | 148.8 | **138.9 (−6.7 %)** |
| evals/s in self-play | 58,730 | 58,485 (**−0.4 %**) |
| `loss` at step 2000 | 2.8914 | 2.9200 |

## Test time, same weights on both sides

512 shared random openings, `n = 128`, evaluation deterministic (`eps = 0`,
`tau_plies = 0`), one network playing both sides so the null is exact.

```
PAIRED over 503 openings finished in both arms:  -9.08 plies  ±1.70   t = -5.34
  80 of 503 games (15.9 %) differed at all; bit-identical elsewhere
  over just those: -57.08 plies ±8.98,  76 shorter / 4 longer
head to head:  collapse side 515.5/1010 = 0.5104 ±0.0157  ->  +7.2 Elo
```

⚠️ The first version of this measurement reported the length delta with **unpaired**
standard errors (−9.93 ± 6.98, t = 1.42) and called it not significant. That was the
wrong test: the arms share openings and evaluation is deterministic, so 84 % of games
are bit-identical and contribute exactly zero variance. Treating them as independent
samples threw that away and inflated the error four-fold.

The −57 plies on games that changed also kills the obvious objection — that the
shortening is mechanical, a mate landing three plies earlier. 28 moves is not
mechanical. Without the collapse those games run on for dozens of plies after the win
was available.

## The Elo, and what it means

Both runs rated against the frozen random-init anchor every league shares.
`t9h-n128-sweep` had never been rated; it is now.

Mean over 20 matched steps: **−14.6 Elo, sd 28.3**, per-point CI95 ±30–37. Head to
head: **+7.2 ± ~15**. Both null.

⚠️ And the league is not a test of the feature: `runner.eval_config` builds
`SearchConfig(n, B, eps=0, tau_plies=0)` with `terminal_collapse` defaulting off, so
**both players played every league game without it**. The league measured only the
effect on the *learned weights* — 0.9 % of records — which was predicted to be
invisible and was. The head-to-head is the measurement that tests the feature, and it
also says zero.

**Converting every available mate-in-1 is worth no measurable Elo.** Those positions
were already won and the mate was being found a few plies later anyway. That is a
real finding about what `n = 128` is and is not missing, and it is the reason to write
this down rather than quietly keep the feature for the number that did move.

What did move is cost: **+11 % games per identical record count**, and 6–7 % shorter
games at evaluation time, for **−0.4 %** throughput. On a project whose deliverable is
a cost-versus-Elo curve, that is the axis it lands on.

## Kept off by default

It changes the move played and the training target, so every number measured before
today was measured without it. `--terminal-collapse` on `train.loop`; it changes the
config hash, so it cannot be flipped silently across a resume.
