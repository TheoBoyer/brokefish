# Where the blunders come from: a dissection of the AlphaGateau matches

*2026-09-09. The 2026-08-19 entry measured that we lose material uncompensated at
2.1× AlphaGateau's rate and left three hypotheses in order of testability. Nobody had
looked at the positions. `scripts/blunder_dissect.py` does, in 54 seconds of card,
and the split it finds is not the one the hypotheses assumed.*

## The detector, committed this time

The 2026-08-19 count was ad hoc and its code is gone. The committed rule: our move,
their reply, our best recapture on the reply's square by piece value, still ≥3 pawns
below the material before our move. Per 1000 plies of the whole game:

| match | ours | theirs | ratio | games with one or more |
|---|---:|---:|---:|---|
| `t12h-gumbel` vs AG, 08-13 | 14.55 | 6.76 | 2.15 | 133 / 200 |
| `t24h-adamw-int8` vs AG, 08-19 | 13.58 | 6.14 | 2.21 | 124 / 200 |

The journal's 13.30 / 6.17 and 12.49 / 5.83 sit between this rule and "any recapture
on the board" (9.36 / 4.60); the exact rule of 08-19 is unrecoverable. The ratio is
2.0 to 2.2 under every variant tried, which is the claim that mattered.

## The dissection

Each blunder position is searched again by the network that played it, at n=128 and
n=1024, Gumbel m=16 with collapse, fp16 encoder as the match server ran. ⚠️ Two
fidelity caveats. `t12h-gumbel-004009.pt` no longer exists, so the 12 h side uses the
run's final checkpoint, which replays the recorded move in only 170 of 269 positions;
the 24 h side has the checkpoint that played and replays 245 of 263. The 18 misses
there may be today's Gumbel floor change (`2026-09-09-core-algorithm-review-fixes.md`),
unverified. Counts below are over the reproduced positions.

| class | 12 h (170) | 24 h (245) | what it means |
|---|---:|---:|---|
| OTHER, forced | 74 | 85 | every legal move already loses ≥3: a fork or pin is on the board, root value −0.31 / −0.44. The three-ply window flags the ply the piece comes off, one ply after the error |
| BUDGET | 55 | 93 | n=1024 plays another move; **32 / 59 of those still hang ≥3** |
| PRIOR | 25 | 32 | n=1024 still plays it and the refutation got zero visits at the child |
| VALUE | 16 | 35 | the refutation was visited and the child's value did not punish it |

Three things the hypotheses of 08-19 did not predict.

**A third of the count is one ply late.** The forced class is the detector's window,
not a decision, and it is the wrong set of positions to dissect. The error is the move
before, which a one-ply-earlier scan would find.

**Budget buys little.** n=1024 changes the move in 55 and 93 cases and fixes the
material in 23 and 34. Eightfold the search removes 14 % of the reproduced blunders.

**The PRIOR class is not a prior failure.** In 54 of the 57 cases the refutation's
prior at the child is above 1/E, median 0.06 to 0.09, and it received zero of the
child's ~240 visits at n=1024. The interior rule is `argmax(π' − N/(1+ΣN))` with
`π' = softmax(logits + σ(completed Q))` and `σ = (c_visit + max N) · c_scale · Q̂`
at `mctx`'s `c_visit = 50`, `c_scale = 0.1`. Once one reply is visited and looks
good, σ spans tens of logits and π' collapses onto it; a 7 % move whose Q is unknown
is never tried. That is a search-and-value interaction, and it is the one class with
a knob attached.

**The VALUE class is the value head not seeing material.** The refutation's prior at
the child is ~0.5, it is visited, and the Q behind it in our frame is −0.16 / −0.19
against a root value of +0.02 / +0.05, and the move is still chosen because every
alternative reads worse. Across all non-forced reproduced blunders, the raw network
value after hanging the piece is at least as high as before the move in **41 %** of
cases (40 / 96 and 66 / 160).

## What this says about the three hypotheses

Representation, value head, self-play diversity, in the 08-19 order. The dissection
puts the value head first on evidence: 41 % of hanging moves do not lower the raw
value, and the interior collapse only bites because the value at the visited reply is
trusted. Whether the head fails to see the material because the tokens cannot express
"this piece is attacked" (representation) or because the head is miscalibrated
(2026-08-24) is the next question, and the rule-feature prototype built today
(`BrokefishNet(rule_features=True)`) is the instrument for the first half: if the
value ceiling on the pinned set rises with the attacked bit as an input, it is
representation.

The diversity hypothesis stays untested: `eval/match.py` writes no PGN, so our
blunder rate against ourselves is still unmeasured.

## Five positions, 24 h network, all reproduced at n=128

1. VALUE. `r1bqkbnr/p1p2ppp/np6/4p3/4Q2P/2P5/PP1P1PP1/RNBK1BNR b kq - 0 6`, played
   `g8f6` at prior 0.89; `e4a8` wins a rook, prior 0.23 at the child, visited 25 times
   at n=1024, value after Qxa8 in our frame **+0.17**. n=1024 plays it too.
2. PRIOR. `rn1qkbnr/pbpp2pp/1p6/4pp2/P5P1/1PN5/2PPPP1P/R1BQKBNR w KQkq - 2 5`, played
   `g4f5` at 0.39; `b7h1` has prior 0.071 at the child and zero of 240 visits.
3. BUDGET, fixed. `r1bqkb1r/pppp1p1p/n3p1p1/8/2BQ4/P3P3/1PPP1PPP/RNB1K1NR b KQkq - 1 6`,
   played `d7d5` into `d4h8`; n=1024 plays `h8g8`.
4. BUDGET, not fixed. `r1b1kbnr/ppp2p1p/2nppq2/6p1/1P2P3/3P1P2/P1P2KPP/RNBQ1BNR w kq - 1 6`,
   played `b4b5` allowing `f6a1`; n=1024 plays `f1e2`, which still loses 5.
5. Forced. `rnbqk1r1/pppp1p1p/4pb2/6p1/2P2N2/PP6/3PPPPP/RN1QKB1R w KQq - 0 8`, played
   `f4d3`, `f6a1` follows; the best alternative already loses 3.

The per-position table is `logs/blunder_dissect.csv`, 532 rows. Nothing here reaches
a training path or selects a checkpoint.
