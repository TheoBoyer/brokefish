# The value head is not broken. The instrument is.

**2026-08-21, during `t12h-wdl`.** The value-head puzzle probe had been flat for four
consecutive 12-hour runs — 0.3530 → 0.3480 on `t12h-flat`, ×0.99, while the policy probe
went ×4.8 — and `t12h-wdl` was launched with that probe preregistered as its **primary
metric**. An end-to-end audit of the value head's life cycle was asked for.

**Verdict: every leg of the value path is correct, the head is learning strongly, and
the probe cannot see it.** Held-out 3-class accuracy rose **0.500 → 0.621** over the same
1,600 steps in which the probe moved **+0.005**. The probe's correlation with real
held-out value quality across nine checkpoints is **−0.07 to −0.37**.

⚠️ **The preregistration of `t12h-wdl` is therefore wrong**, and I wrote it. Its primary
metric measures something other than what it claims to.

## Leg 1 — the head and its two kernels

Verified the same day the WDL option landed (`2026-08-21-the-wdl-value-head.md`):
kernel against torch, both heads, both implementations. Max |Δv| **2.6e-3** scalar and
**1.6e-3** WDL in fp16; 2.6e-2 / 1.2e-2 under int8, which is the documented quantisation
band and not a value-path fact. The row select is pinned by flipping the side to move,
and the class order is pinned by swapping the loss and win rows of `value.weight` and
demanding the *kernel's* value negate exactly, without consulting torch.

## Leg 2 — the collapse into the tree

`search.cuh:1076`: `node_value = (value[b] + 1) / 2`, the documented [-1,1] → [0,1] map.

⚠️ Worth writing down because it looks like a lossy approximation and is not: for the
classifier, `value = p_w − p_l`, so

```
node_value = (1 + p_w − p_l)/2 = (p_w + p_d + p_l + p_w − p_l)/2 = p_w + p_d/2
```

which is **exactly the expected score** in [0,1]. The WDL head enters the tree with no
loss of information about the quantity the tree actually averages.

Terminal nodes take `(tr.result + 1)/2` in the same convention (`search.cuh:1014`), and
the backup negates by path parity, `(L − d) & 1 ? 1 − qleaf : qleaf`. The source itself
carries the warning that inverting that parity "produces a search that reliably plays
the worst available move" and that a two-level test cannot see it. Covered by
`tselect.cu` at 20,000 selections, by exact agreement with the AGZ oracle at n = 512,
and by `tests/test_search.py` + `test_gumbel.py`, 158 passed today.

## Leg 3 — the label

`buffer.py:245`, and it is **not** the distance-from-the-end alternation rule:

```
z = −result · sign(control_record) · sign(control_terminal)
```

i.e. z is a pure function of which side is to move in that record. Checked directly
against the **live** ring of `t12h-wdl`, 4,000 games:

| invariant | violations |
|---|---|
| `|z|` constant within a game | **0** |
| `z` a pure function of the side to move | **0** |
| decisive games verified stm-pure | 3,557 |

Independent evidence that the *sign* is right, not just self-consistent —
`root_value` is the search's own Q at the root in the mover's frame, written but trained
on by nothing:

```
corr(root_value, z) = +0.559
search says losing  (Q < −0.2)   mean z = −0.656
search unsure       (|Q| ≤ 0.2)  mean z = −0.022
search says winning (Q > +0.2)   mean z = +0.602
```

A parity inversion makes all of that negative.

⚠️ **My first pass reported 597 and 1,399 violations and both were my own bug.**
`_g_start` holds an **absolute** ring offset (`buffer.py:293`) and I added `head` to it,
and I asserted sign alternation over the record index, which `buffer.py`'s own comment
says explicitly is not the rule on sparse plies. Retracted in full.

## Leg 4 — the target distribution

No degeneracy. Over the live window: **88.9 % decisive, 11.1 % drawn**,
`games_capped = 0` for the whole run, mean game length 123 plies (median 111, max 403).
The value head is not being asked to predict a constant.

## Leg 5 — training

Nothing starved and nothing saturated:

| counter | at step 1,882 |
|---|---|
| value cross-entropy | 0.759, from `ln 3 = 1.0986` |
| `grad_value_head` | **0.654** |
| `grad_policy_head` | 0.183 |
| `value_saturated_frac` | 0.026 |
| `value_mean` | +0.076 |

The value head is receiving **3.6× the gradient the policy head is**.

## Leg 6 — so is it learning? Yes, a lot

Held out by `weight_gen`: a record carries the generation of the weights that produced
it, so records from generations *after* a checkpoint cannot have trained it. Game-level,
never position-level — a position-level split leaked once already (2026-08-19).

| checkpoint | sign acc | corr(v, z) | MSE | 3-class acc | v std |
|---|---:|---:|---:|---:|---:|
| 201 | 0.8083 | 0.6688 | 0.5348 | **0.5003** | 0.464 |
| 601 | 0.8119 | 0.6802 | 0.5089 | 0.5467 | 0.509 |
| 1002 | 0.8227 | 0.7040 | 0.4686 | 0.5819 | 0.570 |
| 1403 | 0.8283 | 0.7170 | 0.4506 | 0.5947 | 0.592 |
| 1803 | 0.8299 | 0.7243 | 0.4355 | **0.6206** | 0.635 |
| *the search's own `root_value`* | *0.8524* | *0.7567* | — | — | — |

**3-class accuracy rises on 8 of 8 steps.** The head is already within 0.02 sign-accuracy
of what a full n = 128 search produces on the same positions.

⚠️ Control for distribution shift, since the hold-out comes from the newest policy's
self-play: repeated on records generated ~230 generations earlier. Same direction, same
size.

```
EARLY records (weight_gen 680-720)   ckpt 201 -> 1803:  corr 0.496 -> 0.560   acc3 0.395 -> 0.460
LATE  records (weight_gen 905-950)   ckpt 201 -> 1803:  corr 0.481 -> 0.531   acc3 0.356 -> 0.435
```

⚠️ Absolute levels differ between slices and that is composition, not quality: a slice
weighted toward late-game positions is far easier to value. Only within-slice comparison
across checkpoints means anything.

## Leg 7 — the probe, and why it is blind

Its correlation with every measure of real held-out value quality, over the nine
checkpoints of this run:

| held-out series | change over the run | r with the value-puzzle probe |
|---|---:|---:|
| corr(v, z) | +0.0555 | **−0.195** |
| 3-class accuracy | +0.1203 | **−0.066** |
| sign accuracy | +0.0216 | **−0.368** |
| MSE (lower better) | −0.0993 | +0.154 |
| *value-puzzle pass@1* | *+0.0045* | — |

Spearman on ranks: **−0.13**. The probe is not a noisy version of value quality; it is
uncorrelated with it.

Its dynamic range, measured on a common harness (6,000 puzzles, first move of each line
only, so the numbers are lower than the in-loop metric, which averages every step):

| evaluator | value pass@1 |
|---|---:|
| untrained net, random init | 0.0455 |
| `t12h-flat` @201 | 0.1970 |
| `t12h-wdl` @201 | 0.2038 |
| `t12h-flat` @5010 (**the whole 12 h run**) | 0.2035 |
| `t12h-wdl` @1803 | 0.2112 |
| **a hand-written material count, no network at all** | **0.2500** |
| `t24h-adamw-int8`, 24 h | 0.2923 |

Two things fall out. **Every 12-hour checkpoint on this ledger scores below a piece
count**, and a whole 12-hour run moves this metric by +0.007 — a quarter of its own
probe-to-probe noise. The instrument does separate an untrained net from a 24-hour one,
so it is not broken; its resolution simply starts somewhere past 12 hours of training,
and every series we have been reading has been inside its noise floor the entire time.

The mechanism is not mysterious. `value_pick` grades a **static** evaluator on
**tactical** puzzles, where the answer is usually a move that does not maximise
immediate material. Getting better at valuing self-play positions does not move that,
which is precisely what the −0.07 to −0.37 says.

## What this costs and what to do

- ⚠️ `t12h-wdl`'s preregistered primary metric is void. The run is still valid — one
  flag moved, the league is the Elo measurement — but "did the WDL head fix the value
  head" cannot be answered by the value-puzzle probe.
- ⚠️ **Every `val@1` number in every status and every chain header since 2026-08-19 is
  inside this instrument's noise floor**, including the "0.3530 → 0.3480, ×0.99" that
  motivated building the WDL head in the first place. That premise is not supported by
  the data that was used to state it.
- The measurement that *does* work is cheap, on-distribution and already possible with
  no engine, no puzzle file and no search: held-out `corr(v, z)` and 3-class accuracy on
  records whose `weight_gen` post-dates the checkpoint. It is ~1 s per checkpoint against
  the probe's 22 s.
- ⚠️ `evaluation.md` still applies: this is a diagnostic, and it may not select a
  checkpoint.

---

# Corrections, later the same day

⚠️ **Appended rather than edited** (journals are append-only). Two claims above are
wrong or overstated and one large finding was missed entirely. The leg-by-leg audit of
the value *path* — kernel, tree, label, distribution, gradients — stands unchanged.

## Correction 1 — "the probe is blind" is too strong

Théo pushed back with the case the probe was built on: on 2026-08-17 it ranked
`t12h-muon9-int8` below `t12h-gumbel` (0.414 vs 0.486) when the policy probe had them
the wrong way round, and the head-to-head at n = 128 agreed with the probe. That case
is good and it reproduces.

Seven finished nets, one common 47,025-position yardstick from `t12h-wdl` self-play —
a neutral third party for the Muon-vs-AdamW pair:

| net | probe | sign acc | corr(v, z) | acc3 |
|---|---:|---:|---:|---:|
| t24h-adamw-int8 (AdamW, 24 h) | 0.2923 | 0.7367 | 0.5737 | 0.4883 |
| t12h-int8 (AdamW) | 0.2483 | 0.7304 | 0.5692 | 0.4700 |
| t12h-gumbel (AdamW fp8) | 0.2382 | 0.7347 | 0.5694 | 0.4621 |
| t12h-flat (AdamW) | 0.1920 | 0.7177 | 0.5467 | 0.4835 |
| t12h-muon9-int8 (Muon) | 0.1860 | 0.7201 | 0.5195 | 0.5313 |
| t12h-muon9 (Muon) | 0.1855 | 0.7153 | 0.5189 | 0.5149 |
| t12h-vw2 (Muon, vw = 2.0) | 0.1375 | 0.6957 | 0.4614 | 0.5307 |

The 2026-08-17 gap reproduces (+0.052 here against +0.072 then, same direction, 4x the
probe's noise), and `t12h-vw2` — the −231 Elo arm — is last. **The probe is a valid
cross-run instrument.**

What is true is narrower and still matters: **its within-run resolution is below its own
noise.** Within-run movement is **0.59x** the sd of `t12h-flat`'s 25 probes, and 4.8 % of
its across-recipe span, against 44-53 % for the held-out measures. And across runs the
cheap held-out metric agrees with it at **Spearman +0.96** (corr) and **+0.93** (sign
accuracy) — so the probe carries the same information, compressed on the axis we need.

## Correction 2 — I quoted the most flattering statistic

The headline "held-out 3-class accuracy 0.500 → 0.621" used `acc3`, which thresholds at
±1/3 and is therefore **scale-sensitive**: the head's spread grew 0.464 → 0.635 over that
stretch, so part of that rise is confidence, not accuracy. Across recipes `acc3` ranks
Muon *above* AdamW (Spearman **−0.64** against the probe), which the league says is
backwards. The scale-free numbers are the honest ones: **corr +0.056, sign accuracy
+0.022**. Half the size I quoted.

## The finding I missed: the value head peaks at ~6 h and then degrades

Measuring only endpoint-to-endpoint hid this. `t12h-flat`'s full ladder on one fixed
20,000-position yardstick, every checkpoint scored on the identical set:

| step | ~h | sign acc | corr(v, z) | MSE | v_std |
|---:|---:|---:|---:|---:|---:|
| 201 | 0.5 | 0.6387 | 0.4860 | 0.7181 | 0.240 |
| 1002 | 2.4 | 0.7207 | 0.5181 | 0.6268 | 0.423 |
| 1803 | 4.3 | 0.7284 | 0.5425 | 0.6015 | 0.475 |
| **2605** | **6.2** | **0.7378** | **0.5577** | **0.5859** | 0.504 |
| 3408 | 8.1 | 0.7334 | 0.5508 | 0.5956 | 0.554 |
| 4209 | 10.0 | 0.7214 | 0.5417 | 0.6092 | 0.564 |
| 5010 | 11.9 | 0.7189 | 0.5361 | 0.6162 | 0.596 |

**The head improves hard for six hours and then goes backwards for six hours**, while
`v_std` climbs monotonically the whole time: it keeps getting *more confident* while
getting *less accurate*. Net over the run is +0.050 corr, of which roughly half is
given back.

That is why the puzzle probe read as a flat line. The underlying quantity is not flat —
it is a hill, and the probe was sampling its endpoints.

⚠️ Mechanism unknown. Accuracy down + confidence up is what value-head overfitting to
the replay window looks like, and it is consistent with in-buffer value MSE having been
an anti-signal three times, but **that is a hypothesis and nothing here measures it**.

⚠️ **This is not permission to stop at six hours or to select that checkpoint.**
`evaluation.md` forbids it and value-head quality is not Elo.

`t12h-wdl` at the same yardstick, in flight:

| step | ~h | sign acc | corr(v, z) | MSE | v_std |
|---:|---:|---:|---:|---:|---:|
| 201 | 0.5 | 0.7064 | 0.5012 | 0.6515 | 0.355 |
| 1002 | 2.4 | 0.7272 | 0.5383 | 0.6056 | 0.455 |
| 2004 | 4.8 | 0.7327 | 0.5597 | 0.5859 | 0.517 |

At 4.8 h it is already **above the scalar head's whole-run peak** (0.5597 vs 0.5577 at
6.2 h) and still rising, and it starts far better (0.5012 vs 0.4860, sign 0.706 vs
0.639). ⚠️ One yardstick, one run each. **Whether it also turns over at hour six is the
open question and the thing worth watching** — not the puzzle probe.

---

# The same trajectory under Muon, and a calibration signature

**Appended the same evening.** Asked to repeat the ladder on a Muon run. All four
finished runs share one shape, and the differences between them line up with the Elo we
already have.

⚠️ **The yardstick is not the one used in the table above.** It is built from the newest
200 games of `t12h-wdl`'s live ring, which advanced in the intervening hours, so absolute
numbers here do not match the earlier section — `t12h-flat`'s peak reads 0.5253 at step
3608 here against 0.5577 at 2605 there. **Within this table the set is identical for
every row**, which is what the comparison needs. The disagreement between the two
yardsticks is itself the finding that the peak is a **broad plateau (steps ~1400-3600),
not a sharp point**, and that the size of the decline is yardstick-sensitive while its
existence is not.

| run | optimiser | peak corr | at step | final corr | decline | final v_std |
|---|---|---:|---:|---:|---:|---:|
| `t12h-flat` | AdamW, flat lr | **0.5253** | 3608 | 0.5135 | **−0.012** | 0.594 |
| `t12h-muon9` | Muon wd .09, int8-FFN | 0.5149 | 2004 | 0.4820 | −0.033 | 0.634 |
| `t12h-muon9-int8` | Muon wd .09, int8-all | 0.5123 | 2204 | 0.4900 | −0.022 | 0.660 |
| `t12h-vw2` | Muon + `--value-weight 2.0` | 0.5064 | 2204 | **0.4248** | **−0.082** | 0.700 |
| `t12h-wdl` (live, 4.8 h) | AdamW + WDL head | 0.5597 | 2004, still rising | — | — | 0.517 |

**Muon peaks ~1.6x earlier and degrades 2-3x harder than AdamW**, and doubling the value
weight on top of Muon is catastrophic: −0.082 corr, MSE 0.646 → 0.802. ⚠️ `t12h-vw2` is
the arm that lost **−231 Elo**, so the worst value degradation on this instrument belongs
to the worst arm on the league. That is corroboration, not proof — one run, and
correlational.

## The signature: every run overshoots its own calibration

An MSE-optimal predictor with correlation `r` has **exactly** `std = r · std(z)`. Wider
than that is overconfidence in a precise sense: shrinking it lowers the MSE while
changing no ranking at all. On this yardstick `std(z) = 0.9316`.

| run | @201 | @1403 | at its peak | final |
|---|---:|---:|---:|---:|
| `t12h-flat` | −44 % | −3 % | **+13 %** | +24 % |
| `t12h-muon9` | −5 % | +4 % | **+15 %** | +41 % |
| `t12h-muon9-int8` | −1 % | +1 % | **+21 %** | +45 % |
| `t12h-vw2` | −2 % | +5 % | **+12 %** | **+77 %** |
| `t12h-wdl` (live) | −24 % | — | — | **−1 % @2004** |

Every run starts **under**-confident, passes through calibration, and then overshoots
monotonically — and **corr peaks while the excess is between +12 % and +21 %, in all four
runs.** The final excess orders the runs exactly as the decline does: flat +24 %,
muon9 +41 %, muon9-int8 +45 %, vw2 +77 %.

⚠️ `t12h-wdl` sits at **−0.8 % excess at step 2004** — essentially calibrated, with the
highest corr of any row here. ⚠️ But `t12h-flat` was at −3 % at step 1403 and +9 % by
2805, so this may be the same curve shifted rather than a different curve. **Four hours
from now tells the difference and nothing before then does.**

## What is not established

The mechanism. "MSE against ±1 targets keeps pushing the logit outward long after the
ranking has stopped improving" is a plausible story and the observation is consistent
with it, but a cross-entropy also pushes toward one-hot, so it does not obviously predict
that the WDL head escapes. Nothing here measures why.

⚠️ And a confound worth naming: the yardstick is drawn from one net's self-play at one
strength. A later, stronger checkpoint's value function is calibrated for stronger play
and could look worse on weaker-play positions without being worse. `t12h-vw2`'s MSE
reaching 0.80 is too large to be that, but the effect is not zero for the smaller
declines.

---

# Retraction, 2026-08-22: the hill is Muon's, not AdamW's

⚠️ **"The AdamW value head peaks at ~6 h and then degrades" is wrong and this entry
made it the headline.** It was an artefact of the yardstick, which is the confound the
section above named and then failed to act on.

The yardstick was rebuilt each time from `t12h-wdl`'s **live** ring, so it moved between
tables and was drawn from a *weak* net's self-play. Later, stronger checkpoints score
worse on weak-play positions without being worse. Fixed by freezing 40 000 positions to
a file once the run ended, and re-running every ladder on that one set.

| step | t12h-wdl (WDL) | t12h-flat (scalar) | t12h-muon9-int8 | t12h-vw2 |
|---:|---:|---:|---:|---:|
| 201 | 0.4661 | 0.4572 | 0.4252 | 0.4069 |
| 1202 | 0.5272 | 0.5102 | 0.5160 | 0.4908 |
| 2204 | 0.5558 | 0.5327 | **0.5356** peak | **0.5097** peak |
| 3408 | **0.5652** peak | 0.5628 | 0.5183 | 0.4905 |
| 5010 | 0.5626 | **0.5693** peak | 0.5290 | **0.4686** |
| peak → final | −0.003 | **+0.000** | −0.007 | **−0.041** |
| final calibration excess | +4.8 % | +5.0 % | **+24 %** | **+51 %** |

**`t12h-flat` rises monotonically to its last checkpoint.** No hill. The WDL head
plateaus from ~step 2800 and finishes 0.007 behind it. The hill is real for **Muon**,
and worst for `t12h-vw2`, which ends *below* where it was at step 1202 and is the arm
that lost −231 Elo.

⚠️ The overconfidence framing is also weaker than stated: `v_std` does rise monotonically
in every run, but whether that counts as *excess* depends on the reference distribution
through `corr`, and on the pinned yardstick AdamW's final excess is +5 %, not the +24 %
the moving yardstick produced. What survives: **the final excess orders the recipes
flat ≈ wdl ≪ muon ≪ vw2, the same order as their Elo.**

Method note, cheaply learned: **freeze the evaluation set to a file.** Two tables built
from the same live ring hours apart are not comparable, and nothing in the numbers says
so.
