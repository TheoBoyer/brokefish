# The value head never lost the information — it lost its calibration

*2026-08-24. Théo asked for a mechanistic account of the three value-head schemes and
of why Adam and Muon behave differently. This is that investigation: eleven runs,
one pinned record set, five hypotheses killed and one model that survives.*

## The instrument

40 000 full records (policy labels included) drawn uniformly from `t12h-prenorm`'s live
window and frozen to a file, so every checkpoint of every run is scored on **the same
positions**. On that set, per checkpoint:

* **ceiling** — the correlation an OLS head would get from the head's *own* input.
  The upper bound on any linear readout, and therefore a direct measure of how much
  outcome information the trunk carries at the readout.
* **gap** = ceiling − what the run's own head actually gets.
* the head-input geometry: `rho` (constant fraction), participation ratio, spectral tail.
* **gradient noise scale** `B_simple = tr(Σ)/|G|²` (McCandlish), from 32 disjoint chunks
  of 512, for the value head's weight and for the policy head's.
* the routing of `∂L/∂h` at the last-layer residual, split into its value and policy
  terms.

`scratchpad/mech.py`, `trainhead.py`, `transport.py`, `decomp.py`, `material.py`,
`scale2.py`, `window.py`. ⚠️ All fp32 eager, no fused kernel — gradients are the point.

## The one number that orders every run

**The ceiling does not move.** At the final checkpoint it is 0.596–0.607 for every run
measured, whatever the head, the optimiser or the weight decay — and it is flat across
training. The trunk learns value perfectly well in every configuration ever tried here.

What varies is the **gap**, and it ranks the runs by every outcome we have:

| run | head | opt | ceiling | corr | **gap** | Elo n=256 | value_pass@1 drop from peak |
|---|---|---|---:|---:|---:|---:|---:|
| `t24h-adamw-int8` | king, MSE | adamw | 0.607 | 0.603 | **0.003** | — | — |
| `t12h-wdl` | king, CE | adamw | 0.599 | 0.593 | **0.006** | **+1681** | **−0.007** |
| `t12h-flat` | king, MSE | adamw | 0.599 | 0.588 | 0.011 | — | −0.031 |
| `t12h-nsched` | king, MSE | adamw | 0.602 | 0.585 | 0.017 | — | −0.036 |
| `t12h-muong` | king, MSE | muon wd.04 | 0.596 | 0.572 | 0.024 | — | — |
| `t12h-muon9` | king, MSE | muon wd.09 | 0.596 | 0.565 | 0.031 | — | — |
| `t24h-muon` | king, MSE | muon wd.01 | 0.596 | 0.559 | 0.037 | — | — |
| `t12h-prenorm` | prenorm, CE | adamw | 0.596 | 0.556 | **0.041** | **+1391** | −0.087 |
| `t12h-wdb` | pooled, CE | adamw | 0.591 | 0.540 | **0.051** | **+1350** | −0.114 |
| `t12h-muon-wdl` | king, CE | muon wd.09 | 0.577 | 0.518 | 0.059 | — | −0.069 |
| `t12h-vw2` | king, MSE ×2 | adamw | 0.588 | 0.510 | **0.078** | −231 vs ctrl | — |

Within the one joint Gumbel fit that spans three of them, the gap and the Elo rank
identically: 0.006 / 0.041 / 0.051 → +1681 / +1391 / +1350. Three points, so this is a
consistent ordering and not a fitted relationship.

⚠️ `t12h-prenorm`'s **last** checkpoint is contaminated — the pinned set is drawn from
its own final window — and its 5010 row (corr 0.6716 against a 0.6467 ceiling) is out of
family in every table here. Its 4209 row is used instead. That contamination also means
the pinned set *favours* `t12h-prenorm`, so its decline is an underestimate.

## The gap is entirely in the head, and a thousand steps recovers it

Freeze the trunk, precompute the head's input on the pinned set, and train **only** the
value head with the run's own optimiser (AdamW 1e-3, wd 0.01, B = 4096), 32k/8k split:

| checkpoint | run's own head | refit from **random init** | steps to converge |
|---|---:|---:|---:|
| `t12h-wdl` 5010 (king) | +0.5870 | **+0.5906** | < 1000 |
| `t12h-wdb` 5010 (pooled) | **+0.5356** | **+0.5888** | < 1000 |
| `t12h-prenorm` 2605 | +0.5736 | +0.5847 | < 1000 |
| `t12h-wdb` 2605 (pooled) | +0.5758 | +0.5861 | < 1000 |

The pooled head's 0.053 shortfall is **fully recoverable by retraining the head alone**,
from scratch, in under a thousand steps, on the trunk the run itself produced. The
pooled feature is not harder to learn from; `t12h-wdb`'s trunk holds exactly as much
value as `t12h-wdl`'s (0.5888 against 0.5906).

⚠️ **The control that makes this readable is `t12h-wdl`.** Both heads were fit on a
different window than the one they are scored on, so both pay a distribution-shift
penalty. `t12h-wdl` pays **0.005**. `t12h-wdb` pays **0.053**. The shift is the same;
the sensitivity to it is not. (If anything the shift is *smaller* for `t12h-wdb`, since
`t12h-prenorm` is its one-flag descendant.)

Splitting the shortfall into scale and direction, by fitting the single best temperature
on the run's own weight: for `t12h-wdb` at 5010, α\* = 0.36 buys 0.018 and the remaining
**0.036 is direction error**. For `t12h-wdl`, direction costs **0.003**.

## What the head is doing instead: getting more confident

Cross-entropy on a single game outcome has no finite optimum, and the Bayes-optimal
answer in a middlegame is close to "I don't know". Nothing in the loss enforces that;
weight decay brakes the head's *norm*, not its *calibration*. Measured on each run's
**own** training window, `value_saturated_frac` (|v| > 0.99):

| `t12h-flat` (MSE) | `t12h-wdl` (CE) | `t12h-muon-wdl` | `t12h-prenorm` | `t12h-wdb` |
|---:|---:|---:|---:|---:|
| **3.7 %** | 6.8 % | 9.0 % | 9.4 % | **9.6 %** |

and on the pinned set, the fraction of predictions above 90 % confidence rises
0.10 → 0.257 (`t12h-wdl`) against 0.10 → **0.331** (`t12h-wdb`), 0.332 (`t12h-prenorm`),
0.308 (`t12h-muon-wdl`).

**The damage is band-localised and it lands exactly where the labels are noisiest.**
Splitting the pinned set by alive-piece count:

| `t12h-wdb`, corr(v, z) | endgame | **middlegame** | opening |
|---|---:|---:|---:|
| step 2805 (its peak) | 0.8533 | **0.5519** | 0.1533 |
| step 5010 (final) | 0.8446 | **0.4788** | 0.1267 |

The endgame holds. The middlegame collapses — and the middlegame is where the mean
|logit| grows most: 1.11 (`t12h-wdl`) against 1.47 (`t12h-wdb`), 1.39 (`t12h-prenorm`),
1.44 (`t12h-muon-wdl`), while the endgame logits are within 0.4 of each other.

The training loss agrees. Within the three-class family, on **matched windows** (game
length 116–119 plies, decisive fraction 0.908–0.929, `t12h-wdb`'s slightly *longer* and
*less* decisive, i.e. slightly harder):

| | train CE | held-out value_pass@1 | Elo |
|---|---:|---:|---:|
| `t12h-wdl` | **0.668** | **0.3855** | **+1681** |
| `t12h-prenorm` | 0.583 | 0.2833 | +1391 |
| `t12h-wdb` | **0.574** | **0.2634** | +1350 |

Lower training loss, worse everything else, on a window that is not easier.

## Adam and Muon were never compared at matched regularisation

`--adam-wd` feeds **both** `wd` for the Muon group and `aux_wd` for the embeddings and
all three heads (`loop.py:467`), and Muon runs at lr **0.02** against AdamW's 0.001.
`Muon._muon_group` decays with the *unadjusted* lr, so the per-step multiplicative
shrink on the trunk is `lr·wd`:

| runs | opt | lr | wd | shrink/step | vs AdamW | PR at readout |
|---|---|---:|---:|---:|---:|---:|
| every AdamW run | adamw | 1e-3 | 0.01 | 1.0e-5 | 1× | 13–15 |
| `t24h-muon` | muon | 0.02 | 0.01 | 2.0e-4 | **20×** | 14.3 |
| `t12h-muong` | muon | 0.02 | 0.04 | 8.0e-4 | **80×** | 7.4 |
| `t12h-muon9`, `t12h-muon-wdl` | muon | 0.02 | 0.09 | 1.8e-3 | **180×** | 7.3 / 4.5 |

The representation's effective rank at the readout collapses with the dose — and
⚠️ **it costs the ceiling nothing** (0.596 at PR 7.3, 0.596 at PR 15.0). A rank-4.5
representation carries the same decodable value as a rank-15 one. The Muon runs' value
damage is in the **gap** (0.024–0.059 against AdamW's 0.003–0.017), i.e. the same
calibration failure, not a representational one.

So: **there is no measured Adam-versus-Muon difference in the value head.** There is a
weight-decay difference of up to 180×, which `muon.py`'s own docstring flagged on
2026-08-17 and which no comparison has yet controlled for. `t24h-muon`, the single Muon
run at wd 0.01, has an AdamW-shaped participation ratio (14.3).

## Five hypotheses killed

Each of these was proposed, measured, and is wrong. Recording them because the next
person will propose them again.

1. **The trunk loses value information.** No: the ceiling is flat at 0.596–0.607 in
   every run, every checkpoint.
2. **Pooling is a harder linear problem (conditioning).** No: a random-init head reaches
   0.5888 on the pooled feature in under 1000 steps.
3. **The value and policy gradients fight over the shared tokens.** No:
   cos(∂L_val/∂h, ∂L_pol/∂h) = 0.000 ± 0.001 at every checkpoint of every run. This was
   the motivating hypothesis for the pooled head and it is not a real effect.
4. **The pooled head leans on material, which pays in self-play and misleads on
   puzzles.** Backwards: corr(v, material) *falls* for the pooled heads (0.876 → 0.781)
   and holds for the king head (0.886 → 0.872).
5. **The value map transports worse from the pooled feature.** No: the cross-band and
   cross-generation transport penalties are identical (phase 0.069 both; time 0.020
   against 0.025).

And one premise of mine that was simply false: I asserted that the king head's input has
fixed norm because it is a LayerNorm output. `norm_f` has learned affine, so ||x|| runs
15.74 → 14.30 over training. The endgame/opening norm ratio is 1.00–1.06 for every head,
far too small to matter.

⚠️ The 2026-08-21 entry's hypothesis — "overfitting the replay window" — survives all of
this, sharpened: it is not memorisation of records (reuse is 0.815) but **over-confidence
on the current window's distribution, concentrated in the positions whose outcome is
genuinely uncertain.**

## The model

1. The trunk learns value fine, always. **Value information has never been the
   bottleneck in this project**, on any head, optimiser or schedule.
2. The head is what fails, and it fails by drifting toward over-confidence: tighter fit
   to the current window, more saturation, a weight direction that rotates away from the
   held-out optimum, skill lost in the middlegame.
3. The drift rate scales with **how hard the value loss is pushed relative to what
   constrains the channel it reads**. `t12h-vw2` (value weight doubled) has the worst gap
   of any run at 0.078 and lost 231 Elo; `t12h-flat` (bounded tanh-MSE, king token) has
   the lowest saturation at 3.7 %; the pooled heads read a channel no other loss touches
   and sit between.
4. Nothing here is irreversible. Every gap is recoverable by a thousand-step head-only
   refit on the trunk the run already produced.

⚠️ Point 3 is the weakest link: the *ordering* is measured across eleven runs, the
*mechanism* — that the king token is anchored by also driving that token's 64 policy
logits, while the pooled direction is unconstrained — is inference. The instantaneous
gradient cosine says they do not fight at a point in time; it does not say the policy
loss fails to constrain the token over training. I conflated those two claims earlier
and they are different.

⚠️ **No line is closed and no experiment is proposed as settled here.** Nothing in this
entry selects a checkpoint, and nothing in it has been fed back into training.

---

## Correction, same day: the title is wrong, and Théo found it

Théo objected that `value_pass@1` is an argmax over sibling positions, so a monotone
confidence squash cannot move it — and it declines throughout training. He is right, and
the emphasis above is wrong. Three measurements settle it.

**1. The rank ceiling is flat too, so the headline survives.** Everything above measured
the ceiling in Pearson. Re-measured with a CE-refit head scored by Spearman, on frozen
features, 24576/8192 split:

| CE-refit Spearman | step 201 | step 2605 | step 5010 |
|---|---:|---:|---:|
| `t12h-wdl` | 0.5414 | 0.5558 | 0.5549 |
| `t12h-wdb` | 0.5341 | 0.5558 | 0.5565 |
| `t12h-flat` | 0.5360 | 0.5573 | 0.5574 |

Flat, and identical across heads. The trunk's *ranking* information is as unaffected as
its linear information.

**2. But the gap is a rank gap, and confidence is not what opens it.** Refit Spearman
minus the run's own head:

| rank gap | 201 | 1803 | 2605 | 3408 | 4209 | 5010 |
|---|---:|---:|---:|---:|---:|---:|
| `t12h-wdl` | +0.076 | +0.010 | +0.003 | −0.001 | −0.002 | **−0.005** |
| `t12h-flat` | +0.076 | +0.023 | +0.019 | +0.006 | +0.006 | **+0.004** |
| `t12h-wdb` | +0.103 | +0.023 | +0.008 | +0.017 | +0.025 | **+0.043** |
| `t12h-prenorm` | +0.094 | +0.027 | +0.010 | +0.010 | +0.034 | (contaminated) |

Same shape as the Pearson gap: closes by step ~2600, then re-opens for the pooled heads
only. The best single temperature recovers **0.018 of `t12h-wdb`'s 0.053 Pearson gap and
about a sixth of its rank gap**. ⚠️ **Over-confidence is a co-symptom, not the mechanism.
The substance is the direction moving to a worse place, in rank as well as in Pearson.**

Two candidate rank mechanisms were tested and both are dead. The **draw row** — `v` is
not monotone in `w_win − w_loss`, since `v = sinh(s/2)/(cosh(s/2) + e^m)` and `m` varies
per position — costs `spearman(v,z) − spearman(s,z)` = **−0.005** at step 5010 against
`t12h-wdl`'s +0.002. It matters only before step ~2200 (−0.05 at step 201). And the loss
is already present in the linear axis alone: `spearman(s,z)` for `t12h-wdb` peaks 0.5723
at step 2204 and falls to **0.5446**, while `t12h-wdl`'s rises monotonically to 0.5777.

**3. It is not lag either.** Measuring the head's direction against the optimal direction
of every checkpoint, in the predictive metric `rho(u,v) = corr(x·u, x·v)` on the same
features (⚠️ raw Euclidean cosine is meaningless here — `beta = Sigma^-1 c` is whitened,
giving cos ≈ 0.10–0.18 for a head at corr 0.59, and an argmax over it is pure noise; a
first attempt at this measurement was wrong for exactly that reason):

* `rot`, the predictive agreement between consecutive optimal directions, reaches
  **0.98 per 200 steps** by step 2000 in both runs and stays. The target is essentially
  stationary for the last three quarters of the run.
* the head's best-matching target is **the current one — lag 0 steps** — at every
  checkpoint after step 1000, in both runs.
* the head sits at a stable `rho(d, beta_t)` of **0.93**, in the good run and the bad
  one alike, and never approaches 1.

So the head is not behind a moving target. It is converged, onto a point 0.93-aligned
with the optimum, and for the pooled heads that point gets worse after step ~2600 while
the optimum does not move.

⚠️ **This makes a bigger learning rate on the heads the wrong instrument** — there is
nothing to catch up to, and more effective steps is what the decline is made of.
⚠️ And the shape of the gap column is a trap: every failing run peaks near step 2600, and
"take the step-2600 checkpoint" is precisely the one-bit distillation channel
`CLAUDE.md` names. Nothing here licenses that.

⚠️ What remains unexplained is why a CE refit on the pinned distribution lands at 0.5565
while the run's own head, trained on its own window with the same loss and optimiser,
lands at 0.5138. The distribution is the only difference left, and `t12h-wdl`'s head
pays only 0.005 for the same shift. I do not have a mechanism for that asymmetry.

---

## Second correction: what `--terminal-collapse` puts in the buffer, and what it does not

I claimed in chat that "the value head has never once seen a checkmate", which was
sloppy and Théo challenged it against `--terminal-collapse`'s whole purpose. Checked
empirically on `t12h-prenorm`'s ring, 200 000 sampled records:

| | in the buffer? |
|---|---|
| the position a mate is delivered **from** | **yes** — 37 of the last 40 finished games' final record is a position with mate available carrying `z = +1`, and under §6.6a the policy target is a point mass on the mating edge |
| the **checkmated** position itself | **no** — 0 checkmate, 0 stalemate, 0 with no legal move, `policy_len >= 1` everywhere |

Both are true and they are about different positions. `ReplayBuffer.append` stores the
position a move was played *from*, and no move is played from a mated position. So
`--terminal-collapse` does exactly what it was built for; what is absent is the position
*after* the mating move.

⚠️ **And that is harmless in play**, which is the part I should have checked before
raising it: `search.md` §6.3 — *"Games whose descent created no node, or created a
terminal one, have nothing to evaluate. v0 evaluates them anyway and discards the
result."* Terminal positions never contribute their network value in search, collapse or
not. Training and inference agree.

The only thing that asks is **`value_pick`**, which deliberately does not pin terminal
children (documented there: the pin was worth +0.079–0.108 pass@1 and it *compressed the
differences between checkpoints*, which is the one thing a diagnostic must not do). The
consequence is that for the 13.7 % of puzzles whose solution is mate, the probe asks the
head about a position class neither training nor play supplies. Measured: no checkpoint
of any run values a checkmated position below **−0.24**, and the mate bin's mean `v_sol`
at step 5010 is **+0.140 / +0.177 / +0.042** for `t12h-wdl` / `t12h-wdb` / `t12h-muon-wdl`.

⚠️ It does **not** contaminate the finding. Dropping the mate bin makes `t12h-wdb`'s
decline slightly *larger* (0.235 → 0.153 non-mate against 0.232 → 0.144 overall), and the
whole loss localises to **captures**: `t12h-wdb` goes **0.585 → 0.381** on
capture-solutions while its quiet-solution accuracy is flat (0.112 → 0.111), where
`t12h-wdl` holds captures at 0.584 → 0.567 and gains on every kind.

The per-child numbers say the same thing in the other direction: for capture solutions,
`t12h-wdb`'s `v_sol` is **flat** from step 2004 (−0.143 → −0.150) while its `v_pick`
**doubles** (−0.204 → −0.402), so the margin grows 7× over the run. In `t12h-wdl` the two
descend together and the margin is flat after step 2004 (0.063 → 0.077). **The right move
does not stop looking good; a wrong one gets an increasingly extreme evaluation and
overtakes it.**

---

## Third correction: there IS an Adam-vs-Muon difference, and `t12h-vw2` is a Muon run

Two errors above, both mine, found while looking at Muon specifically.

**1. `t12h-vw2` is `optimizer=muon`, `adam_wd=0.09`** — not AdamW as the gap table says. It
is `t12h-muon9` with `--value-weight 2` and nothing else, which makes it a *better*
datapoint than I claimed: a clean one-flag ablation on a Muon run, gap **0.031 → 0.078**.

**2. "There is no measured Adam-versus-Muon difference in the value head" is wrong.** It
was drawn from the `gap` statistic, which the decay dose does move and which is noisier
than I treated it. Splitting the value probe by the kind of move the solution is
separates the optimisers cleanly, and the failure lives entirely in **captures**:

| final checkpoint | opt | adam_wd | head `lr*wd` | capture pass@1 | peak |
|---|---|---:|---:|---:|---:|
| `t12h-int8` | adamw | 0.01 | 1e-5 | **0.646** | rising all run |
| `t12h-flat` | adamw | 0.01 | 1e-5 | 0.521 | 0.636 |
| `t12h-muong` | muon | 0.04 | 4e-5 | **0.454** | 0.454 |
| `t12h-muon9` | muon | 0.09 | 9e-5 | **0.433** | 0.509 |
| `t24h-muon` | muon | **0.01** | **1e-5** | **0.418** | 0.418 (8215 steps, 24 h) |
| `t12h-muon-wdl` | muon | 0.09 | 9e-5 | 0.399 | 0.511 |

⚠️ **The weight-decay dose-response is flat** — 0.433 / 0.454 / 0.418 at wd 0.09 / 0.04 /
0.01 — and `t24h-muon`, whose heads carry *exactly* the AdamW runs' decay (`aux_lr * wd` =
1e-5), is the worst of the three at twice the training. Overall non-mate pass@1:
`t24h-muon` reaches **0.167 at 8215 steps** where `t12h-int8` reaches **0.261 at 4409**.

This is the same regression `muon.py`'s docstring already recorded on 2026-08-17 —
`t12h-muon9-int8` beating `t12h-gumbel` by +0.061 on the *policy* probe and losing
62-45-93 at n = 128, with value sign accuracy 0.615 against 0.783 — now localised to a
single move kind.

**3. And the knob that exists for this has never been used.** `aux_wd` is `None` in
**every run in `runs/`**, all 37 with a config. `muon_param_groups` carries it precisely so
the five embedding tables and the three readout heads can be decoupled from the Muon
group's decay, and the docstring's own warning — *"running one `wd` across both meant
`--adam-wd 0.09` shrank the value head at the same rate as a 256x1024 trunk matrix"* — has
never been acted on. The capture numbers above suggest it would not be sufficient on its
own, since `t24h-muon` already has the matched head decay by accident of `wd = 0.01`.

⚠️ **The structural bind, which is the part worth knowing.** `Muon._muon_group` decays with
the *unadjusted* `lr` (torch's ordering, deliberately reproduced), so the trunk's per-step
shrink is `lr * wd`. Muon needs `lr ~ 0.02` and AdamW runs at 1e-3, so **matching AdamW's
1e-5 trunk shrink under Muon requires `wd = 0.0005`** — and the 2026-08-14 entry measured
that `||W||* = c/wd`, with `wd = 0.04` peaking at 292.9 and `~0.09` being the value that
actually equilibrates. So Muon's 20-180x trunk decay is not a misconfiguration; it is what
the optimiser needs to have an equilibrium at all. There is no setting that matches both
the equilibrium and the decay.
