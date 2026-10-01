# 2026-08-07 (seventh entry) — Muon on the curve: +1.4× on the clock, and a weight norm that triples

The first Elo measurement of an optimiser in this project. `t7h-muon` is
[the survey's](2026-08-05-muon-survey.md) recommendation shipped — Muon on the 98.6 % of
parameters that are matrices, AdamW on the 91 904 that are not, Polar Express
coefficients, 5 Newton-Schulz iterations — run against an AdamW control at otherwise
identical settings, and rated in one joint league with a third run for scale.

⚠️ **The run did not finish.** The disk filled at 05:22, `/` at 99 %, and the training
process died at generation 967 / step 1958 — 5.65 h of a 7 h budget — inside a
`numpy.savez` on the shutdown path (`OSError: [Errno 28]`). The chain's remaining two
stages never ran and `logs/chain-t7h-muon.log` is itself truncated mid-traceback,
because the log was on the disk that was full. The league below was run by hand
afterwards. **No checkpoint after `t7h-muon-001905.pt` exists**, so the muon curve
stops at step 1905 while the control runs to 2206.

## What was compared

| | optimiser | peak lr | sims | base | self-play |
|---|---|---|---|---|---|
| `t7h-muon` | muon + aux adamw at lr × 0.05 | **2.0e-2** | 128 | `b91c8a9` + 438 uncommitted lines | 18.6 s/gen |
| `t7h-fp8` | adamw | 1.0e-3 | 128 | `31ad5e2` + fp8 encoder | 19.6 s/gen |
| `t7h-n128-collapse` | adamw | 1.0e-3 | 128 | `78fb28a` | 22.6 s/gen |

Everything else is the same configuration line: 1024 games in flight, batch 4096 in 4
micro-batches, cadence 0.815 samples/position → 10 240 records and 2.04 steps per
generation, cosine over 2600 steps, warmup 30.

The rating is **one joint fit**, `bt-mm/v1`: 62 players including the frozen random-init
anchor, 355 pairings × 36 games = 12 780 games, **64 eval sims**, 8 random legal plies,
18 distinct openings, interleaved by step so the calendar's offsets cross between runs.
Ratings from this report are comparable across the three runs and to nothing else.
Artifacts: `logs/league-joint-t7h-muon+t7h-fp8+t7h-n128-collapse.{json,log}`,
`logs/curve-joint-muon.{csv,png}`.

## The number

Self-anchored Elo, 0 = random init, ±1 s.e. Draw rate is the pairing's, in the league.

| step | `t7h-fp8` | `t7h-muon` | `t7h-n128-collapse` |
|---:|---|---|---|
| 501 | 26 ± 13 (d 0.82) | **108 ± 14** (d 0.61) | 24 ± 14 (d 0.81) |
| 1002 | 205 ± 15 (d 0.62) | **324 ± 15** (d 0.36) | 129 ± 16 (d 0.61) |
| 1505 | 359 ± 16 (d 0.49) | **488 ± 17** (d 0.41) | 323 ± 16 (d 0.48) |
| 1905 | 399 ± 18 (d 0.43) | **542 ± 19** (d 0.39) | 335 ± 18 (d 0.47) |

**+143 ± 26 Elo at step 1905**, and the gap is open from step 201 onward — it is never
inside the error bars after the first point. Fitted over the whole run, **+489 Elo per
decade of steps against +420 and +362**.

**Read horizontally, which is the axis that pays.** Time to reach a fixed rating:

| target | steps, adamw | steps, muon | ratio | seconds, adamw | seconds, muon | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 710 | 282 | 2.51× | 7 773 | 3 006 | 2.59× |
| 200 | 966 | 760 | 1.27× | 10 515 | 7 860 | 1.34× |
| 300 | 1 315 | 962 | 1.37× | 14 254 | 9 913 | 1.44× |
| 400 | 1 693 | 1 215 | 1.39× | 18 312 | 12 470 | **1.47×** |

So **~1.4× less wall clock to the same strength**, stable over the last three targets;
the 2.5× at Elo 100 is the transient, not the rate. The seconds ratio exceeds the steps
ratio only because muon's self-play generation is 5 % cheaper, which is a code
difference between the two bases and not the optimiser.

**Track E's own landmark — 25 % of games ending in checkmate, 20-generation trailing
window** — puts it higher, and the discrepancy is informative rather than contradictory:

| | positions to 25 % decisive | wall clock | final decisive rate |
|---|---:|---:|---|
| `t7h-muon` | **0.65 M** | 0.36 h | 0.731 |
| `t7h-fp8` | 2.25 M | 1.34 h | 0.766 |
| `t7h-n128-collapse` | 2.90 M | 1.95 h | 0.749 |

**3.5× fewer positions** on the landmark against **1.4×** on the Elo curve. The landmark
fires at Elo ≈ 30 for muon, deep in the transient where the ratio is 2.5×, and both runs
converge on the same final decisive rate. ⚠️ **The landmark overstates a sustained rate
gain by ~2.5×**, and E0's `n = 64 → n = 128` figure of 2.7–3.0× carries the same
optimism until it is re-read against an Elo curve. Also note the landmark has an early
false crossing: the 20-generation window sits at 0.43 around generation 20 in both runs
(short games, random policy, mates that the search finds by accident), dips, and rises
for real later. The number above is the **last** up-crossing, not the first.

The KL direction is healthy in both, which was the thing that could have gone wrong:
muon bottoms at 0.247 at step 596 and rises to 0.312; adamw bottoms at 0.135 at step 480
and rises to 0.213. The search stays ahead of the policy in both, and muon's policy is
chasing a target it is further from throughout.

⚠️ **The weight norm triples and the control's does not.** `108.6 → 369.3` over 1958
steps for muon, `108.6 → 117.2` over 2304 for adamw, and muon's curve is still bending
but not flat at the end. Muon's update has a fixed RMS per step by construction, so
`adam_wd = 0.01` on the aux group restrains nothing on the matrices. Nothing in this run
misbehaves because of it — the gradient norm sits at 0.2–1.3, saturation is nil — but
E0's standing note to *watch `weight_norm` over 24 h* was written for a +5 % drift and
this is +240 %. **A 24 h muon run should not be launched without deciding what bounds it.**

## What this does and does not license

**It licenses**: Muon is a real, free Elo/h gain at this scale, ~1.4×, in the regime the
agentic-RL paper predicted (far from saturation). By Track E's rule it is a *free*
intervention — the Newton-Schulz iteration is under 0.01 % of a step — so it needs no
cost factor to beat and the sample number and the clock number move together.

**It does not license** calling this an optimiser comparison at equal tuning effort. The
muon rate (2.0e-2) was screened the same night by the chain's 780 s KL guard; the AdamW
rate (1.0e-3) is inherited from earlier runs and was never re-screened under that
protocol. Muon's advantage is measured against *an* AdamW, not against a tuned one, and
"Fantastic Pretraining Optimizers II" is precisely the paper that says this is where the
2× claims narrow. The honest statement is **+1.4× against the baseline this project has
been running**, which is the decision-relevant one, and a rate screen on AdamW is what
would upgrade it.

Three further caveats, all in the direction of "smaller than it looks":

- The two runs sit on **different commits**, not on one commit with a flag. `t7h-muon`'s
  base includes the fp8 work that `t7h-fp8` introduced plus later search commits; the
  5 % self-play difference is the visible part of that.
- `t7h-n128-collapse` is in the league for scale, not as a comparison. Its generation
  costs 22.6 s against 18.4, so the wall-clock axis penalises it for reasons that have
  nothing to do with training quality.
- The muon curve's last point is step 1905 of a planned ~2600, and the cosine tail was
  never run. The control's own last three points — 464, 449, 429 at steps 2006/2106/2206
  — are flat inside their error bars, so there is no evidence either way about what the
  tail does. **Nothing here says what the gap is at the end of a schedule**, only that it
  is +143 at 73 % of one.

Puzzle metrics moved the same way (muon's last scan, step ~1905: pass@1 0.338, solve
0.074; the control's last: pass@1 0.281, solve 0.059) and are reported here only as a second signal.
⚠️ Layer-1 output is evaluation and may never select a checkpoint or a hyperparameter.

## What is missing, from the crash

- No muon checkpoints past step 1905; no run to the 7 h budget or the 2600-step schedule.
- The chain's stages 2 and 3 never executed — the puzzle scan and the league were done by
  hand, and `logs/chain-t7h-muon.log` ends inside the traceback that killed it.
- `/` is still at 99 % (2.8 G free) with `checkpoints/` at 3.6 G and `data/` at 7.4 G.
  **Any run launched before that is cleared dies the same way**, and it dies at the
  checkpoint write, which is the one failure that destroys the run's evidence rather than
  the run.
