# 2026-08-14 (sixteenth entry) — muon with the weight decay it needed: the crossover is gone, the Elo is ambiguous

`t12h-muong` is muon + Gumbel at `wd = 0.04`, against `t12h-gumbel` — our best — at
otherwise identical settings. It was built to test one diagnosed mechanism, and the
mechanism behaved exactly as predicted while the Elo refused to commit.

## The measurement it was built for

[2026-08-07's muon curve](2026-08-07-muon-curve.md) had muon leading by
**+143 ± 26 Elo at step 1905** and then losing by **−152 at 24 h**, with the weight
norm running 109 → 530. The diagnosis: with decoupled decay the equilibrium is
`‖W‖* = c/wd`, independent of lr, and muon's fixed-RMS update reaches it fast. In a
pre-norm LayerNorm network the function barely notices the weight scale while the
**relative step `lr/‖W‖` does**, so muon spends most of a run at a fraction of the
control's effective learning rate.

At matched step count the gap was stark — `t24h-muon` at step ~4145 sat at **529.4**
where `t12h-gumbel` at ~4185 sat at **131.1**. `wd = 0.01 × 529/131 ≈ 0.04` was the
sizing.

**It worked.**

| step | `t12h-muong` (wd 0.04) | `t24h-muon` (wd 0.01) | `t12h-gumbel` (AdamW) |
|---:|---:|---:|---:|
| 300 | 199.8 | 220.4 | 111.0 |
| 700 | 262.4 | 317.8 | 115.4 |
| 1100 | 286.3 | 384.6 | 119.5 |
| 1500 | **292.7** | 433.1 | 123.4 |
| 1900 | 289.6 | 467.8 | 126.6 |
| final | **247.8** | 498.2 | 131.1 |

Peak **292.9 at step 1558**, then a genuine turnover — the increments went
`+9.3, +4.9, +1.5, −0.7, −2.4` per 200 steps and it finished at 247.8. Old muon was
still climbing at 468 at the same point and did not peak until ~11.5 h. So `wd`
created an equilibrium where there had been none, and `‖W‖* = c/wd` now has two
points on it.

⚠️ **But 292 is not the 131 that was targeted**, so the effective step still settles
~2.2× below AdamW's rather than matching it. My sizing was undershot because I
computed the ratio from `t24h-muon`'s 529 — a number that was **still rising**, so
`c` is larger than I inferred. The two points say the right value is nearer **0.09**.
This was a *partial* fix and should be read as one.

## The Elo, which does not resolve

One joint league, 53 players, 297 pairings × 36 games = **10 692 games in 1.83 h**.
Both arms 12 h, both ending annealed.

| eval budget | `t12h-muong` | `t12h-gumbel` | difference | z |
|---|---:|---:|---:|---:|
| n = 16 | 1088.1 ± 81 | 1066.5 ± 80 | +21.6 ± 114 | +0.37 |
| n = 64 | **1389.7 ± 83** | 1234.2 ± 81 | **+155.5 ± 116** | **+2.63** |
| n = 256 | 1626.3 ± 90 | **1677.7 ± 93** | −51.4 ± 130 | −0.78 |

⚠️ **The preregistered headline is n = 256, and it is −51 ± 130 — a loss inside
noise. Muon does not clear the bar as written.** The chain script also declared, in
advance, that *"the headline is NOT the verdict if the budgets disagree; a sign flip
across evaluation budgets is itself the finding"* — and the budgets disagree, monotone
downward: **+22 → +156 → −51**.

Two readings, and one run cannot separate them:

- **A real interaction.** Muon produces a stronger *policy*, and search substitutes
  for policy quality — so the advantage is largest in the middle and is bought out by
  n = 256.
- **Noise across three correlated comparisons** on the same pair of networks. With
  ±115 intervals, one z = 2.63 of three is not a result to bank, and it is exactly
  the shape that tempts a fourth run.

I am not going to claim the first. The honest statement is: **level overall, ahead at
n = 64, and the preregistered budget says a small loss.**

## What *is* established

**The crossover did not recur.** That is the clean negative and it is the point of the
run. At 24 h with `wd = 0.01` muon lost by −152; at 12 h with `wd = 0.04` it is
level-to-ahead. The failure mode this arm was built to test **did not happen**, which
supports the mechanism even though it does not deliver Elo.

**Puzzles finished at 0.4228 against 0.3934**, and reached the control's *final* score
at roughly half the steps. ⚠️ Consistent with n = 64, inconsistent with n = 256 — and
the puzzle probe has now called **one of three** comparisons wrong in this project
(right on Gumbel-vs-PUCT, wrong on n = 64, and here it agrees with only one budget of
three). Weight accordingly.

## What is not established

- **Whether the n = 64 advantage is real.** It is one z = 2.63 among three.
- **Whether `wd ≈ 0.09` would finish the job.** The norm target was missed by 2.2×,
  and nobody has run the arm that hits it.
- **Whether any of this survives 24 h**, where the original muon result was measured.
  Both arms here are 12 h.

## Consequence

Our best model remains **`t12h-gumbel-004009`** — 1677.7 at n = 256 in this league,
and the checkpoint that scored 0.400 against AlphaGateau. Nothing about the Gate 2
position changes.

The preregistration said a loss **closes the optimiser line rather than inviting a
third attempt**, and by the pinned headline this is a loss. Honouring that is the
whole reason for pinning it in advance; the n = 64 number is precisely the kind of
result that would otherwise justify a fourth run and a fifth.

⚠️ One thing it does earn: `t12h-muong` is a legitimate **second candidate for the
Gate 2 head-to-head at n = 64**, where its advantage is largest and where the harness
makes the measurement ~75 minutes. That is a use of the existing artefact, not a new
training run.

Artefacts: `logs/league-joint-muong.json`, `logs/curve-joint-muong.{csv,png}`,
`logs/chain-muong.log`.
