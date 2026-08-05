# 2026-08-05 (sixth entry) — Muon: where the literature is, and how to ship one here

A survey, not an experiment. Nothing below was run. It exists because the optimiser is
the one Track E lever that has never been touched — `lr = 0.002` beating AZ's `0.2` is
the only optimiser result this project owns, and it is a learning rate, not a method.

⚠️ **Provenance.** Everything here comes from web search and page fetches on
2026-08-05, and several of the summaries were produced by a small model reading a PDF
rather than by me reading the primary text. Rows marked **(secondary)** should be
re-read before any of them is used to justify a design decision. The reference
implementation details and the Newton-Schulz coefficients came from primary pages and
are safe.

## Where the method is, mid-2026

Muon is SGD-momentum on 2D parameters with the update replaced by its nearest
semi-orthogonal matrix — `UV^T` from the SVD of the momentum buffer, computed not by an
SVD but by a quintic Newton-Schulz iteration `phi(x) = ax + bx^3 + cx^5` applied to the
singular values. Keller Jordan's tuned coefficients are **(3.4445, -4.7750, 2.0315)**,
**5 iterations**, stable in bf16. It is applied to hidden matmul weights only;
embeddings, the classifier head, and all scalars/vectors stay on AdamW.

It stopped being a speedrun curiosity. **Kimi K2, GLM-5 and DeepSeek-V4 were trained
with it** (secondary), and the modded-nanogpt record has gone from 45 min to **1.918
min** across 55 records, with the optimiser doing much of the work.

### The five threads worth knowing

**1. The iteration itself is a solved problem, and it is not Newton-Schulz any more.**
*The Polar Express* (arXiv 2505.16932, ICLR 2026) replaces the fixed quintic with
per-step minimax-optimal coefficients — the polynomial adapts at each iteration to the
current spectrum, from offline-precomputed tables. Strictly better at identical
wall-clock. **Polar Express coefficients are the modern default and the fixed
(3.4445, -4.7750, 2.0315) triple is legacy.**

**2. The iteration got cheaper again, in 2026.** Tri Dao's **Gram Newton-Schulz**
iterates on the small square Gram matrix `XX^T` instead of the rectangular `X`.
Mathematically identical; it turns the rectangular matmuls into one pre- and one
post-processing step and lets the rest run as symmetric GEMMs. **42 % FLOP reduction**,
up to 2x on the NS step, 25-50 % off end-to-end optimiser time. The naive version is
unstable in half precision (spurious negative eigenvalues) and the fix is a **restart
after iteration 2**, rebuilding the Gram matrix. Recommended: **fp16 not bf16**, 5
iterations, Polar Express coefficients, safety factor 1.05.

**3. How much orthogonalisation is actually needed — less than 5.** *How Much
Orthogonalization Does Muon Need?* (arXiv 2606.00371) finds partial orthogonalisation
competitive with full, with diminishing returns setting in early **(secondary)**.
*Spectral Scaling Laws of Muon* (arXiv 2606.04058) is the sharper version: it measures
the singular-value spectra of the momentum buffers across 77M-2.8B and finds the
quantiles stabilise after a burn-in at a value set by **layer type and model size**,
scaling as `M^-0.25` in shallow/mid layers and up to `M^-0.96` in late layers. The
consequence: **5 steps stays fine for shallow/mid layers at any scale**, and only late
layers at frontier scale need more. At 6.4M parameters this project is far inside the
easy regime, and **fewer than 5 iterations is a live option here specifically.**

**4. The best-performing variant is NorMuon, and it is what the speedrun runs.**
Muon flattens the *matrix* condition number but leaves **neuron norms highly
non-uniform**, so a few rows dominate. NorMuon (arXiv 2510.05491) keeps a second-moment
statistic **per neuron** (per output row) and renormalises so the update matrix keeps
its magnitude. Reported **+11.31 % over Muon** and **+21.74 % over Adam** at 1.1B,
optimiser state `m(n+1)` against AdamW's `2mn`, **~2.9 % step overhead** (secondary).
The modded-nanogpt 2.476 -> 1.918 min phase bundles NorMuon with batch scheduling and
multi-token prediction.

**5. Everything else is distributed-systems or stability work, and does not apply
here.** **Dion** — low-rank approximation to cut communication when weights are sharded.
**MuonClip** (Kimi K2) — clips QK logits to stop attention-logit blowup at scale.
**Scion/Gluon** — extend the LMO rule to every layer with explicit per-layer norm
control. **LiMuon, OLion, LoRA-Muon, MuonQ** — memory, sign-composition, low-rank
manifold, quantised states. Single GPU at 6.4M parameters, none of it is load-bearing.

### The two results that argue *against*, and they are the ones to take seriously

⚠️ *Fantastic Pretraining Optimizers and Where to Find Them II* benchmarks the
Muon family against AdamW **with both properly tuned** and finds the gains "meaningful
but modest" — the 2x claims narrow considerably under a fair protocol **(secondary,
and this is the row I would most want to re-read)**. The one detail that cuts our way:
the improvements were reported as most pronounced on *smaller* models.

⚠️ *When Does Muon Help Agentic Reinforcement Learning?* (arXiv 2607.16169) is the
closest thing to our setting — Qwen2.5 0.5B-3B on ALFWorld, sparse reward, GRPO family.
Findings: Muon helps **while optimisation headroom remains** and is neutral-to-behind
**near task saturation**, where a tuned AdamW recovers most of the gap. It needs KL and
clipping regularisation to stay stable, and **learning rates must be screened
empirically — conversion heuristics from AdamW do not work**. "Fan-in Muon" there means
scaling the update by `sqrt(max(1, d_out/d_in))`, which applied **3.53x AdamW's
hidden-matrix update magnitude**.

Self-play is nowhere near saturation, which is the regime the paper says Muon wins in.

## Why this is a good fit for brokefish specifically

**98.6 % of the parameters are Muon-shaped.** Measured from `BrokefishNet`:

| group | params | share |
|---|---|---|
| hidden matmuls: `in_proj` (768,256), `out_proj` (256,256), `linear1` (1024,256), `linear2` (256,1024), x8 | 6,291,456 | **98.6 %** |
| embeddings (`emb_square/type_special/color_turn/clock/rep`) | 47,104 | 0.74 % |
| heads (`policy` 64x256, `promo` 4x256, `value` 1x256) | 17,664 | 0.28 % |
| all biases and LayerNorm gains | 27,136 | 0.43 % |

Compare a GPT-2-small, where the embedding and the tied head are a third of the model
and Muon touches the smaller part. Here the AdamW residue is 91,904 parameters.

**The Newton-Schulz cost is free, and not marginally.** NS is roughly
`15 * min_dim * numel` ~ **24 GFLOP per step** at 5 iterations. At ~18 TFLOPS that is
about **1.3 ms**, against a measured **2.2 s gradient phase** and **18.8 s of
self-play** per step. **Under 0.01 % of the step.** Every argument in the literature
about NS overhead, Gram-NS, Dion's communication cost and fewer iterations is
irrelevant at this scale — we can afford full Polar Express at 5 steps and never look
at the clock.

**And the payoff lands in Track E's currency.** Better sample efficiency = fewer
positions to the same landmark = less self-play, which is 90 % of the wall clock. This
is the same lever `n=64 -> n=128` pulled for 1.6x, in the same units, and it is
independent of it.

**It does not touch the tabula rasa boundary.** An optimiser carries no opinion about
chess.

### One interaction nobody in the literature will flag

Muon's updates are orthogonal, which pushes weight matrices toward **flatter singular
value spectra**. The fp8 encoder quantises weights per 128-column block by amax. A
flatter spectrum plausibly means **fewer outliers and smaller amax per block, i.e.
lower fp8 error at the same `q_max`** — and it could equally go the other way by
raising the typical magnitude relative to the max. Either way it is cheap to measure:
`brokefish/nn/validate.py` on a Muon checkpoint against an AdamW one at matched step.
This is worth doing *because* it is free, not because I expect a particular sign.

## How to ship it

### The seam already exists

`build_optimizer` in `brokefish/train/loop.py:261` dispatches on `cfg.optimizer`
(`"sgd"` / `"adamw"`) and already splits `ndim >= 2` from the rest for weight decay.
Adding `"muon"` is a third branch, not a refactor. `docs/reference/training.md` is
normative and would need the section; `--optimizer muon` is already plumbed through
`chain*.sh`'s argument surface.

### `brokefish/train/muon.py`, self-contained, no new dependency

    Polar Express coefficients, 5 steps, tabulated as a constant
    Gram form on the smaller side, restart after iteration 2, fp16
    NorMuon per-neuron second moment + magnitude renormalisation
    update scale sqrt(max(1, d_out/d_in))          <- the "fan-in" convention
    momentum 0.95, nesterov

NorMuon is in because it is the current best and costs `m` extra floats per matrix
(6,144 floats total here — nothing), not because the +11 % will reproduce at this scale.

### ⚠️ Three decisions that are ours, not the literature's

**1. `in_proj_weight` is fused (768, 256) and must be split into three (256, 256)
blocks before orthogonalisation.** Orthogonalising the stacked QKV matrix is not the
same operation as orthogonalising Q, K and V separately, and `nn.MultiheadAttention`
gives us the fused form by default. modded-nanogpt keeps them separate. This is a
silent-wrongness bug of exactly the shape this repository keeps finding: it runs, it
converges, and it is a different algorithm. **It needs a test that bites.**

**2. `grad_clip = 1.0` means something different under Muon.** Today `grad_norm`
settles at ~0.33 and the clip never fires. Muon sets the update magnitude by the
spectral normalisation, so clipping the *gradient* barely constrains the *step*. The
agentic-RL paper found Muon needed its regularisation to stay stable. Keep the clip,
but stop treating "clip never fires" as evidence of stability, and watch the
update-to-weight norm ratio instead.

**3. The learning rate does not transfer and must be swept.** `lr = 0.002` was found
for AdamW. Muon's reference examples sit near `0.02` for the Muon group with `3e-4` for
the AdamW group, and the RL paper's explicit finding is that conversion heuristics
fail. `train/overfit.py --sweep` is the cheap detector and is the right first step.

### The order, and the instrument

1. `overfit.py --optimizer muon --sweep` — hours, on one batch. Finds the LR band and
   catches the QKV-split bug if the test does not.
2. A short paired run at matched wall clock, Muon against the AdamW control.
3. ⚠️ **Rate them in ONE joint league.** `eval/league.py` already takes repeated
   `--run` and fits all runs on a single Bradley-Terry scale — it was built for exactly
   this after the fp8-vs-control comparison was disavowed for comparing two
   independently-fitted scales as if 20 matched steps were 20 independent samples.
   Two separately-fitted Elo curves are **not** comparable and that mistake is already
   in this journal once.
4. `validate.py` on both checkpoints for the fp8 interaction.

### What would make me drop it

If the LR sweep shows no band where Muon beats tuned AdamW on the same batch, stop —
that is *Fantastic Optimizers II*'s finding reproducing, and it is a real result worth
one journal entry and no more. The measurement that matters is **Elo per hour on the
joint league**, not training loss, and Track E's first lesson applies unchanged: an
intervention has to beat its own cost factor, and Muon's cost factor here is ~1.0000.

## Sources

Primary (read directly): Keller Jordan's Muon post and repository; Tri Dao's Gram
Newton-Schulz post; Varun Neal's modded-nanogpt Muon essay.
Fetched and summarised by a small model **(secondary)**: Spectral Scaling Laws of Muon
(2606.04058), Fantastic Pretraining Optimizers II (2606.16899), When Does Muon Help
Agentic RL (2607.16169), How Much Orthogonalization Does Muon Need (2606.00371).
Search-result summaries only, not fetched: NorMuon (2510.05491), Polar Express
(2505.16932), Moonlight/Muon is Scalable (2502.16982), Optimal Scaling Needs Optimal
Norm (2510.03871), Dion, Scion/Gluon, MuonClip.

---

## Addendum, same day — the granularity question is a first-class published result, and I under-searched it

⚠️ **Théo supplied both of these; the survey above missed them.** Recorded here rather
than folded into the text above, because the survey's "decision 1" presented the QKV
split as *our* implementation detail with a shrug, and it is in fact the most active
sub-thread in the Muon literature right now. That was a search failure, not a judgement
call: I queried variants and scaling and never queried granularity.

### CMuon (arXiv 2608.02502, submitted 2026-08-03 — two days ago)

*Chunked Momentum Orthogonalization*. It names the mechanism the survey only gestured
at: fusing functionally distinct blocks into one tensor and orthogonalising the whole
thing applies **a shared preconditioner `(sum_j G_j^T G_j)^{-1/2}` across all blocks**.
When the dominant principal directions of `G_1` and `G_2` are misaligned, the summed
covariance distorts each block's descent geometry — the paper's term is **subspace
interference**, and it is why vanilla Muon *plateaus late* on DiTs.

Chunking is along the longer dimension: QKV [3456,1152] -> 3 x [1152,1152], FFN gate+up
[6144,1152] -> 2 x [3072,1152], AdaLN [6912,1152] -> 6 x [1152,1152]. Orthogonalise
each chunk, concatenate, scale per chunk with an optional `sqrt(N_chunk)` rescale.

DiT-XL, ImageNet 256, FID: AdamW 1.30 @200ep, Muon 1.29 @200ep, **CMuon 1.18 @200ep and
1.46 @80ep**. ⚠️ Note what that table actually says — **vanilla Muon barely beat AdamW
here (1.29 vs 1.30)**; the entire gain came from the chunking. If that generalises, the
granularity decision is not a refinement of the Muon decision, it *is* the Muon decision.

### Per-head Muon — Kimi K3, GLM-5, and the speedrun

**Kimi K3** (arXiv 2607.24653) partitions the attention momentum **along the head
dimension** and orthogonalises each head's block separately, rather than the full Q/K/V
matrices. Stated motivation: full-matrix orthogonalisation couples all heads through one
polar factor, so **heads with larger momentum scale dominate the shared update direction
and small-scale heads get under-normalised updates**. Reported: more balanced learning
dynamics across heads, better stability at scale, and *cheaper* — NS on tall per-head
blocks costs less than on the full matrix. **GLM-5** ships the same thing as "Muon
Split". A **modded-nanogpt record** was improved by orthogonalising Q and K in **pairs
of heads**.

### And the theory paper says both extremes are wrong

*When and Why Grouping Attention Heads Accelerates Muon Optimization* (arXiv 2605.08933)
makes it a gain-versus-cost inequality. Splitting raises the first-order descent term
from `||G||_*` to `sum_i ||G_i||_*`, and pays a second-order norm penalty in the ranks.
Grouping helps when

    sum_i ||G_i||_*  -  ||G||_*   >   (beta * eta / 2) * ( sum_i r_i  -  r )

⚠️ **The trade-off is stage-dependent and it reverses.** Early, gradients are near
full-rank, the rank cost is negligible and **per-head (g=1) wins**. Late, the whitening
gain decays against accumulated norm cost and **coarser grouping wins**. On GPT-2 Small
/ FineWeb, 12 heads: g=1 best initially then degrades, **g=6 with random grouping is
best overall**, and full-QKV is worse than intermediate grouping throughout. Random
grouping beat adjacent and interval patterns. Their recommendation is explicit: **treat
group size as a hyperparameter, do not assume one.**

### What this changes for the shipping plan

The granularity ladder for `BrokefishNet`, which has 8 heads at `DH = 32`:

| level | `in_proj_weight` (768,256) | `out_proj.weight` (256,256) |
|---|---|---|
| today's naive | 1 x (768,256) | 1 x (256,256) |
| CMuon chunk | 3 x (256,256) | 1 x (256,256) |
| head group g | 3 * (8/g) x (32g, 256) | (8/g) x (256, 32g) |
| per-head | 24 x (32,256) | 8 x (256,32) |

⚠️ **`out_proj` chunks on the other axis.** Its head structure lives on the *input*
dimension — columns `32h..32h+32` are head h's contribution — so "partition along the
head dimension" means rows for `in_proj` and columns for `out_proj`. Getting this
backwards produces a running, converging, wrong optimiser.

**My own reasoning from their inequality, flagged as mine and untested.** The rank cost
term behaves very differently at our two split levels. Splitting the fused QKV into
three (256,256) blocks takes `r <= 256` to `sum_i r_i <= 768` — a real cost. Splitting
one projection into 8 head-blocks of (32,256) takes `r <= 256` to `sum_i r_i <= 8*32 =
256` — **the ranks add up exactly and the cost term is ~zero, so per-head splitting
inside a projection may be nearly free here in a way it is not for the QKV split.** If
that holds it predicts per-head wins at our shape. It is a two-line derivation from
someone else's inequality and should be treated as a hypothesis to test, not a finding.

**And the stage-dependent reversal may not even apply to us.** Their "late training" is
a fixed-horizon pretraining run against a stationary corpus. Self-play has a moving data
distribution and no saturation point in sight, which is arguably permanently "early".
That is a guess. It is also cheap to settle, because grouping `g` is a single integer.

**Revised plan:** `g` joins the learning rate in the first `overfit.py --sweep`. The
ladder is 4 values — `g = 1, 2, 4, 8` on the head axis, plus the QKV split on/off — and
each is one integer in the parameter-grouping function, so the sweep costs sweep time
and no implementation risk beyond the axis bug above. Given CMuon's table, **if only one
knob gets swept it should be `g`, not the learning rate.**
