# 2026-08-13 (fifteenth entry) — how the field does fp8 attention, and fp8 in RL rollouts

Two literature questions, asked because
[this morning's measurement](2026-08-13-massive-activations.md) turned fp8
*activations* from a suspected obstacle into a measured one, and because our loop is
structurally an RL rollout generator running at a different precision from its
trainer. Read the papers rather than the summaries where I could; where I could not,
it is marked.

The short version: **one FA3 technique is the missing piece for our fp8 problem and
costs zero parameters**, the FA3 *kernel* cannot port to this card, FA4 is about a
bottleneck we do not have, and **almost all of the fp8-RL literature's machinery is
inapplicable to AlphaZero-style training for a structural reason worth writing down.**

## FlashAttention-3's fp8 path

[arXiv:2407.08608](https://arxiv.org/abs/2407.08608), Shah et al., NeurIPS 2024.
Read via ar5iv.

**Block quantization.** Not per-tensor: *"we keep one scalar per block, so that for
each of Q, K, V we split the tensor into blocks of size `Br×d` or `Bc×d` and quantize
them separately."*

**Incoherent processing.** Multiply Q and K by a random orthogonal `M` before
quantizing. `MMᵀ = I` so the attention output is unchanged, but *"each entry of QM or
KM is a random sum of entries of Q or K"* — outliers get spread across channels
rather than dominating a scale. Implemented as products of random diagonal matrices
and Hadamard matrices: **O(d log d)**, not O(d²).

Their Table 3:

| | RMSE |
|---|---:|
| FP16 baseline | 3.2e-4 |
| FP16 FA3 | 1.9e-4 |
| FP8 per-tensor | 2.4e-2 |
| FP8 + block quantization | **9.3e-3** |
| FP8 + block quantization + incoherent processing | 9.1e-3 |

⚠️ **Read carelessly this says incoherent processing is worthless** — 9.3e-3 → 9.1e-3
is 2 %. It is not: that is measured on their distribution. The paper separately
reports incoherent processing giving **2.6×** on a synthetic where **0.1 % of entries
are large**, i.e. it is the technique that exists *for* the outlier case and does
nothing without one.

**We have that case.** Four fixed channels at 10–14× the median dimension, in every
position, in every net measured. So for us the two techniques swap importance
relative to their table.

⚠️ **And the kernel does not port.** FA3's fp8 path needs **TMA**, asynchronous
**wgmma**, `setmaxnreg`, an in-kernel V transpose via LDSM/STSM because fp8 wgmma
wants k-major operands, and byte-permute instructions to fix the accumulator →
operand-A layout mismatch between the two GEMMs. `CLAUDE.md`: this card is **Ada
sm89 — no TMA, no wgmma, no clusters.** The *algorithms* transfer; the
implementation is Hopper-shaped throughout.

## FlashAttention-4

[arXiv:2603.05451](https://arxiv.org/abs/2603.05451). Read directly.

Its premise is **asymmetric hardware scaling** on Blackwell, and the numbers are
worth recording because they generalise: tensor cores went to **8192 ops/clock/SM**
(doubled), while **shared memory stayed at 128 B/clock/SM** and the **exponential
unit (MUFU) stayed at 16 ops/clock/SM**. Consequence: *"shared memory traffic and
exponential operations now dominate execution time, exceeding MMA compute by
25–60 %."*

Two portable ideas follow. The **exponential moves off MUFU** onto FMA units — `2^x`
split into integer part by IEEE-754 bit manipulation and fractional part by a degree
3–5 Horner polynomial, a degree-3 giving 3.90e-3 max BF16 error and matching hardware
to 1 ULP on 99 % of inputs; and only **10–25 %** of exponentials are emulated, the
rest staying on MUFU to balance register pressure. And **conditional softmax
rescaling**: rescale only when `m_j − m_{j−1} > τ`, τ ≈ log₂(256) = 8.0, with the
final normalisation correcting the accumulated deviation.

⚠️ **No fp8 content at all** — FA4 is a BF16 paper. It notes SageAttention3 does FP4
on Blackwell consumer parts; FA4 itself does not.

⚠️ **And its bottleneck is not ours.** `encoder.cu` is `math_pipe_throttle`-limited
at 71 % of the 35.5 TFLOPS mma issue rate with `long_scoreboard` second — the tensor
pipe refusing work, then memory. We are not exp-bound, so the FA4 techniques address
a problem we do not have. Recorded so nobody re-derives that.

## What this means for our fp8

**We already implement FA3's first technique.** `csrc/fp8_gemm.cuh`'s `quantise_row`
does per-row dynamic activation scaling with block weight scales fixed at pack time —
that is block quantization, and it is why the fp8 FFN measured 0.66 % max
prior-space error (`2026-08-04-fp8-encoder.md`, `2026-08-05-fp8-per-row.md`).

**The untried one is incoherent processing**, and it is aimed exactly at what was
measured this morning: a Hadamard rotation of the residual before quantizing turns
four 10–14× spikes into a spread distribution, which is the case an e4m3 mantissa
handles worst. It costs **zero parameters** — against 17 M for the per-block logit
lane — and O(d log d) at d = 256 is nothing.

⚠️ Unmeasured here, and two caveats before anyone builds it. The rotation has to be
applied consistently on both sides of every quantized GEMM or the maths does not
cancel, which is a real kernel change rather than a preprocessing step. And FA3
rotates **Q and K only**, where our problem is in the **residual stream feeding the
FFN** — the analogy is close but not identical, and whether the same trick survives
eight sequential layers of it is not something their measurements answer.

## fp8 in RL rollouts

The setup the field describes is ours: generate rollouts in low precision, train in
high precision.

**The problem**, consistently reported: a quantized generator with a bf16 trainer
creates a *rollout-training mismatch* that biases every gradient estimate. Because
the policy gradient sums token-level log-probability terms, small per-token
discrepancies compound over long horizons — severe enough on some reasoning
benchmarks to collapse training.

**The accepted fix** is **truncated importance sampling**: add the ratio
`π_train / π_rollout` to the update and truncate it where the rollout probability is
small. Reported result — fp8 W8A8 with token-level TIS *"preserves the learning
behavior of dense models, aligning closely with BF16 baselines"*, at up to **44 %
rollout throughput**. Variants: masked IS, and
[adaptive IS](https://arxiv.org/html/2605.13907v1) which tunes the correction per
batch on the argument that the mismatch is **non-stationary** — an exploration bonus
early, a destabilising bias once the policy concentrates.
[Jet-RL](https://arxiv.org/html/2601.14243v1) pushes unified train/rollout precision;
NVIDIA ships an end-to-end fp8 RL pipeline.

⚠️ These come from search summaries and abstracts, not from reading each paper end to
end. The TIS mechanism and the mismatch framing are consistent across all of them,
which is why I am willing to state them; the individual numbers are not verified.

### ⚠️ Why almost none of it applies to us

The correction machinery exists because the objective contains a **likelihood
ratio**. `π_train / π_rollout` appears explicitly in the policy gradient, so a
precision gap between the generating network and the trained network biases the
estimator directly.

**AlphaZero-style training has no such ratio.** Our target is the search's improved
policy `π'`, and the loss is a cross-entropy regression onto it (`train/loss.py`,
`training.md`). Self-play precision changes *which data we get* — slightly different
searches, slightly different games — but there is no importance weight in the
objective to be biased, because there is no importance weight. Off-policyness is
already handled the way AZ handles it: as data, with `weight_gen` recording staleness
and the buffer window bounding it.

That is a structural difference, not luck, and it is why running self-play in fp8
against an fp32-master trainer has never destabilised a run here while the same
arrangement is reported as a collapse risk for LLM RL.

What *is* transferable is the throughput framing — **44 % is about what fp8 rollout
generation is worth to anybody**, which brackets what we measured for the fp8 FFN —
and the warning that the failure mode is silent: it shows up as a training curve, not
as an error.

⚠️ One place the analogy could bite and has not been checked: our search is not a
sampler, it is an argmax over `logits + σ(completedQ)`. A precision perturbation that
flips a near-tie changes the *move played*, not just a probability, and that is a
discrete failure mode with no analogue in the token-sampling literature. §12's
differential harness holds the fp8 encoder to the fp16 reference tree-for-tree at
n = 800, which is the check that would catch it, and it passes — but it has never
been run specifically to count near-tie flips under fp8.

## What is worth doing, in order

1. **Nothing yet.** No measurement says the activation spikes cost Elo, and fp8
   activations are not being attempted this week.
2. If fp8 activations are attempted: **incoherent processing first**, because it is
   the technique the literature built for our exact distribution, it costs no
   parameters, and it is testable offline — rotate, quantize, dequantize, compare
   prior-space error against today's 0.66 %, with no training run at all.
3. FA4's ideas stay filed as *not applicable* unless the encoder's stall profile
   changes.
