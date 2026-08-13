# 2026-08-13 (fourteenth entry) — massive activations in the residual stream, and why they are not a bug

Prompted by a hypothesis that the network had an attention-sink pathology. It has
**massive activations**, which are a real and named phenomenon; it does **not** have
an attention sink; and the literature plus one independent internal replication both
say the first is structural rather than a defect. Written down because the natural
next move — "find the bug and remove the spike" — would have been wasted work, and
because the one place it *does* constrain us is a kernel decision already on the
roadmap.

## What was measured

`checkpoints/t12h-gumbel-004009.pt` — the 12 h Gumbel run, our current best — on four
positions taken from `logs/gate2-h2h.pgn`, i.e. games that checkpoint had just
played, so in-distribution by construction. CPU, torch, forward hooks on each
`nn.TransformerEncoderLayer`. Dead slots are excluded: a captured piece is masked out
of attention (`src_key_padding_mask=~alive`) and its residual is never read, so
including it would put a fake mode in every distribution.

Max `|activation|` over live tokens, by depth:

| ply | live | embed | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 29 | 1 | 16 | 19 | 25 | 29 | 36 | 40 | 43 | 42 |
| 30 | 25 | 0 | 21 | 23 | 25 | 30 | 32 | 33 | 34 | 33 |
| 55 | 17 | 1 | 19 | 25 | 26 | 26 | 28 | 33 | 36 | 35 |
| 80 | 9 | 1 | 15 | 21 | 25 | 29 | 33 | 36 | 41 | 46 |

Three facts, and they are the diagnostic ones:

1. **Nothing after the embedding** (max ≈ 1 — it is a LayerNorm-normalised sum), then
   **15–21 at layer 1**. The spike is installed immediately, not accumulated.
2. **The same four channels in every position**: `d190`, `d168`, `d183`, `d244`, with
   `d190` the largest in all four. Against a median dimension-max of 3.2–4.0, that is
   **10–14×**.
3. ⚠️ **The token side does not match.** Token norms are 39.5–66.6 median with the
   largest only **1.3–1.7×** the median, and *which* token is largest changes with the
   position. An attention sink is one token at many times the others, usually a fixed
   one. We do not have that.

### The same measurement on `t24h-muon-008215`, identical inputs

| | `t12h-gumbel` | `t24h-muon` | ratio |
|---|---:|---:|---:|
| final-layer max \|a\| (ply 30) | 33 | **672** | **20×** |
| median dimension-max | 4.02 | 39.2 | 10× |
| median token norm | 45.7 | 800 | 18× |
| `weight_norm` | 131 | 498 | 3.8× |

**20× the activations on 3.8× the weights**, so it compounds through depth rather
than rescaling. And the growth is late where gumbel's is early: muon runs
193 → 287 → **672** over its last three layers. Its dominant channel is a different
one (`d64`, carrying essentially all the excess) though `d190` appears in both.

⚠️ Its token norms are *more* uniform than gumbel's — at ply 80, 1015/1014/1000/988
against a median of 979. A single runaway channel shared by every token is the
opposite of a sink, which needs one token to stand out.

⚠️ **Correlational.** A losing network also has larger activations; four positions and
no ablation cannot say which causes which. What it does is give the muon
weight-norm diagnosis a second, independent symptom.

## What the literature says this is

Three threads, discovered separately and now understood as one cluster.

**Attention sinks** — Xiao et al., *Efficient Streaming Language Models with Attention
Sinks*, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453). LLMs assign large
attention weight to the first token regardless of its semantics. The cause is
softmax normalisation: the weights must sum to one, so a head with nothing useful to
retrieve still has to put its mass somewhere, and a globally visible early token is
the natural dump.

**Massive activations** — Sun et al., *Massive Activations in Large Language Models*,
[arXiv:2402.17762](https://arxiv.org/abs/2402.17762). A very small number of
feature dimensions take values orders of magnitude above the rest, on specific
tokens, appearing at a fixed depth and persisting. They are near-constant across
inputs — the paper's framing is that they act as **implicit bias terms**, learned
parameters smuggled into the activations. Zeroing them or replacing them with their
mean degrades the model badly. The proposed remedy is to give the model an
**explicit attention bias** so it does not have to build one out of activations.

**Outlier features and quantisation** — Bondarenko, Nagel & Blankevoort,
*Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*,
[arXiv:2306.12929](https://arxiv.org/abs/2306.12929), NeurIPS 2023. This is the
mechanistic account: outliers come from heads that want a **no-op**. To get exact
zeros out of a softmax you must push its inputs further and further apart, and that
pressure shows up as outliers elsewhere in the network. Their fixes — **clipped
softmax** and **gated attention** — produce much smaller outliers at equal or better
float performance, and make full INT8 activation quantisation work without tricks.

Two adjacent results complete the picture: **softmax-off-by-one** (Miller, 2023) adds
a constant 1 to the softmax denominator so mass need not be spent at all, and
**registers** (Darcet et al., 2023, for vision transformers) add dedicated tokens to
absorb the role, which is now standard practice.

The current consensus, from the more recent surveys: massive activations and
attention sinks **co-occur and often involve the same tokens but are functionally
distinct** — the activations act globally as implicit parameters, the sinks act
locally on individual heads. Having one without the other, which is our case, is not
a contradiction.

## The structural argument

One reading ties the spike to the readout: as long as one layer is responsible for
turning the residual stream into logits in a single step, that layer's output must
dominate the residual. A remedy in that reading is to let every block contribute
additively to the logits through its own projection, so no single layer has to be the
decoder. Neither the reading nor the remedy is tested in this repository.

## What it means here

`BrokefishNet` has exactly the shape the structural argument describes: a pre-norm
stack, then `norm_f`, then three `nn.Linear` heads producing 32×64 policy logits,
four promotion logits and a value. **One layer must produce all of that in one shot**,
so its output has to dominate. Our measurements fit the prediction — absent at the
embedding, installed at layer 1, fixed channels, monotone to the readout.

⚠️ **Why we have no sink, and it is architectural.** Our tokens are *pieces*, and a
captured piece is masked out of attention entirely. There is no BOS, no always-present
token, and by ply 80 only **9 of 32 slots are live**. There is nothing for a head to
dump mass onto, so the softmax-normalisation pressure that creates sinks has no
target. The Bondarenko mechanism should still apply in principle — a head wanting a
no-op has nowhere to go — and where that pressure lands here is **unmeasured**. The
residual is a proxy; nobody has looked at our attention maps.

**So: not a bug, and not worth removing for its own sake.** The one place it becomes a
real constraint is precision. `2026-08-09-kernel-design-retrospective.md` establishes
that the only variant raising boards per SM without paying occupancy is **fp8
activations** — 25 088 B/board against 50 176, giving 4 boards/SM. A residual stream
with fixed channels at 10–14× the median is exactly what makes that hard, and it is
now a measured obstacle rather than a suspected one. Today's fp8 covers *weights* in
the FFN and measured 0.66 % max prior-space error; activations are a different
question and this is why.

⚠️ **And the obvious port is the wrong size for us.** A per-block logit lane needs one
`d × N_POLICY` projection per layer: 256 × 8192 = 2.1 M parameters × 8 layers =
**17 M**, against a 6.38 M-parameter network. That is a 3.7× parameter increase to
address a symptom, and it would invalidate the parameter count the entire cost-vs-Elo
curve is denominated in. The cheap variant — one shared `W` with a learned per-layer
scalar gate — costs essentially nothing and is the only version worth considering,
and only if fp8 activations are actually attempted.

**The two cheap changes the literature names are both small here.** One is an
explicit **bias on the QKV projection** — and **we already have it**: `nn.TransformerEncoderLayer` defaults to
`bias=True`, so every layer carries `in_proj_bias` (768), `out_proj.bias` (256) and
both FFN biases. Only the three heads are deliberately biasless (`model.py:94`).
So the fix Sun et al. recommend is already in the architecture **and the spike is
there anyway** — which is a real data point against the explicit-bias remedy being
sufficient, at least at this scale.

That leaves the **learned per-head sink gate**, `sigmoid(lse − sink)`: `n_heads = 8`
parameters per layer, **64 in total**. Against 17 M for the logit lane, free. It is
the only untried item of the three.

⚠️ Neither is worth doing on this evidence alone. We have no measurement that the
spike costs Elo, and both would change the trained network, so nothing measured
before them would compare across the flag. They belong to the fp8-activations
question, not to a general clean-up.

## What is not established

- **Whether the spike costs us any Elo.** Nothing here measures that. The literature
  says removing massive activations *naively* hurts.
- **Whether our attention has the no-op pathology at all.** Residual norms are a
  proxy; the attention maps have not been looked at. That is the honest next
  measurement, and it is ~20 minutes on CPU.
- **Whether muon's 20× activations are cause or consequence** of its Elo loss.

Figures: `logs/residuals_*.png` (one bar per residual unit, one panel per layer, both
checkpoints, same four positions). Tensors keyed by FEN in the session scratchpad.
