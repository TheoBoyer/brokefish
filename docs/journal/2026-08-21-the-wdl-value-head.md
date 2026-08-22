# A win/draw/loss value head, as an option

**2026-08-21.** `--value-classes 3` puts a three-way classifier where the scalar
`tanh` head was. It is **off by default**, the scalar branch is unchanged arithmetic,
and every checkpoint written before today still loads. Nothing is measured yet: this
entry records what was built and what it cost, not what it is worth.

## Why

The value head is the one part of this network that has not been observed to train.
Across four 12-hour runs the held-out value-puzzle probe went **0.3530 → 0.3480**
(×0.99) while the policy probe went **0.0844 → 0.4087** (×4.8), and that was
independent of the learning-rate schedule (`t12h-flat`), of the sample reuse
(`t12h-reuse2`) and of the search budget (`t12h-nsched`, whose n=256 phase did not
move it either). The in-buffer value MSE is an **anti-signal** — three cases now have
it inverted against held-out value quality, most recently `t12h-vw2`, which had 19 %
better buffer MSE, −0.045 value-puzzle and −231 Elo.

Three mechanisms make a classifier a candidate rather than a cosmetic change:

1. **A squared error through a `tanh` has a gradient that vanishes as `|x|` grows
   whether or not the prediction is right.** That is the documented mechanism of the
   `lr = 0.2` collapse (`2026-07-31-value-collapse.md`): `value_saturated_frac → 1.0`,
   `grad_value_head → 0.000`. A cross-entropy on logits does not have that failure in
   the same place.
2. **A draw becomes a class instead of a coincidence at zero.** Our leagues are
   draw-heavy — 2 % to 100 % depending on the pairing — and under the scalar head
   "draw" is whatever pre-image of 0 the head happens to find.
3. **It is what the field does.** KataGo and Leela both train win/draw/loss, and
   KataGo's `c_value = 1.5` exists precisely because the classifier's loss scale is
   not the squared error's.

⚠️ **None of that is evidence.** It is the same class of hypothesis as
`--value-weight 2.0`, which cost a 12-hour run and came back −231 Elo.

## What it cost, and the one prediction worth making in advance

**The kernel change is free, and the reason was already in the source.** The packed
head matrix is `[96][256]`: 64 policy rows, then an aux tile **padded from 5 columns
to 32** so it is a whole four-n-tile group. Warp 2 has been computing all 32 of those
columns on every forward since B2 and reading 5 of them. The two extra value columns
come out of the 27 that were zeros — **no extra mma, no extra shared memory, no extra
register, no change to `gemm_direct`**, which is where `perf.md` row 9's +11.8 % lives.

The epilogue branch is a 3-way softmax at `tid == 0`, once per board, in a phase that
is 0.3 % of the network. `expf` and not `__expf`: the fast intrinsic would buy nothing
measurable and spend accuracy against the torch oracle.

> **Prediction, before measuring: zero throughput change**, at or inside the ±3 %
> thermal noise floor of this card.

### Measured

`forward_full` at B = 4096, shipped int8 configuration, ABBA-interleaved, 80 timings
per arm:

| value head | ms / launch (median) | evals/s | vs scalar |
|---|---:|---:|---:|
| scalar | 40.386 | 101 421 | 1.0000x |
| win/draw/loss | 40.402 | 101 381 | **1.0004x** |

**+0.04 %, against a per-timing sd of 1.1 %.** The prediction holds and the gap needs
no explanation, which was the point of making it: the arithmetic was already being
done. `ledger/perf.md`.

## The shape of the change

The contract that made this cheap is that **both heads emit the same `[N]` fp32 in
`[-1, 1]`** — the classifier through `p(win) − p(loss)`. So the search, the terminal
collapse, the value-head puzzle probe, the arbiter and the whole Elo pipeline never
learned that a second head exists. What actually moved:

| | |
|---|---|
| `nn/model.py` | `n_value` on the constructor, `wdl_to_scalar`, `forward(with_logits=)` |
| `train/loss.py` | the value term branches: MSE, or `cross_entropy` on class `z + 1` |
| `train/loop.py` | `TrainConfig.value_classes`, `--value-classes {1,3}` |
| `csrc/encoder.cu` | `n_value` threaded to the epilogue; the 3-way softmax |
| `nn/cuda_impl.py` | pack 1 or 3 rows at `VALUE_ROW` |
| nine loader sites | sized from the file — see below |

**Class order is normative: 0 = loss, 1 = draw, 2 = win, from the side to move**, so a
stored outcome `z ∈ {−1, 0, +1}` has class index `z + 1`. The CUDA epilogue reads
those three columns by position, and a permutation would still produce a finite value
in `[−1, 1]` that nothing downstream could question — so `tests/test_b2.py` pins it by
swapping the loss and win rows and demanding the *kernel's* value negate exactly,
without consulting torch.

## Retro-compatibility is the load-bearing part

A checkpoint is a bare `state_dict` and carries no architecture — `league.py:157`
writes `net.state_dict()` and the shape lives in the constructor's defaults. So a
`[3, 256]` `value.weight` would have made **every checkpoint on the ledger
unloadable**, `checkpoints/anchor.pt` included, and *every Elo scale in this
repository anchors to that file*. There would have been no joint fit spanning the
change and therefore no way to price it.

`nn/model.py:n_value_of` reads the width back off `value.weight.shape[0]` and
`net_for_state` builds the net from it; all nine loaders go through it. Verified on
the real artefacts: `anchor.pt` and `t12h-nsched.pt` load as `n_value=1`, the smoke
run's checkpoints as `n_value=3`, all `strict=True`.

`value_classes` lands in `config_hash`, so a *resume* across the change is refused
without `--allow-config-change`. That is correct — the weights would not even load.

## What is pinned

`tests/test_train.py`: the cross-entropy against a hand-written double-precision
transcription; the class convention; that a perfectly fitted head reaches 0 while a
flat one sits at `ln 3 = 1.0986`; that a target which is not in `{−1, 0, +1}` raises
instead of rounding into a class it does not mean; and that the scalar branch still
reports `n_value = 1` and the same squared error.

`tests/test_b2.py`: the kernel epilogue against torch on both implementations, the
permutation test above, the side-to-move row select, and the retro-compat load.

`csrc/tests/core.cuh` regenerated; `tdirect` and `tint8` green.

## Two things found on the way, neither of them mine

- ⚠️ **`python -m brokefish.train.loop --help` has been crashing** since the `--int8`
  flag landed: its help string contains a literal `~1 %`, and argparse `%`-formats
  every help string, so `% p` is an unsupported format character. One `%%` fixes it.
  Fixed here because it blocked verifying the new flag.
- ⚠️ **11 tests in `tests/test_quant.py` are red on `main`** and have nothing to do
  with this work — confirmed by running them in a worktree at `HEAD`. `SCHEME` became
  `"all"` (quant 3) in b6a79d0, and quant 3 is instantiated at two boards per CTA
  only, so every `two_boards=False` construction now raises in the constructor. The
  tests that prove one board and two boards agree bit-for-bit are among them, which
  means **that invariant is currently unchecked.** Not fixed here.

## What has not been done

- No throughput measurement yet (the prediction above is unmeasured).
- **No Elo.** The head is a hypothesis until a 12-hour run and a joint league price
  it, and the value-puzzle probe is the metric it is supposed to move.
- `--value-weight` is **inherited, not calibrated**, on this branch: the squared error
  starts near 1.0 and the cross-entropy at 1.0986, with a floor of 0 rather than of
  the target's entropy. `gradient/value` and `value_saturated_frac` keep their names
  and change their meanings, so neither is comparable across the two branches.
