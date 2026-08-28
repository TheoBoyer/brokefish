"""The outer loop, ``docs/train.md`` §§6-11.

Play games with the current weights, store what the search produced, sample from the
store, take gradient steps, publish new weights, repeat. Everything inside one move is
already specified and built — ``mcts.md`` §6 for the search, ``spec.md`` §7 for the
network — so what is here is the alternation, the cadence, the optimiser, the
checkpoint and the euro counter.

⚠️ **C2 is the first phase with no oracle.** Perft settled the engine, ``nn/model.py``
settled the encoder, an independently written AGZ search settled C1. Nothing external
says whether a training loop is correct, and its failure mode is not a crash but a
curve that is merely worse than it should have been. ``train.md`` §12 is what replaces
the oracle; :func:`check_weights_propagate` and ``tests/test_train.py`` are where it
lives, and :mod:`brokefish.train.overfit` is check 1.

Three numbers in here are the paper's and not ours, and are the reason to read §7
before changing any of them: **65.2 positions sampled per game generated** (AZ's
700,000 minibatches of 4,096 against 44M games), **batch 4,096** reached by four
accumulated micro-batches of 1,024 — which is exact, not an approximation, because the
network has no batch statistics anywhere — and **SGD with momentum 0.9** at
``lr = 0.2`` dropped three times. That last one is the single most likely value in the
contract to be wrong for a 6.38M-parameter transformer, and check 1 is the fastest way
to find out.

Run from the repository root::

    python -m brokefish.train.loop --run r1                 # the AZ configuration
    python -m brokefish.train.loop --run smoke --smoke      # minutes, end to end
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from brokefish.env import cuda_impl as _cuda_env
from brokefish.env import torch_impl as _torch_env
from brokefish.nn.model import BrokefishNet
from brokefish.search import SearchConfig, search_impl

from .buffer import ReplayBuffer
from brokefish import paths
from .log import Logger
from .loss import TrainBatch, audit_labels, az_loss, l2_penalty, weight_decay_for
from .sync import PackedWeights, weight_fingerprint

# §7.3. Fractions of total training rather than absolute steps, because AZ ran
# 700,000 and we run ~159,000 (§13). The drop points are published nowhere; these
# are the released pseudocode's `learning_rate_schedule` (100k, 300k, 500k of 700k),
# which is the only source with the three drops AZ's prose describes.
LR_SCHEDULE: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.2), (100e3 / 700e3, 0.02), (300e3 / 700e3, 0.002), (500e3 / 700e3, 0.0002))

# Both engines, by the name `TrainConfig.impl` carries. Importing either is free —
# `load_extension` is called lazily by `_ext()`, so nothing compiles here — which is
# why this can be a plain dict rather than the deferred-import registry
# `brokefish.search.search_impl` needs.
ENVIRONMENTS = {"cuda": _cuda_env, "torch": _torch_env}


@dataclass
class TrainConfig:
    """Every parameter ``train.md`` freezes, and nothing it leaves to the caller."""

    # -- the search, mcts.md §4.1
    n_sims: int = 800
    batch_games: int = 4096
    # Must equal `kE` in `csrc/search.cuh` -- the CUDA search refuses a mismatch
    # on construction, since the cap is a compile-time constant there. Raised
    # 64 -> 96 on 2026-08-02: over 80k measured roots the true candidate count
    # was mean 23.8 / p99 58 / max 83, but *mate* roots averaged 45.1 edges and
    # hit the old cap 15x more often than average, so truncation was dropping
    # proved wins. `K_POLICY` in `train/buffer.py` follows it.
    e_cap: int = 96
    tau_plies: int = 30
    eps: float = 0.25
    alpha: float = 0.3
    # §6.6a. Collapse a node onto its proved-winning edges, so the target on a root
    # with a mate is a point mass on it and the mate is actually played. **Off**, so
    # a run started today reproduces every number measured before 2026-08-03; the
    # measurement it exists to fix, over `t9h-n128-sweep`'s finished buffer, is that a
    # mate in one took a **median 0.302** of the visits and was the argmax only
    # **56.1 %** of the time. It touches **0.89 %** of positions, so expect `loss` and
    # `kl` not to move; what should move is the mate rate and the game length.
    terminal_collapse: bool = False
    # `search.md` §11's three seams together: Gumbel MuZero root sampling,
    # sequential halving, and the completed-Q policy target in place of `N / n`.
    # ⚠️ Needs `--impl torch` today: `csrc/search.cuh`'s `descent_kernel` is PUCT,
    # and `search/cuda_impl.py` refuses rather than running one and reading the
    # other. ⚠️ `--tau-plies` and `--eps` stop meaning anything with it on.
    gumbel: bool = False
    gumbel_m: int = 16
    gumbel_scale: float = 1.0
    # §fp8. The FFN's two matmuls in e4m3 during **self-play only** -- the gradient
    # step runs the fp32 master weights through torch, so this cannot destabilise the
    # optimiser and its whole effect is slightly noisier data. Measured on
    # `t7h-n128-collapse@2006`: **1.15x** encoder throughput, **1.05 %** max
    # prior-space error, 1.31 % of top-1 priors moved.
    # `docs/journal/2026-08-04-fp8-encoder.md`. Off, like every other new lever.
    fp8: bool = False

    # The same two matmuls in **int8** rather than e4m3, and the same inference-only
    # property. ⚠️ It is not a trade: measured on `t12h-gumbel-004009`, int8 is
    # **1.016x** e4m3's throughput *and* **2.56x** lower max prior error (1.71e-2
    # against 4.37e-2), because our tiles span ~2 binades of e4m3's 18 and an exponent
    # buys nothing on data with no outliers. Mutually exclusive with `fp8`.
    # `docs/journal/2026-08-14-int8-kernel-spec.md`. Off, because no run has used it.
    int8: bool = False


    # -- §11, playout cap randomisation (KataGo §3.1), `training.md` §11
    #
    # On a proportion `pcr_p` of turns the search runs the full `n_sims` cap and the
    # position is recorded; on the rest it runs `pcr_fast_sims` with the root noise off
    # and **nothing is recorded**. `pcr_p = 0` disables it and is the default, so a run
    # started today reproduces every number measured before 2026-08-07.
    #
    # The tension it relieves is KataGo's: the value target is one noisy binary result
    # per *game*, so value training wants many cheap games, while the policy target
    # wants a search deep enough to actually deviate from the prior. Uniform `n` has to
    # pick one. ⚠️ Only the *positions* of a cheap turn are dropped — the turn is still
    # played, and the games it produces are exactly the extra value data.
    #
    # ⚠️ `n_sims` must be the **full** cap, not the mean: it sizes the node pool
    # (`n_max = n + 1`) and the path arrays. The realised mean is logged per generation
    # as `self_play/sims_mean` rather than assumed, because it is the cost axis.
    pcr_p: float = 0.0
    pcr_fast_sims: int = 64
    # §11a. `(training_seconds, N, n)` in increasing order of the first field, the
    # caps in force from that point on. Empty means the fixed `n_sims` /
    # `pcr_fast_sims` above, which is every run before 2026-08-08.
    #
    # KataGo §3.1 anneals: *"we chose p = 0.25 and (N, n) = (600, 100) initially,
    # annealing up to (1000, 200) after the first two days of training."* The
    # independent argument is ours and is newer than the paper: measured
    # 2026-08-08 over two runs, the Elo value of a doubling of search **grows with
    # the network** -- +15 per 2x at step 101, +112 at 1405, +186 at 8416 -- so a
    # cap that is right at the start is too small later, on both runs.
    #
    # ⚠️ Keyed on **training seconds** (`train.md` §10: self-play plus gradient),
    # not on wall clock and not on steps. Wall clock would include the puzzle probe,
    # which §10 deliberately keeps off the axis; steps would make the switch point
    # move whenever the cadence did, and the thing being scheduled is a *cost*.
    sims_schedule: Tuple[Tuple[float, int, int], ...] = ()

    # -- the buffer, §5
    window_games: int = 500_000
    mean_plies: int = 80
    max_plies: int = 512               # §5.4, scored drawn
    # ⚠️ `None` means `runs/<run>/replay/`, which is what a run should use. It stays
    # overridable because a rented box with a fast scratch disk is a real case.
    # `brokefish/paths.py` owns the layout.
    buffer_dir: Optional[str] = None
    buffer_in_memory: bool = False

    # -- the cadence, §6
    #
    # ⚠️ **Denominated in positions, not games** (changed 2026-07-31). AZ publishes
    # 700,000 x 4096 samples against 44 million games = 65.2 samples per *game*, and
    # that is what `samples_per_game` records. But the quantity that acts on training
    # is samples per *position* — how many times each generated example is trained on
    # — and the two are the same thing only at AZ's game length. `training.md` §6
    # assumed 80 plies for that conversion.
    #
    # Measured on run `t4h-n64`: our games grew from 101 to 144 plies, so at a fixed
    # 65.2 per game the per-position reuse fell **0.641 -> 0.374 inside one run**, a
    # 42 % drop in how much each example is trained on, with the data rate constant at
    # 10,240 records per generation throughout. A run whose reuse halves partway
    # through is uninterpretable, so the cadence now rides the records rather than the
    # games and holds reuse fixed whatever the game length does.
    #
    # ⚠️ The 80-ply figure is **ours, not AZ's** — neither paper publishes a game
    # length (checked against both texts, 2026-07-31). So 0.815 is our estimate of
    # AZ's per-position reuse, not their number.
    samples_per_position: float = 65.2 / 80.0     # = 0.815
    samples_per_game: float = 65.2                # AZ's published ratio, for the log line

    # -- the optimiser, §7
    batch: int = 4096
    micro_batch: int = 1024
    momentum: float = 0.9
    l2: float = 1e-4
    # `sgd` is AGZ's and stays the default: the baseline is the thing being ablated
    # against, so it does not move. `adamw` is ablation 1 (2026-07-30).
    optimizer: str = "sgd"
    betas: Tuple[float, float] = (0.9, 0.95)   # 0.999 is too slow to matter at 10^2 steps
    # ⚠️ **0.01, not the 0.1 transformer default** (changed 2026-07-31). 0.1 comes from
    # supervised settings with orders of magnitude more data than 4.7M samples, and it
    # was judged too aggressive on 2026-07-31 and lowered to 0.01 for run `c2-8h`.
    # That choice then **silently reverted**: it lived only in a command line, so
    # `t60-n256` and `t4h-n64` both picked the 0.1 default back up without anyone
    # noticing. `c2-8h` lost its cosine schedule the same way. A decision that exists
    # only in a shell history is a decision that will be un-made, so it lives here.
    #
    # ⚠️ There is **no measurement** preferring 0.01 to 0.1: the one run that used it
    # was invalidated by the FPU bug (`journal/2026-07-31-value-collapse.md`). This is
    # a restored intention, not a result, and it is a legitimate ablation.
    adam_wd: float = 0.01                      # decoupled, so unrelated to `l2` -- see below
    # ⚠️ **Weight decay for the AdamW matrices, decoupled from Muon's on 2026-08-17.**
    # Under `--optimizer muon` that group is not a footnote: it is the five embedding
    # tables *and all three readout heads*, `value.weight` among them -- one row of 256
    # that was decayed at the same rate as a 256x1024 trunk matrix while training at
    # `aux_lr/lr = 0.05` of its speed. Measured the same day: `grad_policy_head` is
    # identical across the AdamW control and both muon arms (0.127-0.131) while
    # `grad_value_head` falls 0.290 -> 0.174, and value-head puzzle accuracy falls with
    # it (0.486 -> 0.403) even as policy accuracy rises.
    # `None` stays tied to `adam_wd`, so every run before this date reproduces.
    aux_wd: Optional[float] = None
    grad_clip: float = 0.0                     # 0 disables; AGZ specifies no clipping

    # -- ablation 2, `train/muon.py`. Inert unless `optimizer == "muon"`.
    #
    # `aux_lr` is the rate for the 91,904 parameters Muon does not touch -- the five
    # embedding tables, the three readout heads, every bias and LayerNorm gain. It is
    # carried as a *ratio* to `lr` inside the optimiser, so one schedule drives both
    # groups and warmup applies to each. 1e-3 is `t7h-fp8`'s tuned AdamW rate, so the
    # residue is the same optimiser at the same rate in both arms of the ablation.
    #
    # ⚠️ `lr` itself does **not** transfer from the AdamW arm and must be swept. Muon's
    # update has RMS `1/sqrt(fan_in)` = 1/16 here, against AdamW's ~`lr`, so the naive
    # RMS-matched equivalent of `1e-3` is ~`0.016` -- which is where Jordan's 0.02
    # default sits, and where the sweep should be centred.
    aux_lr: float = 1e-3
    muon_momentum: float = 0.95
    # Q, K and V orthogonalised separately rather than as one (768, 256) tensor.
    # ⚠️ Not an ablation arm: Jordan's post reports the split is better, and CMuon
    # names why (one shared preconditioner across misaligned blocks). The fused form
    # is the bug, so this defaults on.
    muon_qkv_split: bool = True
    # 8 = one group holding all eight heads, i.e. no head splitting. 1 = per-head,
    # Kimi K3's and GLM-5's "Muon Split". The ladder is {1, 2, 4, 8} and arXiv
    # 2605.08933 shows the optimum moves during training -- a hyperparameter, not a
    # constant. 8 for the first run: one thing at a time.
    muon_head_group: int = 8
    muon_ns_steps: int = 5
    # "polar" = Polar Express's per-iteration minimax coefficients, "jordan" = the
    # fixed quintic torch and the reference implementation use. Polar is the default
    # because it is what makes the learning rate actually transfer across our layer
    # shapes: measured, the achieved update RMS spread is 2.7 % against Jordan's
    # 11.1 %, for +2.2 % on an iteration that is 0.07 % of a step.
    muon_ns_scheme: str = "polar"
    normuon: bool = False                      # see `muon.Muon` -- reconstructed, off
    lr_schedule: Tuple[Tuple[float, float], ...] = LR_SCHEDULE
    total_steps: int = 159_000
    # Linear warmup, in optimiser steps. 0 is off and is the default, because AGZ
    # specifies none and the SGD baseline must not move underneath the ablation.
    warmup_steps: int = 0
    # "step" is AGZ's three drops and stays the default; "cosine" decays smoothly to
    # `lr_min` at `total_steps` -- which that flag then has to actually describe.
    decay: str = "step"
    lr_min: float = 0.0

    # -- the loop, §8
    moves_per_phase: int = 4
    checkpoint_every: int = 1000
    # Keep every checkpoint as a weights-only snapshot alongside the rolling one, so
    # a finished run yields a *curve* and not a single endpoint. 26 MB each.
    keep_checkpoints: bool = False

    # -- the puzzle probe, `evaluation.md` §2 and `eval/watch.py`
    #
    # ⚠️ An **absolute** progress signal, which is the one thing self-play cannot give:
    # every other number is relative to an opponent that moves with the network, or
    # tied to a search budget that makes two leagues incomparable. Runs after each
    # checkpoint, ~20 s against ~40 min of training. `puzzle_limit = 0` turns it off.
    #
    # ⚠️ Its cost is **not** added to `seconds`, so it never reaches the curve's
    # x-axis (train.md §10: self-play plus gradient and nothing else).
    puzzle_probe: bool = True
    puzzle_limit: int = 20_000
    buffer_snapshot_every: int = 10_000
    autocast: bool = True              # §8.2, bf16 forward and backward
    # The squared-error term's weight against the policy cross-entropy. 1.0 is
    # AlphaZero's and ours; AlphaGateau's code uses `optax.l2_loss`, which is
    # 0.5*(x-y)^2, so their runs weighted value at half -- and their own eq. (10)
    # says otherwise. See `az_loss`.
    value_weight: float = 1.0
    # ⚠️ **The value head's shape, and the only architectural knob in this config.**
    # 1 is spec §7.4's scalar tanh trained with a squared error -- every Elo number on
    # the ledger. 3 is a win/draw/loss classifier trained with a cross-entropy
    # (KataGo's and Leela's shape); the head still hands the search a single scalar,
    # `p(win) - p(loss)`, so nothing but the loss and the packed head matrix knows.
    #
    # ⚠️ It lands in `config_hash`, so a resume across it is refused by
    # `load_checkpoint` without `--allow-config-change`. That is correct: the two
    # heads have different weight shapes and a hybrid run would not even load.
    #
    # ⚠️ `value_weight` is **not** calibrated for the cross-entropy branch. See
    # `az_loss`.
    value_classes: int = 1
    # **Value-label subsampling** (2026-08-25). The value loss is taken over 1 in k
    # records of every game -- every k-th ply, residue rotating per game -- and the
    # policy loss over all of them. Nothing else moves: same records, same cadence,
    # same reuse per record, same optimiser, same throughput. It exists to move one
    # quantity in isolation, how many times a game's one bit of `z` is shown to the
    # head (~118 plies -> ~118 / k), which is the variable the Muon value-head
    # investigation names and which playout cap randomisation could only move at the
    # price of 1/p more reuse. 1 is every run before it, bit for bit.
    value_subsample: int = 1
    # ⚠️ **Where the value head reads from**, and therefore which frame it predicts in.
    # `king` is spec §7.4's row select of the side-to-move king, predicting the mover's
    # result. `pooled` is the masked mean of every live token of both colours,
    # predicting White/draw/Black, which `heads` flips into the mover's frame before
    # anything downstream sees it.
    #
    # `prenorm` pools before `norm_f` instead of after, which fixes the input scale
    # by construction and lets a token with a larger residual count for more.
    #
    # ⚠️ The hypothesis: the policy head touches all 32 tokens and the king select
    # touches one, so value gradient reaches the rest of the board only through that
    # token's attention. It also lands in `config_hash`, so a resume across it is
    # refused -- correct, since the two heads take different inputs.
    value_head: str = "king"
    # `torch.compile` on the gradient step's forward. ~1.25x measured at batch 256;
    # see `Trainer.__init__` for why the dead-end entry in `CLAUDE.md` is about
    # something else. Off by default: it changes training numerics, so a compiled run
    # is not bit-comparable with one before it.
    compile: bool = False
    # §5.5's floor on the buffer before the gradient phase starts, in *records*.
    # 0 keeps the historical behaviour of `cfg.batch` alone. See `steps_owed` for the
    # failure this exists to stop; AlphaGateau's iteration is 131 072 positions, which
    # is the natural value when reproducing their cadence.
    min_records: int = 0

    # -- §10, §11, §12
    euros_per_hour: float = 0.0
    strict_labels: bool = True         # §12 check 3, one synchronisation per micro-batch
    # The engine-side half of check 3. The loss reads its support out of the record
    # and never consults `movegen`, so this is the only thing left that would notice
    # a record whose board and edges disagree. One movegen per sample is 0.13 % of a
    # step, so every hundredth step is free and never is a choice.
    audit_every: int = 100
    collect_search_stats: bool = True
    deterministic: bool = True         # §9's bit-exact resume

    seed: int = 0
    # Two axes, not one. `impl` selects the engine and the search — `ENVIRONMENTS`
    # and `search_impl` both understand "cuda" and "torch". `encoder` selects the
    # *fused* encoder and has no "torch" option on purpose: `search/cuda_impl.py`
    # reads fp16 logits and `nn/model.py` returns fp32, so the reference model is not
    # a self-play evaluator whichever engine is running.
    impl: str = "cuda"
    encoder: str = "cuda"

    def hash(self) -> str:
        """§9: a resume under changed parameters fails loudly instead of hybridising."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def caps_at(self, training_seconds: float) -> Tuple[int, int]:
        """`(N, n)` in force after this many training seconds — §11a's schedule.

        ⚠️ **`n_sims` is the largest `N` the schedule will ever ask for**, because it
        sizes the node pool (`n_max = N + 1`) and the path arrays for the *whole* run;
        `self_play_move(sims=k)` may only go down from it. So a run that anneals
        256 → 512 allocates for 512 from generation 1 and spends the first half not
        using half of it. `config_from_args` raises `n_sims` to the schedule's maximum
        rather than leaving that arithmetic to the caller.
        """
        full, fast = self.n_sims, self.pcr_fast_sims
        for at, n_full, n_fast in self.sims_schedule:
            if training_seconds >= at:
                full, fast = n_full, n_fast
            else:
                break
        return full, fast

    def lr_at(self, step: int) -> float:
        """Warmup, then either AGZ's step schedule or a cosine decay to ``lr_min``.

        ``decay = "step"`` is AGZ's and is the default: three discrete drops at the
        fractions in :data:`LR_SCHEDULE`. ``decay = "cosine"`` is the modern recipe --
        ``lr_schedule[0][1]`` becomes the *peak* and the rate follows half a cosine
        down to :attr:`lr_min` at ``total_steps``, so the run ends at a rate that has
        already gone smoothly to zero rather than being cut off mid-plateau.

        ⚠️ **Cosine makes ``total_steps`` load-bearing, and it was decorative before.**
        Under the step schedule a wrong ``total_steps`` moves the drop points; under
        cosine it sets the entire shape. The default 159,000 with a run that reaches
        step 116 gives ``cos(pi * 116/159000) ~ 1``, i.e. a constant rate and no decay
        at all -- the flag will look broken when it is merely mis-sized. **Set
        ``--total-steps`` to the number of steps the run will actually take.**

        Past ``total_steps`` the progress term is clamped, so a run that overshoots
        holds ``lr_min`` instead of following the cosine back up.

        ⚠️ **Warmup is a fix for Adam specifically, and for a measured failure.**
        Adam's update is gradient-*normalised*: every parameter moves about ``lr`` per
        step no matter how small its gradient is. The value head's pre-tanh activation
        is a sum over ``d = 256`` such moves, so at ``lr = 1e-3`` step 1 alone drove it
        into saturation -- ``value_saturated_frac = 1.0``, ``grad_value_head = 0.000``,
        dead for 62 of 116 steps (run ``t15-adamw``, 2026-07-31). SGD cannot do this,
        because its step is ``lr * g`` and ``g`` is small; that is exactly the
        smallness Adam normalises away, so this ramp restores it by hand.

        The ramp is ``(step + 1) / warmup_steps``, not ``step / warmup_steps``: the
        latter makes step 0 a no-op, which quietly costs a step and, worse, logs a
        gradient the optimiser never applied.
        """
        if self.decay == "cosine":
            peak = self.lr_schedule[0][1]
            # Progress is measured *after* the ramp, so warmup does not eat decay:
            # the cosine starts at the peak the moment warmup hands over.
            span = max(self.total_steps - self.warmup_steps, 1)
            t = min(max(step - self.warmup_steps, 0) / span, 1.0)
            lr = self.lr_min + (peak - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))
        elif self.decay == "step":
            frac = step / max(self.total_steps, 1)
            lr = self.lr_schedule[0][1]
            for at, value in self.lr_schedule:
                if frac >= at:
                    lr = value
        else:
            raise ValueError(f"unknown decay {self.decay!r}, want 'step' or 'cosine'")

        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr *= (step + 1) / self.warmup_steps
        return lr


def build_optimizer(net: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AGZ's optimiser, or ablation 1's.

    ⚠️ **The two weight decays are not the same quantity and do not transfer.**

    AGZ's loss carries an explicit ``c||theta||^2`` term, so its gradient is
    ``2 c theta`` and torch's *coupled* ``weight_decay`` reproduces it at ``2c``
    (hence :func:`weight_decay_for`) -- and, being part of the gradient, it is
    scaled by the learning rate and by Adam's per-parameter normalisation.

    AdamW *decouples* it: the update is ``theta -= lr * wd * theta``, outside the
    second-moment normalisation. Carrying ``2e-4`` across would apply a decay some
    three orders of magnitude weaker than the regulariser it is meant to be, so the
    ablation would silently be "AdamW with no regularisation" -- which is a real
    result about a different question. ``adam_wd`` is the knob that actually
    regularises the AdamW arm.

    ⚠️ **Under AdamW, ``cfg.l2`` reaches nothing.** `az_loss` returns
    ``policy_loss + value_loss`` and that is the only thing ``backward()`` sees, so
    the coupled term exists solely as `weight_decay_for(cfg.l2)` on the **SGD**
    optimiser. `l2_penalty` is evaluated *after* ``opt.step()`` purely to log
    ``gradient/l2``, which is therefore a **weight-norm diagnostic, not a loss term
    being optimised** — a dashboard reader will otherwise take it for one. The two
    arms are: SGD regularised by ``2 * l2`` coupled, AdamW by ``adam_wd`` decoupled,
    and never both.

    The split is the standard one: decay tensors with two or more dimensions, which
    is every matmul weight and the token embedding, and never LayerNorm gains or
    biases -- shrinking a per-channel gain toward zero is not regularisation, it
    scales the layer's output down and the next layer just scales back up.
    """
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            net.parameters(), lr=cfg.lr_at(0), momentum=cfg.momentum,
            weight_decay=weight_decay_for(cfg.l2))
    if cfg.optimizer == "muon":
        # Ablation 2. Three groups, not two: Muon takes the eight layers' hidden
        # matmuls (98.6 % of the network), and the embeddings, the readout heads and
        # every scalar stay on AdamW -- Jordan's rule, and it matters here because
        # `value.weight` is (1, 256), whose nearest semi-orthogonal matrix is its own
        # direction with the magnitude thrown away.
        from .muon import Muon, muon_param_groups
        return Muon(
            muon_param_groups(
                # The schedule's *peak*, not `lr_at(0)`: `lr_scale` is a ratio and
                # `lr_at(0)` is the warmup's first step, `peak / warmup_steps`.
                net, lr=cfg.lr_schedule[0][1], aux_lr=cfg.aux_lr, wd=cfg.adam_wd,
                aux_wd=cfg.aux_wd, qkv_split=cfg.muon_qkv_split, head_group=cfg.muon_head_group,
                betas=cfg.betas, momentum=cfg.muon_momentum),
            ns_steps=cfg.muon_ns_steps, ns_scheme=cfg.muon_ns_scheme,
            normuon=cfg.normuon)
    if cfg.optimizer != "adamw":
        raise ValueError(
            f"unknown optimizer {cfg.optimizer!r}, want 'sgd', 'adamw' or 'muon'")

    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    decay = [p for _, p in named if p.ndim >= 2]
    flat = [p for _, p in named if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.adam_wd},
         {"params": flat, "weight_decay": 0.0}],
        lr=cfg.lr_at(0), betas=cfg.betas, eps=1e-8)


class Trainer:
    """One run. Owns the network, the search, the buffer, the optimiser and the clock."""

    def __init__(self, cfg: TrainConfig, run: str = "run", device: str = "cuda",
                 logger: Optional[Logger] = None, resume: Optional[str] = None,
                 log_dir: Optional[str] = None) -> None:
        self.cfg = cfg
        self.run = run
        self.device = torch.device(device)
        self.log = logger or Logger(run, log_dir=log_dir, config=asdict(cfg))

        if cfg.deterministic:
            # §9 demands a bit-exact resume. `warn_only` so an op with no
            # deterministic kernel degrades to a warning rather than killing a
            # 130-day run — which is what the attention backward does here.
            #
            # ⚠️ **`CUBLAS_WORKSPACE_CONFIG` is deliberately not set.** torch's
            # deterministic mode nominally asks for it, but §12 check 4 was measured
            # to pass without it on this stack (2026-07-31), and setting a process-wide
            # environment variable from library code only works if it happens before
            # the first cuBLAS handle — so a `Trainer` built from a notebook would get
            # a guarantee a `Trainer` built from `main()` had, and neither would say
            # so. A guarantee that holds only when you come through one door is worse
            # than none. If a torch or driver change breaks this, the resume test is
            # what says so, and the fix is to export the variable in the shell.
            torch.use_deterministic_algorithms(True, warn_only=True)

        # §11. Checked here rather than at the first cheap move, because the failure it
        # catches -- a fast cap above the full one -- would otherwise surface as a tree
        # overrun somewhere inside generation 1.
        if cfg.pcr_p:
            if not 0.0 < cfg.pcr_p <= 1.0:
                raise ValueError(f"pcr_p = {cfg.pcr_p} is not in (0, 1]")
            if not 1 <= cfg.pcr_fast_sims <= cfg.n_sims:
                raise ValueError(
                    f"pcr_fast_sims = {cfg.pcr_fast_sims} must sit in "
                    f"[1, n_sims = {cfg.n_sims}]: n_sims is the FULL cap under §11 and "
                    f"it is what sizes the node pool")
            for at, n_full, n_fast in cfg.sims_schedule:
                if not 1 <= n_fast <= n_full <= cfg.n_sims:
                    raise ValueError(
                        f"sims_schedule entry at {at} s asks for (N, n) = "
                        f"({n_full}, {n_fast}); it must satisfy "
                        f"1 <= n <= N <= n_sims = {cfg.n_sims}. n_sims is what "
                        f"allocates the node pool, so no stage may exceed it")
            ats = [a for a, _, _ in cfg.sims_schedule]
            if ats != sorted(ats):
                raise ValueError(f"sims_schedule must be in increasing order of "
                                 f"training seconds, got {ats}")
            if round(cfg.pcr_p * cfg.moves_per_phase) < 1:
                raise ValueError(
                    f"pcr_p = {cfg.pcr_p} over moves_per_phase = {cfg.moves_per_phase} "
                    f"rounds to zero full turns per phase; the schedule would clamp to "
                    f"one and the realised p would be {1 / cfg.moves_per_phase:.3g}, "
                    f"not what was asked for")

        torch.manual_seed(cfg.seed)
        self.net = BrokefishNet(n_value=cfg.value_classes,
                                value_head=cfg.value_head).to(self.device)  # fp32, §8.2
        self.opt = build_optimizer(self.net, cfg)
        # ⚠️ `self.net` stays the **raw** module and only the call is compiled.
        # `PackedWeights.pack`, §12 check 9 and every checkpoint path read the module
        # directly, and an `OptimizedModule` wrapper in their way would either break
        # the state-dict keys or silently pack the wrong object.
        #
        # ⚠️ `CLAUDE.md` lists torch.compile as a measured dead end, and that entry is
        # about a different thing: `perf.md`'s ladder row 0b is the **encoder forward
        # at B = 4096**, compute-bound, where compile has nothing to win. The gradient
        # step at a small batch is a different regime. Measured 2026-08-12 at batch
        # 256, fwd + bwd + fused AdamW under bf16 autocast, interleaved A/B over three
        # separate runs: **x1.28, x1.11, x1.24**, so call it ~1.25x. `reduce-overhead`
        # (CUDA graphs) measured the same as `default`, so the win is kernel fusion
        # and not launch overhead.
        self.fwd = torch.compile(self.net) if cfg.compile else self.net

        self.env = ENVIRONMENTS[cfg.impl]
        self.weight_gen = 0
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=cfg.encoder, fp8=cfg.fp8,
                                          int8=cfg.int8)

        self.search = search_impl("cuda" if cfg.impl == "cuda" else "torch")(
            SearchConfig(n=cfg.n_sims, B=cfg.batch_games, E=cfg.e_cap,
                         tau_plies=cfg.tau_plies, eps=cfg.eps, alpha=cfg.alpha,
                         terminal_collapse=cfg.terminal_collapse,
                         gumbel=cfg.gumbel, gumbel_m=cfg.gumbel_m,
                         gumbel_scale=cfg.gumbel_scale),
            evaluate=self.packed.evaluate(), env=self.env, device=device,
            seed=cfg.seed, check_invariants=False,
            **({"collect_stats": cfg.collect_search_stats} if cfg.impl == "cuda" else {}))
        self.search.weight_gen = self.weight_gen
        self.search.reset()

        path = (None if cfg.buffer_in_memory
                else os.path.join(paths.resolve(cfg.buffer_dir, run, "replay", create=True),
                                  f"{run}.dat"))
        self.buffer = ReplayBuffer(
            path=path, window_games=cfg.window_games, mean_plies=cfg.mean_plies,
            seed=cfg.seed, resume=resume is not None,
            value_subsample=cfg.value_subsample)
        self.buffer.open_games(cfg.batch_games)

        # §9's counters.
        self.step = 0
        self.carry = 0.0
        self.games_completed = 0
        self.positions_generated = 0
        self.samples_drawn = 0
        self.samples_dropped_filling = 0.0
        self.games_capped = 0
        self.generation = 0
        self.seconds = {"self_play": 0.0, "gradient": 0.0}
        # Built here rather than at the call site so the 20 000-puzzle load happens
        # once for the run. `run` returns None -- see `eval/watch.PuzzleProbe`.
        self.probe = None
        if cfg.puzzle_probe and cfg.puzzle_limit > 0:
            from brokefish.eval.watch import PuzzleProbe
            self.probe = PuzzleProbe(
                limit=cfg.puzzle_limit, device=self.device,
                detail_path=paths.artifact(run, f"{run}-puzzles.jsonl", create=True))

        self.t_start = time.time()

        if resume:
            self.load_checkpoint(resume)

    # -- §10 ---------------------------------------------------------------- #

    @property
    def training_seconds(self) -> float:
        """The curve's x-axis: self-play plus gradient, and nothing else (§10)."""
        return self.seconds["self_play"] + self.seconds["gradient"]

    def euros(self) -> dict:
        rate = self.cfg.euros_per_hour / 3600.0
        return {"training": self.training_seconds * rate,
                "total": (time.time() - self.t_start) * rate,
                "training_seconds": self.training_seconds,
                "wall_seconds": time.time() - self.t_start}

    # -- §8, the self-play phase --------------------------------------------- #

    def self_play_phase(self, moves: Optional[int] = None) -> dict:
        """`moves` move-steps over `B` games in flight, harvesting one record each."""
        cfg = self.cfg
        moves = cfg.moves_per_phase if moves is None else moves
        # §8.1. Before a single search runs, and it raises rather than warns.
        self.packed.assert_current(self.net, self.weight_gen)
        self.search.evaluate = self.packed.evaluate()
        self.search.weight_gen = self.weight_gen
        if cfg.collect_search_stats and hasattr(self.search, "reset_counters"):
            self.search.reset_counters()
        else:
            self.search.stats.reset()

        full = self._pcr_schedule(moves)
        # §11a, once per phase rather than per move: a phase is ~18 s of the axis the
        # schedule is keyed on, so finer granularity would buy nothing and would make
        # the caps change inside a phase whose counters are accumulated as one block.
        n_full, n_fast = cfg.caps_at(self.training_seconds)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        closed = 0
        added = 0
        stored = 0
        sims_total = 0
        with torch.no_grad():
            for m in range(moves):
                self.search.reset_finished()
                if full[m]:
                    record = self.search.self_play_move(sims=n_full)
                    sims_total += n_full
                else:
                    record = self.search.self_play_move(sims=n_fast, noise=False)
                    sims_total += n_fast
                done, result = self._apply_ply_cap(record)
                # ⚠️ Called on a cheap move too, with `store=False`. Skipping the call
                # would leave every game that ends on a cheap move — most of them —
                # unclosed, and its records would later be flushed under a different
                # game's result. See `ReplayBuffer.append`.
                closed += self.buffer.append(record, done=done, result=result,
                                             store=full[m])
                # One row per game in flight, per move-step. This is what the §6
                # cadence rides: the *data production rate*, which is constant, rather
                # than the games-closed rate, which falls as games lengthen.
                #
                # ⚠️ **Positions generated, not positions stored**, and under §11 those
                # differ by 1/`pcr_p`. Riding the generated count is what holds the
                # gradient steps per generation fixed against a uniform-`n` control, so
                # a difference in the curve is the data and not the step count; the
                # price is that each *stored* position is trained on `1 / pcr_p` times
                # more often, which is `buffer/reuse` in the log and is part of the
                # intervention rather than an accident.
                added += int(record.board.shape[0])
                stored += int(record.board.shape[0]) if full[m] else 0
                self.positions_generated += int(record.board.shape[0])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.seconds["self_play"] += dt
        self.games_completed += closed

        # ⚠️ `stats_snapshot` rather than `snapshot() | device_counters()`: the two
        # blocks name the same quantities differently, so merging them left the
        # kernel's live value *and* an unwritten torch field reading zero in the same
        # record. See `cuda_impl.COUNTER_ALIASES`.
        if cfg.collect_search_stats and hasattr(self.search, "stats_snapshot"):
            stats = dict(self.search.stats_snapshot())
        else:
            stats = dict(self.search.stats.snapshot())
        stats["games_capped"] = self.games_capped
        stats["seconds"] = dt
        stats["moves"] = moves
        stats["games_closed"] = closed
        stats["records_added"] = added
        # §11's three: what actually reached the buffer, how many turns paid the full
        # cap, and the realised mean budget — the cost axis, measured rather than
        # derived from `pcr_p`.
        stats["records_stored"] = stored
        stats["full_moves"] = int(sum(full))
        stats["sims_mean"] = sims_total / max(moves, 1)
        # §11a's caps as they actually were, so a schedule that failed to fire is
        # visible in the log rather than inferred from a cost that moved.
        stats["full_sims"] = n_full
        stats["fast_sims"] = n_fast
        return stats

    def _pcr_schedule(self, moves: int) -> list:
        """Which of this phase's ``moves`` turns get the full cap (§11).

        **Stratified, not i.i.d.**: exactly ``round(pcr_p * moves)`` of the phase's turns
        are full, with their positions drawn without replacement. Bernoulli draws would
        give the same turns in expectation but a per-generation cost that wanders by
        ±11 % at ``moves = 10``, and the cost is this run's x-axis. Redrawn every
        generation, so a game — whose ply offset within the phase is fixed by whenever
        its slot last reset — does not get a fixed residue class of its plies recorded
        for its whole life.

        Seeded from ``(seed, generation)`` rather than from a carried generator, so §9's
        bit-exact resume gets this for free instead of through another piece of
        checkpoint state that can go stale.
        """
        cfg = self.cfg
        if cfg.pcr_p <= 0.0:
            return [True] * moves
        k = max(1, min(moves, int(round(cfg.pcr_p * moves))))
        rng = np.random.default_rng((cfg.seed, self.generation))
        out = [False] * moves
        for i in rng.choice(moves, size=k, replace=False):
            out[int(i)] = True
        return out

    def _apply_ply_cap(self, record):
        """§5.4: 512 plies, scored as a draw, counted as a completed game.

        Our rules are stricter than AZ's — we implement the fifty-move rule and
        threefold — so this should fire rarely, and its rate is a logged counter
        rather than an assumption. A game that never terminates otherwise leaks its
        whole pending list, which is the reason the cap exists at all.
        """
        over = (self.search.game_ply >= self.cfg.max_plies) & ~record.done
        n = int(over.sum())
        if n:
            self.games_capped += n
            record.done = record.done | over
            record.result = torch.where(over, torch.zeros_like(record.result),
                                        record.result)
            self.search.game_done = self.search.game_done | over
            self.search.game_result = torch.where(
                over, torch.zeros_like(self.search.game_result), self.search.game_result)
        return record.done, record.result

    # -- §6, how much training per game -------------------------------------- #

    def steps_owed(self, records: int) -> int:
        """§6's reuse factor, carried so the long-run ratio is exact.

        ``records`` is the number of positions the self-play phase just appended, not
        the number of games it closed. See ``samples_per_position``: at a fixed
        per-*game* rate the reuse tracks ``65.2 / mean_plies`` and therefore drifts
        with the game length, which is what run `t4h-n64` measured falling 0.641 to
        0.374. Riding the records holds it constant.

        The carry is still needed even though the record count per phase is fixed
        today: ``moves_per_phase x games_in_flight`` is config, a phase can be short
        at the end of a run, and ``0.815 x 10240 / 4096 = 2.04`` is not an integer.
        """
        cfg = self.cfg
        self.carry += cfg.samples_per_position * records
        if self.buffer.n_records < max(cfg.batch, cfg.min_records):
            # §5.5: before the buffer holds one full batch of *sampleable* records the
            # loop is pure self-play. The carry is dropped rather than banked, or the
            # first gradient phase would take a hundred steps over a handful of games.
            #
            # ⚠️ **`cfg.batch` alone is the wrong floor once the cadence is high**, and
            # it fails in the direction that hides itself. The guard scales *with* the
            # batch, so a small-batch high-reuse recipe gets a smaller floor exactly
            # when it needs a larger one. Measured 2026-08-12 on a probe at
            # AlphaGateau's settings (batch 256, 7.6 samples/position): training began
            # at 256 records and took **304 steps over a 465-record buffer** by
            # generation 3, with loss collapsing to 1.68 and KL to 0.17 -- that is
            # memorising a few hundred positions, not learning, and every counter says
            # the run is going well. `min_records` defaults to 0, so no run measured
            # before this moves.
            self.samples_dropped_filling += self.carry
            self.carry = 0.0
            return 0
        steps = int(self.carry // cfg.batch)
        self.carry -= steps * cfg.batch
        return steps

    # -- §7, the gradient phase ---------------------------------------------- #

    def train_step(self, batch: Optional[TrainBatch] = None) -> dict:
        """One optimiser step over `batch`, or over a fresh draw from the buffer.

        Passing a batch is what :mod:`brokefish.train.overfit` does — §12 check 1 is
        this exact path over a frozen sample, which is the point: a check that ran a
        different code path would prove nothing about this one.
        """
        cfg = self.cfg
        lr = cfg.lr_at(self.step)
        # `lr_scale` is 1.0 for every group SGD and AdamW build, so this is the old
        # line for them. Muon uses it to hold its auxiliary AdamW group at a different
        # rate from the orthogonalised group while one schedule -- one warmup, one
        # cosine -- still drives both.
        for group in self.opt.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

        if batch is None:
            batch = self.buffer.sample(cfg.batch, device=self.device)
        self.samples_drawn += len(batch)
        if cfg.audit_every and self.step % cfg.audit_every == 0:
            audit_labels(batch, self.env)
        self.opt.zero_grad(set_to_none=True)

        keys = ("policy", "value", "kl", "entropy", "total")
        parts_sum = None
        total_n = len(batch)
        for i in range(math.ceil(total_n / cfg.micro_batch)):
            mb = batch.slice(i * cfg.micro_batch, (i + 1) * cfg.micro_batch)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=cfg.autocast):
                parts = az_loss(self.fwd, mb, strict=cfg.strict_labels,
                                value_weight=cfg.value_weight)
            # §7.2: four micro-batch means each scaled by 1/4 sum to the gradient of
            # the mean over 4096. The network is pre-norm LayerNorm with no batch
            # statistics anywhere, so this is batch 4096 and not an approximation.
            scaled = len(mb) / total_n
            (parts.total * scaled).backward()
            keep = {k: getattr(parts, k).detach() * scaled for k in keys}
            # Over the whole batch, not just the last micro-batch: saturation is the
            # mechanism that killed the value head at lr = 0.2 and a quarter of the
            # batch is a quarter of the evidence.
            #
            # ⚠️ Under `--value-classes 3` this counter keeps its name and changes its
            # meaning: `value_pred` is `p(win) - p(loss)`, so |v| > 0.99 is a confident
            # classifier, not a saturated tanh. It is still worth watching -- a
            # classifier that puts 0.995 on one class has the same vanishing-gradient
            # problem by a different route -- but it is **not** the same number as the
            # one on the ledger, and `gradient/value` is not either (MSE against a
            # 3-class cross-entropy). See `az_loss`.
            v = parts.value_pred
            keep["value_saturated_frac"] = (v.abs() > 0.99).float().mean() * scaled
            keep["value_mean"] = v.mean() * scaled
            # ⚠️ The pooled head's input is **not** unit-scale: `norm_f` normalises
            # each token, the mean of normed vectors is not normed, and its magnitude
            # moves with how aligned the tokens are and with how many pieces are alive.
            # This is the cheap downstream proxy for that drift.
            if parts.value_logit_rms is not None:
                keep["value_logit_rms"] = parts.value_logit_rms * scaled
            parts_sum = keep if parts_sum is None else {
                k: parts_sum[k] + keep[k] for k in keep}

        grad_norm = torch.sqrt(sum((p.grad.float() ** 2).sum()
                                   for p in self.net.parameters() if p.grad is not None))
        weight_norm = torch.sqrt(sum((p.detach().float() ** 2).sum()
                                     for p in self.net.parameters()))
        # Clip after `grad_norm` is read, so the logged number stays the pre-clip one
        # and remains comparable between an arm that clips and an arm that does not.
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
        self.opt.step()
        self.step += 1

        out = {k: float(v) for k, v in parts_sum.items()}
        out["l2"] = float(l2_penalty(self.net, cfg.l2))
        out["lr"] = lr
        out["grad_norm"] = float(grad_norm)
        out["weight_norm"] = float(weight_norm)
        out["staleness"] = float((self.weight_gen - batch.weight_gen.float()).mean())
        out.update(self._head_health())
        return out

    def _head_health(self) -> dict:
        """§11's "which head has stopped learning", for free.

        AZ weights the two terms 1:1 and AGZ justifies it by the rewards being unit
        scaled — which holds for us, `tanh` into `[-1, 1]` against a target in
        `{-1, 0, +1}`. ⚠️ **Equal loss magnitudes are not equal gradient magnitudes.**
        Measured 2026-07-31 over five seeded initialisations on one fixed batch, the
        policy term's global gradient norm sat at 12-16 every time while the value
        term's ranged over **1.7 to 26.4** — a 14× spread driven entirely by where the
        `tanh` happens to start. So the term that can quietly stop contributing is the
        value one, and this is the counter that says when it has.

        A true per-term decomposition needs a second backward per micro-batch, which
        doubles the phase. These are the three head weights' own gradients, which are
        already computed; the saturation fraction is accumulated in the micro-batch
        loop above.
        """
        out = {}
        for name in ("policy", "value", "promo"):
            grad = getattr(self.net, name).weight.grad
            out[f"grad_{name}_head"] = 0.0 if grad is None else float(grad.norm())
        # ⚠️ `promo` is legitimately zero on most batches: it only receives gradient
        # from positions with a promotion available, and random play produces almost
        # none (D1 found 0 underpromotions in 170,924 positions; a 128-position batch
        # at 24 random plies had 0 promotion edges). Zero here is a fact about chess.
        # Zero *forever*, once the policy sharpens and pawns start queening, would not
        # be — and the head is wired correctly: on `8/6P1/8/8/8/8/8/k6K w`, where one
        # promotion is available, `promo.weight` takes a gradient norm of 16.4.
        return out

    def gradient_phase(self, steps: int) -> dict:
        """`steps` optimiser steps, then publish the weights (§8.1)."""
        if steps <= 0:
            return {"steps": 0}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = {}
        for _ in range(steps):
            last = self.train_step()
            self.log.log({"train": last}, step=self.step)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.seconds["gradient"] += dt

        self.weight_gen += 1
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=self.cfg.encoder,
                                          fp8=self.cfg.fp8, int8=self.cfg.int8)
        last.update({"steps": steps, "seconds": dt,
                     "positions_per_s": steps * self.cfg.batch / max(dt, 1e-9)})
        return last

    # -- §12 check 9 ---------------------------------------------------------- #

    def check_weights_propagate(self, tol: float = 3e-2) -> dict:
        """Take a step, rebuild the packed encoder, require the fused output to move.

        ⚠️ Every other check in §12 passes on a loop whose self-play is frozen at
        generation 0. This one does not, which is why it runs at startup rather than
        living only in the test file.
        """
        boards, control = self.env.initial_boards(8, device=self.device)
        rep = torch.zeros(8, dtype=torch.uint8, device=self.device)
        before = self.packed.encoder.forward_full(boards, control, rep)[0].float().clone()

        # A diagnostic, never a training step: the perturbation is undone below and
        # the optimiser never sees it, so running this at startup does not move the
        # initialisation the run is supposed to begin from.
        saved = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        try:
            with torch.no_grad():
                for p in self.net.parameters():
                    p.add_(torch.randn_like(p) * 0.02)
            packed = PackedWeights.pack(self.net, self.weight_gen + 1, impl=self.cfg.encoder,
                                          fp8=self.cfg.fp8, int8=self.cfg.int8)
            after = packed.encoder.forward_full(boards, control, rep)[0].float().clone()
            with torch.no_grad():
                ref = self.net(boards, control, rep)[0].float()
        finally:
            self.net.load_state_dict(saved)
        moved = float((after - before).abs().max())
        agree = float((after - ref).abs().max())
        scale = float(ref.abs().max())
        if moved == 0.0:
            raise AssertionError(
                "train.md §12 check 9: the fused encoder's output did not change after "
                "a weight update. Self-play is running a stale snapshot (§8.1)")
        # ⚠️ **Explicitly, because `nan > tol` is False.** Without this a fused encoder
        # producing NaN passes check 9 in silence and the run generates garbage for
        # hours. That is not hypothetical: the fp8 FFN produced NaN on 116 boards of
        # 128 during its integration, from a `q_max` that disagreed across two
        # languages, and every isolated component tested clean.
        if not (math.isfinite(moved) and math.isfinite(agree)):
            raise AssertionError(
                f"train.md §12 check 9: the fused encoder produced a non-finite output "
                f"(moved={moved}, agree={agree}). fp16 has no saturating mode, so this "
                f"is an overflow somewhere in the stack rather than a precision loss")
        # §fp8 is a *deliberately* lower-precision path, so it is held to a bar scaled
        # to the logits rather than to fp16's absolute one. Measured at initialisation
        # on 2026-08-04: fp16 disagrees by 0.19 % of the logit maximum and fp8 by
        # 3.28 %, so 8 % leaves fp8 a 2.4x margin while still catching a stale snapshot
        # or a wrong weight slab, both of which are order-one errors. The fp16 bar is
        # untouched: it is the one every existing run was started under.
        if self.cfg.fp8 or self.cfg.int8:
            tol = max(tol, 0.08 * scale)
        if agree > tol:
            raise AssertionError(
                f"train.md §12 check 9: the rebuilt fused encoder disagrees with the "
                f"updated torch model by {agree:.4g} (tolerance {tol}). test_b2.py holds "
                f"the two paths together at a fixed weight set; this is the same check "
                f"after an update, which is the one it does not cover")
        return {"moved": moved, "agree": agree, "tol": tol}

    # -- §9 ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        s = self.search
        return {
            "config": asdict(self.cfg), "config_hash": self.cfg.hash(),
            "net": self.net.state_dict(), "opt": self.opt.state_dict(),
            "step": self.step, "weight_gen": self.weight_gen,
            "generation": self.generation, "carry": self.carry,
            "games_completed": self.games_completed,
            "positions_generated": self.positions_generated,
            "samples_drawn": self.samples_drawn,
            "samples_dropped_filling": self.samples_dropped_filling,
            "games_capped": self.games_capped,
            "lr": self.cfg.lr_at(self.step),
            "fingerprint": weight_fingerprint(self.net),
            "seconds": dict(self.seconds), "euros_per_hour": self.cfg.euros_per_hour,
            # Every RNG state (§9). The search's drives the Dirichlet root noise and
            # the multinomial move sampling below tau_plies; the buffer's rides in
            # the buffer snapshot, which is written separately and less often.
            "search_rng": s._gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all(),
            # The tree is rebuilt from scratch every move (mcts.md: a fresh tree per
            # move), so the whole of self-play's state is the games themselves.
            "games": {name: getattr(s, name) for name in (
                "game_board", "game_control", "game_hash", "game_ring",
                "game_ring_len", "game_ply", "game_done", "game_result")},
        }

    def save_checkpoint(self, path: str, with_buffer: bool = False) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        if with_buffer:
            self.buffer.save(path + ".buffer.npz")

    def load_checkpoint(self, path: str, allow_config_change: bool = False) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        if blob["config_hash"] != self.cfg.hash() and not allow_config_change:
            raise RuntimeError(
                f"the checkpoint was written under config {blob['config_hash']} and this "
                f"run is {self.cfg.hash()}. Resuming across a parameter change produces a "
                f"hybrid run whose curve means nothing (train.md §9); pass "
                f"--allow-config-change if that is really what you want")
        self.net.load_state_dict(blob["net"])
        self.opt.load_state_dict(blob["opt"])
        for key in ("step", "weight_gen", "generation", "carry", "games_completed",
                    "positions_generated", "samples_drawn", "samples_dropped_filling",
                    "games_capped"):
            setattr(self, key, blob[key])
        self.seconds = dict(blob["seconds"])
        # ⚠️ `.cpu()` on all three. `torch.load(map_location=cuda)` moves every tensor
        # in the blob, generator states included, and a generator refuses anything
        # but a CPU ByteTensor — including the state of a CUDA generator.
        self.search._gen.set_state(blob["search_rng"].cpu())
        torch.set_rng_state(blob["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([t.cpu() for t in blob["torch_cuda_rng"]])
        for name, tensor in blob["games"].items():
            getattr(self.search, name).copy_(tensor)
        if hasattr(self.search, "_refresh_tree"):
            self.search._refresh_tree()
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=self.cfg.encoder,
                                          fp8=self.cfg.fp8, int8=self.cfg.int8)

        got = weight_fingerprint(self.net)
        if got != blob["fingerprint"]:
            self.log.note(f"  ⚠️ weight fingerprint after load is {got!r}, checkpoint "
                          f"said {blob['fingerprint']!r}")

        meta = path + ".buffer.npz"
        if os.path.exists(meta):
            self.buffer.load(meta)
            self.log.note(f"  buffer restored: {self.buffer.n_games} games, "
                          f"{self.buffer.n_records} records, "
                          f"{self.buffer.pending_records} pending")
        else:
            # §9: a real discontinuity in the run, and it goes in the log.
            dropped = sum(self.buffer.drop_pending(i)
                          for i in range(self.cfg.batch_games))
            self.log.note(
                f"  ⚠️ no buffer snapshot at {meta}: the replay buffer restarts EMPTY "
                f"and {dropped} pending records were discarded. The games in flight "
                f"continue from the checkpoint and their remaining records are still "
                f"correctly valued, since §4's rule reads the side to move off each "
                f"record and counts nothing.")

    # -- the loop ------------------------------------------------------------ #

    def run_loop(self, generations: Optional[int] = None,
                 max_steps: Optional[int] = None, max_seconds: Optional[float] = None,
                 checkpoint_dir: Optional[str] = None) -> None:
        """Until whichever of the three limits comes first, then checkpoint.

        `checkpoint_dir=None` means `runs/<run>/checkpoints/`.

        ⚠️ A wall-clock limit stops at a **generation boundary**, so the run overruns
        by up to one phase. That is deliberate: a phase interrupted between self-play
        and its gradient steps would leave the §6 carry describing games the buffer
        has and the optimiser has not seen, which is recoverable but is a worse thing
        to hand a resume than fifteen extra seconds.
        """
        cfg = self.cfg
        started = time.time()
        self.log.note(f"  §12 check 9 at startup: {self.check_weights_propagate()}")
        checkpoint_dir = paths.resolve(checkpoint_dir, self.run, "checkpoints",
                                       create=True)
        next_ckpt = self.step + cfg.checkpoint_every
        next_snap = self.step + cfg.buffer_snapshot_every
        g = 0
        while True:
            if generations is not None and g >= generations:
                break
            if max_steps is not None and self.step >= max_steps:
                break
            if max_seconds is not None and time.time() - started >= max_seconds:
                self.log.note(f"  wall-clock limit reached after {g} generations")
                break
            play = self.self_play_phase()
            steps = self.steps_owed(play["records_added"])
            grad = self.gradient_phase(steps)
            self.generation += 1
            g += 1

            buf = self.buffer.stats()
            self.buffer.check()
            # §5's live index: ~320 KB written atomically every generation, so a
            # second process can read the buffer while the run is going. Without it
            # the ring's offsets exist only in this process's memory and the 735 MB
            # mapping is unreadable — which is what a Ctrl-C would have left behind
            # on 2026-07-31, since `save()` only fires every `buffer_snapshot_every`
            # steps. Cheap enough to be unconditional.
            if self.buffer.path:
                self.buffer.save_index(os.path.splitext(self.buffer.path)[0] + ".index.npz")
            euros = self.euros()
            # ⚠️ **No `phase` wrapper.** wandb groups metrics on the *first* path
            # component only, so nesting everything under `phase` put every metric in
            # the run into one useless folder and threw away the grouping that
            # `self_play` / `gradient` / `buffer` / `euros` would have given for free.
            # Removed 2026-07-31; this changes the JSONL keys too (`phase/gradient/kl`
            # is now `gradient/kl`), so a reader of a pre-2026-07-31 log needs the old
            # names.
            self.log.log({"generation": self.generation, "self_play": play,
                          "gradient": grad, "buffer": asdict(buf),
                          "euros": euros}, step=self.step)
            # A phase with no gradient step has no loss to report, which is the
            # normal state until §5.5's startup threshold is met. Printing a dash
            # rather than a formatted absence, because a column of `nan` in a
            # training log reads as divergence and is the wrong thing to shrug at.
            def num(key: str, fmt: str = "9.4f") -> str:
                return f"{grad[key]:{fmt}}" if key in grad else "—".rjust(int(fmt.split('.')[0]))

            # §11's realised cap, on the console rather than only in the JSONL: it is
            # the cost axis, and a schedule that quietly stopped varying would look
            # like a training result rather than a bug.
            pcr = f"  sims {play['sims_mean']:5.1f}" if cfg.pcr_p else ""
            self.log.note(
                f"  gen {self.generation:5d}  step {self.step:7d}  "
                f"games {self.games_completed:8d}  buf {buf.games:7d}g/"
                f"{buf.records:10d}r{pcr}  loss {num('total')}  kl {num('kl')}  "
                f"v {num('value')}  lr {cfg.lr_at(self.step):.4g}  "
                f"{play['seconds']:.1f}s play / {grad.get('seconds', 0.0):.1f}s grad  "
                f"€{euros['training']:.2f}")

            if self.step >= next_ckpt:
                # The buffer is ~13 GB against ~50 MB of weights, so it is
                # checkpointed separately and less often (§9).
                with_buffer = self.step >= next_snap
                self.save_checkpoint(os.path.join(checkpoint_dir, f"{self.run}.pt"),
                                     with_buffer=with_buffer)
                # ⚠️ **A rolling checkpoint has no history, and history is the
                # deliverable.** `{run}.pt` is overwritten every time, so run `c2-8h`
                # produced 2,827 steps and exactly one network — and the hypothesis
                # it was meant to test (did skill peak early and decline?) cannot be
                # asked of a single endpoint. These snapshots are the *weights only*:
                # 26 MB against the full state's 78, because a frozen opponent in a
                # match or a point on the cost-Elo curve needs a network and nothing
                # else. Resume still comes from `{run}.pt`.
                if cfg.keep_checkpoints:
                    torch.save(self.net.state_dict(),
                               os.path.join(checkpoint_dir,
                                            f"{self.run}-{self.step:06d}.pt"))
                # ⚠️ The return value is discarded because there is none: the probe
                # logs directly and returns None, so no evaluation score exists in
                # this process for anything to select on (evaluation.md §2).
                if self.probe is not None:
                    self.probe.run(self.net, self.log, self.step)
                next_ckpt = self.step + cfg.checkpoint_every
                if with_buffer:
                    next_snap = self.step + cfg.buffer_snapshot_every
        self.save_checkpoint(os.path.join(checkpoint_dir, f"{self.run}.pt"),
                             with_buffer=True)


# -- CLI ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="the C2 training loop, docs/train.md")
    p.add_argument("--run", default="c2")
    p.add_argument("--sims", type=int, default=TrainConfig.n_sims)
    p.add_argument("--games", type=int, default=TrainConfig.batch_games)
    p.add_argument("--moves-per-phase", type=int, default=TrainConfig.moves_per_phase)
    p.add_argument("--samples-per-position", type=float,
                   default=TrainConfig.samples_per_position,
                   help="§6's reuse factor: how many times each generated position is "
                        "trained on. 0.815 is AZ's 65.2-per-game at an assumed 80-ply "
                        "game. Pass 65.2/mean_plies to reproduce the old per-game rule")
    p.add_argument("--fp8", action="store_true",
                   help="self-play the FFN in e4m3 (inference only; the gradient step "
                        "is unchanged). ~1.15x encoder throughput for ~1 %% prior error")
    p.add_argument("--int8", action="store_true",
                   help="self-play the FFN in int8 instead of e4m3. Same two matmuls, "
                        "inference only, and better on both axes: 1.016x e4m3's "
                        "throughput at 2.56x lower max prior error")
    p.add_argument("--terminal-collapse", action="store_true",
                   help="§6.6a: send a node's simulations to its proved-winning edges "
                        "and store the target as a point mass on them. Off by default "
                        "-- it changes both the move played and the training target, "
                        "so a run with it is not comparable to one without")
    p.add_argument("--gumbel", action="store_true",
                   help="§11: Gumbel MuZero root sampling, sequential halving and a "
                        "completed-Q policy target instead of N/n. Replaces the "
                        "Dirichlet noise and --tau-plies both. Needs --impl torch "
                        "until descent_kernel learns it")
    p.add_argument("--gumbel-m", type=int, default=TrainConfig.gumbel_m,
                   help="root actions sampled without replacement (default 16)")
    p.add_argument("--gumbel-scale", type=float, default=TrainConfig.gumbel_scale,
                   help="scale on the root Gumbel noise; 0 is deterministic play, "
                        "which is what evaluation uses")
    p.add_argument("--pcr-p", type=float, default=TrainConfig.pcr_p,
                   help="§11 playout cap randomisation: the proportion of turns given "
                        "the full --sims cap and recorded for training. The rest run "
                        "--pcr-fast-sims with no root noise and are not recorded. 0 is "
                        "off. ⚠️ --sims is then the FULL cap, not the mean; "
                        "⚠️ --window-games is in games and only ~pcr_p of each game's "
                        "plies reach the buffer, so raise it or the window shrinks by "
                        "1/pcr_p without saying so")
    p.add_argument("--pcr-fast-sims", type=int, default=TrainConfig.pcr_fast_sims,
                   help="§11: the cheap cap, used on 1 - pcr_p of turns")
    p.add_argument("--sims-schedule", default=None, metavar="SPEC",
                   help="§11a: anneal the caps, as 'SECONDS:N/n,SECONDS:N/n' in "
                        "**training seconds** (self-play + gradient, the curve's own "
                        "x-axis -- not wall clock, which includes the puzzle probe). "
                        "KataGo §3.1 anneals (600,100) -> (1000,200) after two days; "
                        "e.g. '0:256/64,21600:512/128' doubles both at 6 h. "
                        "⚠️ --sims is raised to the largest N automatically, because "
                        "it allocates the node pool for the whole run")
    p.add_argument("--window-games", type=int, default=TrainConfig.window_games)
    p.add_argument("--mean-plies", type=int, default=TrainConfig.mean_plies)
    p.add_argument("--max-plies", type=int, default=TrainConfig.max_plies)
    p.add_argument("--batch", type=int, default=TrainConfig.batch)
    p.add_argument("--micro-batch", type=int, default=TrainConfig.micro_batch)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the gradient step's forward. ~1.25x at batch "
                        "256, measured 2026-08-12; changes training numerics, so a "
                        "compiled run is not bit-comparable with an earlier one")
    p.add_argument("--min-records", type=int, default=TrainConfig.min_records,
                   help="records the buffer must hold before the gradient phase "
                        "starts. 0 uses --batch alone, which is the historical "
                        "behaviour and too small for a high-reuse cadence")
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--lr", type=float, default=None,
                   help="override the whole §7.3 schedule with one constant rate")
    p.add_argument("--aux-lr", type=float, default=TrainConfig.aux_lr,
                   help="muon only: the rate for the 91,904 parameters Muon does not "
                        "touch. Held as a ratio to --lr, so the schedule drives both")
    p.add_argument("--head-group", type=int, default=TrainConfig.muon_head_group,
                   choices=(1, 2, 4, 8),
                   help="muon only: heads per orthogonalisation block. 8 = no head "
                        "split, 1 = per-head (Kimi K3 / GLM-5 'Muon Split')")
    p.add_argument("--no-qkv-split", action="store_true",
                   help="muon only: orthogonalise the fused (768,256) QKV as one "
                        "tensor. Jordan's post reports this is worse; it is here as "
                        "an ablation, not as an option")
    p.add_argument("--ns-scheme", default=TrainConfig.muon_ns_scheme,
                   choices=("polar", "jordan"),
                   help="muon only: Newton-Schulz coefficients. polar = Polar Express "
                        "per-iteration minimax (default), jordan = the fixed quintic "
                        "torch.optim.Muon uses")
    p.add_argument("--normuon", action="store_true",
                   help="muon only: per-neuron second moment. ⚠️ reconstructed from a "
                        "search summary, not the paper -- see train/muon.py")
    p.add_argument("--optimizer", default=TrainConfig.optimizer,
                   choices=("sgd", "adamw", "muon"),
                   help="sgd is AGZ's; adamw is ablation 1")
    p.add_argument("--aux-wd", type=float, default=None,
                   help="weight decay for the AdamW matrices under muon -- the five "
                        "embedding tables and the three readout heads. Defaults to "
                        "--adam-wd, which is what every run before 2026-08-17 used and "
                        "which decays value.weight, a single row of 256, as hard as a "
                        "trunk matrix")
    p.add_argument("--adam-wd", type=float, default=TrainConfig.adam_wd,
                   help="AdamW's decoupled decay -- NOT --l2, see build_optimizer")
    p.add_argument("--value-head", choices=("king", "pooled", "prenorm"),
                   default=TrainConfig.value_head,
                   help="where the value head reads. 'king' (default) is spec 7.4's "
                        "row select of the side-to-move king, predicting the MOVER's "
                        "result. 'pooled' is the masked mean over every live token of "
                        "both colours, predicting WHITE's -- W/D/B -- which is flipped "
                        "into the mover's frame inside the head. The pooled head sends "
                        "value gradient into every token instead of one. 'prenorm' "
                        "pools the RAW residual and applies norm_f to the pooled "
                        "vector, so the head's input scale is fixed by the norm rather "
                        "than drifting with how spread the token cloud is -- measured "
                        "on t12h-wdb, |mean(LN(h))| fell 15.47 -> 10.86 over a run "
                        "while |value.weight| grew 42 %.")
    p.add_argument("--value-classes", type=int, default=TrainConfig.value_classes,
                   choices=(1, 3),
                   help="the value head's shape. 1 (default) is spec 7.4's scalar "
                        "tanh trained on (z - v)^2, which every number on the ledger "
                        "was measured with. 3 is a win/draw/loss classifier trained "
                        "with a cross-entropy; it still gives the search a single "
                        "scalar p(win) - p(loss), so only the loss and the packed head "
                        "matrix change. WARNING: --value-weight is untuned for it.")
    p.add_argument("--value-weight", type=float, default=TrainConfig.value_weight,
                   help="weight on the value term against the policy term. 1.0 is "
                        "AlphaZero's; AlphaGateau's code uses optax.l2_loss = "
                        "0.5*(x-y)^2, so 0.5 reproduces what they ran")
    p.add_argument("--value-subsample", type=int, default=TrainConfig.value_subsample,
                   help="take the value loss over 1 in k records of each game (every "
                        "k-th ply), the policy loss over all of them. Moves how often "
                        "one game's outcome bit is shown to the value head and nothing "
                        "else. 1 = every record, the historical loss")
    p.add_argument("--betas", type=float, nargs=2, default=list(TrainConfig.betas))
    p.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip,
                   help="global grad-norm clip; 0 disables (AGZ specifies none)")
    p.add_argument("--warmup", type=int, default=TrainConfig.warmup_steps,
                   help="linear lr warmup over this many optimiser steps; 0 disables. "
                        "Adam needs it, SGD does not -- see lr_at")
    # The self-generated diversity knobs. Allowed by the tabula rasa boundary
    # (CLAUDE.md) precisely because none of them is an opinion about chess.
    p.add_argument("--tau-plies", type=int, default=TrainConfig.tau_plies,
                   help="plies of tau=1 sampling before argmax takes over. ⚠️ AGZ says "
                        "'the first 30 moves', a ply in Go and a pair in chess")
    p.add_argument("--eps", type=float, default=TrainConfig.eps,
                   help="Dirichlet mixing weight at the root")
    p.add_argument("--alpha", type=float, default=TrainConfig.alpha,
                   help="Dirichlet concentration")
    p.add_argument("--decay", default=TrainConfig.decay, choices=("step", "cosine"),
                   help="step is AGZ's three drops; cosine decays smoothly to --lr-min "
                        "at --total-steps, which must then be the real step count")
    p.add_argument("--lr-min", type=float, default=TrainConfig.lr_min,
                   help="the floor a cosine decay lands on (0 = all the way down)")
    p.add_argument("--euros-per-hour", type=float, default=0.0)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--minutes", type=float, default=None,
                   help="stop at the first generation boundary past this wall clock")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="also write {run}-{step}.pt (weights only, 26 MB) at every "
                        "checkpoint, so skill can be plotted against step")
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every,
                   help="optimiser steps between checkpoints; a short run wants this "
                        "small or it never writes one")
    p.add_argument("--buffer-snapshot-every", type=int,
                   default=TrainConfig.buffer_snapshot_every)
    p.add_argument("--audit-every", type=int, default=TrainConfig.audit_every,
                   help="steps between §12 check 3's engine-side label audit; 0 is off")
    p.add_argument("--resume", default=None)
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--buffer-dir", default=None,
                   help="runs/<run>/replay by default")
    p.add_argument("--puzzle-limit", type=int, default=TrainConfig.puzzle_limit,
                   help="puzzles scored after each checkpoint; 0 disables the probe")
    p.add_argument("--checkpoints", default=None,
                   help="runs/<run>/checkpoints by default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impl", default="cuda", choices=("cuda", "torch"),
                   help="the engine and the search")
    p.add_argument("--encoder", default="cuda", choices=("cuda", "triton"),
                   help="the fused encoder self-play evaluates with; there is no "
                        "'torch' option, since the reference model returns fp32 and "
                        "the CUDA search reads fp16")
    p.add_argument("--no-wandb", action="store_true")
    # Online is the default: wandb persists to disk first and uploads from a retrying
    # background thread, so a dropped link stalls the sync, not the run. `offline` is
    # for a box with no credentials — `python -m wandb sync` pushes it later.
    p.add_argument("--wandb-mode", default="online", choices=("offline", "online"))
    p.add_argument("--nondeterministic", action="store_true",
                   help="faster, and §9's bit-exact resume no longer holds")
    p.add_argument("--smoke", action="store_true",
                   help="a small end-to-end run: n=32, 256 games, an in-memory buffer")
    return p


def parse_sims_schedule(spec: Optional[str]) -> Tuple[Tuple[float, int, int], ...]:
    """``'0:256/64,21600:512/128'`` to §11a's tuple. ``None`` is the empty schedule.

    Raises on anything it cannot read rather than falling back to a fixed cap: a
    mistyped schedule that silently becomes "no schedule" is a twelve-hour run that
    answers the wrong question and looks entirely normal while doing it.
    """
    if not spec:
        return ()
    out = []
    for part in spec.split(","):
        part = part.strip()
        try:
            at, caps = part.split(":")
            full, fast = caps.split("/")
            out.append((float(at), int(full), int(fast)))
        except ValueError as exc:
            raise ValueError(
                f"cannot read sims-schedule stage {part!r}: expected "
                f"'SECONDS:N/n', e.g. '21600:512/128'") from exc
    if out and out[0][0] != 0.0:
        raise ValueError(
            f"the first sims-schedule stage must start at 0 seconds, got {out[0][0]}; "
            f"otherwise the caps before it are the --sims/--pcr-fast-sims defaults and "
            f"the run has a stage nobody wrote down")
    return tuple(out)


def config_from_args(args) -> TrainConfig:
    cfg = TrainConfig(
        n_sims=args.sims, batch_games=args.games, moves_per_phase=args.moves_per_phase,
        samples_per_position=args.samples_per_position,
        window_games=args.window_games, mean_plies=args.mean_plies,
        max_plies=args.max_plies, batch=args.batch, micro_batch=args.micro_batch,
        compile=args.compile, min_records=args.min_records,
        value_weight=args.value_weight,
        value_classes=args.value_classes,
        value_subsample=args.value_subsample,
        value_head=args.value_head,
        total_steps=args.total_steps, euros_per_hour=args.euros_per_hour,
        buffer_dir=args.buffer_dir, seed=args.seed, impl=args.impl,
        encoder=args.encoder, deterministic=not args.nondeterministic,
        checkpoint_every=args.checkpoint_every,
        keep_checkpoints=args.keep_checkpoints,
        puzzle_limit=args.puzzle_limit,
        buffer_snapshot_every=args.buffer_snapshot_every, audit_every=args.audit_every,
        optimizer=args.optimizer, adam_wd=args.adam_wd, aux_wd=args.aux_wd,
        grad_clip=args.grad_clip,
        aux_lr=args.aux_lr, muon_head_group=args.head_group,
        muon_qkv_split=not args.no_qkv_split, normuon=args.normuon,
        muon_ns_scheme=args.ns_scheme,
        betas=tuple(args.betas), warmup_steps=args.warmup,
        decay=args.decay, lr_min=args.lr_min,
        tau_plies=args.tau_plies, eps=args.eps, alpha=args.alpha,
        terminal_collapse=args.terminal_collapse, fp8=args.fp8, int8=args.int8,
        gumbel=args.gumbel, gumbel_m=args.gumbel_m, gumbel_scale=args.gumbel_scale,
        pcr_p=args.pcr_p, pcr_fast_sims=args.pcr_fast_sims,
        sims_schedule=parse_sims_schedule(args.sims_schedule))
    if cfg.sims_schedule:
        # ⚠️ Raised here rather than validated, because `n_sims` is a tree allocation
        # and not a training decision once a schedule exists: leaving it to the caller
        # means a run that anneals to 512 with `--sims 256` dies at the switch, six
        # hours in. The header prints what it became.
        # ⚠️ **Set to the schedule's maximum, not `max(--sims, schedule)`.** The first
        # version kept whichever was larger, so a schedule topping out at 512 run
        # without an explicit `--sims` inherited the 800 default and allocated a
        # 801-node pool for a run that never asks for more than 512 — a gigabyte of an
        # 8 GB card, for nothing, silently. Once a schedule exists it *is* the caps.
        cfg.n_sims = max(f for _, f, _ in cfg.sims_schedule)
        cfg.pcr_fast_sims = cfg.sims_schedule[0][2]
    if args.smoke:
        cfg.n_sims, cfg.batch_games, cfg.moves_per_phase = 32, 256, 8
        cfg.window_games, cfg.mean_plies, cfg.max_plies = 2000, 128, 160
        cfg.batch, cfg.micro_batch, cfg.total_steps = 256, 256, 200
        cfg.buffer_in_memory = True
        cfg.checkpoint_every, cfg.buffer_snapshot_every = 20, 20
    if args.lr is not None:
        cfg.lr_schedule = ((0.0, args.lr),)
    return cfg


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    logger = Logger(args.run, config=asdict(cfg), use_wandb=not args.no_wandb,
                    wandb_mode=args.wandb_mode, append=bool(args.resume))
    logger.note(f"brokefish C2 — docs/train.md, run {args.run!r}")
    logger.note(f"  config {cfg.hash()}, {logger.wandb_why}")
    logger.note(f"  tail -f {logger.text_path}   (records: {logger.jsonl_path})")

    trainer = Trainer(cfg, run=args.run, logger=logger, resume=args.resume)
    # The cadence, stated so a drifting reuse is visible at startup rather than
    # reconstructed from the logs afterwards (§6). `per_gen` is exact because the
    # record rate is `moves_per_phase x batch_games` and does not depend on how long
    # the games turn out to be -- which is the whole point of the change.
    per_gen = cfg.moves_per_phase * cfg.batch_games
    logger.note(f"  {cfg.n_sims} sims, {cfg.batch_games} games in flight, "
                f"batch {cfg.batch} in {math.ceil(cfg.batch / cfg.micro_batch)} "
                f"micro-batches")
    if cfg.sims_schedule:
        stages = "  ".join(
            f"{at / 3600:.1f}h:N={f}/n={s} (mean {cfg.pcr_p * f + (1 - cfg.pcr_p) * s:.0f})"
            for at, f, s in cfg.sims_schedule)
        logger.note(f"  §11a schedule, in TRAINING seconds: {stages}")
        logger.note(f"  ⚠️ the node pool is allocated for N = {cfg.n_sims} from "
                    f"generation 1, so the early stages do not use all of it")
    logger.note(f"  cadence {cfg.samples_per_position:.3f} samples/position -> "
                f"{per_gen} records and {per_gen * cfg.samples_per_position / cfg.batch:.2f} "
                f"steps per generation, constant in game length "
                f"(AZ's published ratio is {cfg.samples_per_game} per *game*, which is "
                f"the same thing only at {cfg.samples_per_game / cfg.samples_per_position:.0f} "
                f"plies)")
    # Print the rate the run will actually see. A cosine sized against the default
    # 159,000 steps in a run that takes 116 is a constant rate wearing a decay's name,
    # and this line is where that becomes obvious instead of being found afterwards.
    marks = [0, cfg.warmup_steps, cfg.total_steps // 4, cfg.total_steps // 2,
             (3 * cfg.total_steps) // 4, max(cfg.total_steps - 1, 0)]
    shape = "  ".join(f"{s}:{cfg.lr_at(s):.2e}" for s in sorted(set(marks)))
    logger.note(f"  {cfg.optimizer}, {cfg.decay} decay over {cfg.total_steps} steps, "
                f"warmup {cfg.warmup_steps}")
    if cfg.optimizer == "muon":
        n_muon = sum(p.numel() for g in trainer.opt.param_groups if g.get("use_muon")
                     for p in g["params"])
        total = sum(p.numel() for p in trainer.net.parameters())
        logger.note(
            f"  muon on {n_muon:,} params ({100 * n_muon / total:.1f} %), aux adamw on "
            f"{total - n_muon:,} at lr x {cfg.aux_lr / cfg.lr_schedule[0][1]:.4g}; "
            f"qkv_split={cfg.muon_qkv_split} head_group={cfg.muon_head_group} "
            f"ns_steps={cfg.muon_ns_steps}({cfg.muon_ns_scheme}) normuon={cfg.normuon}")
    logger.note(f"  lr at step  {shape}")
    try:
        trainer.run_loop(generations=args.generations, max_steps=args.max_steps,
                         max_seconds=None if args.minutes is None else args.minutes * 60,
                         checkpoint_dir=args.checkpoints)
    finally:
        logger.summary({"step": trainer.step, "games": trainer.games_completed,
                        "euros": trainer.euros()})
        logger.close()


if __name__ == "__main__":
    main()
