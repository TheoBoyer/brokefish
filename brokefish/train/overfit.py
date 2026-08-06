"""§12 check 1: overfit one batch.

``train.md`` §12: *"4096 fixed positions, no buffer, no self-play, train to near-zero
loss. Exercises the entire forward/backward/optimiser/accumulation path. If it cannot
memorise 4096 positions, nothing downstream is worth running. Also the fastest
detector of a wrong ``lr``."*

It is first in the list because C2 has no oracle and this is the closest thing to one
that costs minutes: the target is a definite number, zero, and every part of the
gradient path is on the way to it.

**The positions are real search output, not synthetic.** Self-play runs at a small
``n`` until enough games have *finished* — a record is not sampleable until its ``z``
exists (§5.3) — and then one batch is drawn and frozen. Fabricating targets would test
the optimiser and nothing else; harvesting them tests the label decode, the value
parity, the buffer and the record schema on the way in, all of which are §12's other
checks and all of which fail here as an infinite or stuck loss.

**Read the KL, not the cross-entropy.** §3.2: the two have identical gradients and
differ by ``H(pi)``, the target's own entropy, which is a per-sample constant. So the
cross-entropy floor is ``H(pi) > 0`` and varies batch to batch, while the KL goes to
zero at a perfect fit. A run that has memorised the batch shows ``kl -> 0`` and
``value -> 0`` while ``policy`` sits at whatever ``H(pi)`` happens to be.

⚠️ **``lr = 0.2`` is AZ's value for a 46M-parameter convolutional resnet** and ours is
a 6.38M-parameter pre-norm transformer initialised at ``std = 1/sqrt(d)``. Nothing
about that tuning transfers, §7.3 says so, and ``--sweep`` is the intended first move.

Run from the repository root::

    python -m brokefish.train.overfit --run overfit
    python -m brokefish.train.overfit --run of-sweep --sweep 0.2,0.02,0.002
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import asdict

import torch

from .log import Logger
from .loop import build_optimizer, Trainer, TrainConfig


def harvest(trainer: Trainer, positions: int, max_moves: int, log) -> "object":
    """Self-play until the buffer holds `positions` sampleable records, then draw one.

    ⚠️ Sampleable, not generated. Games in flight hold their records outside the
    buffer until they end, so this waits for games to *finish*, which is why it runs
    for a hundred-odd move-steps rather than for ``positions / B`` of them.
    """
    t0 = time.perf_counter()
    for move in range(max_moves):
        stats = trainer.self_play_phase(moves=1)
        buf = trainer.buffer.stats()
        if move % 10 == 0 or buf.records >= positions:
            log(f"    ply {move + 1:4d}  {buf.games:5d} games / {buf.records:7d} records "
                f"sampleable, {buf.pending_records:7d} pending  "
                f"({time.perf_counter() - t0:6.1f} s)")
        if buf.records >= positions:
            break
    else:
        raise SystemExit(
            f"only {trainer.buffer.n_records} records after {max_moves} move-steps; "
            f"raise --max-moves or lower --positions")
    log(f"    mean finished-game length {trainer.buffer.stats().mean_game_length:.1f} plies, "
        f"{trainer.games_capped} games hit the {trainer.cfg.max_plies}-ply cap")
    return trainer.buffer.sample(positions, device=trainer.device)


def overfit(trainer: Trainer, batch, steps: int, lr: float, log, logger: Logger,
            tag: str = "") -> dict:
    """Train on one frozen batch and report the trajectory."""
    trainer.cfg.lr_schedule = ((0.0, lr),)
    trainer.step = 0
    first, last = None, None
    t0 = time.perf_counter()
    for i in range(steps):
        out = trainer.train_step(batch=batch)
        first = first or out
        last = out
        logger.log({"overfit": {**out, "lr_tag": lr}}, step=i)
        if i % 10 == 0 or i == steps - 1:
            log(f"    {tag}step {i:4d}  total {out['total']:9.4f}  kl {out['kl']:9.4f}  "
                f"policy {out['policy']:8.4f}  value {out['value']:8.4f}  "
                f"|g| {out['grad_norm']:9.3f}  |w| {out['weight_norm']:8.2f}")
        if not math.isfinite(out["total"]):
            log(f"    {tag}diverged at step {i}: loss is {out['total']}")
            break
    return {"lr": lr, "steps": steps, "seconds": time.perf_counter() - t0,
            "first": first, "last": last,
            "kl_drop": (first["kl"] - last["kl"]) if first and last else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="overfit")
    p.add_argument("--positions", type=int, default=4096, help="§12's number")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--micro-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.2, help="AZ's value; see §7.3")
    p.add_argument("--sweep", type=str, default="",
                   help="comma-separated learning rates, each from the same init "
                        "on the same batch")
    p.add_argument("--sims", type=int, default=16, help="simulations for the harvest")
    p.add_argument("--games", type=int, default=512, help="games in flight for the harvest")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--max-moves", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="sgd", choices=("sgd", "adamw", "muon"),
                   help="§12 check 1 exercises whichever one the run will use")
    p.add_argument("--aux-lr", type=float, default=TrainConfig.aux_lr,
                   help="muon only, see loop.py")
    p.add_argument("--head-group", type=int, default=TrainConfig.muon_head_group,
                   choices=(1, 2, 4, 8), help="muon only, see loop.py")
    p.add_argument("--no-qkv-split", action="store_true", help="muon only, see loop.py")
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--adam-wd", type=float, default=TrainConfig.adam_wd)
    p.add_argument("--impl", default="cuda", choices=("cuda", "torch"),
                   help="the engine and the search")
    p.add_argument("--encoder", default="cuda", choices=("cuda", "triton"),
                   help="the fused encoder the harvest self-plays with")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        n_sims=args.sims, batch_games=args.games, max_plies=args.max_plies,
        batch=args.positions, micro_batch=args.micro_batch,
        # In memory: a 13 GB mapping for a check that finishes in minutes is absurd,
        # and the harvest only ever holds a few hundred games.
        buffer_in_memory=True, window_games=100_000, mean_plies=args.max_plies,
        total_steps=1, lr_schedule=((0.0, args.lr),), seed=args.seed,
        impl=args.impl, encoder=args.encoder,
        optimizer=args.optimizer, aux_lr=args.aux_lr, adam_wd=args.adam_wd,
        grad_clip=args.grad_clip, muon_head_group=args.head_group,
        muon_qkv_split=not args.no_qkv_split,
        # The label check is the point of running this at all (§12 check 3): a
        # permuted target would memorise the batch just as happily.
        strict_labels=True, collect_search_stats=False, deterministic=True)

    logger = Logger(args.run, config=asdict(cfg), use_wandb=not args.no_wandb)
    log = logger.note
    log("brokefish §12 check 1 — overfit one batch (docs/train.md)")
    log(f"  tail -f {logger.text_path}")
    log(f"  {args.positions} positions, {args.steps} steps, micro-batch "
        f"{args.micro_batch} ({math.ceil(args.positions / args.micro_batch)} "
        f"accumulations per step), {logger.wandb_why}")

    trainer = Trainer(cfg, run=args.run, logger=logger)
    log(f"  §12 check 9 (weights propagate): {trainer.check_weights_propagate()}")
    log(f"  harvesting at n = {args.sims} over {args.games} games in flight")
    batch = harvest(trainer, args.positions, args.max_moves, log)

    z = batch.value
    log(f"  batch: z = {float((z > 0).float().mean()):.3f} win / "
        f"{float((z == 0).float().mean()):.3f} draw / "
        f"{float((z < 0).float().mean()):.3f} loss, "
        f"mean policy_len {float(batch.policy_len.float().mean()):.1f}")

    init = copy.deepcopy(trainer.net.state_dict())
    rates = [float(x) for x in args.sweep.split(",")] if args.sweep else [args.lr]
    results = []
    for lr in rates:
        # Every rate starts from the same initialisation and the same batch, so the
        # comparison is of the rate and of nothing else.
        trainer.net.load_state_dict(init)
        # ⚠️ Rebuild rather than restore a saved fresh state dict. The two were
        # equivalent for SGD and AdamW -- `opt_init` was captured before any step --
        # but Muon derives its auxiliary group's `lr_scale` from the schedule's peak
        # at construction, so a swept rate has to be visible to the constructor or
        # every arm of the sweep would run the auxiliary group at the first arm's rate.
        trainer.cfg.lr_schedule = ((0.0, lr),)
        trainer.opt = build_optimizer(trainer.net, trainer.cfg)
        log(f"\n  lr = {lr}")
        results.append(overfit(trainer, batch, args.steps, lr, log, logger,
                               tag=f"lr={lr} " if len(rates) > 1 else ""))

    log("\n" + "=" * 78)
    log(f"{'lr':>8} {'kl 0':>10} {'kl end':>10} {'value 0':>10} {'value end':>10} "
        f"{'total end':>10} {'s':>7}")
    for r in results:
        log(f"{r['lr']:8.4g} {r['first']['kl']:10.4f} {r['last']['kl']:10.4f} "
            f"{r['first']['value']:10.4f} {r['last']['value']:10.4f} "
            f"{r['last']['total']:10.4f} {r['seconds']:7.1f}")
    log("=" * 78)
    log("  KL, not cross-entropy, is the one with a target of zero (§3.2): the CE")
    log("  floor is H(pi), which this batch fixes and training cannot remove.")
    log("  A KL that does not fall means the gradient path is wrong or the rate is;")
    log("  a KL that falls to ~0 says the whole forward/backward/accumulate/step")
    log("  path memorises 4096 real search targets, which is what check 1 asks.")
    logger.summary({"overfit": results})
    logger.close()


if __name__ == "__main__":
    main()
