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

import torch

from brokefish.env import cuda_impl as _cuda_env
from brokefish.env import torch_impl as _torch_env
from brokefish.nn.model import BrokefishNet
from brokefish.search import SearchConfig, search_impl

from .buffer import ReplayBuffer
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
    e_cap: int = 64
    tau_plies: int = 30
    eps: float = 0.25
    alpha: float = 0.3

    # -- the buffer, §5
    window_games: int = 500_000
    mean_plies: int = 80
    max_plies: int = 512               # §5.4, scored drawn
    buffer_dir: str = "data/replay"
    buffer_in_memory: bool = False

    # -- the cadence, §6
    samples_per_game: float = 65.2

    # -- the optimiser, §7
    batch: int = 4096
    micro_batch: int = 1024
    momentum: float = 0.9
    l2: float = 1e-4
    lr_schedule: Tuple[Tuple[float, float], ...] = LR_SCHEDULE
    total_steps: int = 159_000

    # -- the loop, §8
    moves_per_phase: int = 4
    checkpoint_every: int = 1000
    buffer_snapshot_every: int = 10_000
    autocast: bool = True              # §8.2, bf16 forward and backward

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

    def lr_at(self, step: int) -> float:
        frac = step / max(self.total_steps, 1)
        lr = self.lr_schedule[0][1]
        for at, value in self.lr_schedule:
            if frac >= at:
                lr = value
        return lr


class Trainer:
    """One run. Owns the network, the search, the buffer, the optimiser and the clock."""

    def __init__(self, cfg: TrainConfig, run: str = "run", device: str = "cuda",
                 logger: Optional[Logger] = None, resume: Optional[str] = None,
                 log_dir: str = "logs") -> None:
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

        torch.manual_seed(cfg.seed)
        self.net = BrokefishNet().to(self.device)          # fp32 master weights, §8.2
        self.opt = torch.optim.SGD(
            self.net.parameters(), lr=cfg.lr_at(0), momentum=cfg.momentum,
            # ⚠️ 2c, not c. AGZ writes the penalty as `c||theta||^2` with no half,
            # so its gradient is `2 c theta`, and torch's weight_decay adds `w theta`.
            weight_decay=weight_decay_for(cfg.l2))

        self.env = ENVIRONMENTS[cfg.impl]
        self.weight_gen = 0
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=cfg.encoder)

        self.search = search_impl("cuda" if cfg.impl == "cuda" else "torch")(
            SearchConfig(n=cfg.n_sims, B=cfg.batch_games, E=cfg.e_cap,
                         tau_plies=cfg.tau_plies, eps=cfg.eps, alpha=cfg.alpha),
            evaluate=self.packed.evaluate(), env=self.env, device=device,
            seed=cfg.seed, check_invariants=False,
            **({"collect_stats": cfg.collect_search_stats} if cfg.impl == "cuda" else {}))
        self.search.weight_gen = self.weight_gen
        self.search.reset()

        path = None if cfg.buffer_in_memory else os.path.join(cfg.buffer_dir, f"{run}.dat")
        self.buffer = ReplayBuffer(
            path=path, window_games=cfg.window_games, mean_plies=cfg.mean_plies,
            seed=cfg.seed, resume=resume is not None)
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

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        closed = 0
        with torch.no_grad():
            for _ in range(moves):
                self.search.reset_finished()
                record = self.search.self_play_move()
                done, result = self._apply_ply_cap(record)
                closed += self.buffer.append(record, done=done, result=result)
                self.positions_generated += int(record.board.shape[0])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.seconds["self_play"] += dt
        self.games_completed += closed

        stats = dict(self.search.stats.snapshot())
        if cfg.collect_search_stats and hasattr(self.search, "device_counters"):
            stats.update(self.search.device_counters())
        stats["games_capped"] = self.games_capped
        stats["seconds"] = dt
        stats["moves"] = moves
        stats["games_closed"] = closed
        return stats

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

    def steps_owed(self, games: int) -> int:
        """AZ's 65.2 positions per game, carried so the long-run ratio is exact."""
        cfg = self.cfg
        self.carry += cfg.samples_per_game * games
        if self.buffer.n_records < cfg.batch:
            # §5.5: before the buffer holds one full batch of *sampleable* records the
            # loop is pure self-play. The carry is dropped rather than banked, or the
            # first gradient phase would take a hundred steps over a handful of games.
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
        for group in self.opt.param_groups:
            group["lr"] = lr

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
                parts = az_loss(self.net, mb, strict=cfg.strict_labels)
            # §7.2: four micro-batch means each scaled by 1/4 sum to the gradient of
            # the mean over 4096. The network is pre-norm LayerNorm with no batch
            # statistics anywhere, so this is batch 4096 and not an approximation.
            scaled = len(mb) / total_n
            (parts.total * scaled).backward()
            keep = {k: getattr(parts, k).detach() * scaled for k in keys}
            # Over the whole batch, not just the last micro-batch: saturation is the
            # mechanism that killed the value head at lr = 0.2 and a quarter of the
            # batch is a quarter of the evidence.
            v = parts.value_pred
            keep["value_saturated_frac"] = (v.abs() > 0.99).float().mean() * scaled
            keep["value_mean"] = v.mean() * scaled
            parts_sum = keep if parts_sum is None else {
                k: parts_sum[k] + keep[k] for k in keep}

        grad_norm = torch.sqrt(sum((p.grad.float() ** 2).sum()
                                   for p in self.net.parameters() if p.grad is not None))
        weight_norm = torch.sqrt(sum((p.detach().float() ** 2).sum()
                                     for p in self.net.parameters()))
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
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=self.cfg.encoder)
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
            packed = PackedWeights.pack(self.net, self.weight_gen + 1, impl=self.cfg.encoder)
            after = packed.encoder.forward_full(boards, control, rep)[0].float().clone()
            with torch.no_grad():
                ref = self.net(boards, control, rep)[0].float()
        finally:
            self.net.load_state_dict(saved)
        moved = float((after - before).abs().max())
        agree = float((after - ref).abs().max())
        if moved == 0.0:
            raise AssertionError(
                "train.md §12 check 9: the fused encoder's output did not change after "
                "a weight update. Self-play is running a stale snapshot (§8.1)")
        if agree > tol:
            raise AssertionError(
                f"train.md §12 check 9: the rebuilt fused encoder disagrees with the "
                f"updated torch model by {agree:.4g} (tolerance {tol}). test_b2.py holds "
                f"the two paths together at a fixed weight set; this is the same check "
                f"after an update, which is the one it does not cover")
        return {"moved": moved, "agree": agree}

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
        self.packed = PackedWeights.pack(self.net, self.weight_gen, impl=self.cfg.encoder)

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
                f"correctly valued, since §4's parity depends only on distance from "
                f"the end of the game.")

    # -- the loop ------------------------------------------------------------ #

    def run_loop(self, generations: Optional[int] = None,
                 max_steps: Optional[int] = None, max_seconds: Optional[float] = None,
                 checkpoint_dir: str = "checkpoints") -> None:
        """Until whichever of the three limits comes first, then checkpoint.

        ⚠️ A wall-clock limit stops at a **generation boundary**, so the run overruns
        by up to one phase. That is deliberate: a phase interrupted between self-play
        and its gradient steps would leave the §6 carry describing games the buffer
        has and the optimiser has not seen, which is recoverable but is a worse thing
        to hand a resume than fifteen extra seconds.
        """
        cfg = self.cfg
        started = time.time()
        self.log.note(f"  §12 check 9 at startup: {self.check_weights_propagate()}")
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
            steps = self.steps_owed(play["games_closed"])
            grad = self.gradient_phase(steps)
            self.generation += 1
            g += 1

            buf = self.buffer.stats()
            self.buffer.check()
            euros = self.euros()
            self.log.log({"phase": {"generation": self.generation, "self_play": play,
                                    "gradient": grad, "buffer": asdict(buf),
                                    "euros": euros}}, step=self.step)
            # A phase with no gradient step has no loss to report, which is the
            # normal state until §5.5's startup threshold is met. Printing a dash
            # rather than a formatted absence, because a column of `nan` in a
            # training log reads as divergence and is the wrong thing to shrug at.
            def num(key: str, fmt: str = "9.4f") -> str:
                return f"{grad[key]:{fmt}}" if key in grad else "—".rjust(int(fmt.split('.')[0]))

            self.log.note(
                f"  gen {self.generation:5d}  step {self.step:7d}  "
                f"games {self.games_completed:8d}  buf {buf.games:7d}g/"
                f"{buf.records:10d}r  loss {num('total')}  kl {num('kl')}  "
                f"v {num('value')}  lr {cfg.lr_at(self.step):.4g}  "
                f"{play['seconds']:.1f}s play / {grad.get('seconds', 0.0):.1f}s grad  "
                f"€{euros['training']:.2f}")

            if self.step >= next_ckpt:
                # The buffer is ~13 GB against ~50 MB of weights, so it is
                # checkpointed separately and less often (§9).
                with_buffer = self.step >= next_snap
                self.save_checkpoint(os.path.join(checkpoint_dir, f"{self.run}.pt"),
                                     with_buffer=with_buffer)
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
    p.add_argument("--window-games", type=int, default=TrainConfig.window_games)
    p.add_argument("--mean-plies", type=int, default=TrainConfig.mean_plies)
    p.add_argument("--max-plies", type=int, default=TrainConfig.max_plies)
    p.add_argument("--batch", type=int, default=TrainConfig.batch)
    p.add_argument("--micro-batch", type=int, default=TrainConfig.micro_batch)
    p.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    p.add_argument("--lr", type=float, default=None,
                   help="override the whole §7.3 schedule with one constant rate")
    p.add_argument("--euros-per-hour", type=float, default=0.0)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--minutes", type=float, default=None,
                   help="stop at the first generation boundary past this wall clock")
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every,
                   help="optimiser steps between checkpoints; a short run wants this "
                        "small or it never writes one")
    p.add_argument("--buffer-snapshot-every", type=int,
                   default=TrainConfig.buffer_snapshot_every)
    p.add_argument("--audit-every", type=int, default=TrainConfig.audit_every,
                   help="steps between §12 check 3's engine-side label audit; 0 is off")
    p.add_argument("--resume", default=None)
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--buffer-dir", default=TrainConfig.buffer_dir)
    p.add_argument("--checkpoints", default="checkpoints")
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


def config_from_args(args) -> TrainConfig:
    cfg = TrainConfig(
        n_sims=args.sims, batch_games=args.games, moves_per_phase=args.moves_per_phase,
        window_games=args.window_games, mean_plies=args.mean_plies,
        max_plies=args.max_plies, batch=args.batch, micro_batch=args.micro_batch,
        total_steps=args.total_steps, euros_per_hour=args.euros_per_hour,
        buffer_dir=args.buffer_dir, seed=args.seed, impl=args.impl,
        encoder=args.encoder, deterministic=not args.nondeterministic,
        checkpoint_every=args.checkpoint_every,
        buffer_snapshot_every=args.buffer_snapshot_every, audit_every=args.audit_every)
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
    logger.note(f"  {cfg.n_sims} sims, {cfg.batch_games} games in flight, "
                f"batch {cfg.batch} in {math.ceil(cfg.batch / cfg.micro_batch)} "
                f"micro-batches, {cfg.samples_per_game} samples per game")
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
