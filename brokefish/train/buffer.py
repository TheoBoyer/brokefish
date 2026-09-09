"""The replay buffer of ``docs/train.md`` §5.

The most recent 500,000 games, sampled uniformly at random over all positions in
them (AGZ Methods, Optimisation, inherited by AZ). At ~80 plies and 462 B per
record that is **13.2 GB**, which is why the store is a memory-mapped file rather
than a host allocation: 15.7 GB of RAM cannot hold it and the access pattern never
needed it to. One optimiser step draws 4096 records at ~1.2 steps/s, so the load is
**1.6 MB/s of random reads**, which the page cache absorbs almost entirely.

Three properties are the whole design.

**Eviction is by game, oldest first.** A game's records are therefore written as one
contiguous block, which also makes every write sequential. Games are `B` in flight
and finish at different plies, so arrival order is not game order — the reconciliation
is that a record does not enter the store at all until its game ends.

**A record is not sampleable until its ``value`` exists** (§5.3), which is when its
game ends. Holding the incomplete population outside the mapping rather than inside it
with a flag makes "no record is sampled before it is filled" structural instead of
asserted. The cost is host RAM for the games in flight: 4096 games at the 80-ply mean
is 110 MB, and the 512-ply cap of §5.4 bounds it at 700 MB.

**The value target is the game outcome, flipped by parity** (§4). ``MoveRecord.result``
is from the *new* mover's point of view after the move was played, so with ``L``
consecutive records in a finished game the rule reads::

    z(i) = r  if (L - i) is even, else -r

which is the same parity argument as ``mcts.md`` §6.5's backup flip and fails the same
way if inverted.

⚠️ **That form assumes the recorded plies are consecutive, and under playout cap
randomisation they are not** (`training.md` §11, KataGo §3.1: only full-search turns are
recorded). So the rule is expressed on the quantity it was always really about — *which
side is to move* — rather than on a position in the pending list::

    z = -r * sign(control_record) * sign(control_terminal)

``control_terminal`` is the control word of the position the game-ending move was played
from, which :meth:`ReplayBuffer.append` reads off the record of that very move. The two
forms agree exactly on a dense game (the last record *is* the terminal position, so its
own sign gives ``-r`` and every earlier one alternates), and only the second one survives
a game whose recorded plies are a sparse subset. It is also strictly more robust in the
case that motivated the old wording: a game whose earlier records were lost to a resume
still gets correct values, because nothing here counts anything.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .loss import TrainBatch

# §5.1. The offsets are packed rather than aligned so the record is exactly the
# 462 B the document sizes the window from; numpy handles the unaligned f4 fields
# by copying, and this is a random-read workload where the byte count is what costs.
# §5.1's policy width. **Raised 64 -> 96 on 2026-08-02, with `search.cuh`'s `kE`.**
#
# ⚠️ The two must move together. `training.md` §3.5 makes the record hold the root's
# *whole* edge set -- that is what lets the loss stop recomputing `movegen` to
# rediscover a support the search already knew -- and the claim is only literally
# true while `K_POLICY == E`. A narrower record would silently truncate the softmax
# denominator on exactly the wide roots where §4.3 already bites.
K_POLICY = 96

RECORD = np.dtype({
    "names": ["board", "control", "rep", "policy_len", "policy_move", "policy_prob",
              "value", "root_value", "weight_gen", "value_mask"],
    "formats": [("<i2", 32), "<i2", "u1", "u1", ("<i2", K_POLICY), ("<f2", K_POLICY),
                "<f4", "<f4", "<u2", "u1"],
    "offsets": [0, 64, 66, 67, 68, 68 + 2 * K_POLICY, 68 + 4 * K_POLICY,
                72 + 4 * K_POLICY, 76 + 4 * K_POLICY, 78 + 4 * K_POLICY],
    "itemsize": 79 + 4 * K_POLICY,
})
# `value_mask` (2026-08-25): 1 where the record's `z` is a value-loss target, 0 where
# only the policy is trained on it. Written once, at game close, by `value_subsample`:
# every k-th ply of the game, with the residue rotating per game. ⚠️ It is a property
# of the record and not of the draw, on purpose -- a mask re-rolled per epoch shows the
# head every label eventually and changes nothing about how many times one game's one
# bit is repeated to it, which is the quantity this exists to control.
RECORD_BYTES = RECORD.itemsize
assert RECORD_BYTES == 463, RECORD_BYTES
# `policy_len` is `u1`, so the width may never exceed 255.
assert K_POLICY <= 255, K_POLICY


@dataclass
class BufferStats:
    games: int
    records: int
    pending_records: int
    capacity: int
    evicted_games: int
    capacity_evictions: int
    mean_game_length: float


class ReplayBuffer:
    """A ring of completed games over a memory-mapped record store.

    ``path=None`` keeps the store in host memory, which is what the tests and the
    overfit run of §12 check 1 use; a 13 GB mapping for a check that finishes in
    seconds would be absurd.
    """

    def __init__(self, path: Optional[str] = None, window_games: int = 500_000,
                 mean_plies: int = 80, capacity_records: Optional[int] = None,
                 seed: int = 0, resume: bool = False, value_subsample: int = 1) -> None:
        if value_subsample < 1:
            raise ValueError(f"value_subsample must be >= 1, got {value_subsample}")
        self.value_subsample = int(value_subsample)
        if capacity_records is None:
            capacity_records = window_games * mean_plies
        self.capacity = int(capacity_records)
        self.window_games = int(window_games)
        self.path = path
        self.rng = np.random.default_rng(seed)

        if path is None:
            self.data = np.zeros(self.capacity, dtype=RECORD)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            need = self.capacity * RECORD_BYTES
            free = shutil.disk_usage(os.path.dirname(path) or ".").free
            exists = os.path.exists(path)
            if not (resume and exists) and free < need:
                # §5.2: disk is the resource to check before a long run, and the
                # loop refuses to start rather than discover it at generation 40.
                raise RuntimeError(
                    f"the replay buffer needs {need / 2**30:.1f} GiB at {path} and the "
                    f"filesystem has {free / 2**30:.1f} GiB free. Free space, or lower "
                    f"--window-games / --mean-plies (train.md §5.2)")
            mode = "r+" if (resume and exists) else "w+"
            # ⚠️ A file written at a different `K_POLICY` has a different itemsize, and
            # `np.memmap` would happily reinterpret it: every field would land at the
            # wrong offset and a misparsed record decodes as *a live white pawn on a1*
            # rather than as an error (`CLAUDE.md`). So the size is checked, and an old
            # buffer refuses instead of poisoning a run with plausible garbage.
            if exists and os.path.getsize(path) % RECORD_BYTES:
                raise RuntimeError(
                    f"{path} is {os.path.getsize(path)} bytes, not a multiple of this "
                    f"build's {RECORD_BYTES}-byte record (K_POLICY = {K_POLICY}). It "
                    f"was written by a different edge cap; delete it and let the run "
                    f"start a fresh buffer.")
            self.data = np.memmap(path, dtype=RECORD, mode=mode, shape=(self.capacity,))

        # The ring of completed games. `+1` so a full ring is distinguishable from
        # an empty one without a separate flag.
        self._g_start = np.zeros(self.window_games + 1, dtype=np.int64)
        self._g_count = np.zeros(self.window_games + 1, dtype=np.int64)
        self._g_head = 0
        self.n_games = 0

        self.head = 0          # logical record 0 sits here
        self.tail = 0          # next write
        self.n_records = 0

        self._pending: List[List[np.ndarray]] = []
        self.evicted_games = 0
        self.capacity_evictions = 0
        self.total_games = 0
        self.total_records = 0

    # -- the games in flight ------------------------------------------------ #

    def open_games(self, batch: int) -> None:
        """Allocate the pending lists. One per row of the search's batch."""
        if len(self._pending) != batch:
            self._pending = [[] for _ in range(batch)]

    @property
    def pending_records(self) -> int:
        # By row count, never by list length: a loaded snapshot restores each game's
        # pending records as one block rather than as one entry per ply, and a game's
        # value parity is a function of how many records it has.
        return sum(int(a.shape[0]) for p in self._pending for a in p)

    def append(self, record, done=None, result=None, store: bool = True) -> int:
        """One move-step: `B` rows in, and every game that just ended closed out.

        ``done`` and ``result`` override the record's own, which is how §5.4's
        game-length cap turns an over-long game into a completed drawn one without
        the search knowing about it. Returns the number of games closed.

        ⚠️ **``store = False`` still closes games** (`training.md` §11). Playout cap
        randomisation keeps the *positions* of a cheap turn out of the buffer, and the
        tempting way to write that is to not call this at all on a cheap move — which
        loses the closure of every game that *ends* on one, i.e. of most games at
        ``p = 0.3``. Those games' earlier records would then sit in ``_pending`` while
        the search restarts the slot, and the next game to finish there would flush them
        under *its* result: silent, unlogged, and label noise on a third of the buffer.
        So the two jobs this method does are separated by the flag rather than by the
        call, and the closure path is unconditional.
        """
        b = int(record.board.shape[0])
        self.open_games(b)
        if store:
            rows = self._rows_from(record)
            for i in range(b):
                self._pending[i].append(rows[i:i + 1])

        done = record.done if done is None else done
        result = record.result if result is None else result
        finished = done.nonzero(as_tuple=True)[0].tolist()
        if not finished:
            return 0
        res = result.to(torch.int32).cpu().numpy()
        # The control word of the position the game-ending move was played from. §4's
        # value rule needs the side to move *there*, not the length of the block.
        ctl = record.control.to(torch.int32).cpu().numpy()
        for i in finished:
            self._close(i, int(res[i]), int(ctl[i]))
        return len(finished)

    @staticmethod
    def _rows_from(record) -> np.ndarray:
        """``MoveRecord`` to ``[B]`` of :data:`RECORD`. One device-to-host copy per field."""
        b = int(record.board.shape[0])
        out = np.zeros(b, dtype=RECORD)
        out["board"] = record.board.to(torch.int16).cpu().numpy()
        out["control"] = record.control.to(torch.int16).cpu().numpy()
        out["rep"] = record.rep.to(torch.uint8).cpu().numpy()
        out["policy_len"] = record.policy_len.to(torch.uint8).cpu().numpy()
        # `K = min(E, n)`, so a run at n < 64 produces a narrower policy block than
        # the 64-wide record. The record's width is fixed by §5.1 and the tail stays
        # zero, which `policy_len` already says is unused.
        k = min(K_POLICY, int(record.policy_move.shape[1]))
        out["policy_move"][:, :k] = record.policy_move[:, :k].to(torch.int16).cpu().numpy()
        out["policy_prob"][:, :k] = record.policy_prob[:, :k].to(torch.float16).cpu().numpy()
        out["root_value"] = record.root_value.float().cpu().numpy()
        out["weight_gen"] = np.uint16(record.weight_gen)
        return out

    def _close(self, i: int, result: int, term_control: int) -> None:
        """A game ended: write ``z`` by the §4 parity rule and commit the block.

        ``term_control`` is the control word of the position the last move was played
        from — ``sign`` is the side to move there (`env` §2.2), magnitude is the clock
        and is ignored. ``result`` is from the *new* mover's point of view, i.e. from
        the point of view of ``-sign(term_control)``.
        """
        pending = self._pending[i]
        self._pending[i] = []
        if not pending:
            return
        block = np.concatenate(pending)
        length = block.shape[0]
        # z = -r * sign(control) * sign(term_control): +r for a record whose mover is
        # the one `r` speaks for, -r otherwise. Identical to the old distance-from-the-
        # end rule on a dense game, and correct on a sparse one, which is what §11 makes
        # the recorded plies. A control word is never 0 (magnitude = clock + 1 >= 1), so
        # neither sign can silently vanish.
        s_rec = np.where(block["control"] > 0, 1.0, -1.0)
        s_end = 1.0 if term_control > 0 else -1.0
        block["value"] = -np.float32(result) * s_rec * s_end
        # Which plies carry the value target. `k = 1` marks every record, bit-for-bit
        # the run before this field existed. The residue rotates with the game count so
        # no ply parity is systematically favoured across games.
        k = self.value_subsample
        block["value_mask"] = ((np.arange(length) + self.total_games) % k == 0)
        self._commit(block)
        self.total_games += 1
        self.total_records += length

    def drop_pending(self, i: int) -> int:
        """Discard a game's pending records without committing them. Returns how many.

        The one caller is a resume that found no buffer snapshot: the game state is
        restored and the records that preceded it are gone, so the game must not be
        credited with a block whose parity it cannot vouch for. Everything else that
        ends a game goes through :meth:`append`.
        """
        n = sum(int(a.shape[0]) for a in self._pending[i])
        self._pending[i] = []
        return n

    # -- the ring ----------------------------------------------------------- #

    def _commit(self, block: np.ndarray) -> None:
        length = block.shape[0]
        if length > self.capacity:
            raise ValueError(f"a game of {length} records does not fit a "
                             f"{self.capacity}-record buffer")
        while self.n_games >= self.window_games:
            self._evict(capacity_bound=False)
        while self.capacity - self.n_records < length:
            self._evict(capacity_bound=True)

        start = self.tail
        end = start + length
        if end <= self.capacity:
            self.data[start:end] = block
        else:
            cut = self.capacity - start
            self.data[start:] = block[:cut]
            self.data[:end - self.capacity] = block[cut:]
        self.tail = end % self.capacity
        self.n_records += length

        slot = (self._g_head + self.n_games) % len(self._g_start)
        self._g_start[slot] = start
        self._g_count[slot] = length
        self.n_games += 1

    def _evict(self, capacity_bound: bool) -> None:
        if self.n_games == 0:
            raise RuntimeError("nothing left to evict; the buffer is smaller than one game")
        count = int(self._g_count[self._g_head])
        self._g_head = (self._g_head + 1) % len(self._g_start)
        self.n_games -= 1
        self.head = (self.head + count) % self.capacity
        self.n_records -= count
        self.evicted_games += 1
        if capacity_bound:
            # The window is meant to bind, not the file: if this fires the mapping
            # was sized from a mean game length that is not the one being played,
            # and §14 says to measure it in generation 1 and resize.
            self.capacity_evictions += 1

    # -- sampling ----------------------------------------------------------- #

    def sample(self, n: int, device="cuda"):
        """`n` records drawn uniformly at random over every position in the window."""
        if self.n_records < n:
            raise RuntimeError(f"asked for {n} records and the buffer holds "
                               f"{self.n_records} (train.md §5.5)")
        idx = self.rng.integers(0, self.n_records, size=n)
        rows = self.data[(self.head + idx) % self.capacity]

        def to(field, dtype):
            return torch.from_numpy(np.ascontiguousarray(rows[field])).to(
                device=device, dtype=dtype, non_blocking=True)

        return TrainBatch(
            board=to("board", torch.int16), control=to("control", torch.int16),
            rep=to("rep", torch.uint8), policy_move=to("policy_move", torch.int16),
            policy_prob=to("policy_prob", torch.float16),
            policy_len=to("policy_len", torch.uint8), value=to("value", torch.float32),
            weight_gen=to("weight_gen", torch.int32),
            value_mask=to("value_mask", torch.float32),
            root_value=to("root_value", torch.float32))

    # -- reporting and persistence ------------------------------------------ #

    def stats(self) -> BufferStats:
        return BufferStats(
            games=self.n_games, records=self.n_records,
            pending_records=self.pending_records, capacity=self.capacity,
            evicted_games=self.evicted_games,
            capacity_evictions=self.capacity_evictions,
            mean_game_length=self.n_records / max(self.n_games, 1))

    def check(self) -> None:
        """§12 check 5, cheap enough to run every generation."""
        if self.n_games > self.window_games:
            raise AssertionError(f"5: {self.n_games} games exceeds the window")
        if self.n_records > self.capacity:
            raise AssertionError(f"5: {self.n_records} records exceeds the capacity")
        total = 0
        for j in range(self.n_games):
            slot = (self._g_head + j) % len(self._g_start)
            if int(self._g_start[slot]) != (self.head + total) % self.capacity:
                raise AssertionError(f"5: game {j}'s block is not where the ring says")
            total += int(self._g_count[slot])
        if total != self.n_records:
            raise AssertionError(f"5: game blocks total {total}, ring holds {self.n_records}")

    def _live_records(self) -> np.ndarray:
        """The window's records in logical order, wraparound resolved."""
        if self.n_records == 0:
            return np.zeros(0, dtype=RECORD)
        end = self.head + self.n_records
        if end <= self.capacity:
            return np.array(self.data[self.head:end])
        return np.concatenate([np.array(self.data[self.head:]),
                               np.array(self.data[:end - self.capacity])])

    def save(self, meta_path: str) -> None:
        """The index and the games in flight.

        ⚠️ **An in-memory store saves its records too, and a mapped one does not.**
        For ``path=None`` there is no file the records already live in, so an index
        restored on its own would point at zeros — and a zeroed record decodes as a
        live white pawn on a1 (`CLAUDE.md`), which is a position, not an error. The
        13 GB mapping needs no such copy and must not get one.
        """
        pending = [np.concatenate(p) if p else np.zeros(0, dtype=RECORD)
                   for p in self._pending]
        lengths = np.array([p.shape[0] for p in pending], dtype=np.int64)
        blob = (np.concatenate(pending) if len(pending)
                else np.zeros(0, dtype=RECORD))
        if isinstance(self.data, np.memmap):
            self.data.flush()
            records = np.zeros(0, dtype=RECORD)
        else:
            records = self._live_records()
        np.savez(meta_path, g_start=self._g_start, g_count=self._g_count,
                 records=records, pending=blob, pending_len=lengths,
                 rng=np.frombuffer(json.dumps(self.rng.bit_generator.state).encode(),
                                   dtype=np.uint8),
                 scalars=np.array([self._g_head, self.n_games, self.head, self.tail,
                                   self.n_records, self.evicted_games,
                                   self.capacity_evictions, self.total_games,
                                   self.total_records, self.capacity,
                                   self.window_games], dtype=np.int64))

    # -- the live index, for a reader outside this process -------------------- #

    def save_index(self, path: str) -> None:
        """The ring's *metadata only*, written atomically. For live monitoring.

        ⚠️ **This exists because the ring lives in Python memory and the records do
        not.** ``data/replay/<run>.dat`` is a memory map that a second process can
        open at any time, but ``_g_start``, ``_g_count``, ``head`` and ``tail`` are
        attributes of this object, so without them a reader has 735 MB of records and
        no way to say where a game begins or which region is live. That was found on
        2026-07-31, when the only snapshot mechanism was :meth:`save` at
        ``buffer_snapshot_every = 10_000`` steps — a run 1 292 steps long had never
        written one, and a Ctrl-C would have left the buffer unreadable.

        Metadata only, so it is ~320 KB against :meth:`save`'s 36 MB: no records (the
        mapping already has them) and **no pending blocks** (a game in flight has no
        `z` yet and is not something a reader should show). Cheap enough to write
        every generation.

        ⚠️ **Written to a temporary file and renamed**, because ``os.replace`` is
        atomic on POSIX and a half-written index read by the viewer would point at
        arbitrary offsets. A reader therefore always sees a consistent index, though
        possibly an old one — which is the right failure direction.

        ⚠️ It is written **after** the records it describes, and games are committed
        whole (:meth:`_commit`), so every game the index names is already fully in the
        file. The converse is not true and does not matter: records written since the
        last index are simply invisible until the next one.
        """
        if isinstance(self.data, np.memmap):
            self.data.flush()
        self._index_revision = getattr(self, "_index_revision", 0) + 1
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, g_start=self._g_start, g_count=self._g_count,
                     scalars=np.array([self._g_head, self.n_games, self.head,
                                       self.tail, self.n_records, self.capacity,
                                       self.window_games, self.total_games,
                                       self.total_records, self._index_revision],
                                      dtype=np.int64))
        os.replace(tmp, path)

    def load(self, meta_path: str) -> None:
        z = np.load(meta_path, allow_pickle=False)
        s = z["scalars"]
        if int(s[9]) != self.capacity or int(s[10]) != self.window_games:
            raise RuntimeError(
                f"the snapshot was taken at capacity {int(s[9])} / window {int(s[10])} "
                f"and this buffer is {self.capacity} / {self.window_games}. Resuming "
                f"across a resize would reinterpret the ring, so it refuses")
        self._g_start[:] = z["g_start"]
        self._g_count[:] = z["g_count"]
        (self._g_head, self.n_games, self.head, self.tail, self.n_records,
         self.evicted_games, self.capacity_evictions, self.total_games,
         self.total_records) = (int(x) for x in s[:9])

        records = z["records"]
        if records.size:
            # An in-memory store: replay the blocks through `_commit` rather than
            # restoring the index over them, so the ring is rebuilt by the same code
            # that built it in the first place and cannot disagree with itself.
            counts = [int(self._g_count[(self._g_head + j) % len(self._g_count)])
                      for j in range(self.n_games)]
            keep = (self.evicted_games, self.capacity_evictions,
                    self.total_games, self.total_records)
            self._g_head = self.n_games = self.head = self.tail = self.n_records = 0
            off = 0
            for count in counts:
                self._commit(records[off:off + count])
                off += count
            (self.evicted_games, self.capacity_evictions,
             self.total_games, self.total_records) = keep
        elif self.path is None and self.n_records:
            raise RuntimeError(
                f"the snapshot carries an index for {self.n_records} records and no "
                f"records, and this buffer has no file to read them from. Restoring "
                f"the index alone would sample zeroed records, which decode as a live "
                f"white pawn on a1 rather than as an error")

        self.rng.bit_generator.state = json.loads(z["rng"].tobytes().decode())
        lengths = z["pending_len"]
        blob = z["pending"]
        self._pending = []
        off = 0
        for n in lengths.tolist():
            self._pending.append([blob[off:off + n].copy()] if n else [])
            off += n


# --------------------------------------------------------------------------- #
# Reading a buffer from outside the process that writes it
# --------------------------------------------------------------------------- #

class BufferView:
    """A read-only window onto a replay buffer, live or finished.

    This is the half of the buffer that a viewer needs and the trainer does not: open
    the ``.dat`` mapping and the index :meth:`ReplayBuffer.save_index` wrote, and hand
    out games. It opens the mapping in ``mode="r"``, so a reader can never perturb a
    running trainer.

    ⚠️ **Games are contiguous in the file, but the ring wraps**, so a game near the
    write head is split across the end of the mapping. :meth:`game` resolves that; do
    not index ``.data`` directly.

    ⚠️ **The index is a snapshot and the file is not.** A game this view returns was
    complete when the index was written, because :meth:`ReplayBuffer._commit` writes a
    game whole and the index is written afterwards. But the *oldest* games named by a
    stale index may since have been evicted and overwritten by newer ones. Call
    :meth:`refresh` before a read if freshness matters; ``revision`` says whether
    anything moved. For an evicted game the records are simply different records —
    still valid, still legal positions, just not the game you asked for. Nothing here
    can detect that, which is why a live viewer should refresh rather than cache.

    The debugger imports this; `brokefish` never imports the debugger
    (`docs/ledger/state.md` calls the reverse direction a defect).
    """

    def __init__(self, dat_path: str, index_path: Optional[str] = None) -> None:
        self.dat_path = dat_path
        self.index_path = index_path or (os.path.splitext(dat_path)[0] + ".index.npz")
        self.data = np.memmap(dat_path, dtype=RECORD, mode="r")
        self.refresh()

    @classmethod
    def for_run(cls, run: str, buffer_dir: str = "data/replay") -> "BufferView":
        """The view for a training run by name, using the loop's own paths."""
        return cls(os.path.join(buffer_dir, f"{run}.dat"))

    def refresh(self) -> bool:
        """Re-read the index. Returns whether it moved since the last read."""
        z = np.load(self.index_path, allow_pickle=False)
        s = z["scalars"]
        (self._g_head, self.n_games, self.head, self.tail, self.n_records,
         self.capacity, self.window_games, self.total_games, self.total_records,
         revision) = (int(x) for x in s)
        self._g_start, self._g_count = z["g_start"], z["g_count"]
        moved = revision != getattr(self, "revision", None)
        self.revision = revision
        if self.capacity != self.data.shape[0]:
            raise RuntimeError(
                f"the index describes a {self.capacity}-record ring and "
                f"{self.dat_path} holds {self.data.shape[0]}. They are from different "
                f"runs, or the run was resized.")
        return moved

    def __len__(self) -> int:
        return self.n_games

    def game_length(self, i: int) -> int:
        """Plies in game `i`, where 0 is the oldest game still in the window."""
        return int(self._g_count[(self._g_head + i) % len(self._g_start)])

    def game(self, i: int) -> np.ndarray:
        """Game `i` as a contiguous ``RECORD`` array, wraparound resolved.

        Index 0 is the **oldest** game still in the window and ``len(view) - 1`` the
        newest, so a viewer that wants "what is being played now" reads from the end.
        """
        if not 0 <= i < self.n_games:
            raise IndexError(f"game {i} of {self.n_games}")
        slot = (self._g_head + i) % len(self._g_start)
        start, count = int(self._g_start[slot]), int(self._g_count[slot])
        end = start + count
        if end <= self.capacity:
            return np.array(self.data[start:end])
        return np.concatenate([np.array(self.data[start:]),
                               np.array(self.data[:end - self.capacity])])

    def outcome(self, i: int) -> int:
        """The game's result from **White**'s point of view, in {-1, 0, +1}.

        ``value`` is stored per record from the point of view of the player to move
        there (§4's parity rule), so the last record's `z` belongs to the side that
        delivered the final move, and `control` says who that was.
        """
        g = self.game(i)
        if g.shape[0] == 0:
            return 0
        z = float(g["value"][-1])
        return int(z if int(g["control"][-1]) > 0 else -z)

    def stats(self) -> dict:
        """What a monitor header wants, without walking every game."""
        counts = [self.game_length(i) for i in range(self.n_games)]
        return {"games": self.n_games, "records": self.n_records,
                "capacity": self.capacity, "window_games": self.window_games,
                "total_games": self.total_games, "revision": self.revision,
                "mean_plies": (sum(counts) / len(counts)) if counts else 0.0,
                "max_plies": max(counts) if counts else 0,
                "fill": self.n_records / max(self.capacity, 1)}
