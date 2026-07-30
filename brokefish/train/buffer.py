"""The replay buffer of ``docs/train.md`` §5.

The most recent 500,000 games, sampled uniformly at random over all positions in
them (AGZ Methods, Optimisation, inherited by AZ). At ~80 plies and 334 B per
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
is from the *new* mover's point of view after the move was played, and the record's own
position has the *previous* mover to move, so with ``L`` records in a finished game::

    z(i) = r  if (L - i) is even, else -r

which is the same parity argument as ``mcts.md`` §6.5's backup flip and fails the same
way if inverted. ⚠️ Note it depends only on distance from the *end*, which is what
makes a game whose earlier records were lost to a resume still get correct values.
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
# 334 B the document sizes the window from; numpy handles the unaligned f4 fields
# by copying, and this is a random-read workload where the byte count is what costs.
RECORD = np.dtype({
    "names": ["board", "control", "rep", "policy_len", "policy_move", "policy_prob",
              "value", "root_value", "weight_gen"],
    "formats": [("<i2", 32), "<i2", "u1", "u1", ("<i2", 64), ("<f2", 64),
                "<f4", "<f4", "<u2"],
    "offsets": [0, 64, 66, 67, 68, 196, 324, 328, 332],
    "itemsize": 334,
})
RECORD_BYTES = RECORD.itemsize
assert RECORD_BYTES == 334, RECORD_BYTES

K_POLICY = 64


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
                 seed: int = 0, resume: bool = False) -> None:
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

    def append(self, record, done=None, result=None) -> int:
        """One move-step: `B` rows in, and every game that just ended closed out.

        ``done`` and ``result`` override the record's own, which is how §5.4's
        game-length cap turns an over-long game into a completed drawn one without
        the search knowing about it. Returns the number of games closed.
        """
        rows = self._rows_from(record)
        b = rows.shape[0]
        self.open_games(b)
        for i in range(b):
            self._pending[i].append(rows[i:i + 1])

        done = record.done if done is None else done
        result = record.result if result is None else result
        finished = done.nonzero(as_tuple=True)[0].tolist()
        if not finished:
            return 0
        res = result.to(torch.int32).cpu().numpy()
        for i in finished:
            self._close(i, int(res[i]))
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

    def _close(self, i: int, result: int) -> None:
        """A game ended: write ``z`` by the §4 parity rule and commit the block."""
        pending = self._pending[i]
        self._pending[i] = []
        if not pending:
            return
        block = np.concatenate(pending)
        length = block.shape[0]
        # z(i) = r if (L - i) even else -r. Depends on distance from the end alone.
        sign = np.where(((length - np.arange(length)) % 2) == 0, 1.0, -1.0)
        block["value"] = np.float32(result) * sign
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
            weight_gen=to("weight_gen", torch.int32))

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
