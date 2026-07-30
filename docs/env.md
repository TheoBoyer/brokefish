# The environment

The reference implementation of the engine contract ([spec §4](spec.md#4-the-engine-contract)),
in PyTorch. It is the oracle the CUDA move generator is written against, and
python-chess is in turn the oracle for it. Nothing in the training loop imports
it.

Ported on 2026-07-29 from the ancestor engine in `~/steakfish` (class `TCHESS`),
which is not a dependency and stays where it is.

## Layout

| file | what |
|---|---|
| `brokefish/env/torch_impl.py` | the engine: `movegen`, `step`, `play`, and the constructors |
| `brokefish/env/luts.py` | the four lookup tables, built in torch at first use |
| `brokefish/env/interop.py` | python-chess conversions and the ASCII printers, test scaffolding only |
| `brokefish/env/positions.py` | random positions drawn from parquet shards of human games |
| `tests/test_env.py` | perft, and differential fuzzing against python-chess |
| `scripts/dump_cuda_testset.py` | the differential test set the CUDA kernel is validated against, including the per-stage snapshots and the control-mode attack map |
| `scripts/fetch_test_positions.py` | downloads one Lichess parquet shard |

`brokefish/env/__init__.py` re-exports the names both implementations must
provide, so a caller can be switched from `torch_impl` to `cuda_impl` by changing
one import.

## Interface

State is carried as plain tensors, never as a tuple-of-everything:

```
boards   [N, 32] int16    one piece word per slot, spec §2.1
control  [N]     int16    sign = side to move, magnitude = clock + 1, spec §2.2
mask     [N, 32] int64    bit s of word p set iff the move p -> s is legal
```

⚠️ The mask is `int64` where [spec §3](spec.md#3-moves-and-the-action-space) says
`uint64`. `torch.uint64` exists as a dtype, but on torch 2.13, `<<`, `>>`, `~`,
ordering comparisons, `add` and `nonzero` all raise `NotImplementedError` on it,
on CPU and on CUDA alike, and the move generator is built out of exactly those.
The consequence is that bit 63, square h8, is the sign bit, so a mask word
containing an h8 move is a negative integer. Arithmetic right shift followed by
`& 1` reads it correctly and the bytes written by `dump_cuda_testset.py` are
identical to the `uint64` the CUDA side loads. Anything that tests a mask word
with `> 0` or sorts one is wrong.

⚠️ **Summing mask words is wrong too, and this one shipped.** `mask.sum(-1) == 0`
reads as "no legal move" and is not: the sum overflows. Two pieces that can reach
h8 give 2 × 2⁶³, four that can reach g8 give 4 × 2⁶², and both are 2⁶⁴, which is
zero. `terminal` used it, so
`1r1r2Rk/pp5p/4b2n/1P5P/n1Ppp3/R4P1p/1B1KP3/8 b - - 1 1` was reported as
checkmate with result −1 while having four legal replies, every one of them
capturing the checking rook. That is a false game end and a −1 value target on a
non-terminal position. The test is `(mask == 0).all(-1)`. Found by the CUDA port
on 2026-07-29, one position in a 10 000-position dump, and pinned by
`test_terminal_survives_mask_overflow`.

```python
mask, in_check = movegen(boards, control)
boards, control, hash, irreversible = step(boards, control, move, promo=None, hash=None)
boards, control, mask, in_check = play(boards, control, move)   # step then movegen
code, result = terminal(mask, in_check, control, boards, hash, ring, length)
```

`hash` is optional both ways: pass `None` and you get `None` back, skipping the
work. Perft would otherwise pay a full Zobrist on 119M positions it never reads,
while the search always carries one. `irreversible` is always returned, being
nearly free and needed to bound the repetition window.

The repetition ring is `[N, 100] int64` plus an `[N] int64` length, built by
`empty_history()` and advanced by `push_history(ring, length, hash_before,
irreversible)`. An irreversible move empties it. 100 slots always suffice: the
window is bounded by the fifty-move window, whose reset conditions are a strict
subset, and the game ends at clock 101.

Three things changed on the way in, all deliberate:

* **The mask is the bitset, and 1 means legal.** The ancestor returned a
  `[N,32,64]` bool tensor with `False` meaning legal. That is 64× the traffic,
  and the policy head wants a device-side AND against the engine's own words
  ([spec §3](spec.md#3-moves-and-the-action-space)). `bitset_to_bool()` survives
  as a debug helper, with the polarity the rest of the repository uses.
* **`promo` is the spec §3 field** (`0:N 1:B 2:R 3:Q`), not a piece type code.
  The conversion lives in `interop.move_to_args`.
* **No cache file and no cwd dependency.** The ancestor shelled out to a
  generator script at import time and read `cache/buffers.pt` relative to the
  current working directory, which made it importable from exactly one place.
  The tables are pure arithmetic over 64 squares and are now rebuilt in-process;
  `luts.dump()` writes the flat binaries the CUDA side loads.

## What it is verified to do

Perft, which needs no oracle at all, and is what catches a botched translation:

| position | depth | nodes | in the default run |
|---|---|---|---|
| startpos | 4 / 5 | 197 281 / 4 865 609 | 4 by default, 5 under `--slow` |
| startpos | 6 | 119 060 324 | `--slow`, on its own, 1220 s |
| Kiwipete | 3 / 4 | 97 862 / 4 085 603 | 3 by default |
| position 3 | 4 / 5 | 43 238 / 674 624 | 4 by default |
| position 4 | 3 / 4 | 9 467 / 422 333 | 3 by default |
| position 5 | 3 / 4 | 62 379 / 2 103 487 | 3 by default |
| position 6 | 3 / 4 | 89 890 / 3 894 594 | 3 by default |

⚠️ **The depths above are this module's, and they are not the strongest evidence any
more.** They stop where they do because a CPU perft of Kiwipete at depth 5 takes
about twenty minutes. `csrc/tests/tperft.cu` runs the same six positions one ply
deeper in **4.8 s total: 598M nodes, every count matching**, and that is what the
rules claim now rests on. Deepened 2026-07-30; `scripts/dump_cuda_testset.py` holds
the case table.

Position 5 carries four promotions among its 44 moves at depth 1, so promotion sits
inside a node count rather than only inside the differential harness. `perft()` runs
depth-first over chunks and counts the last ply instead of playing it, so depth 6
holds about 30 MB rather than the 7.6 GB its 119M boards would need.

Differential fuzzing against python-chess, over 256 positions drawn from real
games and walked 5 plies each, singly and as one batch. Every position asserts the
legal-move set as full `chess.Move` objects with the promotion piece included, a
FEN round trip covering castling rights, the en passant square, side to move and
the halfmove clock, then `in_check`, the terminal code, the repetition count at
n = 2 and n = 3, `insufficient_material`, `irreversible`, the castling-rights
vector, the legal en passant file, and the hash.

The hash gets the strongest check available. Grouping every position seen by our
Zobrist and by python-chess's own `_transposition_key()` has to produce the same
partition. A bucket holding two keys is a collision; a key split across two
buckets means two identical positions were given different hashes, which is the
failure that silently loses a threefold repetition and which no single-position
check would find. Both are zero over 7185 positions.

Threefold detection is tested on a forced line rather than left to chance, since
random play reaches one about never: two knight round trips put the start position
on the board three times, and the code has to fire on the third and not the
second.

## Throughput

`python -m bench.bench_env`, batches of distinct mid-game positions, movegen
only. The bar belongs to the CUDA kernel rather than to this, so read it as a
baseline:

| positions | CPU | RTX 4060 |
|---|---|---|
| 256 | 7.5 k boards/s | 37 k boards/s |
| 1024 | 7.6 k | 81 k |
| 4096 | 6.1 k | 98 k |

Gate 1 wants about 45 k positions per second through the environment, and the
reference implementation already does roughly twice that on the GPU. Which is a
reason to hold A1 to correctness first and leave the 50-100M boards/s claim of
`csrc/README.md` where it is, as a write-up claim rather than a gate. Batch size
is capped by memory: the brute-force second order expands N positions into about
35N boards, and `first_order_mask` holds several `[35N, 32] int64` intermediates.

Perft timings on CPU, for scale: 197 k positions expanded in 1.4 s, 4.9 M in 30 s.

## Where it departs from FIDE

Three choices every engine makes, none of them a defect, all of them affecting
self-play results. [`fidelity.md`](fidelity.md) §3.2 has the full argument.

* **Draws are automatic where FIDE makes them claimable.** Threefold and the
  fifty-move rule are claims under Articles 9.2 and 9.3, with automatic thresholds
  at fivefold and seventy-five moves. `terminal` ends the game at three and at 100
  plies, so a position a player would decline to draw is scored 0.
* **FIDE's "dead position" (5.2.2) is reduced to a material rule.** Blocked
  positions with material are not called dead and run to the fifty-move rule
  instead. The approximation errs in the safe direction: nothing is declared drawn
  that FIDE would let continue.
* **No claim mechanics, no Chess960, no agreement, resignation or time.**

## Three places it departs from python-chess, all deliberate

**Promotion is one mask bit and four edges.** Spec §3 fixes the action space at
32×64 and carries the promotion type beside the move, so `movegen` sets a single
bit for a last-rank pawn move. The caller expands it, which is what
`interop.list_legal_moves` and the perft harness both do. This is why perft counts
moves rather than mask bits: position 5 of the standard suite is 44 moves and 41
bits.

**`irreversible` omits python-chess's en passant clause.** `Board.is_irreversible`
is `is_zeroing or _reduces_castling_rights or has_legal_en_passant()`, and spec
§6.2 has only the first two. Measured over 9000 moves, that is the only source of
disagreement, 34 of them. Dropping the clause can only make the repetition window
longer, never shorter, so it cannot cause a missed repetition; and a position with
a legal ep always follows a double push, which is a pawn move and therefore
already irreversible, so the window already starts there.

**Insufficient material follows python-chess, not spec v1.** The four-case list
missed K+2B against K with both bishops on one colour complex. Spec §4.3 was
amended on 2026-07-29 rather than the harness being given an exception.

## What it does not do

Nothing from spec §§2-6 is missing any more. What remains outside this module:

* nothing on the CUDA side. **A1 closed on 2026-07-30**, binding included:
  `brokefish/env/cuda_impl.py` stands in for this module behind the same signatures,
  and `bench/bench_loop.py` puts the environment at **2.2 % of a node** against the
  network. Details: `csrc/movegen.cuh` and `csrc/step.cuh` are
  perft-green on all six standard positions, startpos to depth 6 in 1.01 s against
  1220 s here. They are validated against this module, so a bug here is a bug there,
  which is why `csrc/tests/tperft.cu` exists as an oracle outside the repository.
  `csrc/zobrist.cuh` and `csrc/terminal.cuh` carry the hash, `irreversible` and the
  terminal codes, all bit-exact against this module. The per-game repetition ring is
  the one piece of spec §6 that is not ported, because it is per-game state and
  belongs to the search
* everything above the engine: search, training, evaluation

The two open bugs in `~/steakfish/todo.txt`, including the bishop underpromotion,
replay clean on both the port and the ancestor. They are stale, not fixed.

⚠️ The clock reset lands in this version, so every test set dumped before it is
invalid: `control.bin` carries a ply counter where the engine now writes a
fifty-move clock. The sets under `~/learncuda/` are dead and
`scripts/dump_cuda_testset.py` has to be re-run before anything validates against
them.

## Test positions

`positions.py` draws from parquet shards of human games. Those positions are
coverage for the move generator and never reach training: the tabula rasa
boundary forbids human games as a source of supervision, and this is measurement.
The ancestor's `IterableDataset` and `DataModule`, which existed to feed a
network, were dropped rather than ported.

The shard is not committed. `python scripts/fetch_test_positions.py` fetches a
random shard of the most recent published month into `data/`, so the test set
rolls forward instead of being pinned to one era of play. Recent months are
sharded at about 1 GB per file. `--year 2013 --month 01` pins the smallest one
at 37 MB, and `--shard N` pins the index.

Without any shard the fuzzing tests fall back to positions generated by random
legal play, which needs no download and covers endgames better than openings.
