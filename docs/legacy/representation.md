# Board representation

!!! warning "Superseded by [`spec.md`](../reference/spec.md) on 2026-07-29"

    This document is **no longer normative**. It describes the encoding as it was
    implemented before the v1 freeze, and it is kept only as a record of what
    changed. Where it disagrees with `spec.md`, it is wrong. In particular the
    side-to-move word is now a **halfmove clock**, promotion is complete, and the
    engine emits `in_check` and a terminal code.

The state of a chess position is **32 `uint16` words plus one `int16`**. This
document described the encoding as it was actually implemented, which is not always
what the original design note said. The PyTorch
reference implementation of the engine lives in `~/steakfish`; everything needed to
write or read a CUDA kernel is here.

## Why a piece list and not bitboards

One word per piece slot, 32 slots, fixed for the lifetime of a game. A warp owns a
whole board with one word per lane, and a board-wide OR is a single
`__reduce_or_sync`.

The load-bearing property is that **slots are stable**: a captured piece keeps its
slot forever, and no piece ever migrates. That is what makes the network's 32 piece
tokens index-aligned with the engine's 32 legality masks, so masking the policy is
a device-side AND — no gather, no compaction, no host round-trip.

## Slot layout

| slots | piece |
|---|---|
| 0-7 | white pawns |
| 8-9 | white knights |
| 10-11 | white bishops |
| 12-13 | white rooks |
| 14 | white queen |
| 15 | white king |
| 16-31 | same, black |

⚠️ **The slot tells you where a piece started, not what it is.** A promoted pawn
keeps its pawn slot and has its type bits overwritten, so slot 3 may hold a queen.
Positions imported from FEN follow the same rule: material beyond the initial
count (a third knight, a second queen) is packed into the pawn slots of its color.
**Always read the type from the word.** Only slots 15 and 31 are guaranteed by
construction — there is always exactly one king per side.

## Word encoding (12 bits used out of 16)

| bits | field | meaning |
|---|---|---|
| 11 | `captured` | 1 = the piece is off the board |
| 10 | `color` | 0 = white, 1 = black |
| 9 | `special` | context-dependent, see below |
| 8:6 | `type` | 000 pawn, 001 knight, 010 bishop, 011 rook, 100 queen, 101 king |
| 5:0 | `square` | `row * 8 + col`, 0 = a1, 63 = h8 |

### The `special` bit

- **King, rook** — `1` means *this piece has already moved*, i.e. the castling
  right attached to it is gone. Castling is generated only from a king with
  `special == 0` together with a rook whose word matches the expected
  colour/type/square exactly. Note the polarity: the bit records the loss of a
  right, not the right.
- **Pawn** — `1` means *this pawn just made a double push*, i.e. it is an en
  passant target for exactly one ply. Every move clears the bit on all pawns
  before setting it on the pawn that moved, so at most one pawn carries it.
- **Other pieces** — unused, always 0.

### Captured pieces

When a piece is captured its whole word is overwritten with `1 << 11` (`0x800`).
Colour, type and square are **wiped**, not merely flagged. Every dead slot
therefore looks identical, and the only information a dead token carries is its own
index. Consumers must test bit 11 before trusting any other field.

## The side-to-move word (`magic`)

One `int16` per position, and the name has **nothing to do with magic bitboards**:

- **sign** = side to move. `magic > 0` → white to move; `magic < 0` → black.
- **magnitude** = ply counter, starting at 1.
- update after every move: `magic = -sign(magic) * (|magic| + 1)`.

⚠️ The magnitude increments unconditionally. It is a plain ply count, **not** a
halfmove clock: it is not reset on a capture or a pawn move, so the fifty-move rule
is not tracked by the current state. Threefold repetition is not tracked either.
Both are missing pieces of full game-termination logic, not properties of the
encoding — a self-play loop needs them before its results mean anything.

## Moves and masks

A move is a single integer in `[0, 2048)`:

```
move = slot * 64 + target_square
```

`-1` is the null move. Underpromotion is **not** in this space: promotion defaults
to a queen and any other choice travels in a separate tensor. If the policy ever
needs underpromotion, that is an encoding change, not a parameter.

Legality is returned as **`[n_positions, 32] uint64`**: bit `s` of word `p` is set
iff the move `p -> s` is legal. Expanding it to `[n, 32, 64]` booleans is a
convenience for PyTorch; kernels should stay on the bitset. This is exactly the
shape of the policy head (32 pieces × 64 squares), which is the whole point.

Legality is computed in two passes:

1. **First order** — table lookup per (type, square), then pawn pushes/captures/en
   passant, slider occlusion, castling, and finally the drop of every target
   occupied by a friendly piece.
2. **Second order** — remove the moves that leave one's own king attacked. The
   PyTorch reference does this by brute force: play every candidate, recompute the
   opponent's control mask, test the king square. It also handles castling through
   check by testing the three squares the king crosses. A CUDA implementation
   should instead restrict the second pass to the pieces that can plausibly expose
   the king (the pinned set plus the king itself) — the reference is the
   correctness oracle, not the performance model.

## Invariants a kernel may rely on

- exactly 32 slots, always; no compaction, no reallocation
- exactly one king per colour, in slots 15 and 31, never captured
- a slot's colour bit never changes while the piece is alive
- at most one pawn carries the en passant `special` bit at any time
- `square` is meaningless when `captured` is set
