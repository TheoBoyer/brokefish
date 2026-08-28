# Brokefish specification

**Status: frozen 2026-07-29 (v1).** This document is normative. Where it disagrees
with any other file in the repository, including `docs/legacy/representation.md`
and `csrc/chess.cuh`, this document is correct and the other file is a bug.

Its scope is the contract between the chess engine and the network: how a position
is represented, what the engine produces, what the network consumes and emits. The
RL layer is outside that scope; §11 tracks where its parameters were settled and
what is still open. Sections 1 to 10 do not depend on any of them.

---

## 1. Design criteria

Two consumers constrain the representation.

The network reads a position as 32 tokens, one per piece slot, so the sequence
length is 32 for any material and features are added as summed embeddings.

The CUDA engine reads a position as 64 bytes, one warp per board and one lane per
slot, so a board-wide reduction is a single `__reduce_or_sync` and a position load
is coalesced. No part of a self-play step may touch the host.

Both consumers require slot stability: a piece keeps its slot for the whole game and
never migrates. Slot stability index-aligns the network's 32 policy rows with the
engine's 32 legality masks, which allows the policy to be masked by a device-side
AND without gather, compaction or host synchronisation.

---

## 2. Position state

A position is 32 `uint16` piece words, one `int16` control word, and one `uint64`
hash. Those 82 bytes carry everything the rules require except repetition history,
which is per-game and specified in §6.

### 2.1 Piece word

| bits | field | meaning |
|---|---|---|
| 15:12 | | must be zero |
| 11 | `captured` | 1 = off the board; the rest of the word is wiped |
| 10 | `color` | 0 = white, 1 = black |
| 9 | `special` | context-dependent, see below |
| 8:6 | `type` | 000 pawn, 001 knight, 010 bishop, 011 rook, 100 queen, 101 king |
| 5:0 | `square` | `row * 8 + col`, 0 = a1, 63 = h8 |

Slot layout: 0-7 white pawns, 8-9 knights, 10-11 bishops, 12-13 rooks, 14 queen,
15 king, and 16-31 mirroring that for black.

⚠️ A slot records where a piece started. A promoted pawn keeps its pawn slot with
its type bits overwritten, so slot 3 may hold a queen. Positions imported from FEN
follow the same rule, packing material beyond the initial count into the pawn slots
of its colour. Kernels must read the type from the word. Only slots 15 and 31 are
guaranteed by construction.

The `special` bit records the loss of a right rather than its presence:

- on a king or rook, 1 means the piece has already moved, so its castling right is
  gone
- on a pawn, 1 means the pawn just made a double push and is an en passant target
  for one ply
- on other types it is unused and always 0

A captured piece has its whole word overwritten with `1 << 11` (`0x800`), wiping
colour, type and square. Every dead slot is byte-identical and carries no
information beyond its own index, so kernels must test bit 11 before reading any
other field.

### 2.2 Control word

One `int16` carries two fields. Its sign gives the side to move, positive for white
and negative for black, and it is never zero. Its magnitude is `halfmove_clock + 1`,
where `halfmove_clock` is the FIDE fifty-move counter measured in plies since the
last capture or pawn move. The magnitude therefore lies in `[1, 101]`, and the game
is drawn when it reaches 101.

⚠️ This changed on 2026-07-29. The magnitude was previously a ply counter that
incremented unconditionally, which left the fifty-move rule unrepresentable. It now
resets to ±1 on every capture and every pawn move. `csrc/chess.cuh` now exposes
`advance_control(control, reset)`, which takes the reset as a parameter; no
implementation passes it yet, so both engines still increment unconditionally and
every position they produce carries a ply counter rather than a clock. The reset
lands in A2, and it invalidates every test set dumped before it.

The absolute ply number is a per-game scalar held outside the position. The
self-play temperature schedule and the maximum game length read it; no rule does.

### 2.3 Hash

One `uint64` Zobrist hash per position, maintained incrementally as specified in §6.

### 2.4 Layout in memory

The movegen and the embedding gather both depend on this layout, so it is
normative:

```
boards   [N, 32]  uint16   row-major, 64 B per position, 64 B aligned
control  [N]      int16
hash     [N]      uint64
```

One warp owns a position and lane `i` owns slot `i`, which makes a position exactly
one 64-byte coalesced load.

### 2.5 Invariants a kernel may rely on

- exactly 32 slots at all times, with no compaction and no reallocation
- exactly one king per colour, in slots 15 and 31, never captured
- a slot's colour bit is fixed while the piece is alive
- at most one pawn carries the en passant `special` bit
- `square` is meaningless when `captured` is set
- two live slots never hold the same square, so two live pieces never share an
  identical word. §7.2 relies on this to distinguish tokens without a slot index.

---

## 3. Moves and the action space

A move is an integer in `[0, 2048)`:

```
move = slot * 64 + target_square
```

`-1` is the null move, defined in §9. The action space is exactly 32 × 64, matching
the shape of the policy head, and nothing may be added to it without breaking that
correspondence.

Promotion is carried outside the move space. A pawn move onto the last rank is a
promotion, and the piece it promotes to is a 2-bit field accompanying the move,
produced by the head in §7.4. All four choices are available, so the rules are
complete.

A tree edge is the pair `(move, promo)` with `promo ∈ {0:N, 1:B, 2:R, 3:Q}`, which
is ignored unless the move is a promotion. A promoting move expands to four
children and any other move to one.

Legality is returned as `[N, 32] uint64`, where bit `s` of word `p` is set iff the
move `p -> s` is legal. This inverts the convention used internally by the PyTorch
reference, and it is the only convention that appears in this repository.

---

## 4. The engine contract

Three device-side entry points. None may synchronise with the host, and each
operates on a batch whose positions belong to different games at different stages.

### 4.1 `movegen`

```
in   boards [N,32] u16,  control [N] i16
out  mask   [N,32] u64,  in_check [N] u8
```

The mask is fully legal rather than pseudo-legal, which requires: table lookup per
`(type, square)`; pawn pushes, double pushes, diagonal captures and en passant;
slider occlusion along the four directions; castling, including the rook's identity
and the three squares the king crosses; removal of every target occupied by a
friendly piece; and a second-order pass removing every move that leaves one's own
king attacked.

`in_check` falls out of the second-order pass and is required. An all-zero mask
means checkmate when `in_check` is set and stalemate otherwise, and no other output
distinguishes the two.

Repetition and the fifty-move rule are termination conditions and stay out of the
movegen, which would otherwise carry per-game state through the hottest kernel in
the system.

### 4.2 `step`

```
in   boards, control, hash,  move [N] i16,  promo [N] u8
out  boards', control', hash',  irreversible [N] u8
```

Applying one move per position requires:

- moving the piece and wiping the captured slot to `0x800`, remembering that the en
  passant victim does not stand on the target square
- overwriting the pawn's type bits with `promo` on a promotion
- moving the rook leg on a castle
- setting `special` on a king or rook that moved, then clearing `special` on every
  pawn before setting it on a pawn that just double-pushed
- flipping the sign of the control word, and setting its magnitude to 1 on a capture
  or a pawn move, otherwise incrementing it
- updating the hash incrementally (§6.1)
- setting `irreversible` when the move was a capture, a pawn move, or a change of
  castling rights (§6.2)

### 4.3 `terminal`

```
in   mask, in_check, control, hash, per-game history
out  code [N] u8,  result [N] i8
```

| code | condition |
|---|---|
| 0 | none |
| 1 | checkmate: mask all zero and `in_check` set |
| 2 | stalemate: mask all zero and `in_check` clear |
| 3 | fifty-move: control magnitude reaches 101 |
| 4 | threefold repetition (§6.3) |
| 5 | insufficient material: no pawns, rooks or queens, and then either every bishop on a single colour complex with no knights, or exactly one knight and no bishops |

**This table is the one definition, and `env.TERMINAL_NAMES` is the one copy of it in
code** — `{0: unfinished, 1: checkmate, 2: stalemate, 3: fifty_move, 4: threefold,
5: insufficient}`. Every counter, log key and result table names codes from it rather
than restating them: `search.md` §15's block emits `terminal_checkmate` and
`terminal_threefold`, not `terminal_codes/1` and `terminal_codes/4`, and `eval/`'s
result tables use the same words. Three modules had grown private copies of this
mapping by 2026-07-31, which is how a code eventually gets two names and a plot lies.

⚠️ **Code 0 means two different things by context**, because it is the *absence* of a
terminal rather than a terminal. In a training log it means the ply cap fired, and
[`training.md`](training.md) §5.4 scores that as a draw; in an evaluation log it means
the game was abandoned and [`evaluation.md`](evaluation.md) §5.4.6 drops it rather
than scoring it.

`result` is taken from the side to move's point of view, giving `-1` for checkmate,
since the side to move is the side that is mated, and `0` for every draw. A value of
`+1` never occurs, because a position is never terminal in favour of the player
about to move.

Several draw conditions can hold at once. They all give the same `result`, so the
code reports whichever is found first and nothing downstream depends on the order.

⚠️ The material rule changed on 2026-07-29. It previously listed four cases, K-K,
K+B-K, K+N-K and K+B-K+B with same-colour bishops, which misses K+2B against K with
both bishops on one colour complex. python-chess calls that a draw, underpromotion
to bishop puts it inside reach of self-play, and python-chess is the oracle §10
asserts against, so the rule now follows it.

---

## 5. Termination policy

Termination uses the automatic forms throughout, ending the game at the third
repetition and at halfmove clock 101 without a claim. Claim logic exists to give a
human player a choice, and reproducing it in self-play would double the size of the
state machine for no benefit.

---

## 6. Hashing and repetition

### 6.1 Zobrist

781 random `uint64` keys, 6.25 KB total, generated once from a fixed seed and
shipped as a table:

| keys | count |
|---|---|
| piece-square: colour × type × square | 2 × 6 × 64 = 768 |
| side to move | 1 |
| castling rights, one key per right | 4 |
| en passant file | 8 |

Keys are indexed by `(colour, type, square)`. Indexing by slot is incorrect: two
positions with identical boards but pieces in different slots, such as a promoted
queen in slot 3 against the original queen in slot 14, are the same position under
FIDE and must produce the same hash. This is the point where the piece-list
representation diverges from a bitboard engine's Zobrist.

The en passant file is included only when an en passant capture is actually legal,
because FIDE defines repetition by the same en passant possibility. Hashing a
dangling flag makes real repetitions invisible.

The update is incremental, costing four to six XORs per move: the moving piece out
and in, any captured piece out, the side key, and the deltas of the castling rights
and the en passant file.

The table lives in shared memory. `__constant__` broadcasts efficiently but
serialises on divergent addresses within a warp, and these lookups are per-piece.

### 6.2 The repetition window

Positions before the last irreversible move can never recur, which bounds the search
window. A move is irreversible when it is a capture, a pawn move, or a change of
castling rights.

That third condition differs from the fifty-move rule, whose clock resets only on
captures and pawn moves. Castling rights only ever decrease, so a rights change also
makes every earlier position unreachable. The repetition window is the shorter of
the two spans, and conflating them produces false repetitions.

The window never exceeds 100 plies, since the game ends at clock 101.

### 6.3 Detecting a repetition on device

Repetition is checked in two places, neither of which is the movegen.

Inside the search tree, MCTS already walks root to leaf on every simulation and
therefore already visits every ancestor. Each ancestor's hash is collected on the way
down, accumulation stops at the first ancestor whose move was irreversible, and the
new leaf hash is compared against what was collected. This adds no memory traffic,
since those loads were already happening. Walking up a parent chain at expansion
time instead would cost a serial pointer chase of around ten dependent loads at
hundreds of cycles each, with 31 lanes idle.

Before the root, positions from earlier in the actual game live in a per-game ring
of at most 100 `uint64`, or 1 KB per game. Every simulation of a given game shares
that ring, so it is loaded into shared memory once per block and scanned by 32 lanes
in at most four iterations.

Two early outs keep the average case negligible: a clock below 4 makes repetition
impossible, and an irreversible move empties the ring.

Worst-case traffic at Gate-1 rate is `45k × 100 × 8 B = 36 MB/s` against 272 GB/s of
bandwidth, or 0.013 %. Per node the scheme adds 8 bytes of hash and one bit of
irreversibility.

---

## 7. The network contract

### 7.1 Shape

32 tokens for 32 slots, index-aligned with the engine's 32 mask words, non-causal,
with `d = 256`, `L = 8`, `H = 8` and `FFN = 1024`. That gives 6.32M parameters in the
stack and roughly 400 MFLOPs per evaluation.

The whole network is **6,383,360 parameters**: 6,318,080 in the stack, 47,104 in the
five embedding tables of §7.2, 512 in `norm_f` and 17,664 in the three heads of §7.4.
That total is the one the cost curve is plotted against, so it is worth stating
rather than leaving as a sum a reader has to do.

#### 7.1a The block input has a second form, off by default

⚠️ Added 2026-08-28, as an **option**: `--reinject` gives each block's LayerNorm

$$\mathrm{LN}\Big(h_\ell + \sum_{k=1}^{5} c_{\ell k} E_k\Big)$$

instead of `LN(h_ℓ)`, where the five `E_k` are the embedding lookups of §7.2 at the same
indices the prologue used and `c` is a learned scalar per (site, source). `ln1` is the
attention norm of every block, 8 sites and **40 parameters**; `both` adds the FFN norm,
16 sites and 80. `none` is the default and is the architecture every number on the
ledger was produced by.

**The residual stream is unchanged**: the mix enters the norm's argument only, so
`h_{ℓ+1} = h_ℓ + attn(·) + ffn(·)` still holds exactly. The coefficient on `h_ℓ` is
fixed at 1 and nothing is lost by that -- LayerNorm is scale-invariant, so only the
ratio between the residual and the injected term is observable.

⚠️ **The coefficients are zero-initialised and the kernel is bit-identical to `none` at
`c = 0`**, on both the fp16 and the int8 path. That is a contract, not an accident: it
is what lets a checkpoint written without re-injection be loaded into a net with it, and
what makes an A/B of the flag an A/B of one variable.

⚠️ **The source order is normative** -- square, type_special, color_turn, clock, rep --
because `c` is indexed by it in the `state_dict` and in the kernel's coefficient slab.
It is the order of `EmbOff` in `csrc/encoder.cu` and of `PackedWeights.EMB_TABLES`.

The measured cost is +3.8 % of evals/s at `ln1` and +5.2 % at `both`
(`ledger/perf.md`, `journal/2026-08-28-input-reinjection.md`).

### 7.2 Token features

The token embedding is a sum of table lookups, so each feature costs one gather:

```
token[p] = emb_square[square]            # 64 entries, also the positional encoding
         + emb_type_special[type, spec]  # 12 entries
         + emb_color_turn[color, stm]    #  4 entries
         + emb_clock[halfmove_clock]     # 101 entries, global
         + emb_rep[repetition_count]     #  3 entries, global
```

184 rows and 47k parameters in total.

Three things about this sum are **normative**, because two implementations compute
it and a disagreement between them is a wrong number rather than a crash.

The two-dimensional tables are flattened row-major: `type * 2 + special` and
`color * 2 + stm`, with `stm = 1` for black to move. `type` is three bits but only
six values are defined; an implementation must not let 6 or 7 index past the table.

`emb_clock` and `emb_rep` are indexed per position, not per token, so they
contribute the same vector to all 32 tokens. The clock index is `|control| - 1`.

The summation order is `(square + type_special + color_turn) + (clock + rep)`,
left-associative. In fp16 a different grouping is a different result in the last
bit, and pre-summing the two per-position tables is what lets a kernel pay for
them once per board instead of once per token. `tests/test_b2.py` asserts the
CUDA gather is **bit-identical** to the reference, not merely close, because
neither side has any rounding of its own to hide a regrouping behind.

Several of the choices above need justification:

`emb_square` is the only positional signal, since the square a piece stands on is
its position.

There is no slot-index embedding. The slot carries no information the rules use, and
omitting it leaves the network permutation-equivariant over tokens, so it learns
"pawn on e4" once instead of eight times. Output routing does not need it either,
because token `p`'s logits go to row `p` by construction. §2.5 makes this safe by
guaranteeing that two live pieces never share a word.

`type` and `special` share one 12-entry table because the bit means "castling right
lost" on a king or rook and "en passant target" on a pawn. Two independent tables
summed together cannot express a meaning conditioned on the type. `color` and `stm`
share a 4-entry table for the same reason, since "my piece" is a product of the two.

The clock uses a full 101-entry table rather than buckets. The function is
non-smooth near the boundary and 26k parameters is cheap.

The clock and repetition count are both required inputs. The same 32 words at clock
8 and at clock 98 have different outcomes, and a position seen twice stands one
repetition from a draw. Omitting them leaves the value target ill-defined and asks
the head to fit noise.

### 7.3 Dead tokens

Dead tokens are masked out of attention, and the mask is mandatory.

Under the mask a dead token influences nothing. Other tokens do not see it as a key,
the engine mask kills its policy logits, and the value comes from a king token that
is never dead. No gradient reaches it, so its embedding stays inert and needs no
representation. There is no captured-piece embedding and no dead token to learn.

Without the mask, a dead token's embedding is `emb_square[a1] + emb_type[pawn] + …`,
which avoids colliding with a live piece only because pawns cannot stand on a1, and
which ties dead slots to two live embeddings that then cannot move freely. A
performance pass that wants to drop the mask has to re-open this section.

The cost is a dynamic 32-bit alive mask per board, broadcast into the `[32,32]`
score tile before the softmax. The compiler already eliminates the static
block-diagonal mask at T=32, but this one is dynamic and survives.

### 7.4 Heads

**A final LayerNorm comes first.** The stack is pre-norm, so what it emits is a raw
residual stream whose scale grows with depth, in fp16. `norm_f` normalises it before
any head reads it. This is not optional and it is not part of the stack: the fused
kernels treat it as the first step of the head epilogue, which also gives them a
row-padded buffer to read (`2026-07-29-encoder-kernel.md` §15).

Three heads, all **biasless**, applied to all 32 normed tokens:

| head | shape | meaning |
|---|---|---|
| `W_p` | `[256, 64]` | policy logits, in exact correspondence with the engine's 32 mask words |
| `W_promo` | `[256, 4]` | promotion logits per token, softmax over (N, B, R, Q) |
| `W_value` | `[256, 1]` or `[256, 3]` | read at one token or pooled over all live tokens; see below |

**The value head has two shapes and one output.** `[256, 1]` is the original and the
default: one logit through `tanh`, trained against the outcome with a squared error.
`[256, 3]` is a **win/draw/loss classifier**, trained with a cross-entropy, and is
selected by `--value-classes 3` (added 2026-08-21). Its class order is normative —
**0 = loss, 1 = draw, 2 = win, from the side to move**, so a stored outcome
`z in {-1, 0, +1}` has class index `z + 1`, and `csrc/encoder.cu`'s epilogue reads
those three columns by position.

⚠️ **Both shapes emit the same `[N]` fp32 in `[-1, 1]`**, the classifier through
`p(win) - p(loss)`. That is what makes this an option rather than a rewrite: the
search (`search.md` §3.5), the terminal collapse, the value-head puzzle probe and the
whole Elo pipeline consume one scalar and cannot tell which head produced it. A
draw-aware search — a separate draw term in the utility, as KataGo has — would be a
**different** change, to the search and not to the head.

**The value head has three inputs and two frames.** `king` is the original: a row
select of the side-to-move king's token, slot 15 or 31, which spec §2.5 guarantees is
never captured, predicting from the **mover's** point of view. `pooled` is the
**masked mean of every live token, both colours**, predicting in an **absolute**
frame — White / draw / Black, so `z + 1` is the class index of the outcome *from
White's side* — which the head then flips into the mover's by the sign of the control
word. Selected by `--value-head pooled` (added 2026-08-22). `prenorm` pools the **raw**
residual stream and applies `norm_f` to the pooled vector rather than pooling vectors
`norm_f` has already normalised; same absolute frame, same flip
(`--value-head prenorm`, added 2026-08-23).

⚠️ **`prenorm` exists because a mean of normed vectors is not normed.** Measured on
`t12h-wdb`, `|mean(LN(h))|` drifts **15.47 → 10.86** across a 12 h run as the token
cloud spreads, `|value.weight|` grows **+42 %** compensating, and
`value_saturated_frac` reaches 0.111 — a readout chasing an input whose scale moves.
`LN(mean(h))` has its scale set by the norm and cannot drift. Pooling before the norm
also lets a token with a larger residual contribute more, which an unweighted mean of
normed tokens cannot express.

⚠️ **`prenorm` is the one head that is not free.** `LN(mean(h))` is not a linear
function of the per-token value logits, so an implementation cannot average columns it
already has: `csrc/encoder.cu` pools `bufA`, norms it once and takes three dot products
against an unpacked copy of the value rows kept at `TailOff::w_val`. Measured
ABBA-interleaved at B = 4096: **+0.49 %** against the king select, which is *less* than
the pooled head's +0.80 %.

⚠️ **Dead slots are excluded from the mean.** A captured slot is `1 << 11` with
colour, type and square wiped, so §2.1's warning applies: it decodes as a live white
pawn on a1 and its head output is a real vector that means nothing. Averaging it in
would make the value track how many pieces have been taken, by an accident of the
encoding.

⚠️ **Pooling after the head equals pooling before it.** `norm_f` is per token and sits
upstream of the pool, and the heads are biasless, so `W @ mean(hn) = mean(W @ hn)`
exactly. An implementation may average the 32 per-token value logits it has already
computed instead of running a second GEMM on a pooled vector, and `csrc/encoder.cu`
does. In fp16 the two orders are different numbers, which is what the tolerance in
`tests/test_b2.py` is for — measured, the pooled path is *tighter* than the row select
(9.4e-4 against 3.9e-3), because averaging 32 logits cancels rounding.

⚠️ **A checkpoint does not say which head it has.** Checkpoints are bare
`state_dict`s with the architecture in the constructor, so the width is recovered
from `value.weight.shape[0]` (`nn/model.py:n_value_of`) and the *input* from the
presence of a `value_mode` buffer (`value_head_of`), which is registered only in the
pooled case so that every file written before it still loads under `strict=True`.
Every loader in the repository goes through them, which is the only reason a
scalar-head checkpoint — `checkpoints/anchor.pt` among them, and every Elo scale
anchors to that one — can still be rated against a classifier-head one in a single
Bradley-Terry fit.

`norm_f` keeps its affine, so `beta` is a learned 256-vector every head sees and
each head's effective bias is `W_h @ beta` — any vector in that head's output
space, since every head is wider in than out. What the biasless choice gives up is
independence between the three heads' biases, not expressiveness.

`promo` is per token. A position can have two pawns on the seventh rank, and the
factorisation `P(target | pawn) · P(type | pawn)` gives each its own distribution.
It is indexed by the **moving slot** of the move, not by the pawn slot range: a
promoted queen keeps its pawn slot, so a slot in 0-7 does not imply a pawn and
§2.1's warning applies here more than anywhere.

**The policy output is not masked.** The encoder emits raw logits and the legality
mask is applied by the search. Three reasons, settled 2026-07-29: the search needs
a masked softmax anyway, so masking here means doing it twice or constraining what
C1 can fuse; the encoder stays a pure function of the position, which is what lets
its tests run with no movegen fixture and removes a dependency on a kernel that does
not exist yet; and it costs the same either way — in registers if fused, or 8 KB per
board of traffic if not, which is 0.5 GB/s on a 250 GB/s bus. The requirement this
section used to state, a device-side AND with no host round-trip, is unchanged and
is satisfied wherever that op runs, as long as it runs on the device.

⚠️ Whoever applies it: an all-illegal row is a terminal position that should never
have been expanded. With `-inf` it softmaxes to `NaN` and with `-65504` to uniform.
The failure is worth having loud, so use `-inf` **and** assert `mask.any(dim=-1)`;
with `-65504` that assertion is not optional, it is the only thing that catches it.

The output tensors are part of the contract, because the search is written against
them and the value would otherwise cost every consumer a data-dependent gather:

```
policy_logits [N, 32, 64]  fp16      raw
promo         [N, 32,  4]  fp16      raw
value         [N]          fp32      squashed to [-1, 1], king row already selected
```

At B = 16384 the three heads cost 2.1 GFLOP against a 6.5 TFLOP forward, or 0.03 %.
Placing them in the megakernel epilogue or in a separate small GEMM is an
implementation choice outside this contract, and 64 remains the tile-friendly
policy width either way.

Reading the value from the king token exploits two properties: slots 15 and 31 are
guaranteed alive, and the index is constant given the sign of the control word. The
read is one row select with no reduction and no layout change, and the value arrives
from the mover's point of view. A masked mean over live tokens would require a
reduction along the token axis, which is orthogonal to the axis the LayerNorm
reductions already run along.

Reading the promotion type from the pawn's own token factorises the prior as
`P(target | pawn) · P(type | pawn)`, so a pawn with three promotion targets carries
one shared distribution over types. Legality is unaffected, since the search still
expands all four choices per target square, and the shared prior applies to an event
of frequency around 10⁻⁴.

### 7.5 No canonicalisation

The position is stored and presented as it stands. The board is never flipped,
colours are never swapped, and there is no side-to-move-relative frame.

AlphaZero canonicalises because it has a single-headed policy that needs a fixed
point of view. Brokefish has a per-piece policy in which every token emits its own
logits on every forward and the mask selects who is to move, which removes the
requirement.

Colour-flip augmentation, meaning `square ^= 56` together with `color ^= 1`, is also
unused. The transform preserves legality, value, castling and en passant, but it
breaks reachability parity: the image of "white played e2-e4, black to move" is
"black played e7-e5, white to move", which requires black to have moved first. The
augmented distribution is therefore shifted relative to the inference distribution.
It stays available behind an experiment flag and is decided by measurement.

The accepted cost is that gradient reaches only the mover's tokens, so the network
learns each colour separately. That is a deliberate sample-efficiency tax, and the
flag exists to find out later what it was worth.

---

## 8. Loss and targets

The training objective is regression against fixed targets in the AlphaZero family:
cross-entropy of the policy head against the search's visit distribution, and
regression of the value head against the game's outcome.

The distinction matters when data is reused. PPO's ratio clipping corrects an
expectation taken over trajectories sampled from the behaviour policy, whereas the
search's visit distribution here is a stored label. Reusing it introduces no bias in
the gradient and gives a ratio nothing to correct.

Reuse still costs something, and the two heads pay differently. For the policy head,
the stored label is `I(θ_t)`, a weaker improvement operator than `I(θ_now)`, so
training on it pulls the network backwards and slows convergence in proportion to
staleness without biasing the gradient. For the value head, `V^π` is defined with
respect to a policy, so outcomes from games played by an older network estimate
`V^{π_old}`, which is a genuine bias.

A bounded training window over recent games controls both. Importance weighting
controls neither.

---

## 9. Batch semantics

No part of a self-play step may touch the host, so a batch always holds positions
from different games at different stages.

`move == -1` is a no-op that leaves the board, the control word and the hash
untouched, which is how a finished or paused game rides along in a batch. The
PyTorch reference lacks this, which is why its test harness drops finished games
from the batch instead.

Terminal positions remain in the batch as frozen no-ops until an explicit compaction
step, and compaction belongs to the search layer.

Every kernel is total: no early return skips a lane, and no kernel assumes a batch
is homogeneous in side to move, game phase or terminal status.

---

## 10. Correctness bars

A move generator that is 99 % correct invalidates every downstream Elo measurement,
so the suite below gates the kernel rather than tracking regressions.

python-chess is the oracle. The PyTorch engine in `~/steakfish` serves as a
convenience oracle for first-order legality and as a source of implementation ideas.
It lacks promotion in its mask, repetition, insufficient material and the halfmove
clock as specified here, which disqualifies it as the reference for those.

The suite adapts the fuzzing harness in `~/steakfish/tests/test_moves.py`:

- draw random real positions, assert the legal-move set equals python-chess's, then
  random-walk *t* plies re-asserting at every step, single and batched
- delete `IGNORE_PROMOTIONS`, since comparisons run on full `chess.Move` objects
  including `.promotion`, which is the hole this specification closes
- assert `in_check` against `board.is_check()`
- assert the terminal code against `is_checkmate()`, `is_stalemate()`,
  `is_insufficient_material()`, `is_fifty_moves()` and `is_repetition(3)`
- assert the control word against `board.halfmove_clock` and `board.turn`
- run perft to known node counts: startpos to depth 6, Kiwipete, and positions 3, 4
  and 5 of the standard suite

The performance bar is 50-100M boards/s on the RTX 4060, benchmarked in nodes/s
against a locally recompiled `perft_gpu` and against Stockfish on CPU. The rate the
system needs is around 45k/s, four orders of magnitude lower; the bar exists to keep
the environment under 0.5 % of the self-play budget with margin.

---

## 11. The RL layer

Outside this document's scope, and none of §§1-10 depend on it. This section is a
pointer, not a contract: it records where each parameter was settled and which
document now owns it. **Where a row below and its owning document disagree, the
owning document is correct** — the reverse of the rule in the header, because these
values are not part of the engine-network contract.

| question | where it stands |
|---|---|
| **Value target** | settled 2026-07-30: **the final game outcome**, no bootstrapping and no mixing, per AZ p.3. `root_value` is recorded and trained on by nothing, so KataGo's mix stays a cheap later ablation. [`training.md`](training.md) §4 |
| **Training window** | settled 2026-07-30: **AZ's literal 500,000 games**, uniform over all positions in the window, evicted by game, oldest first. [`training.md`](training.md) §5.2 |
| **Reuse factor R** | settled 2026-07-30: **65.2 positions sampled per game generated**, derived from AZ's 700,000 × 4,096 steps against 44M games. [`training.md`](training.md) §6 |
| **Simulations per move** | settled: **`n = 800`, AlphaZero PUCT**, chosen for convergence rather than for throughput. The sweep downward is a C4 measurement. [`search.md`](search.md) §4.4 |
| **Tree node layout** | settled by C1 and normative there: the node arrays, `E = 96` children per node, and the pool's bump allocator. [`search.md`](search.md) §4.2, §4.3. §6.3 here still owns the hash and the irreversible bit |
| **Learner placement** | settled: **GPU, alternating with self-play**, on 5000 positions/s against 130 on the CPU. Revisit if the sims sweep moves the requirement |
| **Playout cap randomisation** | **still open.** KataGo decouples the cost of value and policy targets this way, which would turn "sims" into `(n_small, n_large, p_large)` and make positions stop costing the same. Priced as a seam in [`search.md`](search.md) §11, not scheduled |

---

## Changelog

**v1, 2026-07-29.** First freeze, superseding `docs/legacy/representation.md`.
Changes against what was previously implemented or documented:

- the control word's magnitude becomes a halfmove clock that resets on captures and
  pawn moves, replacing an unconditional ply counter (§2.2)
- promotion becomes complete, with four choices from a dedicated head replacing
  queen-by-default plus a side-channel tensor (§3, §7.4)
- repetition, insufficient material and the fifty-move rule become representable
  through the Zobrist hash and the per-game ring (§6)
- the engine emits `in_check` and a terminal code, where mate and stalemate were
  previously indistinguishable (§4)
- `move == -1` becomes a no-op, making batches heterogeneous by construction (§9)
- the network gains a value head and a promotion head (§7.4) along with clock and
  repetition features (§7.2)
- dead tokens are masked in attention, mandatorily (§7.3)

**v1.1, 2026-07-29.** Insufficient material widened to agree with python-chess,
which the four-case list of §4.3 contradicted on K+2B against K (§4.3).
- canonicalisation and colour-flip augmentation are both dropped (§7.5)

**v1.2, 2026-07-30.** Written while B2 was implemented, so every clause below is
one an implementation needed and the spec did not have.

- a final LayerNorm before the heads becomes normative, where the pre-norm stack
  previously fed its raw residual stream to a linear head (§7.4)
- the single `W_a : [256, 8]` aux head becomes `W_promo : [256, 4]` and
  `W_value : [256, 1]`; the reserved dims 5-7 are dropped, since padding to a
  tile-friendly width is the kernel's business and not the model's (§7.4)
- all three heads become biasless, with the justification that `norm_f`'s `beta`
  already supplies an effective per-head bias (§7.4)
- the policy output stops being masked here and the mask moves to the search,
  with the all-illegal-row hazard written down (§7.4)
- the output tensor shapes become part of the contract, and `value` arrives as a
  ready `[N] fp32` rather than a slice of an aux tensor (§7.4)
- the embedding sum gains a normative flattening, a normative summation order,
  and the statement that clock and repetition are per position (§7.2)

**v1.3, 2026-07-30.** §11 only. **Nothing normative changed**: §§1-10 are untouched
and the engine-network contract is the same one frozen at v1.

§11 had gone stale in a way that mattered, because it is the frozen document and it
was still calling settled questions open. Six of its seven rows were decided between
2026-07-29 and 2026-07-30 by `search.md` and `training.md` — value target, training
window, reuse factor, simulations per move, tree node layout, learner placement —
and the simulations row in particular still read "32 sits in the Gumbel low-*n*
regime", which v0 is not: the search is AlphaZero PUCT at `n = 800`. The section is
now a pointer to whichever document owns each parameter, and says so, rather than a
second place where those values are written down. Playout cap randomisation is the
one row still genuinely open. The header's reference to `briefing.md` goes with the
file, which was deleted the same day.
