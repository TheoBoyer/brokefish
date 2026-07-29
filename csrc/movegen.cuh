// First-order move generation on the 12-bit piece-list representation.
//
// "First order" is every rule except king safety: the per-(type, square) table,
// pawn pushes and captures and en passant, slider occlusion, castling rights and
// the empty squares they need, and the removal of friendly targets. The
// second-order pass that drops moves leaving one's own king attacked is not here
// yet, which is why this header exports no `movegen`.
//
// The same code instantiated with kControl = true is the attack map: the pawn
// capture diagonals stop being conditional on an enemy standing there, and the
// friendly-occupancy filter is skipped, because a defended square is one the
// enemy king may not take. That instantiation is what `in_check` reads today and
// what the whole second order will read.
//
// One warp owns a position and lane i owns slot i. That is what makes the 32
// piece words a single 64-byte load, turns piece-list -> bitboard into
// `__reduce_or_sync` rather than a loop over 32 slots, and makes "is there an
// unmoved rook on h1" a single `__any_sync` ballot. Half the lanes hold the side
// that is not to move and produce nothing; that is the price of the alignment,
// and it is not paid back until the second order, where every lane will replay a
// different candidate move.
//
// A header rather than a .cu because C1's descent kernel expands nodes in the
// middle of a tree walk and has to call this from inside a kernel, never across
// a launch boundary.
//
// The reference is brokefish/env/torch_impl.py::first_order_mask, which is
// perft-green against python-chess. This file reproduces it bit for bit; the
// three places it deviates are shifts by a negative or out-of-range amount,
// which torch tolerates and C++ leaves undefined. Each is marked below.
#pragma once

#include <cstdint>

#include "chess.cuh"
#include "step.cuh"

namespace brokefish {

// The four tables of brokefish/env/luts.py, in the exact layout `dump()` writes.
struct Luts {
    const uint64_t* move_bitsets;  // [6*64]   (type*64 + square) -> destinations, occupancy ignored
    const int16_t* occl_offsets;   // [4*64*8] (dir, square, i)   -> square of cell i of the line
    const uint8_t* occl_masks;     // [4*64*8] (dir, square, i)   -> cell i is on the board
    const uint8_t* filled_lines;   // [8*256]  (our cell, line occupancy) -> cells we reach
};

// 11 264 B, loaded once per block. Small enough that it never competes with
// occupancy on this card (99 KB per block), so there is no reason to page it.
struct SharedLuts {
    uint64_t move_bitsets[6 * 64];
    int16_t occl_offsets[4 * 64 * 8];
    uint8_t occl_masks[4 * 64 * 8];
    uint8_t filled_lines[8 * 256];
};

// Every thread of the block must call this; it ends on the barrier that makes
// the tables visible to the whole block.
__device__ inline void load_luts(SharedLuts& s, const Luts& g) {
    smem_copy(s.move_bitsets, g.move_bitsets, 6 * 64);
    smem_copy(s.occl_offsets, g.occl_offsets, 4 * 64 * 8);
    smem_copy(s.occl_masks, g.occl_masks, 4 * 64 * 8);
    smem_copy(s.filled_lines, g.filled_lines, 8 * 256);
    __syncthreads();
}

// A rook still holding its castling right, as a whole piece word: alive,
// our colour, type rook, `special` clear, standing on the corner. Comparing the
// word is what checks all five conditions in one instruction.
__device__ inline uint16_t unmoved_rook_word(bool black, int square) {
    return (uint16_t)(((int)black << 10) | (kRook << 6) | square);
}

// ---------------------------------------------------------------------------
// Pawns
// ---------------------------------------------------------------------------

// `ep_targets` is the bitboard of *enemy* pawns carrying the double-push flag.
// A promotion is one bit here and not four: spec §3 fixes the action space at
// 32 x 64 and carries the promotion type beside the move, so a push or capture
// onto the last rank sets a single bit and the caller expands it.
template <bool kControl>
__device__ inline uint64_t pawn_mask(const Piece& p, uint64_t occupancy,
                                     uint64_t opp_occupancy, uint64_t ep_targets) {
    const int row = p.square >> 3;
    const int col = p.square & 7;
    // A black pawn pushes to square - 8 and a white pawn to square + 8;
    // subtracting 16 from a black pawn's square first gives one expression for
    // both, and the same trick then serves the double push and the diagonals.
    const int base = p.square - 16 * (int)p.color;

    const int forward = base + 8;
    const uint64_t forward_free = (~(occupancy >> forward)) & 1ULL;
    uint64_t mask = forward_free << forward;

    // DEVIATION 1: torch computes the double-push target for every pawn and then
    // multiplies by `on_start_row`. A black pawn on rank 2 gives a negative
    // target, which is a negative shift here, so the index is wrapped into range
    // and the result discarded instead.
    const int dbl = (p.square - 32 * (int)p.color + 16) & 63;
    const bool on_start_row = p.color ? (row == 6) : (row == 1);
    const uint64_t dbl_free = (~(occupancy >> dbl)) & forward_free;
    mask |= (on_start_row ? dbl_free : 0ULL) << dbl;

    // In control mode the diagonals are unconditional: an empty square a pawn
    // covers is still a square the enemy king may not step onto.
    const uint64_t targets = kControl ? ~0ULL : opp_occupancy;
    // DEVIATION 2: `base + 7` is negative and `base + 9` overruns 63 exactly on
    // the files where the diagonal leaves the board, so the guard has to come
    // before the shift rather than after it as it does in torch.
    if (col > 0) mask |= (1ULL << (base + 7)) & targets;
    if (col < 7) mask |= (1ULL << (base + 9)) & targets;

    // En passant: an enemy pawn that just double-pushed stands beside us on our
    // own rank, and the capture lands behind it. The rank test is what stops a
    // flag on the far side of the board from being read as an adjacency.
    if (row == (p.color ? 3 : 4)) {
        if (col > 0 && ((ep_targets >> (p.square - 1)) & 1ULL)) mask |= 1ULL << (base + 7);
        if (col < 7 && ((ep_targets >> (p.square + 1)) & 1ULL)) mask |= 1ULL << (base + 9);
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Sliders
// ---------------------------------------------------------------------------

// Occlusion, one direction at a time. Each direction is described not as a ray
// but as the whole 8-cell line through `square`, which turns "where does the ray
// stop" into a byte lookup: pack the line's occupancy into one byte, ask
// `filled_lines` which cells are reachable from our cell index, scatter back.
//
// The reachable set includes the first blocker on each side whatever its colour,
// so friendly captures are still in the mask here and are removed by the
// occupancy filter at the end of `first_order_mask`.
__device__ inline uint64_t slider_mask(const SharedLuts& s, int type, int square,
                                       uint64_t occupancy) {
    // Which directions this piece uses: bishop (2) -> 0b01, the two diagonals;
    // rook (3) -> 0b10, the rank and the file; queen (4) -> 0b11.
    const int dirs = type - kKnight;
    uint64_t mask = 0;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        if (!((dirs >> (d >> 1)) & 1)) continue;
        const int16_t* off = &s.occl_offsets[(d * 64 + square) * 8];
        const uint8_t* valid = &s.occl_masks[(d * 64 + square) * 8];

        int line = 0;   // the line's occupancy, one bit per cell
        int pos = 0;    // our own cell index along the line
        bool found = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            // DEVIATION 3: cells past the edge of the board carry an offset well
            // outside [0, 64), so the shift is wrapped and then killed by
            // `valid` rather than being evaluated raw as it is in torch.
            const unsigned o = (unsigned)off[i];
            line |= (int)(((occupancy >> (o & 63u)) & 1ULL) & valid[i]) << i;
            if (!found && (int)off[i] == square) { pos = i; found = true; }
        }

        const uint8_t filled = s.filled_lines[pos * 256 + line];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (((filled >> i) & 1) && valid[i]) mask |= 1ULL << ((unsigned)off[i] & 63u);
        }
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Castling
// ---------------------------------------------------------------------------

// Only the rook's identity and the empty squares. The three squares the king
// crosses are the second-order pass's job.
//
// Note for the attack map: this runs in control mode too, matching the
// reference, and it cannot corrupt it. A castling bit only ever lands on g1/c1
// or g8/c8, and it is only set when those squares are empty, so it can never
// cover a king and can never turn a quiet position into a false check.
__device__ inline uint64_t castle_mask(bool black, uint64_t occupancy,
                                       bool short_rook, bool long_rook) {
    const int home = black ? 56 : 0;
    uint64_t mask = 0;
    if (short_rook && ((occupancy >> (5 + home)) & 0b11ULL) == 0) mask |= 1ULL << (6 + home);
    if (long_rook && ((occupancy >> (1 + home)) & 0b111ULL) == 0) mask |= 1ULL << (2 + home);
    return mask;
}

// ---------------------------------------------------------------------------
// The first order
// ---------------------------------------------------------------------------

// Warp-collective: every lane of the warp must call it, lane i carrying slot i's
// word, and every lane gets back its own slot's destination bitset.
//
// `black_to_move` is the colour whose moves are being generated, which in
// control mode is the *attacker*, i.e. the side that is not to move.
//
// kStage exists for the test harness only and is a compile-time constant, so the
// shipping instantiation (kStage = 5) pays nothing for it. It stops the pipeline
// after stage 1 base bitsets / 2 pawns / 3 sliders / 4 castling, matching the
// fo_stage*.bin snapshots that scripts/dump_cuda_testset.py writes, because a
// wrong slider is otherwise only visible as a wrong final mask.
template <bool kControl, int kStage = 5>
__device__ inline uint64_t first_order_mask(const SharedLuts& s, uint16_t word,
                                            bool black_to_move) {
    const Piece p = decode(word);
    const bool mine = (p.color == black_to_move) && !p.captured;
    // A captured slot has its whole word wiped to 0x800, so `square` is
    // meaningless and the guard here is what keeps dead pieces out of every
    // reduction below.
    const uint64_t bit = p.captured ? 0ULL : (1ULL << p.square);

    const uint64_t occupancy = warp_or64(bit);
    const uint64_t own_occupancy = warp_or64(mine ? bit : 0ULL);
    const uint64_t opp_occupancy = occupancy ^ own_occupancy;
    const uint64_t ep_targets =
        warp_or64((!p.captured && !mine && p.type == kPawn && p.special) ? bit : 0ULL);

    const bool has_short_rook = __any_sync(
        0xffffffff, word == unmoved_rook_word(black_to_move, black_to_move ? 63 : 7));
    const bool has_long_rook = __any_sync(
        0xffffffff, word == unmoved_rook_word(black_to_move, black_to_move ? 56 : 0));

    // Stage 1. The table is zero for pawns, whose moves depend on colour and
    // occupancy and are built below instead.
    uint64_t mask = mine ? s.move_bitsets[p.type * 64 + p.square] : 0ULL;
    if (kStage >= 2 && mine && p.type == kPawn)
        mask |= pawn_mask<kControl>(p, occupancy, opp_occupancy, ep_targets);
    if (kStage >= 3 && mine && p.type >= kBishop && p.type <= kQueen)
        mask &= slider_mask(s, p.type, p.square, occupancy);
    if (kStage >= 4 && mine && p.type == kKing && !p.special)
        mask |= castle_mask(black_to_move, occupancy, has_short_rook, has_long_rook);
    // Cannot land on one's own piece. In control mode those squares stay set.
    if (kStage >= 5 && !kControl && mine) mask &= ~own_occupancy;
    return mask;
}

// ---------------------------------------------------------------------------
// The second order, and full legality
// ---------------------------------------------------------------------------

struct MovegenResult {
    uint64_t mask;    // this lane's slot: bit s set iff slot -> s is fully legal
    bool in_check;    // the same value on every lane
};

// Fully legal moves and check status, spec §4.1. Warp-collective: every lane must
// call it, lane i carrying slot i's word, and lane i gets slot i's mask back.
//
// The second order is brute force, as in the reference: every pseudo-legal move
// is applied and the resulting position is asked whether our king stands on a
// square the opponent attacks. The whole warp works on one candidate at a time,
// which keeps the mutated board in registers and never touches memory between
// the two halves. The standard pruning (only king moves, pieces on a line
// through the king, and en passant can ever expose it) would cut the ~35 replays
// to a handful and is deliberately not done here: A1 ports the algorithm, and
// nothing has yet measured the environment costing the loop anything.
__device__ inline MovegenResult movegen(const SharedLuts& s, uint16_t word, int lane,
                                        bool black_to_move) {
    const unsigned kAll = 0xffffffff;
    // Our own king. spec §2.5 pins it to slot 15 or 31 and promises it is never
    // captured, so its square is a broadcast from a fixed lane, not a search.
    const int king_slot = black_to_move ? kBlackKingSlot : kWhiteKingSlot;

    // `in_check` is the opponent's attack map over the position as it stands,
    // which is one pass against the ~35 the second order costs. spec §4.1
    // requires it: an all-zero mask is checkmate when it is set and stalemate
    // otherwise, and nothing else separates the two.
    const uint64_t pre_attack = warp_or64(first_order_mask<true>(s, word, !black_to_move));
    const bool in_check =
        (pre_attack >> __shfl_sync(kAll, (int)(word & kSquareMask), king_slot)) & 1ULL;

    const uint64_t pseudo = first_order_mask<false>(s, word, black_to_move);

    uint64_t legal = 0;
    // Only about ten of the 32 slots have any move, so the outer loop walks a
    // ballot of the non-empty ones instead of all 32. Both loops are
    // warp-uniform, which is what lets the collective primitives inside them run
    // on a full warp.
    unsigned slots = __ballot_sync(kAll, pseudo != 0);
    while (slots) {
        const int p = __ffs((int)slots) - 1;
        slots &= slots - 1;
        const uint16_t src = __shfl_sync(kAll, word, p);
        const int src_type = (src >> 6) & 0b111;
        const int src_square = src & kSquareMask;

        uint64_t targets = __shfl_sync(kAll, pseudo, p);
        while (targets) {
            const int target = __ffsll((unsigned long long)targets) - 1;
            targets &= targets - 1;

            // The promotion choice cannot change legality: the promoted piece is
            // ours, and the opponent's attack map sees only its square, through
            // occupancy, never its type. So one replay settles all four edges and
            // the promo argument here is arbitrary.
            const StepResult sr =
                apply_move(word, lane, !black_to_move, p * 64 + target, 3);
            // The side to move always flips, so the new mover is the opponent and
            // the control word does not need rebuilding to know its colour.
            const uint64_t attack =
                warp_or64(first_order_mask<true>(s, sr.word, !black_to_move));

            const int dx = target - src_square;
            bool illegal;
            if (src_type == kKing && (dx == 2 || dx == -2)) {
                // Castling additionally requires that the king neither starts,
                // crosses nor lands on an attacked square. The three squares are
                // contiguous and the destination is among them, so this replaces
                // the test below rather than adding to it.
                //
                // The map is the one *after* the move, with the rook already on
                // f1, which is what the reference does and what perft and the
                // differential harness validate. It agrees with the FIDE rule
                // because neither leg can hide an attack on those three squares:
                // the rook lands inside the ray it would block, and any attacker
                // that the departing king was screening still reaches the square
                // the king left, which is itself one of the three.
                illegal = (attack & (7ULL << (target - (src_square < target ? 2 : 0)))) != 0;
            } else {
                illegal =
                    (attack >> __shfl_sync(kAll, (int)(sr.word & kSquareMask), king_slot)) & 1ULL;
            }
            if (lane == p && !illegal) legal |= 1ULL << target;
        }
    }

    MovegenResult r;
    r.mask = legal;
    r.in_check = in_check;
    return r;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

constexpr int kWarpsPerBlock = 8;

// boards [n,32] u16, control [n] i16 -> out [n,32] u64.
template <bool kControl, int kStage = 5>
__global__ void first_order_kernel(const uint16_t* __restrict__ boards,
                                   const int16_t* __restrict__ control, Luts g,
                                   uint64_t* __restrict__ out, int n) {
    __shared__ SharedLuts s;
    load_luts(s, g);

    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    // Warp-uniform, so the warp-collective primitives below keep a full mask.
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    bool black_to_move = !white_to_move(control[board]);
    if (kControl) black_to_move = !black_to_move;  // the attacker is the side not to move

    const uint16_t word = boards[board * 32 + lane];
    out[board * 32 + lane] = first_order_mask<kControl, kStage>(s, word, black_to_move);
}

// The attack map of the side that is not to move, plus whether it covers the
// king of the side that is. spec §4.1 requires `in_check`: an all-zero legality
// mask is checkmate when it is set and stalemate otherwise, and nothing else
// separates the two.
__global__ void attack_map_kernel(const uint16_t* __restrict__ boards,
                                  const int16_t* __restrict__ control, Luts g,
                                  uint64_t* __restrict__ out_attack,
                                  uint8_t* __restrict__ out_in_check, int n) {
    __shared__ SharedLuts s;
    load_luts(s, g);

    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const bool defender_is_black = !white_to_move(control[board]);
    const uint16_t word = boards[board * 32 + lane];
    const uint64_t mask = first_order_mask<true>(s, word, !defender_is_black);
    out_attack[board * 32 + lane] = mask;

    // spec §2.5 guarantees exactly one king per colour, in slots 15 and 31,
    // never captured, so the king's square is a broadcast from a fixed lane
    // rather than a search.
    const int king_slot = defender_is_black ? kBlackKingSlot : kWhiteKingSlot;
    const int king_square = __shfl_sync(0xffffffff, (int)(word & 63), king_slot);
    // Both reductions run on the full warp before anything narrows to one lane.
    const uint64_t attacked = warp_or64(mask);
    if (lane == 0) out_in_check[board] = (uint8_t)((attacked >> king_square) & 1ULL);
}

// The engine's movegen entry point, spec §4.1.
__global__ void movegen_kernel(const uint16_t* __restrict__ boards,
                               const int16_t* __restrict__ control, Luts g,
                               uint64_t* __restrict__ out_mask,
                               uint8_t* __restrict__ out_in_check, int n) {
    __shared__ SharedLuts s;
    load_luts(s, g);

    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const MovegenResult r = movegen(s, boards[(size_t)board * 32 + lane], lane,
                                   !white_to_move(control[board]));
    out_mask[(size_t)board * 32 + lane] = r.mask;
    if (lane == 0) out_in_check[board] = (uint8_t)r.in_check;
}

}  // namespace brokefish
