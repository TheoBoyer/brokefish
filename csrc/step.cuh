// Applying one move to one position: the board mutation of spec §4.2.
//
// This is the half of `step` the second-order pass calls, once per candidate
// move and therefore about 35 times per position. It moves the piece, wipes the
// captured slot, overwrites the pawn's type bits on a promotion, moves the rook
// leg on a castle, maintains the `special` bits, and advances the control word.
//
// The fifty-move clock rides along because it is one select over two flags the
// mutation already computes. The incremental Zobrist and `irreversible` do not:
// spec §6.1 hashes the en passant file only when the capture is actually legal,
// king safety included, so the hash needs its own attack map and would be a
// second-order pass inside a second-order pass if it were paid 35 times. It gets
// its own entry point (A1.2b), which is why the PyTorch reference takes `hash`
// as an optional argument that perft passes as None.
//
// Warp-collective, lane i owning slot i, matching movegen.cuh so that the two
// compose: the second order will run one warp per candidate move, apply it, then
// build the attack map of the resulting position in the same warp.
//
// Reference: brokefish/env/torch_impl.py::step. The reference is perft-green
// against python-chess and this file reproduces its slot words bit for bit,
// including the two places where it looks wrong and is not, marked below.
#pragma once

#include <cstdint>

#include "chess.cuh"

namespace brokefish {

constexpr uint16_t kCapturedWord = 1 << 11;  // spec §2.1: the whole word is wiped
constexpr uint16_t kSpecialBit = 1 << 9;
constexpr uint16_t kSquareMask = 0b111111;
constexpr uint16_t kNonSquareMask = 0xFC0;  // bits 11:6, i.e. everything but the square

struct StepResult {
    uint16_t word;      // this lane's slot after the move
    bool clock_reset;   // a capture or a pawn move, so the fifty-move clock restarts
};

// `move` is slot * 64 + target and `promo` is the spec §3 field (0:N 1:B 2:R
// 3:Q), both warp-uniform. Every lane of the warp must call this.
__device__ inline StepResult apply_move(uint16_t word, int lane, bool white_to_move,
                                        int move, int promo) {
    const unsigned kAll = 0xffffffff;

    // spec §9: -1 is a no-op, which is how a finished or paused game rides along
    // in a batch. Its lane is computed like every other and then restored, so the
    // kernel stays total and no lane takes an early return.
    const bool is_null = move < 0;
    const int m = is_null ? 0 : move;
    const int source = m >> 6;
    const int target = m & kSquareMask;

    const uint16_t src_word = __shfl_sync(kAll, word, source);
    const int src_type = (src_word >> 6) & 0b111;
    const bool src_is_pawn = src_type == kPawn;
    const int src_square = src_word & kSquareMask;
    const int dx = target - src_square;

    // Capture. Either a live piece standing on the target square, or, when the
    // mover is a pawn, the en passant victim, which stands beside the target and
    // not on it.
    //
    // The victim is matched as a whole word, which demands an enemy pawn
    // carrying the double-push flag on the exact square behind the target. That
    // is what makes the second clause safe next to the first: the square an en
    // passant capture lands on is the one the victim skipped, so it was empty
    // when the victim double-pushed and it is still empty now, which means no
    // ordinary pawn move can ever reach it and the two clauses never both fire.
    const uint16_t ep_victim =
        (uint16_t)(((white_to_move ? 3 : 1) << 9) + target + 8 - (white_to_move ? 16 : 0));
    const bool i_am_victim = (!((word >> 11) & 1) && (word & kSquareMask) == target)
                             || (src_is_pawn && word == ep_victim);
    const unsigned victims = __ballot_sync(kAll, i_am_victim);
    const bool has_capture = victims != 0;
    // Two live slots never share a square (spec §2.5), so this holds at most one
    // bit in any reachable position; taking the lowest matches the reference's
    // argmax if one ever appears anyway.
    const int victim_slot = victims ? (__ffs((int)victims) - 1) : -1;

    uint16_t out = word;
    if (lane == victim_slot) out = kCapturedWord;
    if (lane == source) {
        out = (uint16_t)((src_word & kNonSquareMask) | target);
        // A promotion is the pawn's type bits being overwritten in its own slot:
        // slot 3 may hold a queen and the network reads the type from the word.
        if (src_is_pawn && (target >> 3) == (white_to_move ? 7 : 0))
            out = (uint16_t)(out + ((promo + kKnight) << 6));
        // A king or rook that moves loses its castling right for good.
        if (src_type == kKing || src_type == kRook) out |= kSpecialBit;
    }

    // The rook leg of a castle. `is_castling` is warp-uniform, so the ballot
    // below runs with the full warp inside the branch.
    if (src_type == kKing && (dx == 2 || dx == -2)) {
        const int rook_row = white_to_move ? 0 : 7;
        const int rook_from = rook_row * 8 + (dx > 0 ? 7 : 0);
        const bool i_am_rook =
            ((word >> 6) & 0b111) == kRook && (word & kSquareMask) == rook_from;
        const unsigned rooks = __ballot_sync(kAll, i_am_rook);
        if (rooks && lane == __ffs((int)rooks) - 1) {
            // LOOKS WRONG, IS NOT: the rook does not get its `special` bit set
            // even though it just moved. The king's is set above, and a castling
            // right needs both the king and the rook intact, so the right is
            // already gone through the king. The reference leaves it clear and
            // the words have to match bit for bit.
            out = (uint16_t)((word & kNonSquareMask) | (rook_row * 8 + (dx > 0 ? 5 : 3)));
        }
    }

    // The en passant flag lives for exactly one ply, so every pawn's is cleared
    // before the double-pusher's is set.
    //
    // LOOKS WRONG, IS NOT: this runs after the promotion above, so a piece that
    // has just stopped being a pawn is not cleared. It cannot matter: a pawn
    // carrying the flag stands on the fourth rank and cannot promote.
    if (((out >> 6) & 0b111) == kPawn) out &= (uint16_t)~kSpecialBit;
    if (lane == source && src_is_pawn && (dx == 16 || dx == -16)) out |= kSpecialBit;

    StepResult r;
    r.word = is_null ? word : out;
    // A promotion is a pawn move, so `src_is_pawn` covers it and the clock is
    // reset on every promotion whether or not it captures.
    r.clock_reset = !is_null && (has_capture || src_is_pawn);
    return r;
}

// The control word after the move. spec §2.2: the sign flips, and the magnitude
// restarts at 1 on a capture or a pawn move rather than counting up. It is not
// clamped at 101; `terminal` ends the game there and clamping would silently
// disagree with python-chess, which keeps counting.
__device__ inline int16_t apply_move_control(int16_t control, bool is_null,
                                             bool clock_reset) {
    return is_null ? control : advance_control(control, clock_reset);
}

// ---------------------------------------------------------------------------

constexpr int kStepWarpsPerBlock = 8;

// One warp per entry. `index[e]` selects the position, so the same position can
// appear under many moves without being stored many times, which is the shape
// the second-order pass will use.
__global__ void apply_move_kernel(const uint16_t* __restrict__ boards,
                                  const int16_t* __restrict__ control,
                                  const int32_t* __restrict__ index,
                                  const int16_t* __restrict__ move,
                                  const uint8_t* __restrict__ promo,
                                  uint16_t* __restrict__ out_boards,
                                  int16_t* __restrict__ out_control, int m) {
    const int e = blockIdx.x * kStepWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (e >= m) return;
    const int lane = (int)(threadIdx.x & 31);

    const int pos = index[e];
    const int16_t c = control[pos];
    const int mv = move[e];

    const StepResult r =
        apply_move(boards[(size_t)pos * 32 + lane], lane, white_to_move(c), mv, promo[e]);
    out_boards[(size_t)e * 32 + lane] = r.word;
    if (lane == 0) out_control[e] = apply_move_control(c, mv < 0, r.clock_reset);
}

}  // namespace brokefish
