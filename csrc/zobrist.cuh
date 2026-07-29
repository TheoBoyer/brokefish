// Hashing and the repetition window: spec §6.1, §6.2, and the second half of the
// `step` contract in §4.2.
//
// This is the half of `step` that is paid once per real move rather than once per
// candidate. It is separated from step.cuh for a reason that is not tidiness: the
// incremental hash needs `legal_ep_file`, spec §6.1 hashes the en passant file
// only when the capture is *actually legal* (king safety included), so the hash
// needs an attack map. step.cuh must therefore stay free of movegen.cuh, and this
// header, which includes both, is where the cycle is broken.
//
// The consequence for cost: `step_full` runs up to four extra first-order passes
// (two per `legal_ep_file` call, one per capturing pawn). Paying that inside the
// second-order pass would nest a second order inside a second order, which is
// why the reference takes `hash` as an optional argument that perft passes as
// None. Both en passant tests short-circuit on the common case, where no pawn
// carries the double-push flag: 6 % of the positions in the test dump.
//
// Reference: brokefish/env/torch_impl.py, functions `castling_rights`,
// `legal_ep_file`, `hash_position` and the `track` branch of `step`.
#pragma once

#include <cstdint>

#include "chess.cuh"
#include "movegen.cuh"

namespace brokefish {

// spec §6.1: 781 keys. Indexed by (colour, type, square) and never by slot, so a
// promoted queen in a pawn slot and an original queen hash identically, which is
// where the piece-list representation departs from a bitboard engine's Zobrist.
constexpr int kZobristPiece = 0;     // 768, (colour * 6 + type) * 64 + square
constexpr int kZobristSide = 768;    // 1, xored in when black is to move
constexpr int kZobristCastle = 769;  // 4, FEN order KQkq
constexpr int kZobristEp = 773;      // 8, by file, only when the capture is legal
constexpr int kZobristKeys = 781;

// The table lives in shared memory rather than __constant__: spec §6.1 rules the
// latter out because these lookups are per-piece and therefore divergent within a
// warp, which serialises the constant cache.
struct SharedZobrist {
    uint64_t keys[kZobristKeys];  // 6248 B
};

constexpr uint64_t kSplitmix64Seed = 0x00B4C0FFEE12F00DULL;
constexpr uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ULL;

// Key `i` of the table brokefish/env/luts.py builds, computed directly rather
// than by iterating the generator.
//
// splitmix64 advances its state by adding a constant, so the state before the
// i-th output is seed + (i+1) * gamma, a plain Weyl sequence. That makes the 781
// keys independent and the whole table a parallel map instead of a serial loop.
// The mixing function below is the same shift-multiply-xor chain, which is why
// luts.py chose splitmix64: the CUDA side regenerates the table instead of
// loading one, so the two cannot drift. `tzobrist.cu` pins that they agree.
__device__ inline uint64_t zobrist_key(int i) {
    uint64_t z = kSplitmix64Seed + (uint64_t)(i + 1) * kGoldenGamma;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Every thread of the block must call this; it ends on the barrier that publishes
// the table.
__device__ inline void load_zobrist(SharedZobrist& z) {
    for (int i = threadIdx.x; i < kZobristKeys; i += blockDim.x) z.keys[i] = zobrist_key(i);
    __syncthreads();
}

// One slot's contribution, zero when the slot is dead.
__device__ inline uint64_t piece_key(const SharedZobrist& z, uint16_t word) {
    if ((word >> 11) & 1) return 0;
    const int colour = (word >> 10) & 1;
    const int type = (word >> 6) & 0b111;
    return z.keys[kZobristPiece + ((colour * 6 + type) * 64 + (word & kSquareMask))];
}

// ---------------------------------------------------------------------------
// Castling rights
// ---------------------------------------------------------------------------

// Bits 0..3 in FEN order KQkq: white kingside, white queenside, black kingside,
// black queenside. Warp-collective, uniform result.
//
// A right holds while both the king and that corner's rook are alive and have
// never moved, which is what `special` records by its absence. The rook test is
// one word comparison, since `unmoved_rook_word` pins colour, type, square, and
// both the captured and special bits at once.
__device__ inline unsigned castling_rights(uint16_t word) {
    const unsigned kAll = 0xffffffff;
    // spec §2.5 pins the kings to slots 15 and 31 and promises they are never
    // captured, so "this king has never moved" is one bit of one broadcast word
    // rather than a search over 32 slots.
    const uint16_t wk = __shfl_sync(kAll, word, kWhiteKingSlot);
    const uint16_t bk = __shfl_sync(kAll, word, kBlackKingSlot);
    const bool white_king_ok =
        ((wk >> 6) & 0b111) == kKing && !((wk >> 9) & 1) && !((wk >> 11) & 1);
    const bool black_king_ok =
        ((bk >> 6) & 0b111) == kKing && !((bk >> 9) & 1) && !((bk >> 11) & 1);

    const bool h1 = __any_sync(kAll, word == unmoved_rook_word(false, 7));
    const bool a1 = __any_sync(kAll, word == unmoved_rook_word(false, 0));
    const bool h8 = __any_sync(kAll, word == unmoved_rook_word(true, 63));
    const bool a8 = __any_sync(kAll, word == unmoved_rook_word(true, 56));

    return (unsigned)(white_king_ok && h1) | ((unsigned)(white_king_ok && a1) << 1)
           | ((unsigned)(black_king_ok && h8) << 2) | ((unsigned)(black_king_ok && a8) << 3);
}

// ---------------------------------------------------------------------------
// The en passant file
// ---------------------------------------------------------------------------

// The file of a *legal* en passant capture, or -1. Warp-collective, uniform.
//
// spec §6.1 hashes the file only when the capture is actually available, because
// FIDE defines repetition by the same en passant possibility and a dangling flag
// makes real repetitions invisible. "Available" includes king safety: a capture
// that would expose the king does not count.
//
// At most two pawns can ever reach a given target, so this replays at most two
// moves rather than the ~35 a full second order costs, and it replays none at all
// in the common case where no pawn carries the flag.
__device__ inline int legal_ep_file(const SharedLuts& s, uint16_t word, int lane,
                                    bool black_to_move) {
    const unsigned kAll = 0xffffffff;
    const bool live_pawn = !((word >> 11) & 1) && ((word >> 6) & 0b111) == kPawn;
    const bool colour = (word >> 10) & 1;
    // The flagged pawn belongs to the side that just moved, so not to the mover.
    const unsigned flags =
        __ballot_sync(kAll, live_pawn && ((word >> 9) & 1) && colour != black_to_move);
    if (!flags) return -1;

    // spec §2.5 allows at most one, and the reference takes the first anyway.
    const int victim_slot = __ffs((int)flags) - 1;
    const int victim_square = __shfl_sync(kAll, (int)(word & kSquareMask), victim_slot);
    const int victim_file = victim_square & 7;
    // The capture lands behind the victim, on the square it skipped.
    const int target = victim_square + (black_to_move ? -8 : 8);

    bool legal = false;
    // Either neighbour may be able to capture and both may be, so availability is
    // the OR over the two. The file recorded is the victim's, never the
    // capturer's: that is the file python-chess puts in its transposition key.
    for (int delta = -1; delta <= 1; delta += 2) {
        const int capturer_file = victim_file + delta;
        if (capturer_file < 0 || capturer_file > 7) continue;  // warp-uniform
        const unsigned cands = __ballot_sync(
            kAll, live_pawn && colour == black_to_move
                      && (int)(word & kSquareMask) == victim_square + delta);
        if (!cands) continue;  // warp-uniform

        const int slot = __ffs((int)cands) - 1;
        const StepResult sr = apply_move(word, lane, !black_to_move, slot * 64 + target, 3);
        // After the capture the opponent is to move, so it is the attacker, and
        // the defender is the side that just captured.
        const uint64_t attack = warp_or64(first_order_mask<true>(s, sr.word, !black_to_move));
        const int king_slot = black_to_move ? kBlackKingSlot : kWhiteKingSlot;
        const int king_square = __shfl_sync(kAll, (int)(sr.word & kSquareMask), king_slot);
        legal = legal || !((attack >> king_square) & 1ULL);
    }
    return legal ? victim_file : -1;
}

// ---------------------------------------------------------------------------
// The hash
// ---------------------------------------------------------------------------

// The full Zobrist hash of a position, spec §6.1. Warp-collective, uniform.
// Used to seed a game and to check the incremental update; the search maintains
// the hash through `step_full` instead.
__device__ inline uint64_t hash_position(const SharedLuts& s, const SharedZobrist& z,
                                         uint16_t word, int lane, int16_t control) {
    const bool black_to_move = !white_to_move(control);
    uint64_t h = warp_xor64(piece_key(z, word));
    if (black_to_move) h ^= z.keys[kZobristSide];

    const unsigned rights = castling_rights(word);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        if ((rights >> i) & 1) h ^= z.keys[kZobristCastle + i];

    const int ep = legal_ep_file(s, word, lane, black_to_move);
    if (ep >= 0) h ^= z.keys[kZobristEp + ep];
    return h;
}

struct FullStepResult {
    uint16_t word;      // this lane's slot after the move
    int16_t control;    // uniform
    uint64_t hash;      // uniform
    bool irreversible;  // uniform
};

// `step` in full, spec §4.2: the mutation, the clock, the incremental hash and
// `irreversible`. Warp-collective, `move`/`promo`/`hash` warp-uniform.
//
// kTrackHash = false drops the hash and keeps everything else, which is what
// perft and the second order want: the reference's `hash=None` path. It is a
// compile-time constant, so the untracked instantiation pays nothing for the
// branch and, more to the point, does not run the two `legal_ep_file` calls that
// dominate the tracked one. `irreversible` stays, because it needs only the
// castling-rights delta and no attack map.
template <bool kTrackHash = true>
__device__ inline FullStepResult step_full(const SharedLuts& s, const SharedZobrist& z,
                                           uint16_t word, int lane, int16_t control,
                                           uint64_t hash, int move, int promo) {
    const bool is_null = move < 0;
    const bool black_to_move = !white_to_move(control);
    const StepResult sr = apply_move(word, lane, !black_to_move, move, promo);
    const int16_t new_control = apply_move_control(control, is_null, sr.clock_reset);

    const unsigned rights_before = castling_rights(word);
    const unsigned rights_after = castling_rights(sr.word);
    const unsigned rights_delta = rights_before ^ rights_after;

    // spec §6.2: one condition more than the fifty-move clock resets on. Rights
    // only ever decrease, so a rights change also makes every earlier position
    // unreachable, and the repetition window is the shorter of the two spans.
    const bool irreversible = !is_null && (sr.clock_reset || rights_delta != 0);

    uint64_t h = hash;
    if (kTrackHash) {
        // Only slots whose (colour, type, square) changed contribute: a slot
        // where just `special` flipped xors its key out and straight back in.
        h ^= warp_xor64(piece_key(z, word) ^ piece_key(z, sr.word));
        h ^= z.keys[kZobristSide];  // the side to move always flips
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if ((rights_delta >> i) & 1) h ^= z.keys[kZobristCastle + i];

        const int ep_before = legal_ep_file(s, word, lane, black_to_move);
        const int ep_after = legal_ep_file(s, sr.word, lane, !white_to_move(new_control));
        if (ep_before >= 0) h ^= z.keys[kZobristEp + ep_before];
        if (ep_after >= 0) h ^= z.keys[kZobristEp + ep_after];
    }

    FullStepResult r;
    r.word = sr.word;
    r.control = new_control;
    r.hash = is_null ? hash : h;
    r.irreversible = irreversible;
    return r;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void hash_position_kernel(const uint16_t* __restrict__ boards,
                                     const int16_t* __restrict__ control, Luts g,
                                     uint64_t* __restrict__ out_hash,
                                     unsigned char* __restrict__ out_rights,
                                     signed char* __restrict__ out_ep, int n) {
    __shared__ SharedLuts s;
    __shared__ SharedZobrist z;
    load_luts(s, g);
    load_zobrist(z);

    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const uint16_t word = boards[(size_t)board * 32 + lane];
    const int16_t c = control[board];
    const uint64_t h = hash_position(s, z, word, lane, c);
    // The two intermediates are written out as well, because a wrong hash gives
    // no clue which of the three inputs was wrong.
    const unsigned rights = castling_rights(word);
    const int ep = legal_ep_file(s, word, lane, !white_to_move(c));
    if (lane == 0) {
        out_hash[board] = h;
        if (out_rights) out_rights[board] = (unsigned char)rights;
        if (out_ep) out_ep[board] = (signed char)ep;
    }
}

// One warp per (position, move, promo) case, as in step.cuh's kernel.
//
// `index` may be null, meaning case e applies to position e, which is what the
// torch binding wants; the test set needs the indirection because the same
// position appears under many moves. `hash` may be null when kTrackHash is false.
template <bool kTrackHash = true>
__global__ void step_full_kernel(const uint16_t* __restrict__ boards,
                                 const int16_t* __restrict__ control,
                                 const uint64_t* __restrict__ hash,
                                 const int32_t* __restrict__ index,
                                 const int16_t* __restrict__ move,
                                 const uint8_t* __restrict__ promo, Luts g,
                                 uint16_t* __restrict__ out_boards,
                                 int16_t* __restrict__ out_control,
                                 uint64_t* __restrict__ out_hash,
                                 uint8_t* __restrict__ out_irreversible, int m) {
    __shared__ SharedLuts s;
    __shared__ SharedZobrist z;
    load_luts(s, g);
    if (kTrackHash) load_zobrist(z);

    const int e = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (e >= m) return;
    const int lane = (int)(threadIdx.x & 31);

    const int pos = index ? index[e] : e;
    const FullStepResult r = step_full<kTrackHash>(
        s, z, boards[(size_t)pos * 32 + lane], lane, control[pos],
        kTrackHash ? hash[pos] : 0ULL, move[e], promo ? promo[e] : 3);
    out_boards[(size_t)e * 32 + lane] = r.word;
    if (lane == 0) {
        out_control[e] = r.control;
        if (out_hash) out_hash[e] = r.hash;
        out_irreversible[e] = (uint8_t)r.irreversible;
    }
}

}  // namespace brokefish
