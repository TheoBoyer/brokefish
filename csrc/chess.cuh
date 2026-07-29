// Board representation shared by every Brokefish CUDA kernel.
//
// A position is 32 uint16 words (one per piece slot) plus one int16 side-to-move
// word. Slots are fixed for the lifetime of a game: a captured piece keeps its
// slot and no piece ever migrates, which is what makes the network's 32 piece
// tokens index-aligned with the engine's 32 legality masks.
//
//   bit 11   captured   (1 = off the board; the rest of the word is wiped)
//   bit 10   color      (0 = white, 1 = black)
//   bit  9   special    (king/rook: has already moved, i.e. castling right lost;
//                        pawn: just made a double push, en passant target)
//   bits 8:6 type       (see PieceType)
//   bits 5:0 square     (row * 8 + col, 0 = a1, 63 = h8)
//
// A slot says where a piece started, not what it is: a promoted pawn keeps its
// pawn slot with new type bits. Always read the type from the word.
//
// Legality masks follow one convention throughout: for slot p, bit s of mask[p]
// is set iff the move p -> s is legal. A move is the integer p * 64 + s.
//
// See docs/spec.md for the full specification, including the
// side-to-move word and the invariants a kernel may rely on.
#pragma once

#include <cstdint>

namespace brokefish {

enum PieceType : int {
    kPawn = 0b000,
    kKnight = 0b001,
    kBishop = 0b010,
    kRook = 0b011,
    kQueen = 0b100,
    kKing = 0b101,
};

// Slots guaranteed by construction: there is always exactly one king per color.
constexpr int kWhiteKingSlot = 15;
constexpr int kBlackKingSlot = 31;

struct Piece {
    bool captured;
    bool color;
    bool special;
    int type;
    int square;
};

__device__ inline Piece decode(uint16_t word) {
    Piece p;
    p.captured = (word >> 11) & 1;
    p.color = (word >> 10) & 1;
    p.special = (word >> 9) & 1;
    p.type = (word >> 6) & 0b111;
    p.square = word & 0b111111;
    return p;
}

__device__ inline uint16_t encode(const Piece& p) {
    return (uint16_t)((p.captured << 11) | (p.color << 10) | (p.special << 9)
                      | ((p.type & 0b111) << 6) | (p.square & 0b111111));
}

// White to move iff the control word is positive; its magnitude is
// halfmove_clock + 1, so it IS the fifty-move clock and the game is drawn when
// the magnitude reaches 101. See docs/spec.md §2.2.
__device__ inline bool white_to_move(int16_t control) { return control > 0; }

// Flip the side to move. `reset` restarts the clock at 1 and must be set on a
// capture and on a pawn move (spec §4.2); passing it unconditionally false makes
// this a plain ply counter, which is what the pre-2026-07-29 word was.
__device__ inline int16_t advance_control(int16_t control, bool reset) {
    int16_t magnitude = reset ? 1 : (int16_t)((control > 0 ? control : -control) + 1);
    return (int16_t)(control > 0 ? -magnitude : magnitude);
}

// OR-reduce a 64-bit board mask across a full warp. There is no 64-bit
// __reduce_or_sync, so it runs as two 32-bit reductions.
__device__ inline uint64_t warp_or64(uint64_t x) {
    uint32_t lo = __reduce_or_sync(0xffffffff, (uint32_t)x);
    uint32_t hi = __reduce_or_sync(0xffffffff, (uint32_t)(x >> 32));
    return ((uint64_t)hi << 32) | lo;
}

// XOR-reduce a 64-bit value across a full warp, which is how a Zobrist delta over
// the 32 slots collapses (spec §6.1). Two 32-bit reductions, as above.
__device__ inline uint64_t warp_xor64(uint64_t x) {
    uint32_t lo = __reduce_xor_sync(0xffffffff, (uint32_t)x);
    uint32_t hi = __reduce_xor_sync(0xffffffff, (uint32_t)(x >> 32));
    return ((uint64_t)hi << 32) | lo;
}

// Cooperative global -> shared copy, strided over the block. The caller owns the
// __syncthreads() so that several copies can be issued before paying for one.
template <typename T>
__device__ inline void smem_copy(T* dst, const T* src, int count) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) dst[i] = src[i];
}

}  // namespace brokefish
