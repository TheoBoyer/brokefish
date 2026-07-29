// Termination: spec §4.3 and the repetition detection of §6.3.
//
// Needs no attack map and no move generation, only a legality mask that someone
// else produced, so this header sits on chess.cuh alone.
//
// Reference: brokefish/env/torch_impl.py, `insufficient_material`,
// `repetition_count` and `terminal`.
#pragma once

#include <cstdint>

#include "chess.cuh"

namespace brokefish {

enum TerminalCode : int {
    kNone = 0,
    kCheckmate = 1,
    kStalemate = 2,
    kFiftyMove = 3,
    kRepetition = 4,
    kInsufficient = 5,
};

// spec §6.2: the window never exceeds 100 plies, since the game ends at clock 101
// and the repetition window's reset conditions are a superset of the clock's.
constexpr int kMaxHistory = 100;

// Neither side can deliver mate. Warp-collective, uniform result.
//
// The rule is no pawns, rooks or queens, and then either every bishop on a single
// colour complex with no knights, or exactly one knight and no bishops. The
// single-complex clause is what the first version of spec §4.3 missed: K+2B
// against K is a draw only when both bishops sit on the same complex, and
// underpromotion to bishop puts that inside reach of self-play.
__device__ inline bool insufficient_material(uint16_t word) {
    const unsigned kAll = 0xffffffff;
    const bool live = !((word >> 11) & 1);
    const int type = (word >> 6) & 0b111;
    const int square = word & 0b111111;

    const int pawns = __popc(__ballot_sync(kAll, live && type == kPawn));
    const int rooks = __popc(__ballot_sync(kAll, live && type == kRook));
    const int queens = __popc(__ballot_sync(kAll, live && type == kQueen));
    const int knights = __popc(__ballot_sync(kAll, live && type == kKnight));
    const int bishops = __popc(__ballot_sync(kAll, live && type == kBishop));
    // Warp-uniform, so the ballot below still runs on a full warp.
    if (pawns || rooks || queens) return false;

    const int dark = __popc(__ballot_sync(
        kAll, live && type == kBishop && (((square & 7) + (square >> 3)) & 1) == 0));
    const bool one_complex = dark == 0 || dark == bishops;
    return (knights == 0 && one_complex) || (knights == 1 && bishops == 0);
}

// How many times this position has occurred, counting the present one.
//
// spec §6.3: the ring holds the hashes of earlier positions in this game, an
// irreversible move having emptied it, so it is at most 100 entries and 32 lanes
// scan it in at most four iterations. Also feeds `emb_rep` in spec §7.2.
__device__ inline int repetition_count(uint64_t hash, const uint64_t* ring, int len,
                                       int lane) {
    unsigned hits = 0;
    for (int i = lane; i < len; i += 32) hits += (ring[i] == hash);
    return 1 + (int)__reduce_add_sync(0xffffffff, hits);
}

struct TerminalResult {
    uint8_t code;
    int8_t result;  // from the side to move's point of view: -1 mate, 0 otherwise
};

// spec §4.3. Warp-collective, lane i carrying slot i's mask word and slot word.
// Pass `ring == nullptr` to skip the repetition test, which is what the static
// test set needs: repetition is a property of the game that reached a position,
// not of the position.
//
// The first code that applies wins. A position can satisfy several draw
// conditions at once; they all give the same result, so only the reported code
// differs. `+1` never occurs, because a position is never terminal in favour of
// the player about to move.
__device__ inline TerminalResult terminal(uint64_t mask_word, uint16_t word, int lane,
                                          bool in_check, int16_t control, uint64_t hash,
                                          const uint64_t* ring, int ring_len) {
    // ⚠️ NOT a sum. The mask words carry uint64 bit patterns, and summing them
    // overflows: two pieces able to reach h8 give 2 * 2**63, which is zero. That
    // shipped in the PyTorch reference and called a position with four legal
    // replies checkmate. See docs/env.md.
    const bool no_moves = __all_sync(0xffffffff, mask_word == 0);

    int code = kNone;
    if (no_moves) code = in_check ? kCheckmate : kStalemate;
    const int clock = control > 0 ? control : -control;
    if (code == kNone && clock >= 101) code = kFiftyMove;
    if (code == kNone && ring != nullptr
        && repetition_count(hash, ring, ring_len, lane) >= 3)
        code = kRepetition;
    if (code == kNone && insufficient_material(word)) code = kInsufficient;

    TerminalResult r;
    r.code = (uint8_t)code;
    r.result = (int8_t)(code == kCheckmate ? -1 : 0);
    return r;
}

// ---------------------------------------------------------------------------

// `hash`, `ring` and `lengths` may all be null, which skips the repetition test.
// `ring` is [n, kMaxHistory] and `lengths` says how many entries of each row are
// live (spec §6.2: an irreversible move empties it).
__global__ void terminal_kernel(const uint16_t* __restrict__ boards,
                                const int16_t* __restrict__ control,
                                const uint64_t* __restrict__ mask,
                                const uint8_t* __restrict__ in_check,
                                const uint64_t* __restrict__ hash,
                                const uint64_t* __restrict__ ring,
                                const int64_t* __restrict__ lengths,
                                uint8_t* __restrict__ out_code,
                                signed char* __restrict__ out_result, int n) {
    const int board = blockIdx.x * 8 + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const TerminalResult r = terminal(
        mask[(size_t)board * 32 + lane], boards[(size_t)board * 32 + lane], lane,
        in_check[board] != 0, control[board], hash ? hash[board] : 0ULL,
        ring ? ring + (size_t)board * kMaxHistory : nullptr,
        lengths ? (int)lengths[board] : 0);
    if (lane == 0) {
        out_code[board] = r.code;
        out_result[board] = (signed char)r.result;
    }
}

}  // namespace brokefish
