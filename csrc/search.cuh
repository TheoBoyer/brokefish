// MCTS v0 on device: the four kernels of docs/mcts.md §9.
//
// One warp owns a game and lane i owns slot i, the same layout as the engine
// headers, which is what lets `descent` call `step_full`, `movegen` and
// `terminal` from inside its own tree walk with no launch between them. The edge
// cap E = 64 is a compile-time constant because the selection scan of §6.6 is
// then exactly two edges per lane with no predicate on the second pass; B and
// Nmax are runtime, because the differential harness of §12 runs at B = 8 while
// self-play runs at 4096.
//
// The reference is brokefish/search/torch_impl.py, and "reference" here means
// bit for bit, not to a tolerance. Two consequences that look like fussiness and
// are not:
//
//   * Every step of the selection score is an explicit rounding intrinsic
//     (`__fmul_rn` and friends). The reference is a chain of separate torch
//     elementwise kernels, each a single IEEE-rounded operation, so a
//     compiler-contracted `a * b + c` here differs in the last bit, and a
//     last-bit difference in a near-tie is a different move and a different
//     tree. -fmad=false would do the same job for the whole translation unit;
//     naming the operations keeps it local and visible.
//   * The visit-count sum `N_v` is accumulated as an integer and converted once.
//     The reference sums fp32, but the values are non-negative integers summing
//     to at most `n`, so both are exact and the order does not matter. Doing it
//     in integers is what makes that argument checkable rather than hopeful.
//
// A header rather than a .cu so that csrc/tests/tselect.cu can pin the selection
// scan and the canonical edge enumeration against a host reference with no
// Python and no torch, and so that §9.1's fusion stays a local edit.
#pragma once

#include <cstdint>

#include <cuda_fp16.h>

#include "chess.cuh"
#include "movegen.cuh"
#include "terminal.cuh"
#include "zobrist.cuh"

namespace brokefish {
namespace search {

// §4.3. Two edges per lane in the selection scan; see the file header.
constexpr int kE = 64;
constexpr int kWarps = 8;  // warps per block, matching the engine kernels

// `node_flags`, mirroring brokefish/search/torch_impl.py.
constexpr uint8_t kTerminalMask = 0b111;
constexpr uint8_t kExpanded = 1 << 3;
constexpr uint8_t kIrreversible = 1 << 4;

// spec §3: an 11-bit move with the promotion choice riding above it.
constexpr int kMoveBits = 11;
constexpr int kMoveMask = (1 << kMoveBits) - 1;

// `leaf_flags`, the staging buffer descent writes for the host driver and for
// `expand`. Bit 0 is the only one the kernels read; bit 1 and the terminal code
// are what the §15 counters and the useful-evaluation count are computed from.
constexpr uint8_t kNeedsExpand = 1 << 0;
constexpr uint8_t kFresh = 1 << 1;

constexpr unsigned kAll = 0xffffffffu;

// §15's counter block, accumulated on device so that nothing here costs a host
// synchronisation. `nullptr` turns the whole thing off, which is what the
// throughput benchmark runs so that the number it reports is the search.
struct Counters {
    unsigned long long simulations;
    unsigned long long terminal_descents;   // a descent that ended on a stored terminal
    unsigned long long terminal_children;   // a created node that was already over
    unsigned long long truncated_nodes;     // §4.3's bet, counted rather than assumed
    unsigned long long empty_mask_expansions;  // invariant 5
    unsigned long long pool_overflow;       // invariant 1
    unsigned long long depth_overflow;      // a descent that hit Dmax, §4.1 says never
    unsigned long long depth_sum;
    unsigned int max_edges;
    unsigned int max_depth;
    unsigned int max_nodes;
    float truncated_mass;
};

// Every array of §4.2, in the integer widths brokefish/search/torch_impl.py
// allocates: `edge_N` and `edge_move` are int16 and `path_len` int32 because
// torch has no unsigned arithmetic worth using, and the two implementations
// share tensors.
struct Tree {
    uint16_t* node_board;    // [B, N, 32]
    int16_t* node_control;   // [B, N]
    uint64_t* node_hash;     // [B, N]
    __half* node_value;      // [B, N]
    uint8_t* node_nedges;    // [B, N]
    uint8_t* node_flags;     // [B, N]
    int16_t* node_parent;    // [B, N]
    uint8_t* node_pedge;     // [B, N]

    int16_t* edge_move;      // [B, N, kE]
    __half* edge_prior;      // [B, N, kE]
    int16_t* edge_child;     // [B, N, kE]
    int16_t* edge_N;         // [B, N, kE]
    float* edge_Q;           // [B, N, kE]

    int16_t* path_node;      // [B, D]
    uint8_t* path_edge;      // [B, D]
    int32_t* path_len;       // [B]

    uint64_t* game_ring;     // [B, kMaxHistory]
    int64_t* game_ring_len;  // [B]
    int32_t* node_count;     // [B]
    int32_t* budget;         // [B], §11's structural hook

    // The staging buffers between `descent`, the encoder and `expand`. The
    // encoder is one CTA per board over a contiguous [B, 32] batch, and a
    // simulation's leaf sits at a different node index in every game, so the
    // descent writes the leaf out contiguously rather than the encoder learning
    // to gather.
    uint16_t* leaf_board;    // [B, 32]
    int16_t* leaf_control;   // [B]
    uint8_t* leaf_rep;       // [B], spec §7.2's min(rep - 1, 2)
    int16_t* leaf_node;      // [B]
    uint8_t* leaf_flags;     // [B]

    int B, N, D;
};

struct Params {
    float pb_c_base;
    float pb_c_init;
};

__device__ inline size_t node_at(const Tree& t, int b, int v) {
    return (size_t)b * t.N + v;
}
__device__ inline size_t board_at(const Tree& t, int b, int v) {
    return node_at(t, b, v) * 32;
}
__device__ inline size_t edge_at(const Tree& t, int b, int v) {
    return node_at(t, b, v) * kE;
}

// ---------------------------------------------------------------------------
// Warp helpers
// ---------------------------------------------------------------------------

// Exclusive prefix sum across the warp; `*total` gets the inclusive sum of all
// 32 lanes, uniform.
__device__ inline int warp_exclusive_scan(int v, int lane, int* total) {
    int x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(kAll, x, off);
        if (lane >= off) x += y;
    }
    *total = __shfl_sync(kAll, x, 31);
    return x - v;
}

// ---------------------------------------------------------------------------
// §6.6, the selection score
// ---------------------------------------------------------------------------

// The PUCT argmax over one node's edges. Warp-collective, uniform result; lane
// `i` owns edges `i` and `i + 32`, which is the whole reason E is 64.
//
// `nvis`, `prior` and `qs` are the node's own edge arrays, each of length kE.
//
// Ties go to the lowest edge index. That is a contract and not an accident:
// §12's differential test compares trees edge for edge, and at a freshly created
// node every score is exactly 0 (`sqrt(N_v) == 0` kills the exploration term and
// first-play urgency zeroes Q), so every edge ties and the tie-break alone
// decides the first descent below every new node.
__device__ inline int puct_argmax(const int16_t* nvis, const __half* prior, const float* qs,
                                 int nedges, int lane, const Params& p) {
    // The reference's `torch.where(valid, nvis, 0).sum(-1)`. Non-negative
    // integers summing to at most `n`, so integer and fp32 agree exactly and the
    // reduction order is free.
    unsigned own = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int e = lane + 32 * j;
        if (e < nedges) own += (unsigned)nvis[e];
    }
    const float n_v = (float)__reduce_add_sync(kAll, own);

    // torch evaluates `n_v + base + 1.0` left to right and `/ base` and the log
    // and the `+ init` as separate kernels, so each of these is one rounded op.
    float pb = __fadd_rn(n_v, p.pb_c_base);
    pb = __fadd_rn(pb, 1.0f);
    pb = __fdiv_rn(pb, p.pb_c_base);
    pb = __fadd_rn(logf(pb), p.pb_c_init);
    const float root = sqrtf(n_v);
    const float pb_root = __fmul_rn(pb, root);

    float best = -INFINITY;
    int best_e = kE;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int e = lane + 32 * j;
        if (e >= nedges) continue;
        const float n = (float)nvis[e];
        const float u = __fdiv_rn(pb_root, __fadd_rn(n, 1.0f));
        const float q = n > 0.0f ? qs[e] : 0.0f;
        const float score = __fadd_rn(__fmul_rn(u, __half2float(prior[e])), q);
        // j ascends with the edge index, so a plain `>` already breaks the
        // within-lane tie towards the lower index.
        if (score > best) {
            best = score;
            best_e = e;
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(kAll, best, off);
        const int other_e = __shfl_down_sync(kAll, best_e, off);
        if (other > best || (other == best && other_e < best_e)) {
            best = other;
            best_e = other_e;
        }
    }
    return __shfl_sync(kAll, best_e, 0);
}

// ---------------------------------------------------------------------------
// §6.4, the canonical edge enumeration
// ---------------------------------------------------------------------------

// A candidate's canonical index is `(slot * 64 + square) * 4 + promo`, so
// ascending by slot, then by target square, then by promotion type in spec §3's
// N B R Q order. That ordering is normative: it is the only thing that makes a
// tree comparable between implementations, since PUCT ties are broken by edge
// index.
//
// IEEE floats compare as integers once the sign is folded in, which is what lets
// the truncation of §4.3 select the E-th largest logit by radix rather than by
// sorting. Only the ordering matters, so no inverse is needed.
__device__ inline uint32_t float_key(float f) {
    uint32_t b = __float_as_uint(f);
    // -0.0 and +0.0 are equal as floats and would otherwise get different keys,
    // which would order two candidates the reference calls tied and hand the
    // truncation a threshold the reference never chose. A policy logit of exactly
    // -0.0 is unlikely and is not impossible.
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// log_softmax over the four promotion logits, in the order ATen's
// LogSoftMaxForwardEpilogue evaluates it: `(x - max) - log(sum)`, not
// `x - (max + log(sum))`. The two differ in the last bit, and a promotion edge's
// prior rides on it.
__device__ inline void promo_log_softmax(const __half* row, float out[4]) {
    float x[4];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = __half2float(row[i]);
        m = fmaxf(m, x[i]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) sum = __fadd_rn(sum, expf(__fsub_rn(x[i], m)));
    const float lg = logf(sum);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fsub_rn(__fsub_rn(x[i], m), lg);
}

struct ExpandResult {
    int n_edges;      // written, so at most kE
    int n_candidates; // before truncation, which is what §15.1 watches
    float dropped;    // prior mass discarded by truncation, 0 when none was
};

// Turn one node's legality mask into edges. Warp-collective, lane `i` carrying
// slot `i`'s mask word and slot word; `out_move` and `out_prior` are the node's
// own edge arrays.
//
// The mask is walked four times rather than held in registers. A slot's
// candidate list is up to 27 entries and a dynamically indexed local array that
// size spills to local memory, which costs more than recomputing a bit walk that
// is a handful of iterations on the ten or so lanes that have any move at all.
__device__ inline ExpandResult expand_node(uint64_t mask_word, uint16_t board_word,
                                           int16_t control, const __half* policy_row,
                                           const __half* promo_row, int lane,
                                           int16_t* out_move, __half* out_prior) {
    // §6.4 reads the type from the piece word and never from the slot index: a
    // promoted queen keeps its pawn slot, so slot < 8 does not imply a pawn.
    const bool is_pawn = ((board_word >> 11) & 1) == 0 && ((board_word >> 6) & 0b111) == kPawn;
    const uint64_t last_rank = white_to_move(control) ? (0xffULL << 56) : 0xffULL;
    const uint64_t promo_sq = is_pawn ? (mask_word & last_rank) : 0ULL;

    float lp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // The promotion head is indexed by the *moving slot*, which is this lane, and
    // spec §7.4 keeps it per token because a position can have more than one pawn
    // on the seventh rank.
    if (promo_sq) promo_log_softmax(promo_row + lane * 4, lp);

    const int mine = __popcll(mask_word) + 3 * __popcll(promo_sq);
    int total = 0;
    const int all_base = warp_exclusive_scan(mine, lane, &total);

    // The common path: nothing is dropped, so `keep` is every candidate and the
    // threshold below is never consulted.
    uint32_t t_key = 0;
    int need_eq = 0;
    int eq_base = 0;
    int base = all_base;
    const bool truncating = total > kE;

    if (truncating) {
        // §4.3: keep the kE largest unnormalised priors, ties by canonical
        // order. Radix-select the kE-th largest key from the top bit down,
        // maintaining `count(key >= prefix) >= kE`.
        uint32_t prefix = 0;
#pragma unroll 1
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t cand = prefix | (1u << bit);
            int c = 0;
            uint64_t m = mask_word;
            while (m) {
                const int sq = __ffsll((unsigned long long)m) - 1;
                m &= m - 1;
                const bool pr = (promo_sq >> sq) & 1ULL;
                const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t > 0 && !pr) continue;
                    if (float_key(pr ? __fadd_rn(bl, lp[t]) : bl) >= cand) ++c;
                }
            }
            if (__reduce_add_sync(kAll, (unsigned)c) >= (unsigned)kE) prefix = cand;
        }
        t_key = prefix;

        // `gt` candidates beat the threshold outright; the remaining slots go to
        // the first `kE - gt` of those that tie with it, in canonical order.
        int gt = 0, eq = 0;
        uint64_t m = mask_word;
        while (m) {
            const int sq = __ffsll((unsigned long long)m) - 1;
            m &= m - 1;
            const bool pr = (promo_sq >> sq) & 1ULL;
            const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t > 0 && !pr) continue;
                const uint32_t k = float_key(pr ? __fadd_rn(bl, lp[t]) : bl);
                if (k > t_key) ++gt;
                else if (k == t_key) ++eq;
            }
        }
        int gt_total = 0, eq_total = 0;
        const int gt_base = warp_exclusive_scan(gt, lane, &gt_total);
        eq_base = warp_exclusive_scan(eq, lane, &eq_total);
        need_eq = kE - gt_total;
        base = gt_base + min(eq_base, need_eq);
        (void)eq_total;
    }

    // Whether a candidate survives, given its key and its rank among the ties.
    auto keep = [&](uint32_t k, int eq_rank) -> bool {
        if (!truncating) return true;
        return k > t_key || (k == t_key && eq_rank < need_eq);
    };

    // Pass 2 and 3: the softmax over the survivors. torch's SoftMaxForwardEpilogue
    // is `exp(x - max) / sum`, and the sum here runs over at most kE terms
    // instead of the reference's 8192 masked ones, so the two agree to about a
    // ULP rather than exactly. §12 measures that against the selection margins.
    float local_max = -INFINITY;
    {
        int eq_rank = eq_base;
        uint64_t m = mask_word;
        while (m) {
            const int sq = __ffsll((unsigned long long)m) - 1;
            m &= m - 1;
            const bool pr = (promo_sq >> sq) & 1ULL;
            const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t > 0 && !pr) continue;
                const float lg = pr ? __fadd_rn(bl, lp[t]) : bl;
                const uint32_t k = float_key(lg);
                const int rank = (truncating && k == t_key) ? eq_rank++ : 0;
                if (keep(k, rank)) local_max = fmaxf(local_max, lg);
            }
        }
    }
    float amax = local_max;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(kAll, amax, off));
    amax = __shfl_sync(kAll, amax, 0);

    float local_sum = 0.0f, local_all = 0.0f, local_drop = 0.0f;
    float all_max = -INFINITY;  // only read when truncating
    if (truncating) {
        // The discarded mass has to be read off the distribution *before*
        // truncation (§15.1): the priors written below are renormalised over the
        // survivors, where the dropped tail is zero by construction.
        float lm = -INFINITY;
        uint64_t m = mask_word;
        while (m) {
            const int sq = __ffsll((unsigned long long)m) - 1;
            m &= m - 1;
            const bool pr = (promo_sq >> sq) & 1ULL;
            const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t > 0 && !pr) continue;
                lm = fmaxf(lm, pr ? __fadd_rn(bl, lp[t]) : bl);
            }
        }
        all_max = lm;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            all_max = fmaxf(all_max, __shfl_down_sync(kAll, all_max, off));
        all_max = __shfl_sync(kAll, all_max, 0);
    }

    {
        int eq_rank = eq_base;
        uint64_t m = mask_word;
        while (m) {
            const int sq = __ffsll((unsigned long long)m) - 1;
            m &= m - 1;
            const bool pr = (promo_sq >> sq) & 1ULL;
            const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t > 0 && !pr) continue;
                const float lg = pr ? __fadd_rn(bl, lp[t]) : bl;
                const uint32_t k = float_key(lg);
                const int rank = (truncating && k == t_key) ? eq_rank++ : 0;
                const bool kept = keep(k, rank);
                if (kept) local_sum = __fadd_rn(local_sum, expf(__fsub_rn(lg, amax)));
                if (truncating) {
                    const float w = expf(__fsub_rn(lg, all_max));
                    local_all = __fadd_rn(local_all, w);
                    if (!kept) local_drop = __fadd_rn(local_drop, w);
                }
            }
        }
    }
    float sum = local_sum, sum_all = local_all, drop = local_drop;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(kAll, sum, off);
        sum_all += __shfl_down_sync(kAll, sum_all, off);
        drop += __shfl_down_sync(kAll, drop, off);
    }
    sum = __shfl_sync(kAll, sum, 0);
    sum_all = __shfl_sync(kAll, sum_all, 0);
    drop = __shfl_sync(kAll, drop, 0);

    // Pass 4: write the survivors, in canonical order rather than in prior
    // order, so that the enumeration above is the only ordering an
    // implementation has to reproduce.
    {
        int pos = base;
        int eq_rank = eq_base;
        uint64_t m = mask_word;
        while (m) {
            const int sq = __ffsll((unsigned long long)m) - 1;
            m &= m - 1;
            const bool pr = (promo_sq >> sq) & 1ULL;
            const float bl = __half2float(policy_row[lane * 64 + sq]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t > 0 && !pr) continue;
                const float lg = pr ? __fadd_rn(bl, lp[t]) : bl;
                const uint32_t k = float_key(lg);
                const int rank = (truncating && k == t_key) ? eq_rank++ : 0;
                if (!keep(k, rank)) continue;
                out_move[pos] = (int16_t)((lane * 64 + sq) | (t << kMoveBits));
                out_prior[pos] = __float2half(__fdiv_rn(expf(__fsub_rn(lg, amax)), sum));
                ++pos;
            }
        }
    }

    ExpandResult r;
    r.n_edges = truncating ? kE : total;
    r.n_candidates = total;
    r.dropped = truncating ? __fdiv_rn(drop, sum_all) : 0.0f;
    return r;
}

// ---------------------------------------------------------------------------
// §7, the repetition count during the descent
// ---------------------------------------------------------------------------

// How many times the child position has occurred, counting itself. The game ring
// and the in-tree path are counted separately because no engine-side signature
// joins them: spec §6.3's ring is [N, 100] and the tree half is the search's own.
//
// Warp-collective, uniform result. `length` is the current path length, so the
// path covers the root down to the child's parent.
__device__ inline int tree_repetition(const Tree& t, int b, int length, uint64_t child_hash,
                                      bool child_irrev, int lane) {
    const int16_t* path = t.path_node + (size_t)b * t.D;

    // Node 0's own irreversibility is the real game's business, since the ring
    // was already emptied by it (spec §6.2), so only levels 1 and below cut.
    int deepest = -1;
    for (int d = lane; d < length; d += 32) {
        if (d < 1) continue;
        const uint8_t fl = t.node_flags[node_at(t, b, path[d])];
        if (fl & kIrreversible) deepest = max(deepest, d);
    }
    deepest = __reduce_max_sync(kAll, deepest);

    // An irreversible move into the child makes every earlier position
    // unreachable, which is the empty-window case.
    const int cut = child_irrev ? length : max(deepest, 0);
    unsigned hits = 0;
    for (int d = lane + cut; d < length; d += 32)
        hits += (t.node_hash[node_at(t, b, path[d])] == child_hash);
    const int on_tree = (int)__reduce_add_sync(kAll, hits);

    const bool keep_ring = deepest < 0 && !child_irrev;
    const int ring_len = keep_ring ? (int)t.game_ring_len[b] : 0;
    return repetition_count(child_hash, t.game_ring + (size_t)b * kMaxHistory, ring_len, lane)
           + on_tree;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// §6.1's first half: node 0 from the real game position, and the staging the
// encoder reads. The root then goes through the same `evaluate` and `expand` the
// interior nodes do, which is why there is no root-specific expansion path.
__global__ void root_init_kernel(Tree t, const uint16_t* __restrict__ game_board,
                                 const int16_t* __restrict__ game_control,
                                 const uint64_t* __restrict__ game_hash,
                                 uint8_t* __restrict__ out_root_rep) {
    const int b = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (b >= t.B) return;
    const int lane = (int)(threadIdx.x & 31);

    const uint16_t word = game_board[(size_t)b * 32 + lane];
    t.node_board[board_at(t, b, 0) + lane] = word;
    t.leaf_board[(size_t)b * 32 + lane] = word;

    const size_t e0 = edge_at(t, b, 0);
    for (int e = lane; e < kE; e += 32) {
        t.edge_move[e0 + e] = 0;
        t.edge_prior[e0 + e] = __float2half(0.0f);
        t.edge_child[e0 + e] = -1;
        t.edge_N[e0 + e] = 0;
        t.edge_Q[e0 + e] = 0.0f;
    }

    // The root's repetition count is over the game ring alone: no tree exists
    // yet, and the ring holds every position before this one (§7).
    const uint64_t h = game_hash[b];
    const int rep = repetition_count(h, t.game_ring + (size_t)b * kMaxHistory,
                                     (int)t.game_ring_len[b], lane);
    if (lane == 0) {
        const size_t n0 = node_at(t, b, 0);
        t.node_control[n0] = game_control[b];
        t.node_hash[n0] = h;
        t.node_parent[n0] = -1;
        t.node_pedge[n0] = 0;
        t.node_flags[n0] = 0;
        t.node_nedges[n0] = 0;
        t.node_count[b] = 1;
        // `path_len` is deliberately left alone. It is dead between moves: every
        // descent writes it before anything reads it, and the reference does not
        // clear it here either, so zeroing it would be the one field on which two
        // correct implementations disagree.
        const uint8_t r = (uint8_t)min(rep - 1, 2);
        out_root_rep[b] = r;
        t.leaf_control[b] = game_control[b];
        t.leaf_rep[b] = r;
        t.leaf_node[b] = 0;
        t.leaf_flags[b] = kNeedsExpand;
    }
}

// §6.2: select down to a leaf, apply the move, count repetitions, test for
// termination, allocate. One warp per game, and the tree walk never leaves the
// kernel, which is the whole reason the engine lives in headers.
__global__ void descent_kernel(Tree t, Luts g, Params p, int s,
                               Counters* __restrict__ ctr) {
    __shared__ SharedLuts sl;
    __shared__ SharedZobrist sz;
    load_luts(sl, g);
    load_zobrist(sz);

    const int b = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (b >= t.B) return;
    const int lane = (int)(threadIdx.x & 31);

    // §11's structural hook: the loop reads a per-game budget rather than the
    // constant `n`, which is what playout cap randomisation needs. An exhausted
    // game rides along with an empty path, so every launch shape stays static.
    const bool active = t.budget[b] > s;

    int v = 0, d = 0, parent = -1, chosen = 0, leaf = 0;
    bool fresh = false;
    if (active) {
        int16_t* path_node = t.path_node + (size_t)b * t.D;
        uint8_t* path_edge = t.path_edge + (size_t)b * t.D;
        for (;;) {
            if ((t.node_flags[node_at(t, b, v)] & kTerminalMask) != 0) {
                leaf = v;
                break;
            }
            const size_t ev = edge_at(t, b, v);
            const int e = puct_argmax(t.edge_N + ev, t.edge_prior + ev, t.edge_Q + ev,
                                      (int)t.node_nedges[node_at(t, b, v)], lane, p);
            if (lane == 0) {
                path_node[d] = (int16_t)v;
                path_edge[d] = (uint8_t)e;
            }
            ++d;
            const int c = t.edge_child[ev + e];
            if (c < 0) {
                parent = v;
                chosen = e;
                fresh = true;
                break;
            }
            v = c;
            if (d >= t.D) {
                // §4.1 makes this unreachable: a path over Nmax = n + 1 nodes
                // traverses at most n = Dmax edges, and the node pool would
                // overflow first. Counted rather than capped silently.
                if (lane == 0 && ctr) atomicAdd(&ctr->depth_overflow, 1ull);
                leaf = v;
                break;
            }
        }
    }
    if (lane == 0) t.path_len[b] = d;
    // The path has to be visible to `tree_repetition` below, which reads it
    // through lanes other than the one that wrote it.
    __syncwarp();

    // The move to apply, or spec §9's null move for a game that is creating no
    // node. Its row is computed like any other and the result discarded, which
    // is cheaper than a compaction and is what keeps the encoder batch static
    // (§6.3). A terminal node's edges are cleared, so its label reads as 0.
    const int src = fresh ? parent : leaf;
    const int label = fresh ? (int)(uint16_t)t.edge_move[edge_at(t, b, src) + chosen] : 0;
    const int move = fresh ? (label & kMoveMask) : -1;
    const int promo = (label >> kMoveBits) & 0b11;

    const size_t nsrc = node_at(t, b, src);
    const FullStepResult r =
        step_full<true>(sl, sz, t.node_board[nsrc * 32 + lane], lane, t.node_control[nsrc],
                        t.node_hash[nsrc], move, promo);
    const MovegenResult mg = movegen(sl, r.word, lane, !white_to_move(r.control));

    const int rep = tree_repetition(t, b, d, r.hash, r.irreversible, lane);
    const TerminalResult tr =
        terminal(mg.mask, r.word, lane, mg.in_check, r.control, 0ULL, nullptr, 0);
    // `terminal` is called without a ring because the count above is over the
    // ring *and* the path. Code 4 is folded in at spec §4.3's priority, above
    // insufficient material and below everything else; both are draws, so
    // `result` does not move.
    int code = (int)tr.code;
    if (rep >= 3 && (code == kNone || code == kInsufficient)) code = kRepetition;

    if (fresh) {
        int c = 0;
        if (lane == 0) c = atomicAdd(&t.node_count[b], 1);
        c = __shfl_sync(kAll, c, 0);
        if (c >= t.N) {
            // Invariant 1. Unreachable by §4.1's Nmax = n + 1, so it is counted
            // and the write dropped rather than trapping the whole launch.
            if (lane == 0 && ctr) atomicAdd(&ctr->pool_overflow, 1ull);
            fresh = false;
            leaf = src;
        } else {
            leaf = c;
            t.node_board[board_at(t, b, c) + lane] = r.word;
            const size_t ec = edge_at(t, b, c);
            for (int e = lane; e < kE; e += 32) {
                t.edge_move[ec + e] = 0;
                t.edge_prior[ec + e] = __float2half(0.0f);
                t.edge_child[ec + e] = -1;
                t.edge_N[ec + e] = 0;
                t.edge_Q[ec + e] = 0.0f;
            }
            if (lane == 0) {
                const size_t nc = node_at(t, b, c);
                t.node_control[nc] = r.control;
                t.node_hash[nc] = r.hash;
                t.node_parent[nc] = (int16_t)parent;
                t.node_pedge[nc] = (uint8_t)chosen;
                t.node_nedges[nc] = 0;
                t.node_flags[nc] =
                    (uint8_t)(code | (r.irreversible ? kIrreversible : 0));
                // A terminal node carries its result in place of an evaluation,
                // in the [0,1] convention of §3.5: -1 becomes 0, 0 becomes 0.5.
                if (code != kNone)
                    t.node_value[nc] = __float2half(((float)tr.result + 1.0f) / 2.0f);
                t.edge_child[edge_at(t, b, parent) + chosen] = (int16_t)c;
            }
        }
    }

    t.leaf_board[(size_t)b * 32 + lane] = r.word;
    if (lane == 0) {
        t.leaf_control[b] = r.control;
        t.leaf_rep[b] = (uint8_t)min(rep - 1, 2);
        t.leaf_node[b] = (int16_t)leaf;
        const bool expand = fresh && code == kNone;
        t.leaf_flags[b] = (uint8_t)((expand ? kNeedsExpand : 0) | (fresh ? kFresh : 0));
        if (ctr && active) {
            atomicAdd(&ctr->simulations, 1ull);
            atomicAdd(&ctr->depth_sum, (unsigned long long)d);
            atomicMax(&ctr->max_depth, (unsigned)d);
            atomicMax(&ctr->max_nodes, (unsigned)t.node_count[b]);
            if (!fresh) atomicAdd(&ctr->terminal_descents, 1ull);
            if (fresh && code != kNone) atomicAdd(&ctr->terminal_children, 1ull);
        }
    }
}

// §6.4. One warp per game, on the node the descent staged.
__global__ void expand_kernel(Tree t, Luts g, const __half* __restrict__ policy,
                              const __half* __restrict__ promo,
                              const float* __restrict__ value, Counters* __restrict__ ctr) {
    __shared__ SharedLuts sl;
    load_luts(sl, g);

    const int b = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (b >= t.B) return;
    const int lane = (int)(threadIdx.x & 31);
    if ((t.leaf_flags[b] & kNeedsExpand) == 0) return;

    const int node = (int)t.leaf_node[b];
    // The staged board is the same words as node_board[b][node] and is
    // contiguous, so the mask comes off a coalesced read rather than a stride
    // through the tree.
    const uint16_t word = t.leaf_board[(size_t)b * 32 + lane];
    const int16_t control = t.leaf_control[b];
    const MovegenResult mg = movegen(sl, word, lane, !white_to_move(control));

    const size_t ev = edge_at(t, b, node);
    const ExpandResult er =
        expand_node(mg.mask, word, control, policy + (size_t)b * 32 * 64,
                    promo + (size_t)b * 32 * 4, lane, t.edge_move + ev, t.edge_prior + ev);

    if (lane == 0) {
        const size_t nv = node_at(t, b, node);
        t.node_nedges[nv] = (uint8_t)er.n_edges;
        t.node_flags[nv] |= kExpanded;
        // §3.5: the value head is tanh in [-1, 1] and the tree works in [0, 1].
        t.node_value[nv] = __float2half((value[b] + 1.0f) / 2.0f);
        if (ctr) {
            atomicMax(&ctr->max_edges, (unsigned)er.n_candidates);
            if (er.n_candidates > kE) {
                atomicAdd(&ctr->truncated_nodes, 1ull);
                // The mass matters more than the count: a dropped tail worth
                // 0.1 % of the prior is harmless and one worth 20 % is not.
                atomicAdd(&ctr->truncated_mass, er.dropped);
            }
            if (er.n_candidates == 0) atomicAdd(&ctr->empty_mask_expansions, 1ull);
        }
    }
}

// §6.5. Lane `i` takes path level `i`: every level of a path touches a distinct
// edge and the point of view at each level is fixed by parity, so the updates
// are independent and no atomic is needed.
//
// ⚠️ Inverting the parity produces a search that reliably plays the worst
// available move. `(L - d) % 2` and `d % 2` agree at L = 2, so a two-level test
// cannot see it.
__global__ void backup_kernel(Tree t) {
    const int b = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (b >= t.B) return;
    const int lane = (int)(threadIdx.x & 31);

    const int L = t.path_len[b];
    if (L == 0) return;
    const float qleaf = __half2float(t.node_value[node_at(t, b, (int)t.leaf_node[b])]);
    const int16_t* path_node = t.path_node + (size_t)b * t.D;
    const uint8_t* path_edge = t.path_edge + (size_t)b * t.D;

    for (int d = lane; d < L; d += 32) {
        const size_t e = edge_at(t, b, path_node[d]) + path_edge[d];
        const float q = ((L - d) & 1) ? (1.0f - qleaf) : qleaf;
        const float count = (float)t.edge_N[e] + 1.0f;
        t.edge_N[e] = (int16_t)count;
        const float old = t.edge_Q[e];
        t.edge_Q[e] = old + (q - old) / count;
    }
}

}  // namespace search
}  // namespace brokefish
