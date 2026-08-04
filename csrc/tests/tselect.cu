// Device-level tests for the two warp reductions inside the search.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tselect.cu -o tselect && ./tselect
//
// The differential harness in tests/test_search_cuda.py compares whole trees
// against the PyTorch reference, which is the stronger check and the one that
// matters. It is also the one that cannot say *which* reduction was wrong: a
// mis-scanned PUCT score and a mis-ordered edge enumeration both come out as
// "the trees diverged at simulation 43". These two pin the pieces on their own,
// against a host reference written from docs/mcts.md §6.4 and §6.6 rather than
// copied from the kernel, with inputs a real position would take years to reach:
// every score tied, every logit tied, 218 candidates, 64 candidates exactly.
//
// The host reference computes in double and the kernel in float, so scores are
// compared through their *ordering* and priors to a tolerance. Where the ordering
// is genuinely ambiguous the case is counted and reported rather than being
// silently called a pass; §6.6's tie-break is then tested separately on inputs
// that are exact in both precisions.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_fp16.h>

#include "harness.cuh"
#include "search.cuh"

using namespace brokefish;
using namespace brokefish::search;

namespace {

constexpr float kBase = 19652.0f;
constexpr float kInit = 1.25f;

// splitmix64, so the cases are reproducible without a dump.
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int below(int n) { return (int)(next() % (uint64_t)n); }
    double unit() { return (double)(next() >> 11) / 9007199254740992.0; }
};

// --------------------------------------------------------------------------- //
// §6.6, the selection score
// --------------------------------------------------------------------------- //

// `win` is §6.6a's mask, or nullptr for the plain scan. The kernel takes it rather
// than two kernels taking one each, so the collapse is exercised through the same
// entry point the descent uses.
__global__ void puct_kernel(const int16_t* nvis, const __half* prior, const float* qs,
                            const uint8_t* win, const int* nedges, Params p, int* out,
                            int cases) {
    const int c = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (c >= cases) return;
    const int lane = (int)(threadIdx.x & 31);
    const size_t off = (size_t)c * kE;
    const int e = puct_argmax(nvis + off, prior + off, qs + off,
                              win ? win + off : nullptr, nedges[c], lane, p);
    if (lane == 0) out[c] = e;
}

// The same formula in double, from §6.6 rather than from the kernel. Returns the
// winning edge and, through `margin`, how far ahead of the runner-up it was.
int host_puct(const int16_t* nvis, const __half* prior, const float* qs, int nedges,
              double* margin) {
    double n_v = 0.0;
    for (int e = 0; e < nedges; ++e) n_v += (double)nvis[e];
    const double pb = std::log((n_v + (double)kBase + 1.0) / (double)kBase) + (double)kInit;
    const double pb_root = pb * std::sqrt(n_v);

    int best_e = -1;
    double best = -INFINITY, second = -INFINITY;
    for (int e = 0; e < nedges; ++e) {
        const double n = (double)nvis[e];
        const double u = pb_root / (n + 1.0);
        // ⚠️ §6.6's first-play urgency, and it must be `kFpuDraw`. This line read a
        // literal 0.0 until 2026-08-03 and the check had been failing 71 of 20 000
        // cases since the FPU fix of 2026-07-31 -- the **fourth** independent copy of
        // this expression to carry AGZ's `[-1, 1]` zero into the `[0, 1]` tree, after
        // `tests/oracle.py`, `search/trace.py` and `debugger/static/trace.js`. It
        // survived because `csrc/tests` is a separate no-Python build that `pytest`
        // never runs, so nothing said so out loud.
        const double q = n > 0.0 ? (double)qs[e] : (double)kFpuDraw;
        const double score = u * (double)__half2float(prior[e]) + q;
        if (score > best) {
            second = best;
            best = score;
            best_e = e;
        } else if (score > second) {
            second = score;
        }
    }
    *margin = nedges > 1 ? best - second : INFINITY;
    return best_e;
}

int test_puct() {
    const int cases = 20000;
    std::vector<int16_t> nvis((size_t)cases * kE, 0);
    std::vector<__half> prior((size_t)cases * kE);
    std::vector<float> qs((size_t)cases * kE, 0.0f);
    std::vector<int> nedges(cases);

    Rng rng{0xC1C1C1C1ULL};
    for (int c = 0; c < cases; ++c) {
        // Four regimes, because the exploration term and the value term dominate
        // in different ones and a bug in either is invisible in the other.
        //   0  a fresh node: no visits at all, so every score is exactly 0
        //   1  early: a handful of visits, exploration dominates
        //   2  late: hundreds of visits, Q dominates
        //   3  equal priors and equal Q, so the tie-break decides
        const int regime = c % 4;
        const int ne = 1 + rng.below(kE);
        nedges[c] = ne;
        const float flat_q = (float)(rng.below(5)) / 4.0f;
        for (int e = 0; e < ne; ++e) {
            const size_t i = (size_t)c * kE + e;
            int n = 0;
            if (regime == 1) n = rng.below(6);
            else if (regime == 2) n = rng.below(400);
            else if (regime == 3) n = 1 + rng.below(3);
            nvis[i] = (int16_t)n;
            prior[i] = __float2half(regime == 3 ? 0.125f : (float)rng.unit());
            qs[i] = n > 0 ? (regime == 3 ? flat_q : (float)rng.unit()) : 0.0f;
        }
    }

    int16_t* d_nvis = to_device(nvis);
    __half* d_prior = to_device(prior);
    float* d_qs = to_device(qs);
    int* d_nedges = to_device(nedges);
    int* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, cases * sizeof(int)));

    puct_kernel<<<(cases + kWarps - 1) / kWarps, kWarps * 32>>>(
        d_nvis, d_prior, d_qs, nullptr, d_nedges, Params{kBase, kInit}, d_out, cases);
    CHECK(cudaGetLastError());
    std::vector<int> got(cases);
    CHECK(cudaMemcpy(got.data(), d_out, cases * sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0, ambiguous = 0, ties = 0;
    for (int c = 0; c < cases; ++c) {
        double margin = 0.0;
        const size_t off = (size_t)c * kE;
        const int want = host_puct(&nvis[off], &prior[off], &qs[off], nedges[c], &margin);
        if (margin == 0.0) ++ties;
        if (got[c] == want) continue;
        // fp32 against fp64: a decision closer than the float epsilon of the
        // score is not attributable, so it is counted rather than failed.
        if (margin < 1e-6) {
            ++ambiguous;
            continue;
        }
        if (bad++ < 5)
            printf("  case %d: kernel chose %d, host %d, margin %.3g\n", c, got[c], want,
                   margin);
    }
    printf("  %-34s %s  (%d cases, %d exact ties, %d inside the fp32 floor)\n",
           "puct_argmax", bad ? "FAIL" : "OK", cases, ties, ambiguous);
    CHECK(cudaFree(d_nvis));
    CHECK(cudaFree(d_prior));
    CHECK(cudaFree(d_qs));
    CHECK(cudaFree(d_nedges));
    CHECK(cudaFree(d_out));
    return bad;
}

// The tie-break on its own, on inputs that are exact in every precision, so
// "lowest index wins" is an equality and not a tolerance. §6.6 fixes it because
// §12 compares trees edge for edge, and because at a freshly created node every
// score is exactly zero and the tie-break alone decides the first descent.
int test_tie_break() {
    struct Case {
        int nedges;
        int visits;   // the same on every edge
        const char* what;
    };
    const Case cases[] = {
        {1, 0, "one edge, no visits"},
        {20, 0, "a fresh node: sqrt(N_v) is 0, so every score is 0"},
        {64, 0, "a full fresh node"},
        {33, 4, "equal priors and equal Q, across the lane boundary"},
        {64, 7, "a full node, all equal"},
        {2, 1, "two equal edges"},
    };
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));

    std::vector<int16_t> nvis((size_t)n * kE, 0);
    std::vector<__half> prior((size_t)n * kE, __float2half(0.0f));
    std::vector<float> qs((size_t)n * kE, 0.0f);
    std::vector<int> nedges(n);
    for (int c = 0; c < n; ++c) {
        nedges[c] = cases[c].nedges;
        for (int e = 0; e < cases[c].nedges; ++e) {
            const size_t i = (size_t)c * kE + e;
            nvis[i] = (int16_t)cases[c].visits;
            // Powers of two, exact in fp16 and fp32 alike, so the scores tie
            // exactly rather than nearly.
            prior[i] = __float2half(0.25f);
            qs[i] = cases[c].visits > 0 ? 0.5f : 0.0f;
        }
    }

    int16_t* d_nvis = to_device(nvis);
    __half* d_prior = to_device(prior);
    float* d_qs = to_device(qs);
    int* d_nedges = to_device(nedges);
    int* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));
    puct_kernel<<<(n + kWarps - 1) / kWarps, kWarps * 32>>>(
        d_nvis, d_prior, d_qs, nullptr, d_nedges, Params{kBase, kInit}, d_out, n);
    CHECK(cudaGetLastError());
    std::vector<int> got(n);
    CHECK(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int c = 0; c < n; ++c) {
        if (got[c] != 0) {
            printf("  %s: every score ties and the kernel chose edge %d, not 0\n",
                   cases[c].what, got[c]);
            ++bad;
        }
    }
    printf("  %-34s %s  (%d all-tied nodes)\n", "puct_argmax tie-break",
           bad ? "FAIL" : "OK", n);
    CHECK(cudaFree(d_nvis));
    CHECK(cudaFree(d_prior));
    CHECK(cudaFree(d_qs));
    CHECK(cudaFree(d_nedges));
    CHECK(cudaFree(d_out));
    return bad;
}

// §6.6a, the collapse. Two claims, and the second is the one a bug would hide in:
//
//   * with at least one winning edge, the scan returns the **least-visited** one,
//     ties to the lowest index -- that is what makes the visits round-robin and
//     `pi` come out uniform over the winners;
//   * with none, the scan is bit-for-bit the plain PUCT scan, so passing the mask
//     costs nothing on the 99 % of nodes that have no proved win.
//
// Exact, no tolerance: the collapsed branch compares small integers and never
// touches a prior, so there is no fp32-against-fp64 floor to hide behind.
int test_collapse() {
    const int cases = 20000;
    std::vector<int16_t> nvis((size_t)cases * kE, 0);
    std::vector<__half> prior((size_t)cases * kE, __float2half(0.0f));
    std::vector<float> qs((size_t)cases * kE, 0.0f);
    std::vector<uint8_t> win((size_t)cases * kE, 0);
    std::vector<int> nedges(cases);

    Rng rng{0x5EED5EEDULL};
    for (int c = 0; c < cases; ++c) {
        // Regime 0 has no winner at all, which is the "identical to the plain scan"
        // half. The rest carry 1..4 winners, placed anywhere in the edge set so the
        // warp reduction has to cross lane boundaries to find them.
        const int regime = c % 4;
        const int ne = 1 + rng.below(kE);
        nedges[c] = ne;
        for (int e = 0; e < ne; ++e) {
            const size_t i = (size_t)c * kE + e;
            const int n = rng.below(200);
            nvis[i] = (int16_t)n;
            prior[i] = __float2half((float)rng.unit());
            qs[i] = n > 0 ? (float)rng.unit() : 0.0f;
        }
        if (regime != 0) {
            const int w = 1 + rng.below(4);
            for (int k = 0; k < w; ++k) win[(size_t)c * kE + rng.below(ne)] = 1;
        }
    }

    int16_t* d_nvis = to_device(nvis);
    __half* d_prior = to_device(prior);
    float* d_qs = to_device(qs);
    uint8_t* d_win = to_device(win);
    int* d_nedges = to_device(nedges);
    int* d_out = nullptr;
    int* d_off = nullptr;
    CHECK(cudaMalloc(&d_out, cases * sizeof(int)));
    CHECK(cudaMalloc(&d_off, cases * sizeof(int)));

    puct_kernel<<<(cases + kWarps - 1) / kWarps, kWarps * 32>>>(
        d_nvis, d_prior, d_qs, d_win, d_nedges, Params{kBase, kInit}, d_out, cases);
    CHECK(cudaGetLastError());
    // The same inputs with the mask withheld, so "no winner behaves exactly like
    // the plain scan" is checked against the kernel itself rather than against the
    // host reference's rounding.
    puct_kernel<<<(cases + kWarps - 1) / kWarps, kWarps * 32>>>(
        d_nvis, d_prior, d_qs, nullptr, d_nedges, Params{kBase, kInit}, d_off, cases);
    CHECK(cudaGetLastError());
    std::vector<int> got(cases), plain(cases);
    CHECK(cudaMemcpy(got.data(), d_out, cases * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(plain.data(), d_off, cases * sizeof(int), cudaMemcpyDeviceToHost));

    int bad = 0, collapsed = 0, passthrough = 0;
    for (int c = 0; c < cases; ++c) {
        const size_t off = (size_t)c * kE;
        int want = -1, fewest = 1 << 30;
        for (int e = 0; e < nedges[c]; ++e) {
            if (!win[off + e]) continue;
            if ((int)nvis[off + e] < fewest) {
                fewest = (int)nvis[off + e];
                want = e;
            }
        }
        if (want < 0) {
            ++passthrough;
            want = plain[c];
        } else {
            ++collapsed;
        }
        if (got[c] == want) continue;
        if (bad++ < 5)
            printf("  case %d: kernel chose %d, expected %d (%s)\n", c, got[c], want,
                   fewest < (1 << 30) ? "collapsed" : "pass-through");
    }
    printf("  %-34s %s  (%d collapsed, %d pass-through)\n", "puct_argmax collapse",
           bad ? "FAIL" : "OK", collapsed, passthrough);
    CHECK(cudaFree(d_nvis));
    CHECK(cudaFree(d_prior));
    CHECK(cudaFree(d_qs));
    CHECK(cudaFree(d_win));
    CHECK(cudaFree(d_nedges));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_off));
    return bad;
}

// --------------------------------------------------------------------------- //
// §6.4, the canonical edge enumeration
// --------------------------------------------------------------------------- //

__global__ void expand_probe_kernel(const uint64_t* masks, const uint16_t* boards,
                                    const int16_t* control, const __half* policy,
                                    const __half* promo, int16_t* out_move,
                                    __half* out_prior, int* out_n, int* out_cand,
                                    float* out_drop, int cases) {
    const int c = blockIdx.x * kWarps + (int)(threadIdx.x >> 5);
    if (c >= cases) return;
    const int lane = (int)(threadIdx.x & 31);
    const ExpandResult r = expand_node(
        masks[(size_t)c * 32 + lane], boards[(size_t)c * 32 + lane], control[c],
        policy + (size_t)c * 32 * 64, promo + (size_t)c * 32 * 4, lane,
        out_move + (size_t)c * kE, out_prior + (size_t)c * kE);
    if (lane == 0) {
        out_n[c] = r.n_edges;
        out_cand[c] = r.n_candidates;
        out_drop[c] = r.dropped;
    }
}

// The gap between adjacent fp16 values at `x`. Priors are stored as fp16 (§4.2)
// and a truncated node's tail lands in the subnormal range, where the spacing is
// a flat 2^-24 and a relative tolerance would be meaningless: two adjacent fp16
// numbers near 5e-6 are 0.6 % apart.
double ulp16(double x) {
    const double a = std::fabs(x);
    if (a < 6.103515625e-5) return 5.9604644775390625e-8;  // subnormal
    int exp = 0;
    std::frexp(a, &exp);
    return std::ldexp(1.0, exp - 11);
}

struct Cand {
    int index;      // (slot * 64 + square) * 4 + promo, the canonical order
    int move;       // the spec §3 label with the promotion field
    double logit;
};

// The enumeration of §6.4, written from the document. Ascending by slot, then by
// target square, then by promotion type in spec §3's N B R Q order; a promoting
// move emits four edges and every other move emits one. Truncation keeps the E
// largest logits with ties broken by canonical order, and writes the survivors in
// canonical order rather than in prior order.
std::vector<Cand> host_expand(const uint64_t* mask, const uint16_t* board,
                              int16_t control, const __half* policy,
                              const __half* promo, int* n_candidates, double* dropped) {
    const int last_rank = control > 0 ? 7 : 0;
    std::vector<Cand> all;
    for (int slot = 0; slot < 32; ++slot) {
        const uint16_t w = board[slot];
        // The type comes from the word and never from the slot index: a promoted
        // queen keeps its pawn slot.
        const bool is_pawn = ((w >> 11) & 1) == 0 && ((w >> 6) & 0b111) == 0;
        double lp[4] = {0, 0, 0, 0};
        {
            double x[4], m = -INFINITY, sum = 0.0;
            for (int i = 0; i < 4; ++i) {
                x[i] = (double)__half2float(promo[slot * 4 + i]);
                m = std::max(m, x[i]);
            }
            for (int i = 0; i < 4; ++i) sum += std::exp(x[i] - m);
            for (int i = 0; i < 4; ++i) lp[i] = (x[i] - m) - std::log(sum);
        }
        for (int sq = 0; sq < 64; ++sq) {
            if (!((mask[slot] >> sq) & 1ULL)) continue;
            const bool pr = is_pawn && (sq >> 3) == last_rank;
            const double base = (double)__half2float(policy[slot * 64 + sq]);
            for (int t = 0; t < (pr ? 4 : 1); ++t)
                all.push_back({(slot * 64 + sq) * 4 + t, (slot * 64 + sq) | (t << kMoveBits),
                               pr ? base + lp[t] : base});
        }
    }
    *n_candidates = (int)all.size();

    std::vector<Cand> keep = all;
    std::stable_sort(keep.begin(), keep.end(), [](const Cand& a, const Cand& b) {
        if (a.logit != b.logit) return a.logit > b.logit;
        return a.index < b.index;
    });
    *dropped = 0.0;
    if ((int)keep.size() > kE) {
        double m = -INFINITY, sum = 0.0, drop = 0.0;
        for (const Cand& c : all) m = std::max(m, c.logit);
        for (const Cand& c : all) sum += std::exp(c.logit - m);
        for (size_t i = kE; i < keep.size(); ++i) drop += std::exp(keep[i].logit - m);
        *dropped = drop / sum;
        keep.resize(kE);
    }
    std::sort(keep.begin(), keep.end(),
              [](const Cand& a, const Cand& b) { return a.index < b.index; });
    return keep;
}

int test_expand() {
    // 0-3 plain, 4-7 with promoting pawns, 8-9 engineered to truncate on ties.
    const int cases = 400;
    std::vector<uint64_t> masks((size_t)cases * 32, 0);
    std::vector<uint16_t> boards((size_t)cases * 32, 0x800);  // every slot captured
    std::vector<int16_t> control(cases, 1);
    std::vector<__half> policy((size_t)cases * 32 * 64, __float2half(0.0f));
    std::vector<__half> promo((size_t)cases * 32 * 4, __float2half(0.0f));

    Rng rng{0x5E1EC7ULL};
    int with_promo = 0, truncating = 0, tied = 0;
    for (int c = 0; c < cases; ++c) {
        const int regime = c % 10;
        const bool flat = regime >= 8;                 // every logit identical
        const bool pawns = regime >= 4;
        const bool many = regime == 3 || regime >= 6;  // enough candidates to truncate
        control[c] = (c & 1) ? 1 : -1;
        const int last = control[c] > 0 ? 7 : 0;

        const int n_slots = 1 + rng.below(many ? 16 : 6);
        for (int k = 0; k < n_slots; ++k) {
            const int slot = rng.below(32);
            const int bits = many ? 4 + rng.below(20) : 1 + rng.below(6);
            uint64_t m = 0;
            for (int j = 0; j < bits; ++j) m |= 1ULL << rng.below(64);
            if (pawns && (k & 1)) {
                boards[(size_t)c * 32 + slot] = (uint16_t)((0 << 6) | rng.below(64));
                m |= 0xffULL << (8 * last);  // land on the promotion rank
            } else {
                boards[(size_t)c * 32 + slot] = (uint16_t)((4 << 6) | rng.below(64));
            }
            masks[(size_t)c * 32 + slot] |= m;
        }
        for (int i = 0; i < 32 * 64; ++i)
            policy[(size_t)c * 32 * 64 + i] =
                __float2half(flat ? 0.5f : (float)(rng.unit() * 8.0 - 4.0));
        for (int i = 0; i < 32 * 4; ++i)
            promo[(size_t)c * 32 * 4 + i] =
                __float2half(flat ? 0.25f : (float)(rng.unit() * 4.0 - 2.0));
    }

    uint64_t* d_masks = to_device(masks);
    uint16_t* d_boards = to_device(boards);
    int16_t* d_control = to_device(control);
    __half* d_policy = to_device(policy);
    __half* d_promo = to_device(promo);
    int16_t* d_move = nullptr;
    __half* d_prior = nullptr;
    int *d_n = nullptr, *d_cand = nullptr;
    float* d_drop = nullptr;
    CHECK(cudaMalloc(&d_move, (size_t)cases * kE * sizeof(int16_t)));
    CHECK(cudaMalloc(&d_prior, (size_t)cases * kE * sizeof(__half)));
    CHECK(cudaMalloc(&d_n, cases * sizeof(int)));
    CHECK(cudaMalloc(&d_cand, cases * sizeof(int)));
    CHECK(cudaMalloc(&d_drop, cases * sizeof(float)));
    CHECK(cudaMemset(d_move, 0, (size_t)cases * kE * sizeof(int16_t)));

    expand_probe_kernel<<<(cases + kWarps - 1) / kWarps, kWarps * 32>>>(
        d_masks, d_boards, d_control, d_policy, d_promo, d_move, d_prior, d_n, d_cand,
        d_drop, cases);
    CHECK(cudaGetLastError());

    std::vector<int16_t> move((size_t)cases * kE);
    std::vector<__half> prior((size_t)cases * kE);
    std::vector<int> n(cases), cand(cases);
    std::vector<float> drop(cases);
    CHECK(cudaMemcpy(move.data(), d_move, move.size() * sizeof(int16_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(prior.data(), d_prior, prior.size() * sizeof(__half),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(n.data(), d_n, cases * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cand.data(), d_cand, cases * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(drop.data(), d_drop, cases * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    double worst_prior = 0.0;
    for (int c = 0; c < cases; ++c) {
        int n_cand = 0;
        double dropped = 0.0;
        const std::vector<Cand> want =
            host_expand(&masks[(size_t)c * 32], &boards[(size_t)c * 32], control[c],
                        &policy[(size_t)c * 32 * 64], &promo[(size_t)c * 32 * 4], &n_cand,
                        &dropped);
        if (n_cand > kE) ++truncating;
        for (const Cand& x : want)
            if ((x.move >> kMoveBits) != 0) { ++with_promo; break; }

        if (cand[c] != n_cand || n[c] != (int)want.size()) {
            if (bad++ < 5)
                printf("  case %d: %d edges from %d candidates, host says %zu from %d\n", c,
                       n[c], cand[c], want.size(), n_cand);
            continue;
        }
        double sum = 0.0;
        for (const Cand& x : want) sum += std::exp(x.logit);
        (void)sum;
        int local = 0;
        for (int e = 0; e < (int)want.size(); ++e) {
            const int got_move = (int)(uint16_t)move[(size_t)c * kE + e];
            if (got_move != want[e].move && local++ < 3) {
                if (bad++ < 8)
                    printf("  case %d edge %d: move %d, host %d (canonical index %d)\n", c,
                           e, got_move, want[e].move, want[e].index);
            }
        }
        // The priors are compared to a tolerance because the host reference works
        // in double and the kernel in float, and both then round to fp16.
        double denom = 0.0, m = -INFINITY;
        for (const Cand& x : want) m = std::max(m, x.logit);
        for (const Cand& x : want) denom += std::exp(x.logit - m);
        for (int e = 0; e < (int)want.size(); ++e) {
            const double w = std::exp(want[e].logit - m) / denom;
            const double g = (double)__half2float(prior[(size_t)c * kE + e]);
            // Two fp16 steps: one because the host works in double and the kernel
            // in float, one because both then round to fp16.
            const double err = std::fabs(w - g) / ulp16(w);
            worst_prior = std::max(worst_prior, err);
            if (err > 2.0) {
                if (bad++ < 8)
                    printf("  case %d edge %d: prior %.6g, host %.6g (%.2f fp16 ULP)\n", c,
                           e, g, w, err);
            }
        }
        if (n_cand > kE) {
            if (std::fabs(drop[c] - dropped) > 1e-3 * std::max(dropped, 1e-6)) {
                if (bad++ < 8)
                    printf("  case %d: dropped mass %.6g, host %.6g\n", c, drop[c], dropped);
            }
            if (drop[c] <= 0.0 && bad++ < 8)
                printf("  case %d: truncated %d candidates and reported no mass\n", c,
                       n_cand);
        }
        if (c % 10 >= 8) ++tied;
    }
    printf("  %-34s %s  (%d cases, %d truncating, %d all-logits-tied, "
           "worst prior error %.2f fp16 ULP)\n",
           "expand_node", bad ? "FAIL" : "OK", cases, truncating, tied, worst_prior);
    if (!truncating) {
        printf("  expand_node: no case truncated, §4.3's radix select went untested\n");
        ++bad;
    }
    if (!with_promo) {
        printf("  expand_node: no promotion edge, §6.4's four-per-move fan-out untested\n");
        ++bad;
    }
    CHECK(cudaFree(d_masks));
    CHECK(cudaFree(d_boards));
    CHECK(cudaFree(d_control));
    CHECK(cudaFree(d_policy));
    CHECK(cudaFree(d_promo));
    CHECK(cudaFree(d_move));
    CHECK(cudaFree(d_prior));
    CHECK(cudaFree(d_n));
    CHECK(cudaFree(d_cand));
    CHECK(cudaFree(d_drop));
    return bad;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_puct();
    failures += test_tie_break();
    failures += test_collapse();
    failures += test_expand();
    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
