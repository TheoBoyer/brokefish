// The e4m3 FFN GEMM: packed weight layout, activation quantisation, and the tile.
//
// The scheme is specified in **brokefish/nn/quant.py** and measured in
// docs/journal/2026-08-04-fp8-encoder.md and 2026-08-05-fp8-per-row.md. Settled
// parameters, from measurement and not from taste:
//
//     matmuls      linear1, linear2      66.7 % of encoder-body FLOPs
//     scaling      activations 1 x 256 online, weights 128 x K offline
//     q_max        8           flat from 4 to 32 in prior space; see the bound below
//     accumulator  fp16 for the whole 256-deep GEMM, descaled once at the end
//
// ⚠️ **This file is checked against `quant.py`, not the other way round.** The device
// test in csrc/tests/tfp8.cu pins the layouts and the plumbing; the number that
// decides whether the kernel is *right* is the prior-space comparison through
// `brokefish/nn/validate.py`, because that is the quantity the search consumes.
//
// ## Why there is only one accumulator, and why that is the whole point
//
// The first version scaled activations per 128-element tile, which forced a descale
// every 128 k-elements and therefore an `float acc[2][4][4]` running total on top of
// the fp16 mma fragment: 32 registers where the fp16 path needs 16. Under
// `__launch_bounds__(THREADS, 2)` ptxas has exactly 65536 / (2 * 256) = **128
// registers** and it paid the difference in local memory -- 960 B of spill loads
// against the fp16 path's 64.
//
// ⚠️ Buying the registers back is a measured loss. `__launch_bounds__(THREADS, 1)`
// removes every spill (200 registers, 0 B) and makes the kernel **slower**: 0.978x
// against fp16 where the spilling version was 1.154x. Two CTAs per SM is 16 warps and
// that is the entire latency-hiding budget. So the fp32 accumulator had to go, not
// grow room.
//
// One scale per row over the whole 256-deep reduction removes it: every k-group then
// shares a scale, so the fp16 mma fragment *is* the accumulator and one descale at the
// end suffices. Measured, `t7h-fp8-002206`, 2041 non-terminal positions:
//
//     spill loads   960 B -> 200 B      (the fp16 path's is 64 B)
//     throughput    1.154x -> 1.214x    = 82.2k evals/s at B = 4096
//     max |dp|      1.05e-2 -> 1.81e-2, p95 1.22e-3, top-1 moved 1.32 %
//
// The emulation in quant.py predicted 1.83e-2 / 1.25e-3 / 1.14 % for this scheme, so
// the kernel now lands on its specification -- the per-128-tile version was 1.6x worse
// than its own prediction, and losing seven of the eight descales is what closed it.
//
// ## The accumulator bound, which is what sets q_max
//
// One `mma.m16n8k32` sums 32 products; 256 elements of k is eight of them into one
// fp16 accumulator, so the worst case is `q_max^2 * 256` against fp16's 65504. That is
// a hard ceiling, not a guideline: fp16 has no saturation, an overflow becomes `inf`,
// ReLU passes `inf` through, and the *next* tile's amax is then `inf` -- so `inv` is 0
// and `inf * 0` is NaN, two tiles downstream of the cause. Measured at q_max = 448:
// 116 boards of 128 came out NaN with every component testing clean in isolation.
//
// `q_max = 8` gives 16384, a 4x margin, and the prior-space sweep found 4, 6 and 8
// indistinguishable. `linear2`'s input is post-ReLU and therefore **non-negative**, so
// its partial sums get no sign cancellation and sit nearer the worst case than
// `linear1`'s; that margin is where it is doing real work rather than being slack.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "mma.cuh"

namespace brokefish {
namespace fp8 {

constexpr int kRowK = 256;       // k-depth one activation scale covers, = one GEMM call
constexpr int kBlockN = 128;     // weight scale block, over the *whole* k
constexpr int kMmaK = 32;        // one mma
constexpr int kTileK = 128;      // the quantiser's write granularity; see quantise_row
constexpr float kQMax = 8.0f;

static_assert(kQMax * kQMax * kRowK <= 65504.0f / 3.0f,
              "the fp16 accumulator has no saturation: q_max^2 * kRowK must stay well "
              "under 65504 or an overflow becomes inf and then NaN two tiles later");
static_assert(kRowK % kTileK == 0 && kTileK % kMmaK == 0, "the k widths must nest");

// ---------------------------------------------------------------------------
// Packed weights
//
// Mirrors `gemm_direct`'s fp16 packing with half the bytes. For each n-group of 8
// and k-group of 32, the 32 lanes hold one `uint2` each:
//
//     packed[(n8 * NK32 + k32) * 32 + lane] = { B[4t..4t+3][g], B[4t+16..4t+19][g] }
//
// with g = lane / 4, t = lane % 4 and B[k][n] = W[n][k] -- the mma's B operand is
// k-major, and the weight is stored [out, in], so the transpose is free at pack time.
//
// 8 n x 32 k x 1 byte = 256 B per group against the fp16 path's 512 B. That halving
// is the second-order win: the FFN weights are 8.4 MB and become 4.2 MB, which is
// bandwidth *and* whatever occupancy the smaller footprint buys.
//
// ⚠️ The scale blocking changed with the accumulator. It is now **one float per 128
// output columns for the whole of k**, not per (128 out, 128 k) tile, because every
// k-group has to share a scale for the single fp16 accumulator to be meaningful. A
// warp owns four consecutive n8-tiles = 32 columns, 32-aligned, so its four tiles
// always fall inside one 128-block and one `sw` covers the whole call.

// ---------------------------------------------------------------------------
// Activations
//
// `ldmatrix` on sm_89 is 16-bit only -- there is no `.b8` form until sm_90 -- so the
// A fragment cannot be loaded the way the fp16 path loads it. It does not need to be:
// the scheme requires a per-row amax and a convert before the GEMM anyway, so the
// quantised activations are written to shared memory **already in the fragment's
// order**, and each thread then reads its four contiguous bytes with one 32-bit load.
//
// Row pitch is padded by 4 bytes. Without it, lanes sharing a `t` and differing in
// `g` stride by the row pitch: at 32 B that is 8 banks, so eight lanes hit four banks
// and every load is a 4-way conflict.
constexpr int kARowPad = 4;

// ⚠️ `__host__` too: the packer and whoever sizes the shared-memory allocation both
// need this, and a pitch computed in two places is a pitch that will disagree.
__host__ __device__ __forceinline__ int a_pitch(int k_bytes) { return k_bytes + kARowPad; }

/// Quantise `rows x WIDTH` fp16 activations into e4m3 **in place**, one scale per row.
///
/// `src` is row-major fp16 with stride `src_stride`; `dst` is row-major bytes with
/// stride `dst_pitch`, and the two may alias -- the caller relies on that, because
/// SMEM_HALVES is already at its static_assert and a separate byte buffer would cost
/// the second CTA. `scale` gets one float per row. Called by all 32 lanes of a warp,
/// which between them own `rows` rows.
///
/// The amax is a full-row reduction over WIDTH elements. Each lane takes a contiguous
/// eighth of the row, so the reduction is over the eight lanes sharing a row --
/// `__shfl_xor_sync` with the three low bits, not the full five-step tree.
///
/// ## The three aliasing hazards, and why the shape is what it is
///
/// A row's WIDTH halves occupy `2 * WIDTH` bytes and its quantised form occupies
/// `WIDTH`, so the bytes land on the row's own first half. Three separate orderings
/// have to hold and only one of them is a hardware question:
///
///  1. **The amax pass reads the whole row, so it must finish in every lane before
///     any lane writes.** That is the `__syncwarp` below. A shuffle orders the lanes
///     but is *not* a memory barrier.
///  2. **Inside one 128-element tile, lane c writes bytes [16c, 16c+16) while lane
///     c/2 is still reading bytes [32(c/2), +32).** They overlap, so the sixteen
///     source halves are held in registers across the write -- which is the whole
///     reason `v[16]` exists and why this is not written as a two-pass loop over the
///     full row. Measured when the barrier before it was missing: NaN on 116 of 128
///     boards, and the separate-buffer test could not see it.
///  3. **Across tiles there is no hazard and so no barrier.** Tile t writes bytes
///     [128t, 128t+128) and tile t+1 reads halves [128(t+1), ...) = bytes
///     [256t+256, ...), which is strictly above. Ascending tile order is load-bearing.
template <int WIDTH>
__device__ __forceinline__ void quantise_row(uint8_t* dst, int dst_pitch,
                                             float* scale, int scale_stride,
                                             const half* src, int src_stride,
                                             int rows, int lane) {
    static_assert(WIDTH % kTileK == 0, "the row must be whole 128-element tiles");
    constexpr int kTiles = WIDTH / kTileK;
    constexpr int kPerLane = kTileK / 8;      // sixteen columns per lane per tile
    const int col_group = lane % 8;           // eight lanes per row
    const int row_step = 32 / 8;              // four rows in flight per warp pass
    const int row0 = lane / 8;

    for (int r = row0; r < rows; r += row_step) {
        const half* s = src + (size_t)r * src_stride;

        // Hazard 1: read-only, so the whole row can be scanned before anything moves.
        float m = 0.0f;
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
#pragma unroll
            for (int i = 0; i < kPerLane; i += 2) {
                const __half2 v = *reinterpret_cast<const __half2*>(
                    s + t * kTileK + col_group * kPerLane + i);
                m = fmaxf(m, fabsf(__half2float(__low2half(v))));
                m = fmaxf(m, fabsf(__half2float(__high2half(v))));
            }
        }
#pragma unroll
        for (int off = 1; off < 8; off <<= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

        const float sc = (m > 0.0f) ? m / kQMax : 1.0f;
        const float inv = 1.0f / sc;
        if (col_group == 0) scale[(size_t)r * scale_stride] = sc;

        __syncwarp();                          // hazard 1

#pragma unroll 1
        for (int t = 0; t < kTiles; ++t) {     // hazard 3: ascending, no barrier
            const half* st = s + t * kTileK + col_group * kPerLane;
            half v[kPerLane];                  // hazard 2: the reads outlive the writes
#pragma unroll
            for (int i = 0; i < kPerLane; ++i) v[i] = st[i];

            uint8_t* d = dst + (size_t)r * dst_pitch + t * kTileK + col_group * kPerLane;
#pragma unroll
            for (int i = 0; i < kPerLane; i += 4) {
                // ⚠️ The scaling is done in **fp32**, not by an fp16 multiply against
                // `__float2half(inv)`. A row whose maximum is below 2^-14 gives
                // `inv > 65504`, so the fp16 form makes `inv` itself infinite, and
                // then `0 * inf` is NaN -- a single quiet row poisons the whole
                // board's logits. The e4m3 convert is `satfinite`, so it clamps rather
                // than producing NaN; this is the only place a NaN could be made.
                const __half2 lo = __floats2half2_rn(__half2float(v[i]) * inv,
                                                     __half2float(v[i + 1]) * inv);
                const __half2 hi = __floats2half2_rn(__half2float(v[i + 2]) * inv,
                                                     __half2float(v[i + 3]) * inv);
                *reinterpret_cast<uint32_t*>(d + i) = cvt_e4m3x4(lo, hi);
            }
        }
    }
}

/// One thread's A fragment for m-tile row base `m0` and k-tile `k0`.
///
/// a[i] takes rows g / g+8 and the k-halves at 4t and 4t+16, which is the layout
/// csrc/tests/tfp8.cu pins against a host product.
__device__ __forceinline__ void load_a_frag(uint32_t (&a)[4], const uint8_t* base,
                                            int pitch, int m0, int k0, int lane) {
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = m0 + ((i & 1) ? g + 8 : g);
        const int col = k0 + ((i >> 1) ? 4 * t + 16 : 4 * t);
        a[i] = *reinterpret_cast<const uint32_t*>(base + (size_t)row * pitch + col);
    }
}

/// `acc[m][p] = (or +=) A W^T` over exactly `kRowK` elements of k, in fp16 throughout.
///
/// `acc` is the same `uint32_t [2][4][2]` fragment `gemm_direct` produces, so the two
/// are substitutable at the call site and the epilogue -- bias, ReLU, `store_frags` --
/// does not care which ran. `a_scale` is one float per row; `w_scale` one float per
/// 128 output columns, covering the whole k.
///
/// `ADD` folds this call into a running total, which is what `linear2` needs: its k is
/// 1024 but only 256 hidden units are materialised at a time, so it arrives as four
/// calls. Each carries its own scale and is descaled before the add, so the four never
/// have to agree on one -- and the running total is a *network activation*, which is
/// stored as fp16 two instructions later anyway.
///
// ⚠️ `<K32N, NK32>` -- **depth first, pitch second**, matching `gemm_direct` in
// encoder.cu. The two are called side by side and one has to be substitutable for the
// other; reversed, the `ff2` call site `<8, 32>` reads a matrix of row pitch 8 as one
// of pitch 32, which is in range, finite, and wrong. The device test only had them
// equal, so it could not have caught this.
template <int K32N, int NK32, bool ADD>
__device__ __forceinline__ void gemm_fp8_row(uint32_t acc[2][4][2], const uint8_t* a_base,
                                             int a_pitch_, const float* a_scale,
                                             int a_scale_stride, const uint2* w,
                                             const float* w_scale,
                                             int n8_0, int k32_0, int lane) {
    static_assert(K32N * kMmaK == kRowK,
                  "one activation scale has to cover exactly this call's k-depth, or "
                  "the single fp16 accumulator is summing incommensurable units");
    const uint2* wp = w + ((size_t)n8_0 * NK32 + k32_0) * 32 + lane;
    constexpr int PS = NK32 * 32;                    // one n-group of 8, in uint2
    const int g = lane / 4;

    // ⚠️ **No ping-pong prefetch here, and that is a measurement rather than an
    // omission.** `gemm_direct` prefetches the next k-group's weights while the mma
    // runs, and its comments price that highly. Adding the same thing here -- two
    // `uint2` buffers of four n-tiles, 16 registers -- took the encoder from **1.147x
    // to 0.835x**, a 27 % regression. Halving the arithmetic does not halve the
    // register pressure, so this path sits against the 128-register cap and buying
    // latency hiding with registers buys spilling instead.
    uint32_t hacc[2][4][2] = {};

#pragma unroll 1
    for (int k32 = 0; k32 < K32N; ++k32) {
        uint32_t af[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m)
            load_a_frag(af[m], a_base, a_pitch_, m * 16, k32 * kMmaK, lane);

#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint2 bw = wp[(size_t)p * PS + (size_t)k32 * 32];
            const uint32_t b[2] = {bw.x, bw.y};
#pragma unroll
            for (int m = 0; m < 2; ++m) mma_e4m3(hacc[m][p], af[m], b);
        }
    }

    // One descale for the whole GEMM. ⚠️ In fp32, not by an fp16 multiply against
    // `__float2half2_rn(sa * sw)`: the product of two scales is routinely below fp16's
    // smallest normal (6.1e-5) -- a weight block of amax 0.05 and an activation row of
    // amax 0.4 already gives 3.9e-5 -- and rounding the *scale* to a subnormal throws
    // away mantissa the accumulator still has. The accumulator itself stays fp16
    // because its dynamic range is bounded by construction; the scale's is not.
#pragma unroll
    for (int m = 0; m < 2; ++m) {
        const float sa_lo = a_scale[(size_t)(m * 16 + g) * a_scale_stride];
        const float sa_hi = a_scale[(size_t)(m * 16 + g + 8) * a_scale_stride];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            // A warp's four n8-tiles are 32 consecutive, 32-aligned columns, so they
            // never straddle a 128-wide scale block -- `n8_0 + p` and `n8_0` agree.
            const float sw = w_scale[(n8_0 + p) / (kBlockN / 8)];
            const __half2 d0 = h2(hacc[m][p][0]), d1 = h2(hacc[m][p][1]);
            float o[4] = {__half2float(__low2half(d0)) * sa_lo * sw,
                          __half2float(__high2half(d0)) * sa_lo * sw,
                          __half2float(__low2half(d1)) * sa_hi * sw,
                          __half2float(__high2half(d1)) * sa_hi * sw};
            if (ADD) {
                const __half2 a0 = h2(acc[m][p][0]), a1 = h2(acc[m][p][1]);
                o[0] += __half2float(__low2half(a0));
                o[1] += __half2float(__high2half(a0));
                o[2] += __half2float(__low2half(a1));
                o[3] += __half2float(__high2half(a1));
            }
            acc[m][p][0] = u32(__floats2half2_rn(o[0], o[1]));
            acc[m][p][1] = u32(__floats2half2_rn(o[2], o[3]));
        }
    }
}

}  // namespace fp8
}  // namespace brokefish
