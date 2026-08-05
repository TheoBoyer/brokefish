// The e4m3 FFN GEMM: packed weight layout, activation quantisation, and the tile.
//
// The scheme is specified in **brokefish/nn/quant.py** and measured in
// docs/journal/2026-08-04-fp8-encoder.md. Settled parameters, from that measurement
// and not from taste:
//
//     matmuls      linear1, linear2      66.7 % of encoder-body FLOPs
//     scaling      activations 1 x 128 online, weights 128 x 128 offline
//     q_max        16          flat from 8 to 32; 48 degrades, 64 overflows
//     accumulator  fp16 inside a 128-tile, fp32 across tiles
//
// ⚠️ **This file is checked against `quant.py`, not the other way round.** The device
// test in csrc/tests/tfp8.cu pins the layouts and the plumbing; the number that
// decides whether the kernel is *right* is the prior-space comparison through
// `brokefish/nn/validate.py`, because that is the quantity the search consumes.
//
// ## Why the accumulator is split
//
// One `mma.m16n8k32` sums 32 products into an fp16 accumulator. With `q_max = 16`
// the worst-case partial is 32 * 16^2 = 8192 against fp16's 65504, so a 128-element
// tile -- four mma -- is safe with 8x headroom. `linear2`'s K is 1024, i.e. 32 mma,
// which is not, and `q_max = 64` was measured to produce 121,912 non-finite outputs
// on the real network. So the fp16 accumulator runs for exactly one 128-tile and is
// then descaled by `sa * sw` and folded into an fp32 running total.
//
// ⚠️ `linear2`'s input is post-ReLU and therefore **non-negative**, so its partial
// sums get no sign cancellation and sit nearer the worst case than `linear1`'s. That
// is the one place the 8x headroom above is doing real work rather than being slack.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "mma.cuh"

namespace brokefish {
namespace fp8 {

constexpr int kTileK = 128;      // activation scale tile, and the descale period
constexpr int kBlockN = 128;     // weight scale block
constexpr int kMmaK = 32;        // one mma
constexpr float kQMax = 16.0f;

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

// ---------------------------------------------------------------------------
// Activations
//
// `ldmatrix` on sm_89 is 16-bit only -- there is no `.b8` form until sm_90 -- so the
// A fragment cannot be loaded the way the fp16 path loads it. It does not need to be:
// the scheme requires a per-row-per-tile amax and a convert before the GEMM anyway,
// so the quantised activations are written to shared memory **already in the
// fragment's order**, and each thread then reads its four contiguous bytes with one
// 32-bit load.
//
// Row pitch is padded by 4 bytes. Without it, lanes sharing a `t` and differing in
// `g` stride by the row pitch: at 32 B that is 8 banks, so eight lanes hit four banks
// and every load is a 4-way conflict.
constexpr int kARowPad = 4;

// ⚠️ `__host__` too: the packer and whoever sizes the shared-memory allocation both
// need this, and a pitch computed in two places is a pitch that will disagree.
__host__ __device__ __forceinline__ int a_pitch(int k_bytes) { return k_bytes + kARowPad; }

/// Quantise `rows x kTileK` fp16 activations into e4m3, one scale per row.
///
/// `src` is row-major fp16 with stride `src_stride`; `dst` is row-major bytes with
/// stride `dst_pitch`; `scale` gets one float per row. Called by all 32 lanes of a
/// warp, which between them own `rows` rows.
///
/// The amax is a full-row reduction over 128 elements. Each lane takes four columns,
/// so the reduction is over the eight lanes sharing a row -- `__shfl_xor_sync` with
/// the three low bits, not the full five-step tree, because the row is owned by a
/// contiguous group of eight.
__device__ __forceinline__ void quantise_tile(uint8_t* dst, int dst_pitch,
                                              float* scale, int scale_stride,
                                              const half* src, int src_stride,
                                              int rows, int lane) {
    const int col_group = lane % 8;          // eight lanes per row, four columns each
    const int row_step = 32 / 8;             // four rows in flight per warp pass
    const int row0 = lane / 8;

    for (int r = row0; r < rows; r += row_step) {
        const half* s = src + (size_t)r * src_stride;
        // 16 columns per lane: 128 / 8. Read as four `__half2` pairs per 4-column
        // chunk so the loads are 32-bit rather than sixteen scalar LDS.
        float m = 0.0f;
        half v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = s[col_group * 16 + i];
            m = fmaxf(m, fabsf(__half2float(v[i])));
        }
#pragma unroll
        for (int off = 1; off < 8; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

        // ⚠️ **The barrier that makes an in-place quantisation legal**, and it is a
        // compiler barrier before it is a hardware one. Lane 1 writes bytes [16, 32)
        // of a row that lane 0 is reading as halves [0, 16) = bytes [0, 32), so every
        // lane's reads must complete before any lane's writes. A shuffle orders the
        // lanes but is **not** a memory barrier: `v[]` is consumed only in the write
        // loop below, so ptxas is free to sink those sixteen loads past the shuffle to
        // save registers -- and under this kernel's 128-register cap it does.
        //
        // Measured when this was missing: the FFN produced NaN on 116 of 128 boards
        // with fp8 on ff1 alone, and the separate-buffer test could not see it.
        __syncwarp();
        const float sc = (m > 0.0f) ? m / kQMax : 1.0f;
        const float inv = 1.0f / sc;
        if (col_group == 0) scale[(size_t)r * scale_stride] = sc;

        uint8_t* d = dst + (size_t)r * dst_pitch + col_group * 16;
#pragma unroll
        for (int i = 0; i < 16; i += 4) {
            // ⚠️ The scaling is done in **fp32**, not by an fp16 multiply against
            // `__float2half(inv)`. A row whose maximum is below 2^-14 gives
            // `inv > 65504`, so the fp16 form makes `inv` itself infinite, and then
            // `0 * inf` is NaN -- a single quiet row poisons the whole board's logits.
            // The e4m3 convert is `satfinite`, so it clamps rather than producing NaN;
            // this is the only place a NaN could be manufactured.
            const __half2 lo = __floats2half2_rn(__half2float(v[i]) * inv,
                                                 __half2float(v[i + 1]) * inv);
            const __half2 hi = __floats2half2_rn(__half2float(v[i + 2]) * inv,
                                                 __half2float(v[i + 3]) * inv);
            *reinterpret_cast<uint32_t*>(d + i) = cvt_e4m3x4(lo, hi);
        }
    }
}

/// `float acc[2][4][4]` back into the `uint32_t [2][4][2]` fp16 fragments the rest of
/// encoder.cu passes around. `acc[..][0..1]` is row g and `[2..3]` row g+8, which is
/// exactly what `h2()`'s low and high halves mean for a D fragment.
__device__ __forceinline__ void frags_from_f32(uint32_t (&d)[2][4][2],
                                               const float (&s)[2][4][4]) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            d[m][p][0] = u32(__floats2half2_rn(s[m][p][0], s[m][p][1]));
            d[m][p][1] = u32(__floats2half2_rn(s[m][p][2], s[m][p][3]));
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

/// `acc[m][p] += (A W^T)` over `K32N` k-groups, descaling once per 128-element tile.
///
/// `acc` is fp32 and lives across the whole GEMM; the fp16 accumulator is local to a
/// tile. `a_scale` is indexed `[row][ktile]`, `w_scale` `[n8 / 16][ktile]` -- sixteen
/// n-groups of 8 to a 128-wide block.
// ⚠️ `<K32N, NK32>` -- **depth first, pitch second**, matching `gemm_direct` in
// encoder.cu. The two are called side by side and one has to be substitutable for the
// other; reversed, the `ff2` call site `<8, 32>` reads a matrix of row pitch 8 as one
// of pitch 32, which is in range, finite, and wrong. The device test only had them
// equal, so it could not have caught this.
template <int K32N, int NK32>
__device__ __forceinline__ void gemm_fp8(float acc[2][4][4], const uint8_t* a_base,
                                         int a_pitch_, const float* a_scale,
                                         int a_scale_stride, const uint2* w,
                                         const float* w_scale, int w_scale_stride,
                                         int n8_0, int k32_0, int lane) {
    const uint2* wp = w + ((size_t)n8_0 * NK32 + k32_0) * 32 + lane;
    constexpr int PS = NK32 * 32;                    // one n-group of 8, in uint2
    constexpr int kMmaPerTile = kTileK / kMmaK;      // four
    static_assert(K32N % kMmaPerTile == 0, "the k depth must be whole 128-tiles");

    const int g = lane / 4;

    // ⚠️ **No ping-pong prefetch here, and that is a measurement rather than an
    // omission.** `gemm_direct` prefetches the next k-group's weights while the mma
    // runs, and its comments price that highly. Adding the same thing here -- two
    // `uint2` buffers of four n-tiles, 16 registers -- took the encoder from **1.147x
    // to 0.835x**, a 27 % regression. The fp8 path halves the arithmetic but not the
    // register pressure, so it sits against the 128-register cap that
    // `__launch_bounds__(THREADS, 2)` imposes, and buying latency hiding with
    // registers buys spilling instead. The fp16 tile is latency-bound; this one is not.
    uint32_t hacc[2][4][2] = {};                 // fp16, one 128-tile's worth

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

        if ((k32 + 1) % kMmaPerTile != 0) continue;

        // Descale and promote, once per 128-element tile. ⚠️ Two different tile
        // indices: the weight matrix is addressed globally so its scale takes `k32_0`
        // into account, the activation buffer holds only this call's k-slice so its
        // scale is indexed from zero.
        const int tile = k32 / kMmaPerTile;
        const int wtile = (k32_0 / kMmaPerTile) + tile;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float sa_lo = a_scale[(size_t)(m * 16 + g) * a_scale_stride + tile];
            const float sa_hi = a_scale[(size_t)(m * 16 + g + 8) * a_scale_stride + tile];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int n8 = n8_0 + p;
                const float sw = w_scale[(size_t)(n8 / (kBlockN / 8)) * w_scale_stride
                                         + wtile];
                const __half2 d0 = h2(hacc[m][p][0]), d1 = h2(hacc[m][p][1]);
                acc[m][p][0] = fmaf(__half2float(__low2half(d0)), sa_lo * sw, acc[m][p][0]);
                acc[m][p][1] = fmaf(__half2float(__high2half(d0)), sa_lo * sw, acc[m][p][1]);
                acc[m][p][2] = fmaf(__half2float(__low2half(d1)), sa_hi * sw, acc[m][p][2]);
                acc[m][p][3] = fmaf(__half2float(__high2half(d1)), sa_hi * sw, acc[m][p][3]);
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) { hacc[m][p][0] = 0; hacc[m][p][1] = 0; }
        }
    }
}

}  // namespace fp8
}  // namespace brokefish
