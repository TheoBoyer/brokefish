// int8 for the FFN's two matmuls, in place of e4m3.
//
// docs/journal/2026-08-14-int8-kernel-spec.md is why. In one line: `mma.m16n8k32`
// with s8 operands issues at **72.2 TFLOPS**, exactly the e4m3-with-fp16-accumulator
// rate, and is **2.3x more accurate** in prior space -- FFN flip risk 4.43 % -> 1.93 %
// on `t12h-gumbel-004009`, 11.20 % -> 5.88 % on `t24h-muon-008215`.
//
// ## Why a format with no exponent wins here
//
// e4m3 spends four bits on an exponent and keeps three of mantissa, so its error is
// *relative*, ~2^-4 across eighteen binades. int8 spends everything on magnitude, so
// its error is *absolute* inside a block. Which is better is decided entirely by how
// much dynamic range a block actually has, and ours has almost none -- measured three
// separate ways in this repository:
//
//   * `q_max` is flat from 4 to 448 (2026-08-14): 0.6-0.9 % of Q entries subnormal at
//     4, 0.09 % at 32. There is no underflow for an exponent to rescue.
//   * a random-sign Hadamard bought nothing (2026-08-05), which is the same fact from
//     the other side: the tiles are not spiky enough for range tricks to pay.
//   * Q/K/V span ~2 binades of e4m3's 18.
//
// That is exactly the Gaussian case of arXiv:2303.17951, where INT8 wins.
//
// ## The scheme, and what it deletes
//
// Same granularity as the e4m3 path, because the same kernel constraint produces it:
// one activation scale per **row** over the whole `kRowK = 256` reduction, one weight
// scale per **128 output channels** over the whole K. `gemm_s8_row` reads
// `w_scale[n8/16]` with no k index, so a packer emitting a scale per k-tile would
// silently scale every output by k-tile 0's constant.
//
// ⚠️ **`kQMax` has no analogue here and that is a robustness result, not a detail.**
// The e4m3 path exists inside a squeeze: an fp16 accumulator sums `kRowK = 256`
// products, so `q_max^2 * 256 <= 65504` forces `q_max = 8`, and getting that constant
// wrong in one of two languages produced `inf`, then `inf * 0 = NaN`, on 116 boards of
// 128 while every isolated component tested clean. The worst s32 partial here is
// `127 * 127 * 256 = 4.13e6` against `2.147e9` -- a **520x margin**. int8 has no
// infinity, no NaN, no saturating-convert trap and no reachable overflow. The whole
// failure mode is unreachable.
//
// ⚠️ **The accumulator is `s32` and there is no narrower form** (see `mma.cuh`). Four
// registers per tile against the packed-fp16 path's two is what forces one CTA per SM.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "fp8_gemm.cuh"          // load_a_frag, and the packed-B contract it documents
#include "mma.cuh"

namespace brokefish {
namespace int8q {

constexpr int kMmaK = 32;
constexpr int kTileK = 128;      // the row-quantisation tile; an aliasing constraint,
                                 // not a scaling one -- see quantise_row_int8
constexpr int kBlockN = 128;     // one weight scale per 128 output channels
constexpr int kRowK = 256;       // one activation scale per row per 256 of k

//: ⚠️ Must equal `Q_MAX_S8` / `Q_MAX_U8` in brokefish/nn/quant.py.
constexpr float kQMaxS = 127.0f;
constexpr float kQMaxU = 255.0f;

// The bound that `kQMax` exists to enforce on the e4m3 path, stated once and then
// never thought about again.
static_assert(kQMaxU * kQMaxS * kRowK < 2147483647.0f,
              "the s32 accumulator must not be reachable; if this ever fires the "
              "scheme has grown a k-depth it was not designed for");

/// `[rows][WIDTH]` halves to `[rows][WIDTH]` bytes plus one scale per row.
///
/// A transcription of `fp8::quantise_row` with the convert swapped, and it keeps that
/// function's three aliasing hazards **verbatim**, because they are properties of the
/// buffer geometry rather than of the format: the quantised row lands on top of the
/// halves that produced it, so
///
///  1. the amax pass must finish in every lane before any lane writes (`__syncwarp`;
///     a shuffle orders lanes but is not a memory barrier);
///  2. inside one 128-element tile lane c writes bytes lane c/2 is still reading, so
///     the sixteen source halves are held in registers across the write -- that is
///     what `v[kPerLane]` is for, and why this is not two passes over the row;
///  3. across tiles there is no hazard, which is why ascending tile order is
///     load-bearing and there is no barrier between them.
///
/// Getting (1) wrong on the e4m3 path produced NaN on 116 of 128 boards and the
/// separate-buffer test could not see it.
///
/// `UNSIGNED` is for the post-ReLU hidden: the amax is then a plain max, the scale is
/// `amax/255`, and the operand carries 256 levels instead of 127.
template <int WIDTH, bool UNSIGNED, int ROWS>
__device__ __forceinline__ void quantise_row_int8(uint8_t* dst, int dst_pitch,
                                                  float* scale, int scale_stride,
                                                  const half* src, int src_stride,
                                                  int lane) {
    static_assert(WIDTH % kTileK == 0, "the row must be whole 128-element tiles");
    constexpr int kTiles = WIDTH / kTileK;
    constexpr int kPerLane = kTileK / 8;      // sixteen columns per lane per tile
    constexpr float kQMax = UNSIGNED ? kQMaxU : kQMaxS;
    const int col_group = lane % 8;           // eight lanes per row
    const int row_step = 32 / 8;              // four rows in flight per warp pass
    const int row0 = lane / 8;

    // ⚠️ **Rolled, deliberately.** `ROWS` is a template parameter only so the bound is
    // known; unrolling it measured **1.341x against 1.353x** because two rows in flight
    // means two sets of sixteen held halves, and this loop already spends its registers
    // on holding one row across both passes.
#pragma unroll 1
    for (int r = row0; r < ROWS; r += row_step) {
        const half* s = src + (size_t)r * src_stride;

        // ⚠️ **The whole of this lane's row, held in registers across both passes.**
        // The e4m3 original reads every element **twice** -- once to find the amax and
        // again to convert -- because it only had eight registers to spare against the
        // 128-register cap. At one CTA per SM there are ~50 free, and `kTiles *
        // kPerLane` halves is 16 of them, so the second read simply goes away.
        //
        // It also collapses the aliasing argument. The original needed three separate
        // orderings because its reads and writes interleaved; here **every read
        // happens before every write**, so hazards 2 and 3 cannot arise at all and
        // only hazard 1 -- other lanes still reading while this one writes -- remains.
        // ⚠️ **`uint4`, not scalar halves.** A lane's sixteen columns per tile are
        // contiguous and 16-byte aligned (`col_group * 16` halves = 32 B, and every row
        // pitch here is 528 B = 33 x 16), so a tile is two 128-bit loads, not sixteen.
        static_assert(kPerLane % 8 == 0, "a lane's slice must be whole uint4s");
        union { uint4 q[kTiles][kPerLane / 8]; half h[kTiles][kPerLane]; } rb;
        // ⚠️ **Four accumulators, not one.** A single `m` makes the amax a 32-long
        // dependent `fmaxf` chain, and `wait` -- fixed-latency dependency stalls -- is
        // the second stall in this kernel's ncu profile at 2.17 cycles per issue.
        // Four independent chains of eight cost three extra `fmaxf` at the end and are
        // **bit-identical**, because max is associative and commutative and exact.
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int j = 0; j < kPerLane / 8; ++j)
                rb.q[t][j] = *reinterpret_cast<const uint4*>(
                    s + t * kTileK + col_group * kPerLane + j * 8);
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int i = 0; i < kPerLane; ++i) {
                // For UNSIGNED the input is post-ReLU, so a plain max is the amax and
                // a negative can only be a denormal artefact -- the convert saturates
                // at 0 regardless.
                const float a = __half2float(rb.h[t][i]);
                m[i & 3] = fmaxf(m[i & 3], UNSIGNED ? a : fabsf(a));
            }
        float mm = fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3]));
#pragma unroll
        for (int off = 1; off < 8; off <<= 1)
            mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, off));

        const float sc = (mm > 0.0f) ? mm / kQMax : 1.0f;
        const float inv = 1.0f / sc;
        if (col_group == 0) scale[(size_t)r * scale_stride] = sc;

        __syncwarp();                          // hazard 1, and now the only one

#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            uint8_t* d = dst + (size_t)r * dst_pitch + t * kTileK + col_group * kPerLane;
#pragma unroll
            for (int i = 0; i < kPerLane; i += 4) {
                // ⚠️ fp32 scaling, for the same reason the e4m3 path gives: a row whose
                // maximum is below 2^-14 makes an fp16 `inv` infinite and `0 * inf` is
                // NaN. `cvt.rni.sat` cannot produce a NaN, but it can produce garbage
                // from one, so the arithmetic upstream of it still has to be clean.
                *reinterpret_cast<uint32_t*>(d + i) = cvt_int8x4<UNSIGNED>(
                    __half2float(rb.h[t][i]) * inv, __half2float(rb.h[t][i + 1]) * inv,
                    __half2float(rb.h[t][i + 2]) * inv, __half2float(rb.h[t][i + 3]) * inv);
            }
        }
    }
}

/// `acc[m][p] = (or +=) A W^T` over exactly `kRowK` elements of k, in s32 throughout.
///
/// The int8 twin of `fp8::gemm_fp8_row`, and deliberately the same shape: same packed
/// B layout, same `load_a_frag`, same single descale at the end, same absence of a
/// ping-pong prefetch. That last one is inherited rather than re-derived --
/// `fp8_gemm.cuh` records that adding two `uint2` buffers, sixteen registers, took the
/// encoder from **1.147x to 0.835x**. This path has more registers to play with (255
/// against 128) but the same warning applies until somebody measures otherwise.
///
/// `MT` is deduced from the accumulator, so a caller cannot silently ask for more
/// m-tiles than it allocated -- which is exactly the mistake available when the same
/// helper serves a 32-row and a 64-row buffer.
template <int K32N, int NK32, bool ADD, bool UNSIGNED_A, int MT>
__device__ __forceinline__ void gemm_s8_row(uint32_t (&acc)[MT][4][2],
                                            const uint8_t* a_base, int a_pitch_,
                                            const float* a_scale, int a_scale_stride,
                                            const uint2* w, const float* w_scale,
                                            int n8_0, int k32_0, int lane) {
    static_assert(K32N * kMmaK == kRowK,
                  "one activation scale has to cover exactly this call's k-depth, or "
                  "the single accumulator is summing incommensurable units");
    const uint2* wp = w + ((size_t)n8_0 * NK32 + k32_0) * 32 + lane;
    constexpr int PS = NK32 * 32;                    // one n-group of 8, in uint2
    const int g = lane / 4;

    int32_t hacc[MT][4][4] = {};

    // ⚠️ **Unrolled by two, and the exact factor is the measurement.** Rolled, the four
    // weight loads of iteration `k32 + 1` cannot issue until the back edge, so their
    // ~250-cycle L2 latency is exposed with only 2 warps per scheduler to cover it --
    // this kernel runs at 16.7 % occupancy and 70 % of its cycles have no eligible warp
    // (ncu, 2026-08-28). At two, the compiler overlaps one iteration's loads with the
    // other's `mma` without a hand-written ping-pong buffer, which is the thing that
    // regressed twice (88 204 -> 72 704 evals/s, above).
    //
    // Measured 2026-08-28, `bench_phases --impl int8`, cycles per board (warp 0):
    // rolled **401,343**, unroll 2 **393,872 (-1.86 %)**, unroll 4 **408,651** -- worse
    // than rolled, with `ff2_gemm` alone going 89,601 -> 101,244 as it spills. The
    // window that fits in the register budget is exactly two.
    //
    // ⚠️ `perf.md` records "`#pragma unroll 2` on the same loop is inside noise". That
    // was measured against the *ping-pong* form of this loop, not this one.
#pragma unroll 2
    for (int k32 = 0; k32 < K32N; ++k32) {
        uint32_t af[MT][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
            fp8::load_a_frag(af[m], a_base, a_pitch_, m * 16, k32 * kMmaK, lane);

#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint2 bw = wp[(size_t)p * PS + (size_t)k32 * 32];
            const uint32_t b[2] = {bw.x, bw.y};
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if constexpr (UNSIGNED_A) mma_u8s8(hacc[m][p], af[m], b);
                else                      mma_s8(hacc[m][p], af[m], b);
            }
        }
    }

    // One descale for the whole GEMM, in fp32 -- the product of two scales is
    // routinely below fp16's smallest normal (a weight block of amax 0.05 against an
    // activation row of amax 0.4 already gives 3.9e-5), and rounding the *scale* to a
    // subnormal throws away mantissa the accumulator still holds.
    //
    // The s32 D fragment is c0,c1 on row g and c2,c3 on row g+8, which is the same
    // row split the packed-fp16 fragment expresses as two registers.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const float sa_lo = a_scale[(size_t)(m * 16 + g) * a_scale_stride];
        const float sa_hi = a_scale[(size_t)(m * 16 + g + 8) * a_scale_stride];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            // A warp's four n8-tiles are 32 consecutive, 32-aligned columns, so they
            // never straddle a 128-wide scale block.
            const float sw = w_scale[(n8_0 + p) / (kBlockN / 8)];
            float o[4] = {(float)hacc[m][p][0] * sa_lo * sw,
                          (float)hacc[m][p][1] * sa_lo * sw,
                          (float)hacc[m][p][2] * sa_hi * sw,
                          (float)hacc[m][p][3] * sa_hi * sw};
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

}  // namespace int8q
}  // namespace brokefish
