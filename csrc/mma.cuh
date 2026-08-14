// Tensor-core primitives for sm89: ldmatrix, mma.sync, and the fragment layouts
// that let the encoder chain them without moving data.
//
// Everything here is m16n8k16 fp16. The three fragment shapes, with
//   g = lane / 4   (0..7)      t = lane % 4   (0..3)
//
//   A (16x16, row-major)   a[0] = A[g  ][2t], A[g  ][2t+1]
//                          a[1] = A[g+8][2t], A[g+8][2t+1]
//                          a[2] = A[g  ][2t+8], A[g  ][2t+9]
//                          a[3] = A[g+8][2t+8], A[g+8][2t+9]
//
//   B (16x8, k-major)      b[0] = B[2t  ][g], B[2t+1][g]
//                          b[1] = B[2t+8][g], B[2t+9][g]
//
//   D (16x8, fp16 acc)     d[0] = D[g  ][2t], D[g  ][2t+1]
//                          d[1] = D[g+8][2t], D[g+8][2t+1]
//
// Two consequences the encoder lives on:
//
//   * D and A agree. The D fragments of n-tiles j and j+1 are exactly the
//     a[0..1] and a[2..3] of one A fragment, so a GEMM result feeds the next
//     mma as the A operand with a register move and nothing else. That is what
//     carries S -> P -> O and O -> W_o without touching memory.
//   * D and B agree along the other axis. For S = Q K^T the B operand is
//     B[k][n] = K[n][k], and d[0] of K's n-tile at dims 0..7 is exactly b[0].
//     So K needs no transpose either. V does, and gets one via ldmatrix.trans.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace brokefish {

__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Four 8x8 fp16 matrices. Lanes 0-7 address the rows of the first, 8-15 the
// second, and so on, so one instruction fills a whole A fragment.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Same, transposed inside each 8x8. Turns a [row][col] tile into the fragment
// that wants [col][row], which is how V reaches the P*V matmul.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

// D += A * B, fp16 accumulate. Half the issue cost of the fp32-accumulate form
// on GeForce Ada (35.5 against 18.0 TFLOPS measured), which is the whole reason
// this kernel exists in the shape it does. docs/perf.md has the error budget.
__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The fp32-accumulate form, for the acc_dtype="fp32" build.
__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// e4m3, for the FFN. docs/journal/2026-08-04-fp8-encoder.md is the why:
// `mma.m16n8k32.f16.e4m3.e4m3.f16` measures **71.4 TFLOPS** against `mma_f16`'s 35.6
// on this card, and one of these consumes k = 32, so it replaces *two* `mma_f16`.
//
// The fragment layout is the m16n8k32 8-bit one, and it is **verified by
// csrc/tests/tfp8.cu against a host reference** rather than transcribed from the
// manual. With g = lane / 4 and t = lane % 4, each thread holding four contiguous k:
//
//   A (16x32, row-major)   a[0] = A[g  ][4t .. 4t+3]
//                          a[1] = A[g+8][4t .. 4t+3]
//                          a[2] = A[g  ][4t+16 .. 4t+19]
//                          a[3] = A[g+8][4t+16 .. 4t+19]
//
//   B (32x8, k-major)      b[0] = B[4t .. 4t+3][g]
//                          b[1] = B[4t+16 .. 4t+19][g]
//
//   D (16x8, fp16 acc)     unchanged from m16n8k16 -- d[0] = D[g][2t], D[g][2t+1]
//                                                     d[1] = D[g+8][2t], D[g+8][2t+1]
//
// ⚠️ **D and A no longer agree.** The fp16 path chains a GEMM result straight into
// the next mma because D's registers *are* an A fragment; at 8 bits an A fragment is
// four registers of sixteen values and a D fragment is two registers of four, so the
// chain has to go through a quantisation step. That is not a loss here: the scheme in
// brokefish/nn/quant.py requires a per-tile amax and a convert between the two GEMMs
// anyway, so the step that breaks the identity is one we were paying for regardless.

__device__ __forceinline__ void mma_e4m3(uint32_t (&d)[2], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four halves to four packed e4m3 bytes, low half first.
//
// ⚠️ The byte order inside `cvt.rn.satfinite.e4m3x2.f16x2` is asserted by nothing in
// the PTX text that is easy to misread, so tfp8.cu checks it against
// `__nv_cvt_float_to_fp8`. `.satfinite` is not optional: e4m3 has **no infinity**, so
// an unsaturated convert of an out-of-range value yields NaN, and one NaN in one tile
// poisons a whole row of logits.
__device__ __forceinline__ uint32_t cvt_e4m3x4(__half2 lo, __half2 hi) {
    uint32_t a, b;
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }"
        : "=r"(a) : "r"(u32(lo)));
    asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t; }"
        : "=r"(b) : "r"(u32(hi)));
    return (a & 0xffffu) | (b << 16);
}

// ---------------------------------------------------------------------------
// int8. Same shape as the e4m3 mma above -- A is four registers of sixteen bytes,
// B is two of eight -- and therefore the **same packed B layout**, which is why
// `pack_b_fp8` is reused verbatim for int8 weights: the m16n8k32 fragment map is a
// property of the element *width*, not of how the bytes are interpreted.
//
// ⚠️ The accumulator is `s32` and there is no narrower form. `f16`, `s16` and `f32`
// accumulators all fail to assemble with s8 operands, while e4m3 has both `f16` and
// `f32` (probed, 2026-08-14). That is arithmetic rather than an omission: fp16
// accumulation is affordable for e4m3 only because `kQMax = 8` discards range we do
// not use, and an integer's range *is* its precision. The cost is 4 accumulator
// registers per tile against the packed-fp16 path's 2, and it is the whole reason the
// int8 kernel runs one CTA per SM.
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4],
                                       const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A unsigned, B signed. The FFN hidden is post-ReLU and therefore provably
// non-negative, so its sign bit is dead weight: u8 gives 256 levels where s8 gives
// 127. A float cannot make this trade -- its sign bit buys no mantissa -- so this is
// an int8-only gain. ⚠️ Measured effect is inside noise (flip 1.93 % -> 1.64 %); it is
// here because it is free, not because it was shown to help.
__device__ __forceinline__ void mma_u8s8(int32_t (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Four scaled floats to four packed bytes, low byte first -- the int8 twin of
// `cvt_e4m3x4`. `cvt.rni.sat` clamps rather than wrapping, which matters: a value a
// hair past 127 from a rounding division would otherwise become -128 and change sign.
// ⚠️ Unlike e4m3 there is no NaN to make here, because there is no NaN encoding.
// ⚠️ Two values per instruction, via `cvt.pack`, and that matters more than it looks.
// The e4m3 twin converts two at a time (`cvt.rn.satfinite.e4m3x2.f16x2`) and this one
// used to convert **one**, then assemble the word by hand with shifts and ors --
// ~10 instructions against e4m3's 3. That asymmetry is why quantisation measured
// **10.3 %** of the int8 kernel against fp8's 6.8 %. This is 4 `cvt.rni` + 2 packs.
//
// ⚠️ **`srcB` lands in byte 0 and `srcA` in byte 1 -- reversed from the operand
// order**, and nothing in the instruction's name says so. Measured on the card, and
// `tint8.cu` pins it, exactly as `tfp8.cu` pins the e4m3 convert's byte order for the
// same reason: a swapped pair still assembles, still produces bytes in range, and is
// wrong by something that reads as quantisation error.
//
// ⚠️ `__float2int_rn` is unsaturated where `cvt.rni.sat.s8.f32` was; the saturation
// now happens inside the pack, which is where 300 -> 255 and -200 -> -128 come from.
// The inputs cannot be out of range by construction (`x * inv` is bounded by `q_max`
// because `inv` is built from that row's own amax), so the only behaviour that changed
// is on a NaN, which the fp32 scaling upstream already precludes.
template <bool UNSIGNED>
__device__ __forceinline__ uint32_t cvt_int8x4(float a, float b, float c, float d) {
    const int ia = __float2int_rn(a), ib = __float2int_rn(b);
    const int ic = __float2int_rn(c), id = __float2int_rn(d);
    uint32_t lo, out;
    if constexpr (UNSIGNED) {
        asm("cvt.pack.sat.u8.s32.b32 %0, %2, %1, 0;" : "=r"(lo) : "r"(ic), "r"(id));
        asm("cvt.pack.sat.u8.s32.b32 %0, %2, %1, %3;"
            : "=r"(out) : "r"(ia), "r"(ib), "r"(lo));
    } else {
        asm("cvt.pack.sat.s8.s32.b32 %0, %2, %1, 0;" : "=r"(lo) : "r"(ic), "r"(id));
        asm("cvt.pack.sat.s8.s32.b32 %0, %2, %1, %3;"
            : "=r"(out) : "r"(ia), "r"(ib), "r"(lo));
    }
    return out;
}

// Two scaled halves to two packed bytes -- the tail of the above, for the register-side
// quantiser, which has a `__half2` in hand and wants a 16-bit store.
template <bool UNSIGNED>
__device__ __forceinline__ uint32_t cvt_int8x2(float a, float b) {
    const int ia = __float2int_rn(a), ib = __float2int_rn(b);
    uint32_t out;
    if constexpr (UNSIGNED)
        asm("cvt.pack.sat.u8.s32.b32 %0, %2, %1, 0;" : "=r"(out) : "r"(ia), "r"(ib));
    else
        asm("cvt.pack.sat.s8.s32.b32 %0, %2, %1, 0;" : "=r"(out) : "r"(ia), "r"(ib));
    return out;
}

// ---------------------------------------------------------------------------
// Fragment plumbing

// Two D fragments, covering n-tiles at column offsets 0..7 and 8..15, become
// one A fragment for k = 0..15. No shuffle: the register files already agree.
__device__ __forceinline__ void d_pair_to_a(uint32_t (&a)[4], const uint32_t (&lo)[2],
                                            const uint32_t (&hi)[2]) {
    a[0] = lo[0];
    a[1] = lo[1];
    a[2] = hi[0];
    a[3] = hi[1];
}

// Half of an A fragment, for a k of 8 rather than 16. Used nowhere yet; kept
// because the 32-wide attention tiles are one k-step and it documents the split.
__device__ __forceinline__ void d_to_b(uint32_t (&b)[2], const uint32_t (&lo)[2],
                                       const uint32_t (&hi)[2]) {
    b[0] = lo[0];
    b[1] = hi[0];
}


}  // namespace brokefish
