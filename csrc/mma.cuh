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

namespace brokefish {

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

__device__ __forceinline__ __half2 h2(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t u32(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }

}  // namespace brokefish
