// What fp8 is worth *after* paying for the scheme, not at the raw issue rate.
//
//   /usr/local/cuda-13.2/bin/nvcc -arch=sm_89 -O3 -std=c++17 fp8_loop.cu -o fp8_loop
//
// The 72.0 TFLOPS of `mma.m16n8k32.f16.e4m3.e4m3.f16` is a pure-mma number. The
// scheme in brokefish/nn/quant.py adds work per 128-element k-tile that a naive
// ceiling ignores, and it lands on the *issue slots* rather than on the tensor pipe,
// which is where it can hurt:
//
//   * the fp16 accumulator has to be promoted to fp32 and descaled once per k-tile
//     (`q_max <= 16` is forced, so the fp16 accumulator cannot run the whole K), and
//   * the activation tile's amax is a warp reduction that must finish before the
//     tile can be converted to e4m3 -- a serialisation point, not just an add.
//
// Four variants, all counting only the mma FLOPs as useful work, so the numbers are
// directly comparable to perf.md's roofline.
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kIters = 4096;
constexpr int kAcc = 8;
constexpr unsigned kAll = 0xffffffffu;

#define MMA_F16 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
#define MMA_FP8 "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 " \
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"

// (a) what ships today: fp16 operands, fp16 accumulate, k = 16.
__global__ void k_today(float* sink, int iters) {
    unsigned a[2] = {0x3c003c00u, 0x3c003c00u}, b = 0x3c003c00u;
    unsigned c[kAcc][2] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int i = 0; i < kAcc; ++i)
            asm volatile(MMA_F16 : "+r"(c[i][0]), "+r"(c[i][1])
                         : "r"(a[0]), "r"(a[1]), "r"(a[0]), "r"(a[1]), "r"(b), "r"(b));
    float s = 0;
#pragma unroll
    for (int i = 0; i < kAcc; ++i) s += (float)__low2half(*(__half2*)&c[i][0]);
    if (threadIdx.x == 1024) sink[0] = s;
}

// (b) the raw fp8 ceiling: four mma per 128-element tile, nothing else.
__global__ void k_fp8_raw(float* sink, int iters) {
    unsigned a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    unsigned b[2] = {0x38383838u, 0x38383838u};
    unsigned c[kAcc][2] = {};
    for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int i = 0; i < kAcc; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile(MMA_FP8 : "+r"(c[i][0]), "+r"(c[i][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
    float s = 0;
#pragma unroll
    for (int i = 0; i < kAcc; ++i) s += (float)__low2half(*(__half2*)&c[i][0]);
    if (threadIdx.x == 1024) sink[0] = s;
}

// (c) + the descale: promote the fp16 accumulator to fp32 once per k-tile and fold in
//     `sa * sw`, which is what `quant.py`'s tile loop does and what `q_max <= 16`
//     forces. Four cvt and four FFMA per four mma, per accumulator.
__global__ void k_fp8_descale(float* sink, int iters) {
    unsigned a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    unsigned b[2] = {0x38383838u, 0x38383838u};
    unsigned c[kAcc][2];
    float f[kAcc][4] = {};
    float scale = 1.0009765625f;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < kAcc; ++i) {
            c[i][0] = 0; c[i][1] = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile(MMA_FP8 : "+r"(c[i][0]), "+r"(c[i][1])
                             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                               "r"(b[0]), "r"(b[1]));
            const __half2 lo = *(__half2*)&c[i][0], hi = *(__half2*)&c[i][1];
            f[i][0] = fmaf((float)__low2half(lo), scale, f[i][0]);
            f[i][1] = fmaf((float)__high2half(lo), scale, f[i][1]);
            f[i][2] = fmaf((float)__low2half(hi), scale, f[i][2]);
            f[i][3] = fmaf((float)__high2half(hi), scale, f[i][3]);
        }
        scale = __fmaf_rn(scale, 1.0000001f, 0.0f);   // keep it live
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < kAcc; ++i) s += f[i][0] + f[i][1] + f[i][2] + f[i][3];
    if (threadIdx.x == 1024) sink[0] = s;
}

// (d) + the activation quantisation: a warp amax over the k-tile, the reciprocal, and
//     the fp16 -> e4m3 conversion of the tile. This is the part modded-nanogpt found
//     shows up in wall-clock, and it is a *serialisation* point: the reduction has to
//     finish before the tile can be converted.
__global__ void k_fp8_full(float* sink, int iters) {
    unsigned a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    unsigned b[2] = {0x38383838u, 0x38383838u};
    unsigned aq[4];
    unsigned c[kAcc][2];
    float f[kAcc][4] = {};
    __half2 raw = __floats2half2_rn(1.5f, -2.25f);
    for (int it = 0; it < iters; ++it) {
        // amax over the 128-element tile: eight halves per thread, then a full warp
        // reduction, then broadcast. Five shuffles is the log2(32) tree.
        float m = fabsf((float)__low2half(raw));
        m = fmaxf(m, fabsf((float)__high2half(raw)));
#pragma unroll
        for (int off = 16; off; off >>= 1) m = fmaxf(m, __shfl_xor_sync(kAll, m, off));
        const float inv = __frcp_rn(fmaxf(m, 1e-30f)) * 16.0f;   // q_max = 16
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            // fp16x2 -> e4m3x2, twice, packed: the cvt the kernel would issue.
            __half2 s0 = __hmul2(raw, __float2half2_rn(inv));
            unsigned packed;
            asm volatile("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; "
                         "cvt.u32.u16 %0, t; }" : "=r"(packed) : "r"(*(unsigned*)&s0));
            aq[j] = packed | (packed << 16);
        }
#pragma unroll
        for (int i = 0; i < kAcc; ++i) {
            c[i][0] = 0; c[i][1] = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile(MMA_FP8 : "+r"(c[i][0]), "+r"(c[i][1])
                             : "r"(aq[0]), "r"(aq[1]), "r"(aq[2]), "r"(aq[3]),
                               "r"(b[0]), "r"(b[1]));
            const __half2 lo = *(__half2*)&c[i][0], hi = *(__half2*)&c[i][1];
            const float sc = m;
            f[i][0] = fmaf((float)__low2half(lo), sc, f[i][0]);
            f[i][1] = fmaf((float)__high2half(lo), sc, f[i][1]);
            f[i][2] = fmaf((float)__low2half(hi), sc, f[i][2]);
            f[i][3] = fmaf((float)__high2half(hi), sc, f[i][3]);
        }
        raw = __hmul2(raw, __float2half2_rn(1.0009765625f));
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < kAcc; ++i) s += f[i][0] + f[i][1] + f[i][2] + f[i][3];
    if (threadIdx.x == 1024) sink[0] = s + (float)__low2half(raw) + (float)a[0];
}

template <typename F>
double rate(F k, int blocks, int threads, double flops_per_iter, float* sink) {
    k<<<blocks, threads>>>(sink, 64);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    k<<<blocks, threads>>>(sink, kIters);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    const double warps = (double)blocks * threads / 32.0;
    return warps * kIters * flops_per_iter / (ms * 1e-3) / 1e12;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    const int blocks = p.multiProcessorCount * 4, threads = 256;
    float* sink = nullptr;
    cudaMalloc(&sink, sizeof(float));
    printf("%s\n\n", p.name);

    const double f16 = 2.0 * 16 * 8 * 16 * kAcc;         // one mma per accumulator
    const double fp8 = 2.0 * 16 * 8 * 32 * 4 * kAcc;     // four per accumulator

    const double a = rate(k_today,       blocks, threads, f16, sink);
    const double b = rate(k_fp8_raw,     blocks, threads, fp8, sink);
    const double c = rate(k_fp8_descale, blocks, threads, fp8, sink);
    const double d = rate(k_fp8_full,    blocks, threads, fp8, sink);
    printf("  %-44s %6.1f TFLOPS   %5.2fx\n", "(a) fp16 mma, fp16 acc  [ships today]", a, 1.0);
    printf("  %-44s %6.1f TFLOPS   %5.2fx\n", "(b) fp8 mma, fp16 acc   [raw ceiling]", b, b / a);
    printf("  %-44s %6.1f TFLOPS   %5.2fx\n", "(c) + per-tile descale to fp32", c, c / a);
    printf("  %-44s %6.1f TFLOPS   %5.2fx\n", "(d) + activation amax and cvt to e4m3", d, d / a);
    cudaFree(sink);
    return 0;
}
