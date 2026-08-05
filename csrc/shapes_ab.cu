// How much of the megakernel's time is *not* matmul.
//
//   /usr/local/cuda-13.2/bin/nvcc -arch=sm_89 -O3 -std=c++17 -I. -Itests \
//       shapes_ab.cu -o shapes_ab && ./shapes_ab
//
// docs/journal/2026-08-05 estimates the non-matmul share by comparing the whole
// kernel's 25.5 TFLOPS against `gemm_direct`'s 31.9 on the ff1 shape alone. That
// assumes every other matmul runs at ff1's rate, which is exactly the kind of
// assumption that turns into a plan. The four shapes differ:
//
//   qkv   N = 768,  K = 256     6/24 of the flops
//   out   N = 256,  K = 256     2/24
//   ff1   N = 1024, K = 256     8/24
//   ff2   N = 256,  K = 1024    8/24, and the kernel walks it in four k-slices
//
// Each is timed exactly as `encoder_kernel` issues it -- same warp decomposition,
// same n-tile offsets, same k-depth -- so the flop-weighted rate here is the ceiling
// the megakernel would hit if it were matmul and nothing else.
//
// ⚠️ Order-balanced across rounds. perf.md records a naive before/after producing a
// fake -4.5 % on this card, whose clocks drift 1.38-1.5 GHz under sustained load.
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

#include "tests/core.cuh"

using namespace brokefish;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_) { \
    printf("cuda error %s at %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

constexpr int NK = Dm / 32;          // 8 k-groups of 32 for the K = 256 shapes

// One layer's four matmuls, issued the way the kernel issues them. `which` selects
// so each shape can be timed alone; -1 runs all four, which is the per-layer total.
template <int WHICH>
__global__ void k_shape(const half* __restrict__ w, float* sink, int reps) {
    __shared__ half A[T * AROW];
    __shared__ half Hbuf[T * HROW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < T * AROW; i += blockDim.x)
        A[i] = __float2half(0.01f * float(i % 7) - 0.03f);
    for (int i = threadIdx.x; i < T * HROW; i += blockDim.x)
        Hbuf[i] = __float2half(0.01f * float(i % 5));
    __syncthreads();
    float s = 0.0f;
    for (int r = 0; r < reps; ++r) {
        if (threadIdx.x == 0) A[r % (T * AROW)] = __float2half(0.01f * float(r % 7));
        __syncthreads();
        uint32_t acc[2][4][2] = {};
        if (WHICH == 0 || WHICH < 0) {          // qkv: three [32,256]x[256,256] slabs
            gemm_direct<NK, NK>(acc, A, AROW, w, warp * 4, 0, lane);
            gemm_direct<NK, NK>(acc, A, AROW, w, 32 + warp * 4, 0, lane);
            gemm_direct<NK, NK>(acc, A, AROW, w, 64 + warp * 4, 0, lane);
        }
        if (WHICH == 1 || WHICH < 0)            // out: one [32,256]x[256,256]
            gemm_direct<NK, NK>(acc, A, AROW, w, warp * 4, 0, lane);
        if (WHICH == 2 || WHICH < 0)            // ff1: four passes of four n-tiles
            for (int c = 0; c < 4; ++c)
                gemm_direct<NK, NK>(acc, A, AROW, w, c * 32 + warp * 4, 0, lane);
        if (WHICH == 3 || WHICH < 0)            // ff2: K = 1024 in four k-slices
            for (int c = 0; c < 4; ++c)
                gemm_direct<8, DFF / 32>(acc, Hbuf, HROW, w, warp * 4, c * 8, lane);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                s += __half2float(__low2half(h2(acc[m][q][0])));
        __syncthreads();
    }
    if (threadIdx.x == 1024) sink[0] = s;
}

template <typename F>
float time_ms(F launch) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    launch();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    const int blocks = p.multiProcessorCount * 2, reps = 2000;
    printf("%s, %d SMs, %d CTAs x %d threads, %d reps\n\n", p.name,
           p.multiProcessorCount, blocks, THREADS, reps);

    const size_t nel = (size_t)DFF * Dm;                 // the widest slab needed
    std::vector<half> wh(nel);
    for (size_t i = 0; i < nel; ++i) wh[i] = __float2half(0.02f * float(i % 11) - 0.1f);
    half* dw = nullptr; float* sink = nullptr;
    CHECK(cudaMalloc(&dw, nel * sizeof(half)));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemcpy(dw, wh.data(), nel * sizeof(half), cudaMemcpyHostToDevice));

    // Flops per rep per CTA, 2*M*N*K, M = 32 tokens.
    const double f_qkv = 2.0 * T * (3 * Dm) * Dm;
    const double f_out = 2.0 * T * Dm * Dm;
    const double f_ff1 = 2.0 * T * DFF * Dm;
    const double f_ff2 = 2.0 * T * Dm * DFF;
    const double f_all = f_qkv + f_out + f_ff1 + f_ff2;

    auto r0 = [&] { k_shape<0><<<blocks, THREADS>>>(dw, sink, reps); };
    auto r1 = [&] { k_shape<1><<<blocks, THREADS>>>(dw, sink, reps); };
    auto r2 = [&] { k_shape<2><<<blocks, THREADS>>>(dw, sink, reps); };
    auto r3 = [&] { k_shape<3><<<blocks, THREADS>>>(dw, sink, reps); };
    auto ra = [&] { k_shape<-1><<<blocks, THREADS>>>(dw, sink, reps); };
    r0(); r1(); r2(); r3(); ra(); CHECK(cudaDeviceSynchronize());   // reach clocks

    double a[5] = {};
    const int rounds = 6;
    for (int i = 0; i < rounds; ++i) {
        if (i % 2 == 0) {
            a[0] += time_ms(r0); a[1] += time_ms(r1); a[2] += time_ms(r2);
            a[3] += time_ms(r3); a[4] += time_ms(ra);
        } else {   // reversed, so a monotone clock drift cancels
            a[4] += time_ms(ra); a[3] += time_ms(r3); a[2] += time_ms(r2);
            a[1] += time_ms(r1); a[0] += time_ms(r0);
        }
    }
    CHECK(cudaGetLastError());
    const char* nm[5] = {"qkv  [32,256]x[768,256]", "out  [32,256]x[256,256]",
                         "ff1  [32,256]x[1024,256]", "ff2  [32,1024]x[256,1024]",
                         "one whole layer"};
    const double fl[5] = {f_qkv, f_out, f_ff1, f_ff2, f_all};
    const double share[5] = {6.0/24, 2.0/24, 8.0/24, 8.0/24, 1.0};
    for (int i = 0; i < 5; ++i) {
        const double ms = a[i] / rounds;
        const double tf = fl[i] * reps * blocks / (ms * 1e-3) / 1e12;
        if (i == 4) printf("\n");
        printf("  %-26s %7.2f ms   %5.1f TFLOPS", nm[i], ms, tf);
        if (i < 4) printf("   (%4.1f %% of layer flops)", 100 * share[i]);
        printf("\n");
    }

    // What the megakernel would run at if it were these matmuls and nothing else.
    const double ms_all = a[4] / rounds;
    const double tf_all = f_all * reps * blocks / (ms_all * 1e-3) / 1e12;
    printf("\n  matmul-only ceiling for the encoder body: %.1f TFLOPS\n", tf_all);
    printf("  the whole kernel measured 25.5 TFLOPS (62.0k evals/s, bench_model)\n");
    printf("  => matmul is %.1f %% of kernel time, everything else %.1f %%\n",
           100 * 25.5 / tf_all, 100 * (1.0 - 25.5 / tf_all));
    cudaFree(dw); cudaFree(sink);
    return 0;
}
