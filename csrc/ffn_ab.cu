// gemm_direct against gemm_fp8 on the FFN's real shape, interleaved.
//
// ff1 is [32 tokens, 256] x [1024, 256]^T per board: 8 warps, four passes of four
// n-tiles, which is exactly encoder.cu's `for c in 0..3: gemm_direct(hacc, bufB, ...,
// c*32 + warp*4, 0)`. Same launch shape, weights resident in L2 -- the compute
// comparison the megakernel would see, not a microbenchmark of the issue rate.
//
// ⚠️ Order-balanced. perf.md records a naive before/after producing a fake -4.5 % on
// this card, because clocks drift 1.38-1.5 GHz under sustained load.
//
// ⚠️ The fp8 path allocates the **same** shared memory as the fp16 one: the quantised
// activations reuse the fp16 row pitch of 528 B in place. encoder.cu's SMEM budget has
// zero headroom -- 50,176 B is the largest block that still fits twice on an SM -- so
// a design needing a separate byte buffer would have cost the second CTA.
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "tests/core.cuh"
#include "fp8_gemm.cuh"

using namespace brokefish;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_) { \
    printf("cuda error %s at %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

constexpr int NK = Dm / 32;          // 8 k-groups
constexpr int APITCH = AROW * 2;     // 528 B, the fp16 row pitch reused for bytes

__global__ void k_fp16(const half* __restrict__ w, float* sink, int reps) {
    __shared__ half A[T * AROW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < T * AROW; i += blockDim.x)
        A[i] = __float2half(0.01f * float(i % 7) - 0.03f);
    __syncthreads();
    float s = 0.0f;
    for (int r = 0; r < reps; ++r) {
        if (threadIdx.x == 0) A[r % (T * AROW)] = __float2half(0.01f * float(r % 7));
        __syncthreads();
        uint32_t acc[2][4][2] = {};
        for (int c = 0; c < 4; ++c)
            gemm_direct<NK, NK>(acc, A, AROW, w, c * 32 + warp * 4, 0, lane);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                s += __half2float(__low2half(h2(acc[m][q][0])))
                   + __half2float(__high2half(h2(acc[m][q][1])));
        __syncthreads();
    }
    if (threadIdx.x == 1024) sink[0] = s;
}

__global__ void k_fp8(const uint2* __restrict__ w, const float* __restrict__ ws,
                      float* sink, int reps) {
    __shared__ uint8_t A[T * APITCH];
    __shared__ float as[T * (Dm / fp8::kTileK)];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < T * APITCH; i += blockDim.x) A[i] = uint8_t(i % 120);
    for (int i = threadIdx.x; i < T * (Dm / fp8::kTileK); i += blockDim.x) as[i] = 0.01f;
    __syncthreads();
    float s = 0.0f;
    for (int r = 0; r < reps; ++r) {
        if (threadIdx.x == 0) A[r % (T * APITCH)] = uint8_t(r % 120);
        __syncthreads();
        float acc[2][4][4] = {};
        for (int c = 0; c < 4; ++c)
            fp8::gemm_fp8<NK, NK>(acc, A, APITCH, as, Dm / fp8::kTileK, w, ws,
                                  Dm / fp8::kTileK, c * 32 + warp * 4, 0, lane);
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                s += acc[m][q][0] + acc[m][q][1] + acc[m][q][2] + acc[m][q][3];
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
    printf("%s, %d SMs, %d CTAs x %d threads, %d reps\n", p.name,
           p.multiProcessorCount, blocks, THREADS, reps);
    printf("smem: fp16 %zu B, fp8 %zu B (the fp8 activations reuse the fp16 pitch)\n\n",
           T * AROW * sizeof(half), T * APITCH + T * (Dm / fp8::kTileK) * sizeof(float));

    // ff1: [1024][256]. fp16 packed is 8 halves per lane per (n8, k32); fp8 is 8 bytes.
    const size_t nel = (size_t)DFF * Dm;
    std::vector<half> w16(nel);
    std::vector<uint32_t> w8(nel / 4);
    for (size_t i = 0; i < nel; ++i) w16[i] = __float2half(0.02f * float(i % 11) - 0.1f);
    for (size_t i = 0; i < w8.size(); ++i) w8[i] = 0x38383838u;
    std::vector<float> ws((DFF / fp8::kBlockN) * (Dm / fp8::kTileK), 0.01f);

    half* d16 = nullptr; uint32_t* d8 = nullptr; float *dws = nullptr, *sink = nullptr;
    CHECK(cudaMalloc(&d16, nel * sizeof(half)));
    CHECK(cudaMalloc(&d8, w8.size() * 4));
    CHECK(cudaMalloc(&dws, ws.size() * 4));
    CHECK(cudaMalloc(&sink, 4));
    CHECK(cudaMemcpy(d16, w16.data(), nel * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d8, w8.data(), w8.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));

    auto run16 = [&] { k_fp16<<<blocks, THREADS>>>(d16, sink, reps); };
    auto run8 = [&] { k_fp8<<<blocks, THREADS>>>(
        reinterpret_cast<const uint2*>(d8), dws, sink, reps); };
    run16(); run8(); CHECK(cudaDeviceSynchronize());   // warm up and reach clocks

    // 2 * 32 tokens * 1024 outputs * 256 k per rep per CTA.
    const double flop = 2.0 * T * DFF * Dm * (double)reps * blocks;
    double a16 = 0, a8 = 0;
    const int rounds = 6;
    for (int i = 0; i < rounds; ++i) {
        // Both orders, so a monotone clock drift cancels instead of favouring one.
        if (i % 2 == 0) { a16 += time_ms(run16); a8 += time_ms(run8); }
        else            { a8 += time_ms(run8);  a16 += time_ms(run16); }
    }
    CHECK(cudaGetLastError());
    const double t16 = a16 / rounds, t8 = a8 / rounds;
    printf("  %-28s %8.2f ms   %6.1f TFLOPS\n", "gemm_direct (fp16)", t16, flop / (t16 * 1e-3) / 1e12);
    printf("  %-28s %8.2f ms   %6.1f TFLOPS\n", "gemm_fp8    (e4m3)", t8, flop / (t8 * 1e-3) / 1e12);
    printf("\n  speedup on the ff1 shape: %.2fx\n", t16 / t8);
    return 0;
}
