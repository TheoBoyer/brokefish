// Does gemm_direct read the host's packed weights the way pack_b writes them?
//
// This test owns the reference implementation of the packing, written straight
// from the fragment definition rather than copied from cuda_impl.py, so a
// mismatch between kernel and Python shows up here rather than as plausible
// wrong numbers eight layers downstream.
#include <cstdio>
#include <cstdlib>
#include "core.cuh"
using namespace brokefish;

#ifndef KDIM
#define KDIM 256
#endif
#ifndef NDIM
#define NDIM 256
#endif

// packed[n8][k32][lane][j], j = k16*4 + half8*2 + pair
static void pack_b_host(const float* w, half* out, int N, int K) {
    for (int n8 = 0; n8 < N / 8; ++n8)
        for (int k32 = 0; k32 < K / 32; ++k32)
            for (int lane = 0; lane < 32; ++lane) {
                int g = lane / 4, t = lane % 4;
                for (int j = 0; j < 8; ++j) {
                    int k16 = j / 4, h8 = (j / 2) % 2, pair = j % 2;
                    int row = n8 * 8 + g;
                    int col = k32 * 32 + k16 * 16 + h8 * 8 + t * 2 + pair;
                    out[((n8 * (K / 32) + k32) * 32 + lane) * 8 + j] =
                        __float2half(w[row * K + col]);
                }
            }
}

__global__ void k_direct(const half* A, const half* Wp, half* out) {
    extern __shared__ half smem[];
    half* a = smem;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < T * KDIM; i += THREADS) a[a_idx(i / KDIM, i % KDIM)] = A[i];
    __syncthreads();
    uint32_t acc[2][4][2];
    zero_frags(acc);
    // The packed row pitch is a template argument now: it makes p * NK32 * 32 a
    // compile-time constant, so the four B loads are one register plus an LDG
    // immediate offset instead of 7 IMAD + 4 LEA per k-step.
    gemm_direct<KDIM / 32, KDIM / 32>(acc, a, AROW, Wp, warp * 4, 0, lane);
    __syncthreads();
    store_frags(a, AROW, acc, warp * 32, lane);
    __syncthreads();
    for (int i = tid; i < T * NDIM; i += THREADS) out[i] = a[a_idx(i / NDIM, i % NDIM)];
}

int main() {
    float *Ah = (float*)malloc(T * KDIM * 4), *Wh = (float*)malloc((size_t)NDIM * KDIM * 4);
    for (int i = 0; i < T * KDIM; ++i) Ah[i] = ((i * 37) % 13 - 6) * 0.125f;
    for (int i = 0; i < NDIM * KDIM; ++i) Wh[i] = ((i * 17) % 11 - 5) * 0.125f;

    half *A, *Wp, *O;
    cudaMallocManaged(&A, T * KDIM * 2);
    cudaMallocManaged(&Wp, (size_t)NDIM * KDIM * 2);
    cudaMallocManaged(&O, T * NDIM * 2);
    for (int i = 0; i < T * KDIM; ++i) A[i] = __float2half(Ah[i]);
    pack_b_host(Wh, Wp, NDIM, KDIM);

    size_t sm = T * AROW * sizeof(half);
    cudaFuncSetAttribute(k_direct, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    k_direct<<<1, THREADS, sm>>>(A, Wp, O);
    printf("launch: %s / %s\n", cudaGetErrorString(cudaGetLastError()),
           cudaGetErrorString(cudaDeviceSynchronize()));

    double worst = 0; int wm = -1, wn = -1;
    for (int m = 0; m < T; ++m)
        for (int n = 0; n < NDIM; ++n) {
            float ref = 0;
            for (int k = 0; k < KDIM; ++k) ref += Ah[m * KDIM + k] * Wh[n * KDIM + k];
            double d = fabs(ref - __half2float(O[m * NDIM + n]));
            if (d > worst) { worst = d; wm = m; wn = n; }
        }
    printf("worst |gemm_direct - ref| = %g at (%d,%d)   [N=%d K=%d]\n", worst, wm, wn, NDIM, KDIM);
    return worst > 0.0;
}
