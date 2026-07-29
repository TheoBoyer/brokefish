// gemm_full at the real shape: A[32][256] in SMEM, W[256][256] staged 32 k at a
// time, 8 warps each taking 32 of the 256 output columns.
#include <cstdio>
#include "core.cuh"
using namespace brokefish;
#ifndef KDIM
#define KDIM 32
#endif

__global__ void k_full(const half* A, const half* W, half* out) {
    extern __shared__ half smem[];
    half* a = smem;                 // [32][AROW]
    half* wt = a + T * AROW;        // [256][WROW]
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int i = tid; i < T * Dm; i += THREADS) a[a_idx(i / Dm, i % Dm)] = A[i];
    __syncthreads();
    uint32_t acc[2][4][2];
    zero_frags(acc);
    gemm_full(acc, a, AROW, wt, W, KDIM, Dm, tid, warp, lane);
    __syncthreads();
    store_frags(a, AROW, acc, warp * 32, lane);
    __syncthreads();
    for (int i = tid; i < T * Dm; i += THREADS) out[i] = a[a_idx(i / Dm, i % Dm)];
}

int main() {
    half *A, *W, *O;
    cudaMallocManaged(&A, T * Dm * 2);
    cudaMallocManaged(&W, Dm * Dm * 2);
    cudaMallocManaged(&O, T * Dm * 2);
    for (int i = 0; i < T * Dm; ++i) A[i] = __float2half(((i * 37) % 13 - 6) * 0.125f);
    for (int i = 0; i < Dm * Dm; ++i) W[i] = __float2half(((i * 17) % 11 - 5) * 0.125f);
    size_t sm = (T * AROW + Dm * WROW) * sizeof(half);
    k_full<<<1, THREADS, sm>>>(A, W, O);
    printf("launch: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    double worst = 0; int wm = -1, wn = -1;
    for (int m = 0; m < T; ++m)
        for (int n = 0; n < Dm; ++n) {
            float ref = 0;
            for (int k = 0; k < KDIM; ++k) ref += __half2float(A[m * Dm + k]) * __half2float(W[n * Dm + k]);
            double d = fabs(ref - __half2float(O[m * Dm + n]));
            if (d > worst) { worst = d; wm = m; wn = n; }
        }
    printf("worst |gemm_full - ref| = %g at (%d,%d)\n", worst, wm, wn);
    return 0;
}
