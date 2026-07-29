#include <cstdio>
#include "core.cuh"
using namespace brokefish;
__global__ void k_load(const half* W, half* out) {
    extern __shared__ half smem[];
    load_wtile(smem, W, 0, 0, Dm, threadIdx.x);
    __syncthreads();
    for (int i = threadIdx.x; i < Dm * KTILE; i += THREADS)
        out[i] = smem[w_idx(i / KTILE, i % KTILE)];
}
int main() {
    half *W, *O;
    cudaMallocManaged(&W, Dm * Dm * 2); cudaMallocManaged(&O, Dm * KTILE * 2);
    for (int i = 0; i < Dm * Dm; ++i) W[i] = __float2half((float)(i % 251));
    k_load<<<1, THREADS, Dm * WROW * 2>>>(W, O);
    printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    int bad = 0;
    for (int n = 0; n < Dm && bad < 5; ++n)
        for (int k = 0; k < KTILE; ++k)
            if (__half2float(O[n * KTILE + k]) != __half2float(W[n * Dm + k])) {
                printf("mismatch at row %d col %d: got %g want %g\n", n, k,
                       __half2float(O[n * KTILE + k]), __half2float(W[n * Dm + k]));
                if (++bad >= 5) break;
            }
    if (!bad) printf("load_wtile OK\n");
    return 0;
}
