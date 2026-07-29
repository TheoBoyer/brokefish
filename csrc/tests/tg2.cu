// Does gemm_tile compute A[32][32] * Wt[32][32]^T with the fragment layouts we
// think it does? One warp, one tile, compared against a host reference.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include "mma.cuh"
#ifndef ASTRIDE
#define ASTRIDE 40
#endif
using namespace brokefish;

constexpr int KTILE = 32, PAD = 8, WROW = KTILE + PAD, AROW = ASTRIDE;

__device__ __forceinline__ uint32_t a_frag_addr(const half* b, int r0, int c0, int st, int l) {
    return smem_addr(b + (r0 + (l & 7) + ((l & 8) ? 8 : 0)) * st + c0 + ((l & 16) ? 8 : 0));
}
__device__ __forceinline__ uint32_t b_frag_addr(const half* b, int n0, int k0, int st, int l) {
    return smem_addr(b + (n0 + (l & 7) + ((l & 16) ? 8 : 0)) * st + k0 + ((l & 8) ? 8 : 0));
}

__global__ void k_test(const half* A, const half* W, half* out) {
    __shared__ half sa[32 * AROW], sw[32 * WROW];
    int l = threadIdx.x;
    for (int i = l; i < 32 * 32; i += 32) sa[(i / 32) * AROW + i % 32] = A[i];
    for (int i = l; i < 32 * 32; i += 32) sw[(i / 32) * WROW + i % 32] = W[i];
    __syncwarp();

    uint32_t acc[2][4][2] = {};
    uint32_t af[2][2][4];
    for (int m = 0; m < 2; ++m)
        for (int k = 0; k < 2; ++k) ldmatrix_x4(af[m][k], a_frag_addr(sa, m * 16, k * 16, AROW, l));
    uint32_t bf[4][2][2];
    for (int p = 0; p < 2; ++p)
        for (int k = 0; k < 2; ++k) {
            uint32_t raw[4];
            ldmatrix_x4(raw, b_frag_addr(sw, p * 16, k * 16, WROW, l));
            bf[2 * p][k][0] = raw[0]; bf[2 * p][k][1] = raw[1];
            bf[2 * p + 1][k][0] = raw[2]; bf[2 * p + 1][k][1] = raw[3];
        }
    for (int k = 0; k < 2; ++k)
        for (int m = 0; m < 2; ++m)
            for (int n = 0; n < 4; ++n) mma_f16(acc[m][n], af[m][k], bf[n][k]);

    int g = l >> 2, t = l & 3;
    for (int m = 0; m < 2; ++m)
        for (int n = 0; n < 4; ++n) {
            *(uint32_t*)(out + (16 * m + g) * 32 + n * 8 + t * 2) = acc[m][n][0];
            *(uint32_t*)(out + (16 * m + g + 8) * 32 + n * 8 + t * 2) = acc[m][n][1];
        }
}

int main() {
    half *A, *W, *O;
    cudaMallocManaged(&A, 1024 * 2); cudaMallocManaged(&W, 1024 * 2); cudaMallocManaged(&O, 1024 * 2);
    for (int i = 0; i < 1024; ++i) {
        A[i] = __float2half(((i * 37) % 13 - 6) * 0.25f);
        W[i] = __float2half(((i * 17) % 11 - 5) * 0.25f);
    }
    k_test<<<1, 32>>>(A, W, O);
    cudaDeviceSynchronize();
    double worst = 0;
    for (int m = 0; m < 32; ++m)
        for (int n = 0; n < 32; ++n) {
            float ref = 0;
            for (int k = 0; k < 32; ++k) ref += __half2float(A[m * 32 + k]) * __half2float(W[n * 32 + k]);
            worst = fmax(worst, fabs(ref - __half2float(O[m * 32 + n])));
        }
    printf("worst |gemm - reference| = %g\n", worst);
    printf("O[0][0]=%g O[0][1]=%g O[1][0]=%g\n", __half2float(O[0]), __half2float(O[1]), __half2float(O[32]));
    return 0;
}
