// Device-level tests for the int8 primitives the FFN kernel is built on.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tint8.cu -o tint8 && ./tint8
//
// The int8 twin of `tfp8.cu`, and it pins the same three things, for the same reason:
// every one of them still issues, still produces finite numbers, and is wrong by an
// amount that looks like quantisation error.
//
//   * the **byte order** of `cvt_int8x4`, against a host convert;
//   * the **fragment layout** of `mma.m16n8k32` for s8 and u8 operands -- including
//     the s32 D fragment's row split, which `gemm_s8_row`'s descale assumes and which
//     was reasoned about rather than measured when it was written;
//   * `quantise_row_int8` and `gemm_s8_row` end to end against a host product.
//
// ⚠️ Everything here is computed in **exact integer arithmetic** on the host. int8
// operands and an s32 accumulator have no rounding at all once the inputs are
// integers, so any mismatch is a layout or a scale error and never a tolerance
// question. That is a sharper test than `tfp8.cu` can write, and it is a property of
// the format rather than of the test.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_fp16.h>

#include "int8_gemm.cuh"
#include "mma.cuh"

using namespace brokefish;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_) { \
    printf("cuda error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    return 1; } } while (0)

namespace {

// Round-to-nearest-even then saturate, which is what `cvt.rni.sat` does.
int host_s8(float x) {
    float r = std::nearbyint(x);
    return (int)fminf(fmaxf(r, -128.0f), 127.0f);
}
int host_u8(float x) {
    float r = std::nearbyint(x);
    return (int)fminf(fmaxf(r, 0.0f), 255.0f);
}

// -- 1. the convert ------------------------------------------------------- //

__global__ void cvt_kernel(const float* in, uint32_t* out, int unsigned_) {
    if (unsigned_) *out = cvt_int8x4<true>(in[0], in[1], in[2], in[3]);
    else           *out = cvt_int8x4<false>(in[0], in[1], in[2], in[3]);
}

int test_cvt() {
    // Deliberately includes both saturation directions and a .5 tie, because
    // round-to-nearest-*even* and round-half-away differ there and the host and the
    // device have to agree on which.
    const std::vector<std::vector<float>> cases = {
        {0.0f, 1.0f, -1.0f, 2.5f}, {3.5f, -3.5f, 126.6f, -126.6f},
        {200.0f, -200.0f, 0.4f, -0.4f}, {127.0f, -127.0f, 254.7f, 0.5f}};
    float* d_in = nullptr; uint32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_in, 4 * sizeof(float)));
    CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));
    int bad = 0;
    for (int u = 0; u < 2; ++u) {
        for (const auto& c : cases) {
            CHECK(cudaMemcpy(d_in, c.data(), 4 * sizeof(float), cudaMemcpyHostToDevice));
            cvt_kernel<<<1, 1>>>(d_in, d_out, u);
            CHECK(cudaGetLastError());
            uint32_t got = 0;
            CHECK(cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost));
            for (int j = 0; j < 4; ++j) {
                const int want = u ? host_u8(c[j]) : host_s8(c[j]);
                const int g = u ? (int)((got >> (8 * j)) & 0xff)
                                : (int)(int8_t)((got >> (8 * j)) & 0xff);
                if (g != want && bad++ < 6)
                    printf("  cvt %s %g: device %d, host %d\n",
                           u ? "u8" : "s8", c[j], g, want);
            }
        }
    }
    printf("  %-42s %s\n", "cvt_int8x4 byte order and rounding", bad ? "FAIL" : "OK");
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    return bad;
}

// -- 2. the mma fragment layout ------------------------------------------- //

__global__ void mma_kernel(const uint32_t* af, const uint32_t* bf, int32_t* out,
                           int unsigned_a) {
    const int lane = threadIdx.x;
    uint32_t a[4], b[2];
    int32_t d[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = af[lane * 4 + i];
#pragma unroll
    for (int i = 0; i < 2; ++i) b[i] = bf[lane * 2 + i];
    if (unsigned_a) mma_u8s8(d, a, b); else mma_s8(d, a, b);
    // ⚠️ This mapping is the claim under test, and `gemm_s8_row`'s descale depends on
    // it: d[0],d[1] are row g and take the row-g activation scale, d[2],d[3] are row
    // g+8 and take the other one. Swapping them would apply each row's scale to the
    // other row -- finite, plausible, and wrong by roughly a quantisation step.
    const int g = lane / 4, t = lane % 4;
    out[g * 8 + 2 * t] = d[0];
    out[g * 8 + 2 * t + 1] = d[1];
    out[(g + 8) * 8 + 2 * t] = d[2];
    out[(g + 8) * 8 + 2 * t + 1] = d[3];
}

int test_mma_layout(int unsigned_a) {
    constexpr int M = 16, N = 8, K = 32;
    // Asymmetric in m, n and k, so a transposed or half-swapped fragment cannot
    // survive. Values stay small enough that the host product is obviously exact.
    std::vector<int> A(M * K), B(K * N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A[m * K + k] = unsigned_a ? (m * 5 + k * 3) % 251
                                      : ((m * 5 + k * 3) % 201) - 100;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) B[k * N + n] = ((k * 7 + n * 2) % 201) - 100;

    std::vector<uint32_t> af(32 * 4, 0), bf(32 * 2, 0);
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) {
            const int row = (i & 1) ? g + 8 : g;
            const int k0 = (i >> 1) ? 4 * t + 16 : 4 * t;
            uint32_t packed = 0;
            for (int j = 0; j < 4; ++j)
                packed |= uint32_t(A[row * K + k0 + j] & 0xff) << (8 * j);
            af[lane * 4 + i] = packed;
        }
        for (int i = 0; i < 2; ++i) {
            const int k0 = i ? 4 * t + 16 : 4 * t;
            uint32_t packed = 0;
            for (int j = 0; j < 4; ++j)
                packed |= uint32_t(B[(k0 + j) * N + g] & 0xff) << (8 * j);
            bf[lane * 2 + i] = packed;
        }
    }

    uint32_t *d_a = nullptr, *d_b = nullptr;
    int32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_a, af.size() * 4));
    CHECK(cudaMalloc(&d_b, bf.size() * 4));
    CHECK(cudaMalloc(&d_out, M * N * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_a, af.data(), af.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, bf.data(), bf.size() * 4, cudaMemcpyHostToDevice));
    mma_kernel<<<1, 32>>>(d_a, d_b, d_out, unsigned_a);
    CHECK(cudaGetLastError());
    std::vector<int32_t> got(M * N);
    CHECK(cudaMemcpy(got.data(), d_out, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            long want = 0;
            for (int k = 0; k < K; ++k) want += (long)A[m * K + k] * B[k * N + n];
            if (got[m * N + n] != want && bad++ < 5)
                printf("  D[%2d][%d]: device %d, host %ld\n", m, n, got[m * N + n], want);
        }
    printf("  %-42s %s  (16x8x32, exact integers)\n",
           unsigned_a ? "mma.m16n8k32.u8.s8 layout" : "mma.m16n8k32.s8.s8 layout",
           bad ? "FAIL" : "OK");
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_out));
    return bad;
}

// -- 3. the quantiser, in place, with the aliasing the kernel relies on ---- //

constexpr int QM = 32, QK = int8q::kRowK;
constexpr int QPITCH = QK * (int)sizeof(__half);          // the fp16 row pitch, reused

__global__ void quant_kernel(__half* buf, float* scale, int unsigned_) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(buf);
    if (unsigned_)
        int8q::quantise_row_int8<QK, true>(dst, QPITCH, scale, 1, buf, QK, QM,
                                           threadIdx.x & 31);
    else
        int8q::quantise_row_int8<QK, false>(dst, QPITCH, scale, 1, buf, QK, QM,
                                            threadIdx.x & 31);
}

int test_quantise(int unsigned_) {
    // ⚠️ `dst` aliases `src`, exactly as it does in the kernel, because that aliasing
    // is the thing three documented hazards exist to make safe. Testing it against a
    // separate output buffer is what let the e4m3 version ship a missing `__syncwarp`
    // that produced NaN on 116 boards of 128.
    std::vector<float> ref(QM * QK);
    std::vector<__half> h(QM * QK);
    for (int r = 0; r < QM; ++r)
        for (int k = 0; k < QK; ++k) {
            float v = (float)(((r * 31 + k * 17) % 401) - 200) * (0.001f * (r + 1));
            if (unsigned_) v = fabsf(v);
            ref[r * QK + k] = __half2float(__float2half(v));   // what the device sees
            h[r * QK + k] = __float2half(v);
        }
    __half* d_buf = nullptr; float* d_scale = nullptr;
    CHECK(cudaMalloc(&d_buf, h.size() * sizeof(__half)));
    CHECK(cudaMalloc(&d_scale, QM * sizeof(float)));
    CHECK(cudaMemcpy(d_buf, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    quant_kernel<<<1, 32>>>(d_buf, d_scale, unsigned_);
    CHECK(cudaGetLastError());
    std::vector<uint8_t> bytes(QM * QPITCH);
    std::vector<float> scale(QM);
    CHECK(cudaMemcpy(bytes.data(), d_buf, bytes.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scale.data(), d_scale, QM * 4, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int r = 0; r < QM; ++r) {
        float amax = 0.0f;
        for (int k = 0; k < QK; ++k)
            amax = fmaxf(amax, unsigned_ ? ref[r * QK + k] : fabsf(ref[r * QK + k]));
        const float qmax = unsigned_ ? int8q::kQMaxU : int8q::kQMaxS;
        const float want_sc = amax > 0.0f ? amax / qmax : 1.0f;
        if (scale[r] != want_sc && bad++ < 5)
            printf("  row %d scale: device %g, host %g\n", r, scale[r], want_sc);
        const float inv = 1.0f / want_sc;
        for (int k = 0; k < QK; ++k) {
            const int want = unsigned_ ? host_u8(ref[r * QK + k] * inv)
                                       : host_s8(ref[r * QK + k] * inv);
            const uint8_t raw = bytes[(size_t)r * QPITCH + k];
            const int g = unsigned_ ? (int)raw : (int)(int8_t)raw;
            if (g != want && bad++ < 8)
                printf("  q[%d][%d]: device %d, host %d\n", r, k, g, want);
        }
    }
    printf("  %-42s %s  (in place, %s)\n", "quantise_row_int8 vs host",
           bad ? "FAIL" : "OK", unsigned_ ? "u8" : "s8");
    CHECK(cudaFree(d_buf)); CHECK(cudaFree(d_scale));
    return bad;
}

// -- 4. gemm_s8_row end to end, including the descale --------------------- //
//
// ⚠️ This test exists because fault injection found the hole it fills. On 2026-08-14
// two deliberate mutations of `gemm_s8_row` -- swapping the `sa_lo`/`sa_hi` row split,
// and running the signed mma on the unsigned operand -- were **both missed** by the
// three tests above and caught only by a prior-space comparison in Python. The
// primitives were pinned and the function that composes them was not.

constexpr int GM = 32, GK = int8q::kRowK, GN = 32;
constexpr int GK32 = GK / 32, GNK32 = GK / 32;

__global__ void gemm_kernel(const uint8_t* a, const float* as, const uint2* w,
                            const float* ws, __half* out, int unsigned_a) {
    const int lane = threadIdx.x & 31;
    uint32_t acc[2][4][2] = {};
    if (unsigned_a)
        int8q::gemm_s8_row<GK32, GNK32, false, true>(acc, a, GK, as, 1, w, ws, 0, 0, lane);
    else
        int8q::gemm_s8_row<GK32, GNK32, false, false>(acc, a, GK, as, 1, w, ws, 0, 0, lane);
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const __half2 d0 = h2(acc[m][p][0]), d1 = h2(acc[m][p][1]);
            const int c = p * 8 + 2 * t;
            out[(16 * m + g) * GN + c] = __low2half(d0);
            out[(16 * m + g) * GN + c + 1] = __high2half(d0);
            out[(16 * m + g + 8) * GN + c] = __low2half(d1);
            out[(16 * m + g + 8) * GN + c + 1] = __high2half(d1);
        }
}

int test_gemm_row(int unsigned_a) {
    std::vector<int> A(GM * GK), W(GN * GK);
    std::vector<float> as(GM), ws(1);
    for (int m = 0; m < GM; ++m) {
        for (int k = 0; k < GK; ++k)
            A[m * GK + k] = unsigned_a ? (m * 13 + k * 7) % 256
                                       : ((m * 13 + k * 7) % 255) - 127;
        as[m] = 0.001f * (float)(m + 1);
    }
    for (int n = 0; n < GN; ++n)
        for (int k = 0; k < GK; ++k) W[n * GK + k] = ((n * 11 + k * 5) % 255) - 127;
    ws[0] = 0.002f;

    // The packed B order, derived here from the layout documented in encoder.cu rather
    // than by re-running `pack_b_fp8` -- an independent transcription is the only kind
    // that can disagree.
    std::vector<uint32_t> packed(GN / 8 * GNK32 * 32 * 2, 0);
    for (int n8 = 0; n8 < GN / 8; ++n8)
        for (int k32 = 0; k32 < GNK32; ++k32)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane / 4, t = lane % 4;
                for (int h = 0; h < 2; ++h) {
                    uint32_t v = 0;
                    for (int j = 0; j < 4; ++j)
                        v |= uint32_t(W[(n8 * 8 + g) * GK + k32 * 32 + h * 16 + t * 4 + j]
                                      & 0xff) << (8 * j);
                    packed[((size_t)(n8 * GNK32 + k32) * 32 + lane) * 2 + h] = v;
                }
            }
    std::vector<uint8_t> abytes(GM * GK);
    for (int i = 0; i < GM * GK; ++i) abytes[i] = (uint8_t)(A[i] & 0xff);

    uint8_t* d_a = nullptr; float *d_as = nullptr, *d_ws = nullptr;
    uint32_t* d_w = nullptr; __half* d_out = nullptr;
    CHECK(cudaMalloc(&d_a, abytes.size()));
    CHECK(cudaMalloc(&d_as, GM * 4));
    CHECK(cudaMalloc(&d_w, packed.size() * 4));
    CHECK(cudaMalloc(&d_ws, 4));
    CHECK(cudaMalloc(&d_out, GM * GN * sizeof(__half)));
    CHECK(cudaMemcpy(d_a, abytes.data(), abytes.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_as, as.data(), GM * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, packed.data(), packed.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws, ws.data(), 4, cudaMemcpyHostToDevice));
    gemm_kernel<<<1, 32>>>(d_a, d_as, reinterpret_cast<const uint2*>(d_w), d_ws,
                           d_out, unsigned_a);
    CHECK(cudaGetLastError());
    std::vector<__half> got(GM * GN);
    CHECK(cudaMemcpy(got.data(), d_out, GM * GN * sizeof(__half), cudaMemcpyDeviceToHost));

    int bad = 0;
    float worst = 0.0f;
    for (int m = 0; m < GM; ++m)
        for (int n = 0; n < GN; ++n) {
            long dot = 0;                       // exact
            for (int k = 0; k < GK; ++k) dot += (long)A[m * GK + k] * W[n * GK + k];
            const float want = (float)dot * as[m] * ws[0];
            const float g = __half2float(got[m * GN + n]);
            // The only rounding in the whole path is the final store to fp16, so the
            // bar is one fp16 ulp at this magnitude and nothing else.
            const float tol = fmaxf(fabsf(want), 1.0f) * 1e-3f;
            worst = fmaxf(worst, fabsf(g - want));
            if (fabsf(g - want) > tol && bad++ < 5)
                printf("  out[%2d][%2d]: device %g, host %g\n", m, n, g, want);
        }
    printf("  %-42s %s  (%s A, worst |delta| %g)\n", "gemm_s8_row vs exact host product",
           bad ? "FAIL" : "OK", unsigned_a ? "u8" : "s8", worst);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_as)); CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_ws)); CHECK(cudaFree(d_out));
    return bad;
}

}  // namespace

int main() {
    printf("int8 device tests\n");
    int bad = 0;
    bad += test_cvt();
    bad += test_mma_layout(0);
    bad += test_mma_layout(1);
    bad += test_quantise(0);
    bad += test_quantise(1);
    bad += test_gemm_row(0);
    bad += test_gemm_row(1);
    printf("%s\n", bad ? "FAILED" : "all int8 device tests passed");
    return bad ? 1 : 0;
}
