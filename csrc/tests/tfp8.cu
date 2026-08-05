// Device-level tests for the e4m3 primitives the FFN kernel is built on.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tfp8.cu -o tfp8 && ./tfp8
//
// ⚠️ Requires CUDA >= 12.1: `mma` with `.e4m3` entered PTX ISA 8.1. `/usr/bin/nvcc`
// on this machine is 12.0 and rejects it with "Unexpected instruction types
// specified for 'mma'"; `brokefish/nn/_build.py` already resolves 13.2 for the torch
// extension, so only a hand-typed nvcc hits this.
//
// Two things are pinned here, and everything in the fp8 FFN rests on both:
//
//   * the **byte order** of `cvt.rn.satfinite.e4m3x2.f16x2`, against NVIDIA's own
//     host converter, so the packer and the kernel agree on which half lands in
//     which byte;
//   * the **fragment layout** of `mma.m16n8k32` for 8-bit operands, against a host
//     matrix product. This is the one that cannot be reasoned about safely: a wrong
//     layout still issues, still produces finite numbers, and is wrong by an amount
//     that looks like quantisation error.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "fp8_gemm.cuh"
#include "mma.cuh"

using namespace brokefish;

#define CHECK(x) do { cudaError_t e_ = (x); if (e_) { \
    printf("cuda error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    return 1; } } while (0)

namespace {

uint8_t host_e4m3(float x) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
float host_from_e4m3(uint8_t b) {
    __half h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3);
    return __half2float(h);
}

// -- 1. the convert ------------------------------------------------------- //

__global__ void cvt_kernel(const __half* in, uint32_t* out) {
    const __half2 lo = __halves2half2(in[0], in[1]);
    const __half2 hi = __halves2half2(in[2], in[3]);
    out[0] = cvt_e4m3x4(lo, hi);
}

int test_cvt() {
    // Values chosen to be distinguishable in every byte, to span both signs, and to
    // include one that must saturate rather than become NaN.
    const float want[4] = {1.5f, -2.25f, 0.109375f, 1e4f};
    std::vector<__half> h(4);
    for (int i = 0; i < 4; ++i) h[i] = __float2half(want[i]);

    __half* d_in = nullptr;
    uint32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_in, 4 * sizeof(__half)));
    CHECK(cudaMalloc(&d_out, sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_in, h.data(), 4 * sizeof(__half), cudaMemcpyHostToDevice));
    cvt_kernel<<<1, 1>>>(d_in, d_out);
    CHECK(cudaGetLastError());
    uint32_t got = 0;
    CHECK(cudaMemcpy(&got, d_out, sizeof(got), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < 4; ++i) {
        const uint8_t g = (got >> (8 * i)) & 0xff;
        const uint8_t w = host_e4m3(__half2float(h[i]));
        if (g != w) {
            printf("  byte %d: device 0x%02x (%g), host 0x%02x (%g)\n", i, g,
                   host_from_e4m3(g), w, host_from_e4m3(w));
            ++bad;
        }
    }
    // The saturating case is the one that matters most: 1e4 is past e4m3's 448 and
    // must come back as 448, not as NaN.
    if (host_from_e4m3((got >> 24) & 0xff) != 448.0f) {
        printf("  1e4 did not saturate to 448: got %g\n",
               host_from_e4m3((got >> 24) & 0xff));
        ++bad;
    }
    printf("  %-38s %s  (byte order low-half-first, satfinite)\n",
           "cvt.e4m3x2.f16x2", bad ? "FAIL" : "OK");
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    return bad;
}

// -- 2. the fragment layout ------------------------------------------------ //

__global__ void mma_kernel(const uint32_t* af, const uint32_t* bf, __half* out) {
    const int lane = threadIdx.x;
    uint32_t a[4], b[2], d[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = af[lane * 4 + i];
#pragma unroll
    for (int i = 0; i < 2; ++i) b[i] = bf[lane * 2 + i];
    mma_e4m3(d, a, b);
    // D (16x8): d[0] holds row g, columns 2t and 2t+1; d[1] holds row g+8.
    const int g = lane / 4, t = lane % 4;
    const __half2 d0 = h2(d[0]), d1 = h2(d[1]);
    out[g * 8 + 2 * t] = __low2half(d0);
    out[g * 8 + 2 * t + 1] = __high2half(d0);
    out[(g + 8) * 8 + 2 * t] = __low2half(d1);
    out[(g + 8) * 8 + 2 * t + 1] = __high2half(d1);
}

int test_mma_layout() {
    constexpr int M = 16, N = 8, K = 32;
    // Integers 0..7 are exact in e4m3 (three mantissa bits carry four significant
    // binary digits), so the reference product is exact and any mismatch is layout
    // and not rounding. The pattern is deliberately asymmetric in k as well as in
    // m and n: a transposed or half-swapped fragment must not survive it.
    std::vector<float> A(M * K), B(K * N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) A[m * K + k] = float((m * 5 + k * 3) % 8);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) B[k * N + n] = float((k * 7 + n * 2) % 8) - 3.0f;

    // The layout under test, written once, here.
    std::vector<uint32_t> af(32 * 4, 0), bf(32 * 2, 0);
    for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        for (int i = 0; i < 4; ++i) {
            const int row = (i & 1) ? g + 8 : g;
            const int k0 = (i >> 1) ? 4 * t + 16 : 4 * t;
            uint32_t packed = 0;
            for (int j = 0; j < 4; ++j)
                packed |= uint32_t(host_e4m3(A[row * K + k0 + j])) << (8 * j);
            af[lane * 4 + i] = packed;
        }
        for (int i = 0; i < 2; ++i) {
            const int k0 = i ? 4 * t + 16 : 4 * t;
            uint32_t packed = 0;
            for (int j = 0; j < 4; ++j)
                packed |= uint32_t(host_e4m3(B[(k0 + j) * N + g])) << (8 * j);
            bf[lane * 2 + i] = packed;
        }
    }

    uint32_t *d_a = nullptr, *d_b = nullptr;
    __half* d_out = nullptr;
    CHECK(cudaMalloc(&d_a, af.size() * 4));
    CHECK(cudaMalloc(&d_b, bf.size() * 4));
    CHECK(cudaMalloc(&d_out, M * N * sizeof(__half)));
    CHECK(cudaMemcpy(d_a, af.data(), af.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, bf.data(), bf.size() * 4, cudaMemcpyHostToDevice));
    mma_kernel<<<1, 32>>>(d_a, d_b, d_out);
    CHECK(cudaGetLastError());
    std::vector<__half> got(M * N);
    CHECK(cudaMemcpy(got.data(), d_out, M * N * sizeof(__half), cudaMemcpyDeviceToHost));

    int bad = 0;
    float worst = 0.0f;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float want = 0.0f;
            for (int k = 0; k < K; ++k) want += A[m * K + k] * B[k * N + n];
            const float g = __half2float(got[m * N + n]);
            worst = fmaxf(worst, fabsf(g - want));
            if (g != want && bad++ < 5)
                printf("  D[%2d][%d]: device %g, host %g\n", m, n, g, want);
        }
    printf("  %-38s %s  (16x8x32, worst |delta| %g)\n", "mma.m16n8k32.e4m3 layout",
           bad ? "FAIL" : "OK", worst);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_out));
    return bad;
}

// -- 3. the activation quantiser ------------------------------------------ //

constexpr int QM = 32, QK = fp8::kRowK;

__global__ void quant_kernel(const __half* src, uint8_t* dst, float* scale, int pitch) {
    extern __shared__ uint8_t smem[];
    fp8::quantise_row<QK>(smem, pitch, scale, 1, src, QK, QM, threadIdx.x);
    __syncwarp();
    for (int i = threadIdx.x; i < QM * pitch; i += 32) dst[i] = smem[i];
}

// ⚠️ The case encoder.cu actually runs: `dst` and `src` are **the same memory**, the
// bytes landing over the front of the fp16 row that produced them. That is what makes
// the whole scheme cost no shared memory, and it is a different test: with separate
// buffers no lane can clobber another's source, so the read/write ordering inside
// `quantise_row` is never exercised. It was wrong, and this is what found it.
//
// The row is now 256 wide, so it is two write-tiles rather than one, and the ordering
// has a second half to get right: tile 1's writes land on bytes [128, 256) while its
// own reads are at bytes [512, 1024). That direction is safe; the reverse would not be,
// which is why `quantise_row` walks tiles in ascending order and says so.
__global__ void quant_inplace_kernel(const __half* src, uint8_t* dst, float* scale,
                                     int hpitch) {
    extern __shared__ __half hsmem[];
    uint8_t* bytes = reinterpret_cast<uint8_t*>(hsmem);
    for (int i = threadIdx.x; i < QM * hpitch; i += 32) hsmem[i] = src[i];
    __syncwarp();
    fp8::quantise_row<QK>(bytes, hpitch * (int)sizeof(__half), scale, 1,
                          hsmem, hpitch, QM, threadIdx.x);
    __syncwarp();
    for (int i = threadIdx.x; i < QM * hpitch * (int)sizeof(__half); i += 32)
        dst[i] = bytes[i];
}

int test_quantise() {
    // A different magnitude per row, so one scale shared between rows fails, and one
    // loud column per row so the amax never sits at a predictable lane.
    std::vector<float> a(QM * QK);
    for (int m = 0; m < QM; ++m)
        for (int k = 0; k < QK; ++k)
            a[m * QK + k] = (float((m * 13 + k * 7) % 17) - 8.0f)
                            * ldexpf(1.0f, m % 5) * ((k == (m * 3) % QK) ? 9.0f : 1.0f);
    std::vector<__half> h(QM * QK);
    for (size_t i = 0; i < a.size(); ++i) h[i] = __float2half(a[i]);

    const int pitch = fp8::a_pitch(QK);
    __half* d_src = nullptr; uint8_t* d_dst = nullptr; float* d_scale = nullptr;
    CHECK(cudaMalloc(&d_src, h.size() * sizeof(__half)));
    CHECK(cudaMalloc(&d_dst, QM * pitch));
    CHECK(cudaMalloc(&d_scale, QM * sizeof(float)));
    CHECK(cudaMemcpy(d_src, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    quant_kernel<<<1, 32, QM * pitch>>>(d_src, d_dst, d_scale, pitch);
    CHECK(cudaGetLastError());
    std::vector<uint8_t> bytes(QM * pitch);
    std::vector<float> scale(QM);
    CHECK(cudaMemcpy(bytes.data(), d_dst, bytes.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scale.data(), d_scale, QM * 4, cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int m = 0; m < QM && bad < 6; ++m) {
        float amax = 0.0f;
        for (int k = 0; k < QK; ++k) amax = fmaxf(amax, fabsf(__half2float(h[m * QK + k])));
        const float want_scale = amax > 0 ? amax / fp8::kQMax : 1.0f;
        if (fabsf(scale[m] - want_scale) > 1e-6f * fmaxf(want_scale, 1.0f)) {
            printf("  row %d: scale %g, want %g\n", m, scale[m], want_scale);
            ++bad;
            continue;
        }
        for (int k = 0; k < QK && bad < 6; ++k) {
            const uint8_t want = host_e4m3(__half2float(h[m * QK + k]) / want_scale);
            const uint8_t got = bytes[m * pitch + k];
            if (got != want) {
                printf("  [%d][%d]: 0x%02x (%g), want 0x%02x (%g)\n", m, k, got,
                       host_from_e4m3(got), want, host_from_e4m3(want));
                ++bad;
            }
        }
    }
    printf("  %-38s %s  (32 rows x 256, byte-exact)\n", "quantise_row",
           bad ? "FAIL" : "OK");

    // The same rows again, quantised over themselves.
    const int hpitch = QK;
    std::vector<uint8_t> ib(QM * hpitch * sizeof(__half));
    std::vector<float> isc(QM);
    uint8_t* d_ib = nullptr; float* d_isc = nullptr;
    CHECK(cudaMalloc(&d_ib, ib.size()));
    CHECK(cudaMalloc(&d_isc, QM * 4));
    quant_inplace_kernel<<<1, 32, QM * hpitch * sizeof(__half)>>>(d_src, d_ib, d_isc,
                                                                  hpitch);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(ib.data(), d_ib, ib.size(), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(isc.data(), d_isc, QM * 4, cudaMemcpyDeviceToHost));
    int ibad = 0;
    for (int m = 0; m < QM && ibad < 6; ++m)
        for (int k = 0; k < QK; ++k) {
            const uint8_t want = host_e4m3(__half2float(h[m * QK + k]) / isc[m]);
            const uint8_t got = ib[(size_t)m * hpitch * sizeof(__half) + k];
            if (got != want && ibad++ < 6)
                printf("  in place [%d][%d]: 0x%02x, want 0x%02x\n", m, k, got, want);
        }
    printf("  %-38s %s  (dst and src are the same memory)\n", "quantise_row, in place",
           ibad ? "FAIL" : "OK");
    CHECK(cudaFree(d_ib)); CHECK(cudaFree(d_isc));
    CHECK(cudaFree(d_src)); CHECK(cudaFree(d_dst)); CHECK(cudaFree(d_scale));
    return bad + ibad;
}

// -- 4. the tile ----------------------------------------------------------- //

constexpr int GM = 32, GK = fp8::kRowK, GN = 256;
constexpr int NK32 = GK / 32, NB = GN / fp8::kBlockN;

__global__ void gemm_kernel(const uint8_t* a, const float* as, const uint2* w,
                            const float* ws, __half* out, int pitch) {
    const int lane = threadIdx.x;
    for (int n8_0 = 0; n8_0 < GN / 8; n8_0 += 4) {
        uint32_t acc[2][4][2];
        fp8::gemm_fp8_row<NK32, NK32, /*ADD=*/false>(acc, a, pitch, as, 1, w, ws,
                                                     n8_0, 0, lane);
        const int g = lane / 4, t = lane % 4;
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int col = (n8_0 + p) * 8 + 2 * t;
                const __half2 d0 = h2(acc[m][p][0]), d1 = h2(acc[m][p][1]);
                out[(size_t)(m * 16 + g) * GN + col] = __low2half(d0);
                out[(size_t)(m * 16 + g) * GN + col + 1] = __high2half(d0);
                out[(size_t)(m * 16 + g + 8) * GN + col] = __low2half(d1);
                out[(size_t)(m * 16 + g + 8) * GN + col + 1] = __high2half(d1);
            }
    }
}

// `exact`: A and W take values in {0, 1}, so after scaling every quantised value is
// 0 or `kQMax`, every product is 0 or `kQMax^2`, and every partial sum is a multiple
// of `kQMax^2` well inside fp16's exactly-representable integers -- at q_max = 8 the
// largest is 256 * 64 = 16384, where fp16's spacing is 16 and 64 is a multiple of it.
// The comparison is then an **equality** whatever the hardware does internally, which
// is what pins the layout, the packing and the two scale axes.
//
// ⚠️ The realistic case cannot be an equality and the reason is a finding in itself:
// modelling the accumulator as `fp16(C + exact_32_products)` -- what quant.py assumes
// and what the PTX text suggests -- left *more* residual than summing in double, so
// the fp8 mma's internal reduction is neither of those. DeepSeek-V3 documents Hopper's
// fp8 tensor cores accumulating at an effective 14 bits; this is consistent with Ada
// doing something similar, and it is unmeasured here. The authority for the numerical
// question is brokefish/nn/validate.py in prior space, not this tolerance.
int test_gemm(bool exact) {
    // Host-side quantisation, the scheme of brokefish/nn/quant.py: **one** activation
    // scale per row and **one** weight scale per 128 output columns, both covering the
    // whole reduction. Magnitudes vary by row and by n-block, so a scale applied along
    // the wrong axis is caught; they deliberately also vary along k, which the scheme
    // now absorbs into the single scale rather than tracking.
    std::vector<float> A(GM * GK), W((size_t)GN * GK);
    for (int m = 0; m < GM; ++m)
        for (int k = 0; k < GK; ++k)
            A[m * GK + k] = exact ? float((m * 11 + k * 5) % 2)
                                  : (float((m * 11 + k * 5) % 13) - 6.0f)
                                    * ldexpf(1.0f, (m + k / 128) % 4);
    for (int n = 0; n < GN; ++n)
        for (int k = 0; k < GK; ++k)
            W[(size_t)n * GK + k] = exact ? float((n * 7 + k * 3) % 2)
                                          : (float((n * 7 + k * 3) % 11) - 5.0f)
                                            * ldexpf(1.0f, (n / 128) % 3);

    std::vector<float> as(GM), ws(NB);
    std::vector<uint8_t> aq(GM * GK), wq((size_t)GN * GK);
    for (int m = 0; m < GM; ++m) {
        float amax = 0;
        for (int k = 0; k < GK; ++k) amax = fmaxf(amax, fabsf(A[m * GK + k]));
        const float sc = amax > 0 ? amax / fp8::kQMax : 1.0f;
        as[m] = sc;
        for (int k = 0; k < GK; ++k) aq[m * GK + k] = host_e4m3(A[m * GK + k] / sc);
    }
    for (int b = 0; b < NB; ++b) {
        float amax = 0;
        for (int n = 0; n < 128; ++n)
            for (int k = 0; k < GK; ++k)
                amax = fmaxf(amax, fabsf(W[(size_t)(b * 128 + n) * GK + k]));
        const float sc = amax > 0 ? amax / fp8::kQMax : 1.0f;
        ws[b] = sc;
        for (int n = 0; n < 128; ++n)
            for (int k = 0; k < GK; ++k)
                wq[(size_t)(b * 128 + n) * GK + k] =
                    host_e4m3(W[(size_t)(b * 128 + n) * GK + k] / sc);
    }

    // Pack: (n8, k32, lane) -> {B[4t..4t+3][g], B[4t+16..4t+19][g]}, B[k][n] = W[n][k].
    std::vector<uint32_t> packed((size_t)(GN / 8) * NK32 * 32 * 2);
    for (int n8 = 0; n8 < GN / 8; ++n8)
        for (int k32 = 0; k32 < NK32; ++k32)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane / 4, t = lane % 4;
                for (int half = 0; half < 2; ++half) {
                    uint32_t v = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int k = k32 * 32 + (half ? 4 * t + 16 : 4 * t) + j;
                        v |= uint32_t(wq[(size_t)(n8 * 8 + g) * GK + k]) << (8 * j);
                    }
                    packed[((size_t)(n8 * NK32 + k32) * 32 + lane) * 2 + half] = v;
                }
            }

    const int pitch = fp8::a_pitch(GK);
    std::vector<uint8_t> a_smem((size_t)GM * pitch, 0);
    for (int m = 0; m < GM; ++m)
        for (int k = 0; k < GK; ++k) a_smem[(size_t)m * pitch + k] = aq[m * GK + k];

    uint8_t* d_a = nullptr; float *d_as = nullptr, *d_ws = nullptr;
    __half* d_out = nullptr;
    uint32_t* d_w = nullptr;
    CHECK(cudaMalloc(&d_a, a_smem.size()));
    CHECK(cudaMalloc(&d_as, as.size() * 4));
    CHECK(cudaMalloc(&d_w, packed.size() * 4));
    CHECK(cudaMalloc(&d_ws, ws.size() * 4));
    CHECK(cudaMalloc(&d_out, (size_t)GM * GN * sizeof(__half)));
    CHECK(cudaMemcpy(d_a, a_smem.data(), a_smem.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_as, as.data(), as.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, packed.data(), packed.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));
    gemm_kernel<<<1, 32>>>(d_a, d_as, reinterpret_cast<const uint2*>(d_w), d_ws,
                           d_out, pitch);
    CHECK(cudaGetLastError());
    std::vector<__half> got((size_t)GM * GN);
    CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__half),
                     cudaMemcpyDeviceToHost));

    // Reference: the same quantised bytes, with **both roundings modelled** -- the
    // fp16 mma accumulator every 32 products, and the fp16 the descaled result is
    // written back into. The second one is new: the output of a GEMM is now an fp16
    // fragment rather than an fp32 array, because that is what removed the running
    // accumulator and its 960 B of spill.
    int bad = 0;
    double worst = 0.0, scale = 0.0, scale_hint = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        scale_hint = fmax(scale_hint, fabs((double)__half2float(got[i])));
    for (int m = 0; m < GM; ++m)
        for (int n = 0; n < GN; ++n) {
            float acc16 = 0.0f;
            for (int c = 0; c < GK / 32; ++c) {
                double part = 0.0;
                for (int k = 0; k < 32; ++k) {
                    const int kk = c * 32 + k;
                    part += (double)host_from_e4m3(aq[m * GK + kk])
                            * host_from_e4m3(wq[(size_t)n * GK + kk]);
                }
                acc16 = __half2float(__float2half(acc16 + (float)part));
            }
            // The kernel's own order: (acc * sa) * sw in fp32, then one fp16 store.
            const float want = __half2float(__float2half(acc16 * as[m] * ws[n / 128]));
            const double d = fabs(__half2float(got[(size_t)m * GN + n]) - want);
            worst = fmax(worst, d);
            scale = fmax(scale, fabs((double)want));
            const double tol = exact ? 0.0 : 3e-3 * scale_hint;
            if (d > tol && bad++ < 5)
                printf("  out[%d][%d]: %g, want %g\n", m, n,
                       __half2float(got[(size_t)m * GN + n]), want);
        }
    printf("  %-38s %s  (32x256x256, worst %.3g on %.3g = %.1e rel)\n",
           exact ? "gemm_fp8_row, exactly representable"
                 : "gemm_fp8_row, realistic values",
           bad ? "FAIL" : "OK", worst, scale, worst / fmax(scale, 1.0));
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_as)); CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_ws)); CHECK(cudaFree(d_out));
    return bad;
}

// -- 5. the ff2 call shape: a k-slice of a wider matrix -------------------- //
//
// ⚠️ This is the case the first four could not reach. `encoder.cu` calls the FFN's
// second matmul as `gemm_direct<8, DFF/32>(ff, hid, HROW, w_ff2, warp*4, c*8, lane)`:
// eight k-groups of a matrix whose row pitch is thirty-two, starting at `k32_0 = 8c`,
// with an activation buffer holding only that chunk. Every other test used
// `k32_0 = 0`, where a global and a local k index are the same number -- and two bugs
// lived in exactly that difference.
constexpr int SM_ = 32, SK = 1024, SN = 256, SCHUNK = fp8::kRowK;
constexpr int SNK32 = SK / 32, SNB = SN / fp8::kBlockN;

__global__ void slice_kernel(const uint8_t* a, const float* as, const uint2* w,
                             const float* ws, __half* out, int pitch) {
    const int lane = threadIdx.x;
    for (int n8_0 = 0; n8_0 < SN / 8; n8_0 += 4) {
        uint32_t acc[2][4][2] = {};
        for (int c = 0; c < SK / SCHUNK; ++c)
            // The activation chunk is local and carries its own per-row scale; the
            // weight is addressed globally from `k32_0`. ADD folds the four descaled
            // chunks together in fp16, which is what encoder.cu does with `ff`.
            fp8::gemm_fp8_row<SCHUNK / 32, SNK32, /*ADD=*/true>(
                acc, a + (size_t)c * SM_ * pitch, pitch, as + c * SM_, 1,
                w, ws, n8_0, c * (SCHUNK / 32), lane);
        const int g = lane / 4, t = lane % 4;
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int col = (n8_0 + p) * 8 + 2 * t;
                const __half2 d0 = h2(acc[m][p][0]), d1 = h2(acc[m][p][1]);
                out[(size_t)(m * 16 + g) * SN + col] = __low2half(d0);
                out[(size_t)(m * 16 + g) * SN + col + 1] = __high2half(d0);
                out[(size_t)(m * 16 + g + 8) * SN + col] = __low2half(d1);
                out[(size_t)(m * 16 + g + 8) * SN + col + 1] = __high2half(d1);
            }
    }
}

int test_gemm_slice() {
    // Binary inputs, so every mma partial is exact and the only rounding left is the
    // one this test exists to pin: the fp16 the running total is kept in between
    // chunks. A per-chunk magnitude step makes the four scales differ, so a kernel
    // that quietly shared one across chunks fails here.
    std::vector<float> A(SM_ * SK), W((size_t)SN * SK);
    for (int m = 0; m < SM_; ++m)
        for (int k = 0; k < SK; ++k)
            A[m * SK + k] = float((m * 11 + k * 5) % 2) * ldexpf(1.0f, k / SCHUNK);
    for (int n = 0; n < SN; ++n)
        for (int k = 0; k < SK; ++k) W[(size_t)n * SK + k] = float((n * 7 + k * 3) % 2);

    const int NC = SK / SCHUNK;
    std::vector<float> as(NC * SM_), ws(SNB);
    std::vector<uint8_t> aq(SM_ * SK), wq((size_t)SN * SK);
    for (int c = 0; c < NC; ++c)
        for (int m = 0; m < SM_; ++m) {
            float amax = 0;
            for (int k = 0; k < SCHUNK; ++k)
                amax = fmaxf(amax, fabsf(A[m * SK + c * SCHUNK + k]));
            const float sc = amax > 0 ? amax / fp8::kQMax : 1.0f;
            as[c * SM_ + m] = sc;
            for (int k = 0; k < SCHUNK; ++k)
                aq[m * SK + c * SCHUNK + k] =
                    host_e4m3(A[m * SK + c * SCHUNK + k] / sc);
        }
    for (int b = 0; b < SNB; ++b) {
        float amax = 0;
        for (int n = 0; n < 128; ++n)
            for (int k = 0; k < SK; ++k)
                amax = fmaxf(amax, fabsf(W[(size_t)(b * 128 + n) * SK + k]));
        const float sc = amax > 0 ? amax / fp8::kQMax : 1.0f;
        ws[b] = sc;
        for (int n = 0; n < 128; ++n)
            for (int k = 0; k < SK; ++k)
                wq[(size_t)(b * 128 + n) * SK + k] =
                    host_e4m3(W[(size_t)(b * 128 + n) * SK + k] / sc);
    }

    std::vector<uint32_t> packed((size_t)(SN / 8) * SNK32 * 32 * 2);
    for (int n8 = 0; n8 < SN / 8; ++n8)
        for (int k32 = 0; k32 < SNK32; ++k32)
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane / 4, t = lane % 4;
                for (int half = 0; half < 2; ++half) {
                    uint32_t v = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int k = k32 * 32 + (half ? 4 * t + 16 : 4 * t) + j;
                        v |= uint32_t(wq[(size_t)(n8 * 8 + g) * SK + k]) << (8 * j);
                    }
                    packed[((size_t)(n8 * SNK32 + k32) * 32 + lane) * 2 + half] = v;
                }
            }

    // The activation buffer is laid out chunk by chunk, as `hid` is refilled.
    const int pitch = fp8::a_pitch(SCHUNK);
    std::vector<uint8_t> a_smem((size_t)NC * SM_ * pitch, 0);
    for (int c = 0; c < NC; ++c)
        for (int m = 0; m < SM_; ++m)
            for (int k = 0; k < SCHUNK; ++k)
                a_smem[((size_t)c * SM_ + m) * pitch + k] = aq[m * SK + c * SCHUNK + k];

    uint8_t* d_a = nullptr; float *d_as = nullptr, *d_ws = nullptr;
    __half* d_out = nullptr;
    uint32_t* d_w = nullptr;
    CHECK(cudaMalloc(&d_a, a_smem.size()));
    CHECK(cudaMalloc(&d_as, as.size() * 4));
    CHECK(cudaMalloc(&d_w, packed.size() * 4));
    CHECK(cudaMalloc(&d_ws, ws.size() * 4));
    CHECK(cudaMalloc(&d_out, (size_t)SM_ * SN * sizeof(__half)));
    CHECK(cudaMemcpy(d_a, a_smem.data(), a_smem.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_as, as.data(), as.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, packed.data(), packed.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));
    slice_kernel<<<1, 32>>>(d_a, d_as, reinterpret_cast<const uint2*>(d_w), d_ws,
                            d_out, pitch);
    CHECK(cudaGetLastError());
    std::vector<__half> got((size_t)SM_ * SN);
    CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__half),
                     cudaMemcpyDeviceToHost));

    int bad = 0;
    double worst = 0.0;
    for (int m = 0; m < SM_; ++m)
        for (int n = 0; n < SN; ++n) {
            float acc = 0.0f;                     // the fp16 running total, modelled
            for (int c = 0; c < NC; ++c) {
                double part = 0.0;
                for (int k = 0; k < SCHUNK; ++k) {
                    const int kk = c * SCHUNK + k;
                    part += (double)host_from_e4m3(aq[m * SK + kk])
                            * host_from_e4m3(wq[(size_t)n * SK + kk]);
                }
                const float o = (float)part * as[c * SM_ + m] * ws[n / 128] + acc;
                acc = __half2float(__float2half(o));
            }
            const double d = fabs(__half2float(got[(size_t)m * SN + n]) - acc);
            worst = fmax(worst, d);
            if (d != 0.0 && bad++ < 5)
                printf("  out[%d][%d]: %g, want %g\n", m, n,
                       __half2float(got[(size_t)m * SN + n]), acc);
        }
    printf("  %-38s %s  (32x1024x256 in 4 chunks, worst %g)\n",
           "gemm_fp8_row, ff2 k-slice + ADD", bad ? "FAIL" : "OK", worst);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_as)); CHECK(cudaFree(d_w));
    CHECK(cudaFree(d_ws)); CHECK(cudaFree(d_out));
    return bad;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_cvt();
    failures += test_mma_layout();
    failures += test_quantise();
    failures += test_gemm(true);
    failures += test_gemm(false);
    failures += test_gemm_slice();
    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
