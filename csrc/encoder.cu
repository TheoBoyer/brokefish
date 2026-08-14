// Fused encoder forward: 8 layers, 32 tokens, d=256, in one launch.
//
// One CTA owns one board and carries it through the whole stack, so activations
// never reach HBM. Eight warps, and warp w owns head w and model dimensions
// [32w, 32w+32) of every output. That assignment is what makes attention
// entirely warp-local: no CTA barrier anywhere between Q and O.
//
// Why this shape. The Triton kernel it replaces reaches 42.6k evals/s and is
// bounded by 90 bar.sync per layer, because every weight tile travels
// global -> registers -> SMEM -> ldmatrix -> mma and each transit through shared
// memory costs a CTA barrier. Here the residual stream lives in SMEM and the
// weights are the only thing staged, which puts the count at 14 per layer.
//
// Layout facts this file depends on, all in mma.cuh: the mma D fragment agrees
// with the A fragment, so GEMM results chain into the next mma with a register
// move; and D agrees with B along the other axis, so K needs no transpose for
// Q*K^T. V does, and gets one from ldmatrix.trans.
//
// Accumulation is fp16 by default: fp32 accumulate is half rate on GeForce Ada
// (18.0 against 35.5 TFLOPS measured) and the error lands in the same band as
// torch's own fp16 forward. docs/perf.md carries the numbers and the overflow
// analysis; ACC32 restores the wide path.

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include "fp8_gemm.cuh"
#include "int8_gemm.cuh"
#include "mma.cuh"


namespace brokefish {

constexpr int T = 32;       // tokens per board
constexpr int Dm = 256;     // model width
constexpr int DFF = 1024;
constexpr int NH = 8;       // heads
constexpr int DH = Dm / NH; // 32
constexpr int NWARPS = 8;
constexpr int THREADS = NWARPS * 32;

constexpr int PAD = 8;              // 16 bytes: shifts each row by 4 banks so the
                                    // 8 rows an ldmatrix touches never collide
constexpr int AROW = Dm + PAD;      // 264 halves
constexpr int HCHUNK = 256;         // FFN hidden columns held at once
constexpr int HROW = AROW;          // same row pitch, so hid can alias vbuf

// Three activation buffers and nothing else -- no weight tile, because the
// weights never pass through shared memory. The total is a hard budget, not an
// outcome: 50,176 B is the largest block that still fits twice on an SM
// (measured with cudaOccupancyMaxActiveBlocksPerMultiprocessor, 128 B
// granularity, 100,352 B usable of the 102,400), and the second CTA is worth
// 5.4 %. bufA is the one buffer no ldmatrix reads, so it is the one that can
// drop its row padding -- which is exactly the 512 bytes that buys the fit.
constexpr int AROWA = Dm;               // bufA is never read by ldmatrix
constexpr int SM_A = T * AROWA;         // residual stream
constexpr int SM_B = T * AROW;          // normed input / attention output
constexpr int SM_S = T * AROW;          // scratch: V transpose, then FFN hidden
constexpr int SMEM_HALVES = SM_A + SM_B + SM_S;
static_assert(SMEM_HALVES * sizeof(half) <= 50176,
              "over 50,176 B the block stops fitting twice on an sm89 SM, which "
              "costs more than any use of the extra memory has been worth");

// §fp8. A parallel weight slab for the FFN only, in e4m3, plus its 128x128 block
// scales. docs/journal/2026-08-04-fp8-encoder.md measured 1.70x on the ff1 shape and
// 0.66 % max prior-space error for FFN-only quantisation; the other two matmuls stay
// fp16 because that is where three quarters of the error came from.
struct Fp8Off {                                  // bytes, per layer
    static constexpr int w_ff1 = 0;              // [DFF][D], packed B-fragment order
    static constexpr int w_ff2 = w_ff1 + DFF * Dm;
    static constexpr int stride = w_ff2 + Dm * DFF;
};
struct Fp8SOff {                                 // floats, per layer
    // One scale per 128 output columns, covering the **whole** reduction -- the fp16
    // accumulator runs the full 256-deep GEMM, so every k-group must share a scale.
    static constexpr int s_ff1 = 0;              // [DFF/128]
    static constexpr int s_ff2 = s_ff1 + DFF / 128;
    static constexpr int stride = s_ff2 + Dm / 128;
};

// The quantised activations are written **over** the fp16 row that produced them, so
// they cost no shared memory -- which they had to, because SMEM_HALVES is already at
// the static_assert. Reusing the fp16 row pitch is also better for banks than a packed
// 260 would be: 528 / 4 = 132 words, and 132 mod 32 = 4, so the eight lanes sharing a
// `t` land on eight distinct banks. The per-row scale goes in the row padding that
// AROW already carries.
namespace fp8cfg {
constexpr int kAPitch = AROW * (int)sizeof(half);            // 528 B
constexpr int kScaleOff = Dm * (int)sizeof(half);            // 512 B, into the padding
constexpr int kScaleStride = kAPitch / (int)sizeof(float);   // 132 floats
static_assert(Dm == fp8::kRowK, "one scale per row means one scale for the whole row");
static_assert(kAPitch - kScaleOff >= (int)sizeof(float),
              "the row padding cannot hold this row's scale");
static_assert(kAPitch % sizeof(float) == 0 && kScaleOff % sizeof(float) == 0,
              "the scales must land 4-byte aligned inside the row");
static_assert(HCHUNK == Dm, "the ff2 activation chunk reuses ff1's row geometry");
}  // namespace fp8cfg

// Weight buffer offsets, in halves, within one layer's slab.
struct Off {
    static constexpr int ln1_w = 0;
    static constexpr int ln1_b = ln1_w + Dm;
    static constexpr int w_qkv = ln1_b + Dm;             // [3D][D], torch native
    static constexpr int b_qkv = w_qkv + 3 * Dm * Dm;
    static constexpr int w_o = b_qkv + 3 * Dm;           // [D][D]
    static constexpr int b_o = w_o + Dm * Dm;
    static constexpr int ln2_w = b_o + Dm;
    static constexpr int ln2_b = ln2_w + Dm;
    static constexpr int w_ff1 = ln2_b + Dm;             // [DFF][D]
    static constexpr int b_ff1 = w_ff1 + DFF * Dm;
    static constexpr int w_ff2 = b_ff1 + DFF;            // [D][DFF]
    static constexpr int b_ff2 = w_ff2 + Dm * DFF;
    static constexpr int stride = b_ff2 + Dm;
};

// --------------------------------------------------------------------------
// B2: the embedding tables of spec 7.2 and the heads of spec 7.4.
//
// Five tables in one slab, in this order, each row Dm halves. The flattening of
// the two two-dimensional tables is normative -- type*2+special and
// color*2+stm -- because brokefish/nn/model.py computes the same arithmetic and
// a mismatch is a wrong number, not a crash.
constexpr int N_SQUARE = 64, N_TYPE_SPECIAL = 12, N_COLOR_TURN = 4;
constexpr int N_CLOCK = 101, N_REP = 3;

struct EmbOff {
    static constexpr int square       = 0;
    static constexpr int type_special = square + N_SQUARE * Dm;
    static constexpr int color_turn   = type_special + N_TYPE_SPECIAL * Dm;
    static constexpr int clock        = color_turn + N_COLOR_TURN * Dm;
    static constexpr int rep          = clock + N_CLOCK * Dm;
    static constexpr int stride       = rep + N_REP * Dm;
};

// The three heads are one packed matrix. Policy is 64 columns, exactly eight mma
// n-tiles, and gemm_direct emits four n-tiles per call -- so policy is two warps
// and the aux head is a third. Padding the aux head out to 32 columns (promo at
// 0-3, value at 4, then 27 zero columns) wastes 24 rows of zeros and one warp's
// mma, which is 0.1 % of an epilogue that is itself 0.3 % of the network. What it
// buys is gemm_direct untouched: that function is where the whole +11.8 % of
// perf.md row 9 lives, and templating its n-tile count to recover 0.2 % would
// spend the 12 spare registers the two hard budgets leave.
constexpr int NPOL = 64;                  // policy columns
constexpr int NAUX = 32;                  // aux tile, padded from 5
constexpr int NHEAD = NPOL + NAUX;        // 96 packed rows
constexpr int N_PROMO = 4;
constexpr int PROMO_COL = NPOL;           // where promo logit 0 lands in staging
constexpr int VALUE_COL = NPOL + 4;
// 104 halves is 52 words, so eight rows sit 20 banks apart and 20 is coprime
// enough with 32 that all eight land in distinct banks -- same test as AROW.
constexpr int HDROW = NHEAD + 8;
static_assert(T * HDROW <= SM_S, "the head staging block has to fit inside scratch");

struct TailOff {
    static constexpr int lnf_w  = 0;
    static constexpr int lnf_b  = Dm;
    static constexpr int w_head = 2 * Dm;      // packed [NHEAD][Dm]
    static constexpr int stride = w_head + NHEAD * Dm;
};

// --------------------------------------------------------------------------
// SMEM addressing. Rows are padded rather than XOR-swizzled: 264 halves is 528
// bytes, which is 4 banks off a multiple of 128, so the eight rows one
// ldmatrix reads land in eight different bank groups.

__device__ __forceinline__ int a_idx(int row, int col) { return row * AROW + col; }
__device__ __forceinline__ int ai_idx(int row, int col) { return row * AROWA + col; }
__device__ __forceinline__ int h_idx(int row, int col) { return row * HROW + col; }

// Address of the 16x16 A fragment at (row0, col0), for ldmatrix.x4: lanes 0-7
// give rows 0-7 of the k-low half, lanes 8-15 rows 8-15, lanes 16-23 rows 0-7
// of the k-high half, 24-31 rows 8-15.
__device__ __forceinline__ uint32_t a_frag_addr(const half* base, int row0, int col0,
                                                int stride, int lane) {
    int r = row0 + (lane & 7) + ((lane & 8) ? 8 : 0);
    int c = col0 + ((lane & 16) ? 8 : 0);
    return smem_addr(base + r * stride + c);
}

// --------------------------------------------------------------------------
// Weights, straight from global memory into mma B fragments.
//
// This is the change that matters. The staged version put every weight tile
// through global -> registers -> shared -> ldmatrix -> registers, which cost
// three passes over the LSU (one LDG, one STS, one LDSM) for one pass of
// arithmetic, plus two CTA barriers per k-tile to protect the shared buffer.
// At d=256 that came to ~202 barriers and 4.7 MB of shared-memory traffic per
// layer, against 1.58 MB of weights actually consumed.
//
// A B fragment is only 4 bytes per lane, and the lane -> element map is fixed
// and known at pack time. So the host pre-permutes each weight matrix into
// exactly that order and the kernel reads it with one coalesced 128-bit load:
// no shared memory, no barrier, no ldmatrix. The LSU sees one pass instead of
// three and the barrier disappears with the buffer it was protecting.
//
// Packed layout for a matrix [N][K], in halves:
//
//     packed[n8][k32][lane][8]      n8 = n/8, k32 = k/32
//
// where lane L (g = L/4, t = L%4) holds, for the 8 halves j = 0..7:
//
//     j = 0,1   W[8*n8 + g][32*k32      + 2t], +1     -> b[0] of k-step 2*k32
//     j = 2,3   W[8*n8 + g][32*k32 +  8 + 2t], +1     -> b[1] of k-step 2*k32
//     j = 4,5   W[8*n8 + g][32*k32 + 16 + 2t], +1     -> b[0] of k-step 2*k32+1
//     j = 6,7   W[8*n8 + g][32*k32 + 24 + 2t], +1     -> b[1] of k-step 2*k32+1
//
// One uint4 per lane is therefore two complete B fragments, and the 32 lanes of
// a warp read 512 contiguous bytes. `pack_b` in brokefish/nn/cuda_impl.py is
// the other half of this contract; changing one without the other is silent.

// acc[2][4] (32 tokens x 32 output columns) += A[32][k] * W[n0..n0+31][k]^T,
// contracting over k32_n groups of 32 starting at k32_0. n8_0 = n0/8, and nk32
// is the packed matrix's full K/32 (its row pitch), not the slice length.
// K32N is a template parameter rather than an argument so the k-loop unrolls:
// with a runtime trip count ptxas keeps it rolled, which serialises each
// iteration's B loads behind the previous iteration's mma instead of letting
// them overlap. Every call site knows its depth at compile time.
template <int K32N, int NK32>
__device__ __forceinline__ void gemm_direct(uint32_t acc[2][4][2], const half* a_base,
                                            int a_stride, const half* w, int n8_0,
                                            int k32_0, int lane) {
    // Base pointer for (p=0, j=0). NK32 (the packed matrix's row pitch in
    // k-groups) is a template parameter so p * NK32 * 32 is a compile-time
    // constant and the four loads become one register plus an LDG immediate
    // offset. As a runtime argument it cost 7 IMAD + 4 LEA per k-iteration.
    const uint4* wp = reinterpret_cast<const uint4*>(w) + ((size_t)n8_0 * NK32 + k32_0) * 32 + lane;
    constexpr int PS = NK32 * 32;
    static_assert(K32N % 2 == 0, "the ping-pong prefetch needs an even k-depth");

    // Two prefetch buffers, alternating, instead of one buffer plus a copy.
    // Same 32 registers; the 15 MOV per iteration the copy used to cost are gone.
    uint4 pre[2][4];
#pragma unroll
    for (int p = 0; p < 4; ++p) pre[0][p] = wp[p * PS];

#pragma unroll 1
    for (int j = 0; j < K32N; j += 2) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            uint32_t af[2][2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int kk = 0; kk < 2; ++kk)
                    ldmatrix_x4(af[m][kk],
                                a_frag_addr(a_base, m * 16, (j + u) * 32 + kk * 16, a_stride, lane));

            if (j + u + 1 < K32N) {
                const uint4* nx = wp + (u + 1) * 32;
#pragma unroll
                for (int p = 0; p < 4; ++p) pre[u ^ 1][p] = nx[p * PS];
            }

#pragma unroll
            for (int p = 0; p < 4; ++p) {
                uint32_t b0[2] = {pre[u][p].x, pre[u][p].y};
                uint32_t b1[2] = {pre[u][p].z, pre[u][p].w};
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    mma_f16(acc[m][p], af[m][0], b0);
                    mma_f16(acc[m][p], af[m][1], b1);
                }
            }
        }
        wp += 64;
    }
}

// --------------------------------------------------------------------------
// LayerNorm, warp-local. Warp w owns rows w, w+8, w+16, w+24, and one lane owns
// eight contiguous columns of a row, so the whole reduction is two shuffles and
// never leaves the warp. This is why the residual stream lives in SMEM rather
// than in registers: a register-resident stream would spread each row across
// all eight warps and cost a CTA barrier per norm.
/// LayerNorm over one row, warp-local.
///
/// `AFFINE` is a template parameter because the encoder body does not have an affine
/// any more. §CODA (arXiv 2605.19269) observes that `gamma` scales the *input axis* of
/// whatever GEMM consumes this, so `(gamma . x_hat + beta) W^T = x_hat (W . gamma)^T +
/// (W beta)`: both halves fold into the next matmul's weights and bias, exactly, at
/// pack time. `brokefish/nn/cuda_impl.py` does the folding and writes ones and zeros
/// into the slab's affine slots, so a path that still applies it stays *correct* --
/// the failure mode of forgetting is slow, not wrong.
///
/// Measured, `cuobjdump -sass` on `encoder_kernel<true>`: **4056 -> 3888 instructions**,
/// of which FFMA 261 -> 165. That is the currency -- this kernel is issue-limited, and
/// `docs/journal/2026-08-05-ffn-handoff-negative.md` is what happens to a change that
/// trades instructions for elapsed cycles instead of removing them.
///
/// ⚠️ Only the *body* folds. `tail_epilogue`'s `norm_f` would need a bias vector added
/// to the three heads, which have none, so it keeps `AFFINE = true` -- it runs once per
/// board against the body's sixteen, and buying it would cost a new slab field.
template <bool AFFINE>
__device__ __forceinline__ void layernorm(half* dst, const half* src, const half* gamma,
                                          const half* beta, float eps, int warp, int lane) {
    // One 128-bit load each instead of eight scalar LDG.E.U16: a lane's eight
    // columns are contiguous and the same for every row it owns, so these are
    // loop-invariant and 16-byte aligned (every offset in Off is a multiple of
    // 8 halves). As scalar loads they were long-latency global reads feeding
    // arithmetic that depends on them immediately.
    uint4 graw = {}, braw = {};
    if constexpr (AFFINE) {
        graw = *reinterpret_cast<const uint4*>(gamma + lane * 8);
        braw = *reinterpret_cast<const uint4*>(beta + lane * 8);
    }
    const half* gv = reinterpret_cast<const half*>(&graw);
    const half* bv = reinterpret_cast<const half*>(&braw);
#pragma unroll
    for (int i = 0; i < T / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        const half* p = src + ai_idx(row, lane * 8);
        uint4 raw = *reinterpret_cast<const uint4*>(p);
        const half* v = reinterpret_cast<const half*>(&raw);
        float sum = 0.f, sq = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x = __half2float(v[j]);
            sum += x;
            sq += x * x;
        }
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
            sq += __shfl_xor_sync(0xffffffff, sq, off);
        }
        float mean = sum / Dm;
        // The clamp is not cosmetic. E[x^2] - mu^2 is one pass instead of two,
        // but at a row of constant 65504 both terms are 4.29e9, where an fp32
        // ulp is 512, so the true zero can round negative and rsqrtf returns
        // NaN. Dead tokens are allowed to hold any finite value, the attention
        // mask multiplies their V by an exact zero -- and 0 * NaN is NaN, so a
        // single dead row would poison every live token of the board.
        float var = fmaxf(sq / Dm - mean * mean, 0.f);
        float rstd = rsqrtf(var + eps);
        half out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float xn = (__half2float(v[j]) - mean) * rstd;
            if constexpr (AFFINE)
                out[j] = __float2half(xn * __half2float(gv[j]) + __half2float(bv[j]));
            else
                out[j] = __float2half(xn);
        }
        *reinterpret_cast<uint4*>(dst + a_idx(row, lane * 8)) = *reinterpret_cast<uint4*>(out);
    }
}

// Add a bias to a [2][4] block of D fragments. Element (m, n) of the fragment
// covers rows 16m + {g, g+8} and columns 8n + {2t, 2t+1}; the bias depends only
// on the column, so both halves of both registers take the same pair.
__device__ __forceinline__ void add_bias(uint32_t acc[2][4][2], const half* bias, int col0,
                                         int lane) {
    int t = lane & 3;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        __half2 b = *reinterpret_cast<const __half2*>(bias + col0 + n * 8 + t * 2);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            acc[m][n][0] = u32(__hadd2(h2(acc[m][n][0]), b));
            acc[m][n][1] = u32(__hadd2(h2(acc[m][n][1]), b));
        }
    }
}

// Scatter a [2][4] D-fragment block into an SMEM activation buffer at column
// offset col0.
__device__ __forceinline__ void store_frags(half* dst, int stride, uint32_t acc[2][4][2],
                                            int col0, int lane) {
    int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c = col0 + n * 8 + t * 2;
            *reinterpret_cast<uint32_t*>(dst + (16 * m + g) * stride + c) = acc[m][n][0];
            *reinterpret_cast<uint32_t*>(dst + (16 * m + g + 8) * stride + c) = acc[m][n][1];
        }
}

// Residual, row-wise. bufA is the only buffer no ldmatrix reads, which is what
// lets it drop its padding and bring the block under the 50,176 B that fits
// twice on an SM -- but at pitch 256 the old in-place fragment-order
// read-modify-write collides 8 ways (the 8 rows a lane group touches land in
// one bank). Staging the fragment block in a padded buffer, where store_frags
// is conflict-free, and adding it a row at a time costs 20 shared accesses per
// warp against 32, with no conflicts. It needs no extra barrier either: every
// row of bufA belongs to exactly one warp, so the add and the LayerNorm that
// reads it back are warp-local.
__device__ __forceinline__ void residual_rows(half* x, const half* src, int warp, int lane) {
#pragma unroll
    for (int i = 0; i < T / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        half* dst = x + ai_idx(row, lane * 8);
        uint4 a = *reinterpret_cast<uint4*>(dst);
        uint4 b = *reinterpret_cast<const uint4*>(src + a_idx(row, lane * 8));
        __half2* pa = reinterpret_cast<__half2*>(&a);
        const __half2* pb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) pa[j] = __hadd2(pa[j], pb[j]);
        *reinterpret_cast<uint4*>(dst) = a;
    }
}

__device__ __forceinline__ void zero_frags(uint32_t acc[2][4][2]) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n) acc[m][n][0] = acc[m][n][1] = 0;
}

// --------------------------------------------------------------------------
// Attention for one head, entirely inside one warp.
//
// S = Q K^T needs no transpose of K: the B operand wants B[k][n] = K[n][k],
// and the D fragment K came out of already holds exactly that, so the B
// fragments are picked out of K's registers. V does need one, and takes the
// only memory round trip in the block -- warp-private, so __syncwarp() rather
// than a CTA barrier.
__device__ __forceinline__ void attention(uint32_t o[2][4][2], const uint32_t q[2][4][2],
                                          const uint32_t k[2][4][2], uint32_t v[2][4][2],
                                          half* vbuf, int vcol0, uint32_t alive, int lane) {
    // Warp w writes its head into columns [32w, 32w+32) of the shared scratch
    // buffer. Pitch AROW = 264 halves keeps the eight rows of an ldmatrix in
    // eight different bank groups, exactly as for the activation buffers, so
    // the per-warp padded tile the old layout needed is gone.
    store_frags(vbuf, AROW, v, vcol0, lane);
    __syncwarp();

    uint32_t s[2][4][2];
    zero_frags(s);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {          // dims 16kk .. 16kk+15
        uint32_t a[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) d_pair_to_a(a[m], q[m][2 * kk], q[m][2 * kk + 1]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {         // keys 8j .. 8j+7
            uint32_t b[2] = {k[j / 2][2 * kk][j % 2], k[j / 2][2 * kk + 1][j % 2]};
#pragma unroll
            for (int m = 0; m < 2; ++m) mma_f16(s[m][j], a[m], b);
        }
    }

    // Softmax over the 32 keys. A lane holds four rows (two m-tiles x two
    // halves) and eight of the columns of each, so the reduction is a register
    // pass over the four n-tiles plus two shuffles inside the quad.
    int t = lane & 3;
    float p[2][2][4][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int hi = 0; hi < 2; ++hi) {
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                __half2 val = h2(s[m][n][hi]);
                float lo_v = __low2float(val), hi_v = __high2float(val);
                int c = n * 8 + t * 2;
                p[m][hi][n][0] = (alive >> c) & 1 ? lo_v : -INFINITY;
                p[m][hi][n][1] = (alive >> (c + 1)) & 1 ? hi_v : -INFINITY;
                mx = fmaxf(mx, fmaxf(p[m][hi][n][0], p[m][hi][n][1]));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            float sum = 0.f;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                p[m][hi][n][0] = __expf(p[m][hi][n][0] - mx);
                p[m][hi][n][1] = __expf(p[m][hi][n][1] - mx);
                sum += p[m][hi][n][0] + p[m][hi][n][1];
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            float inv = 1.f / sum;
#pragma unroll
            for (int n = 0; n < 4; ++n)
                s[m][n][hi] = u32(__floats2half2_rn(p[m][hi][n][0] * inv, p[m][hi][n][1] * inv));
        }

    // O = P V. P's D fragments are already A fragments; V arrives transposed.
    zero_frags(o);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {          // keys 16kk .. 16kk+15
        uint32_t a[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) d_pair_to_a(a[m], s[m][2 * kk], s[m][2 * kk + 1]);
#pragma unroll
        for (int qd = 0; qd < 2; ++qd) {      // dims 16qd .. 16qd+15, two n-tiles
            uint32_t raw[4];
            ldmatrix_x4_trans(raw, a_frag_addr(vbuf, kk * 16, vcol0 + qd * 16, AROW, lane));
            uint32_t b0[2] = {raw[0], raw[1]}, b1[2] = {raw[2], raw[3]};
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                mma_f16(o[m][2 * qd], a[m], b0);
                mma_f16(o[m][2 * qd + 1], a[m], b1);
            }
        }
    }
}

// --------------------------------------------------------------------------
// The B2 prologue and epilogue, deliberately not inlined.
//
// Both run once per board against eight layers of loop, so their own speed is
// irrelevant -- but their register demand is not, and that is the trap. ptxas
// allocates one budget for the whole kernel, so cold code that wants registers
// does not slow itself down: it shrinks what the k-loop has left and makes
// *that* spill. Inlined, these two took the kernel from 113 registers and no
// spill to the 128 cap with 56 bytes of spill stores landing inside the mma
// blocks. As ABI calls they carry their own frames, the loop keeps its
// allocation, and the call overhead is paid once per board.
//
// Measured, because none of the above was obvious enough to assume: see the
// table in docs/perf.md under "the B2 prologue and the register cap".

// The embedding sum of spec 7.2. Returns the alive mask, which comes out of the
// same word: bit 11 is `captured`, so it is a ballot over a register every lane
// already holds and the pre-B2 [N,32] int8 input disappears.
__device__ __noinline__ uint32_t embed_board(half* bufA, const uint16_t* __restrict__ boards,
                                             const int16_t* __restrict__ control,
                                             const uint8_t* __restrict__ rep_ptr,
                                             const half* __restrict__ emb,
                                             int board, int warp, int lane) {
    // A position is one 64-byte coalesced load and lane l owns slot l, which is
    // what spec 2.4 exists to guarantee.
    uint32_t word = boards[(size_t)board * T + lane];
    uint32_t alive = __ballot_sync(0xffffffffu, ((word >> 11) & 1) == 0);

    int ctl = control[board];
    int stm = ctl < 0;
    // Clamped because an out-of-range index here faults rather than lying: clock
    // 0 would read the row before the table. Spec 2.2 guarantees the magnitude
    // is in [1,101] and 6.2 that the count is in [0,2]; this is insurance
    // against a malformed batch, in a path that runs once per board.
    int clk = min(max((ctl < 0 ? -ctl : ctl) - 1, 0), N_CLOCK - 1);
    int rp = min((int)rep_ptr[board], N_REP - 1);

    // The clock and repetition tables are indexed per board, not per token, so
    // they are the same vector for all 32 rows. Summing them into one uint4 here
    // costs four half2 adds per board instead of eight per token and holds eight
    // registers instead of sixteen. It is also why the normative order is
    // (square + type_special + color_turn) + (clock + rep) rather than a plain
    // left fold: model.py sums the same way, and in fp16 the difference between
    // the two groupings is a real difference in the last bit.
    uint4 g = *reinterpret_cast<const uint4*>(emb + EmbOff::clock + clk * Dm + lane * 8);
    {
        uint4 r = *reinterpret_cast<const uint4*>(emb + EmbOff::rep + rp * Dm + lane * 8);
        __half2* pg = reinterpret_cast<__half2*>(&g);
        const __half2* pr = reinterpret_cast<const __half2*>(&r);
#pragma unroll
        for (int j = 0; j < 4; ++j) pg[j] = __hadd2(pg[j], pr[j]);
    }

#pragma unroll
    for (int i = 0; i < T / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        // row is uniform across the warp, so this is a broadcast out of the lane
        // that owns the slot rather than 32 lanes reloading the word.
        uint32_t w = __shfl_sync(0xffffffffu, word, row);
        // A dead slot is 0x800 exactly, so it decodes to square a1, pawn, white
        // -- a valid set of indices. The gather runs for it anyway, branchless
        // and divergence-free, and produces the finite vector spec 7.3 requires;
        // the attention mask is what makes it irrelevant.
        int sq = w & 63;
        // `type` is three bits but only six values are defined, so 6 and 7 are
        // reachable from a malformed word. Unclamped they would read a row of
        // emb_clock -- in bounds by an accident of slab order, which is worse than
        // a fault because it lies. `square` and `color_turn` cannot go out of
        // range by construction.
        int ts = min(((w >> 6) & 7) * 2 + ((w >> 9) & 1), N_TYPE_SPECIAL - 1);
        int ct = ((w >> 10) & 1) * 2 + stm;
        uint4 a = *reinterpret_cast<const uint4*>(emb + EmbOff::square + sq * Dm + lane * 8);
        __half2* pa = reinterpret_cast<__half2*>(&a);
        const __half2* pg = reinterpret_cast<const __half2*>(&g);
        // Accumulated one table at a time rather than five loads then four adds:
        // 16 live registers instead of 40, for the same arithmetic in the same
        // order.
#pragma unroll
        for (int tbl = 0; tbl < 2; ++tbl) {
            int off = tbl == 0 ? EmbOff::type_special + ts * Dm : EmbOff::color_turn + ct * Dm;
            uint4 b = *reinterpret_cast<const uint4*>(emb + off + lane * 8);
            const __half2* pb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
            for (int j = 0; j < 4; ++j) pa[j] = __hadd2(pa[j], pb[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) pa[j] = __hadd2(pa[j], pg[j]);
        *reinterpret_cast<uint4*>(bufA + ai_idx(row, lane * 8)) = a;
    }
    return alive;
}

// The final LayerNorm and the three heads of spec 7.4. Called by the whole CTA,
// so the barrier inside it is safe; returns after the norm when `policy` is null,
// which is how debug stage 8 gets the normed stream on its own.
__device__ __noinline__ void tail_epilogue(half* bufA, half* bufB, half* scratch,
                                           const half* __restrict__ tail,
                                           const int16_t* __restrict__ control,
                                           half* __restrict__ policy,
                                           half* __restrict__ promo,
                                           float* __restrict__ value, float eps,
                                           int board, int warp, int lane, int tid) {
    // Two independent reasons this norm is not optional. The stack is pre-norm,
    // so what falls out of it is a raw residual stream whose scale grows with
    // depth, in fp16, and a linear head on top of that is the classic pre-norm
    // mistake. And bufA is the one buffer carrying no row padding -- that missing
    // 512 B is what bought the second CTA per SM -- so a head GEMM reading it
    // through ldmatrix would collide eight ways. The norm lands the stream in
    // bufB, which is padded, and both problems go away at once.
    layernorm<true>(bufB, bufA, tail + TailOff::lnf_w, tail + TailOff::lnf_b, eps,
                    warp, lane);
    __syncthreads();
    if (!policy) return;

    // One packed [96][256]. Warps 0 and 1 take policy columns 0-31 and 32-63,
    // warp 2 takes the padded aux tile; warps 3-7 have nothing to do, which is
    // the right answer for 0.3 % of the network's arithmetic.
    if (warp < 3) {
        uint32_t hd[2][4][2];
        zero_frags(hd);
        gemm_direct<Dm / 32, Dm / 32>(hd, bufB, AROW, tail + TailOff::w_head,
                                      warp * 4, 0, lane);
        // scratch last held an FFN hidden chunk and two barriers have passed
        // since anything read it. Staging here rather than storing straight out
        // turns a fragment-order scatter into coalesced rows.
        store_frags(scratch, HDROW, hd, warp * 32, lane);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < T / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        // 64 halves is exactly 32 lanes x one uint32, so a row leaves as one
        // 128-byte transaction. Raw logits: the legality mask belongs to the
        // search, which needs a masked softmax anyway.
        *reinterpret_cast<uint32_t*>(policy + ((size_t)board * T + row) * NPOL + lane * 2) =
            *reinterpret_cast<const uint32_t*>(scratch + row * HDROW + lane * 2);
        // Four halves is eight bytes, so one lane carries a whole row.
        if (promo && lane == 0)
            *reinterpret_cast<uint2*>(promo + ((size_t)board * T + row) * N_PROMO) =
                *reinterpret_cast<const uint2*>(scratch + row * HDROW + PROMO_COL);
    }
    // The value is a row select decided by the sign of the control word, then
    // tanh in fp32. Doing it here rather than shipping the whole [N,32,8] aux
    // tensor saves every consumer in the search a data-dependent gather of two
    // bytes out of every 512, on every backup.
    if (value && tid == 0)
        value[board] = tanhf(__half2float(scratch[(control[board] < 0 ? 31 : 15) * HDROW
                                                  + VALUE_COL]));
}

// --------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase accounting.
//
// `docs/ledger/perf.md` gets the non-matmul share of this kernel by *subtraction*:
// the four real GEMM shapes run at 32.1 TFLOPS in isolation (csrc/shapes_ab.cu), the
// whole kernel at 25.5, so 20.6 % of the time is something else. That is a residual,
// not a measurement -- part of it could be the GEMMs themselves running slower in
// situ -- and it says nothing about *which* something else. This measures directly.
//
// ## Why per-warp `clock64` deltas are a sound decomposition here
//
// Each warp's phase deltas sum to that warp's own lifetime, so the fractions are
// exact by construction and nothing has to be assumed about overlap. Two CTAs share
// an SM, so a phase's cycles include time the SM spent on the other CTA -- which is
// what makes them *fractions of elapsed time* rather than of issue slots, and
// fractions of elapsed time are what a speedup is denominated in.
//
// Marks sit at boundaries the kernel already has, so **no barrier is added**. Adding
// one would serialise warps that currently overlap and inflate the total it is
// supposed to be dividing. Where a mark precedes an existing `__syncthreads()`, the
// barrier wait lands in the *following* phase, which is why the phases are named for
// what follows the barrier and not for what precedes it.
//
// ⚠️ Warps are symmetric everywhere except `tail_epilogue`, where warps 0-2 do the
// three heads and 3-7 idle. Slot `kNumPhases + p` therefore records **warp 0 alone**,
// which works in every phase; comparing the two columns is how that asymmetry is read
// off rather than guessed at.
namespace prof {

enum Phase {
    kPrologue = 0,  // entry, embedding gather or activation load, first barrier
    kLn1,           // LayerNorm 1 and the cross-warp handoff barrier
    kQkvGemm,       // three [32,256]x[256,256]
    kQkvBias,       // three add_bias over the fragments
    kAttention,     // V transpose, scores, softmax, AV -- warp-local, no barrier
    kAttnStore,     // store_frags(o) between two barriers
    kProjGemm,      // out_proj [32,256]x[256,256]
    kProjEpi,       // its bias and the staged store
    kRes1Ln2,       // residual add and LayerNorm 2
    kQuantA,        // quantise_row over the normed input          (fp8 only)
    kFf1Gemm,       // linear1, x4 chunks
    kFf1Epi,        // its bias and the ReLU, x4
    kHidStore,      // store_frags(hid), x4
    kQuantH,        // quantise_row over the hidden chunk, x4      (fp8 only)
    kFf2Gemm,       // linear2, x4 chunks
    kFf2Epi,        // its bias and the staged store
    kRes2,          // the second residual add
    kEpilogue,      // final LayerNorm and the three heads
    kNumPhases
};
// ⚠️ **Sharded by CTA, and this is not an optimisation.** A mark fires ~250 times per
// warp per CTA, so an unsharded counter takes 8 * 250 * n_boards atomics onto 36
// addresses -- same-address atomics serialise, and the profiler would then be
// measuring its own contention and attributing it to whichever phase it landed in.
// 32 shards summed on the host cuts that 32-fold and costs 9 KB of device globals.
constexpr int kShards = 32;
constexpr int kRows = 2;          // row 0: every warp. row 1: warp 0 alone.
constexpr int kSlots = kShards * kRows * kNumPhases;

__device__ unsigned long long g_prof[kSlots];

__device__ __forceinline__ long long now() {
    long long t;
    // ⚠️ `volatile` **and** the memory clobber. Without them ptxas may move loads and
    // stores across the read, which is exactly what makes a phase boundary a fiction.
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

template <bool ON>
struct Timer {
    long long t;
    unsigned long long* base;     // this CTA's shard, row 0
    bool warp0, lane0;
    __device__ __forceinline__ Timer(int block, int warp, int lane)
        : t(0), base(g_prof + (size_t)(block & (kShards - 1)) * kRows * kNumPhases),
          warp0(warp == 0), lane0(lane == 0) {
        if constexpr (ON) t = now();
    }
    __device__ __forceinline__ void mark(int phase) {
        if constexpr (ON) {
            const long long n = now();
            const unsigned long long d = (unsigned long long)(n - t);
            if (lane0) {
                atomicAdd(base + phase, d);
                if (warp0) atomicAdd(base + kNumPhases + phase, d);
            }
            t = n;
        }
    }
};

}  // namespace prof

// `FP8` is a template parameter and not a runtime flag on purpose. Under
// `__launch_bounds__(THREADS, 2)` ptxas must fit 128 registers, and a runtime branch
// would make it allocate for the union of both paths and spill the fp16 one -- which
// is the path every existing measurement was taken on.
//
// ⚠️ **Buying the fp8 path more registers is a measured loss.** It carries a second
// accumulator -- `float acc[2][4][4]` running across k-tiles on top of the fp16 mma
// fragment -- so it wants 32 registers where fp16 wants 16, and at 2 CTAs/SM ptxas
// holds it to 128 and pays in local memory: 960 B of spill loads against fp16's 64.
// `__launch_bounds__(THREADS, FP8 ? 1 : 2)` removes every spill (200 registers, 0 B)
// and makes the kernel **slower**: 0.978x against fp16 where the spilling version is
// 1.154x. Two CTAs per SM is only 16 warps and it is already the whole of the latency
// hiding; halving it costs more than the spill traffic ever did. 2 CTAs/SM needs
// 65536 / (2 * 256) = 128 registers exactly, so 128 is not a tuning knob -- the way
// out is to *need* fewer registers, i.e. to drop the fp32 running accumulator.
// `INT8` swaps e4m3 for s8 in the FFN's two matmuls and nowhere else -- the same two
// places, the same block/row scale granularity, the same packed B layout. It is a
// separate template parameter rather than a mode of `FP8` because the two slabs are
// laid out identically and a runtime switch between them would be invisible: int8
// bytes read as e4m3 stay in range and produce plausible logits.
// docs/journal/2026-08-14-int8-kernel-spec.md.
template <bool FP8, bool INT8, bool PROF = false>
__global__ __launch_bounds__(THREADS, 2) void encoder_kernel(
    half* __restrict__ y, const half* __restrict__ weights,
    const uint8_t* __restrict__ wq8, const float* __restrict__ sq8,
    const int8_t* __restrict__ alive_ptr,
    // B2. Every pointer below may be null, and the null-ness is the mode switch:
    // `boards` non-null runs the embedding prologue instead of reading
    // activations out of y, and `policy` non-null runs the final norm and the
    // heads instead of writing activations back. All four combinations are legal
    // and all four are tested. The backbone-only one is the pre-B2 kernel
    // instruction for instruction, which is what keeps the Triton A/B and the
    // eleven tests of tests/test_model.py meaningful across this change.
    // `debug_stage` is orthogonal to all of it: a manual hook, set through
    // FusedEncoder.debug_stage, that dumps one intermediate buffer instead of
    // finishing. Nothing in tests/ drives it.
    const uint16_t* __restrict__ boards, const int16_t* __restrict__ control,
    const uint8_t* __restrict__ rep_ptr, const half* __restrict__ emb,
    const half* __restrict__ tail, half* __restrict__ policy,
    half* __restrict__ promo, float* __restrict__ value,
    int n_layers, float eps, int debug_stage) {
    static_assert(!(FP8 && INT8), "the FFN is quantised once, in one format");
    // Both quantised paths read the same byte slab and the same scale slab.
    constexpr bool Q8 = FP8 || INT8;
    extern __shared__ half smem[];
    half* bufA = smem;                    // residual stream
    half* bufB = bufA + SM_A;             // normed input, then attention output
    // One scratch buffer serves both roles: the V transpose during attention
    // and the FFN hidden chunk afterwards. Their lifetimes do not overlap, and
    // sharing the allocation is what brings the block down to 49.5 KB.
    half* scratch = bufB + SM_B;
    half* hid = scratch;
    half* vbuf = scratch;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int board = blockIdx.x;

    // Declared before the first goto: jumping over an initialised declaration in
    // the same scope does not compile, and the debug stages jump to `dump`.
    half* gy = y ? y + (size_t)board * T * Dm : nullptr;
    prof::Timer<PROF> tm(board, warp, lane);

    // One 32-bit word says which slots hold a live piece (spec 7.3). Dead keys
    // leave the softmax as -inf; dead rows are computed and never read.
    uint32_t alive = 0xffffffffu;

    if (boards) {
        alive = embed_board(bufA, boards, control, rep_ptr, emb, board, warp, lane);
    } else {
        if (alive_ptr) {
            // One coalesced byte per lane and a ballot, not 32 scalar loads per
            // thread. T == warpSize is what makes the ballot the whole reduction.
            alive = __ballot_sync(0xffffffffu, alive_ptr[(size_t)board * T + lane] != 0);
        }
        // Board in: [T][Dm] contiguous, one uint4 per lane per row.
#pragma unroll
        for (int i = 0; i < T / NWARPS; ++i) {
            int row = warp + i * NWARPS;
            *reinterpret_cast<uint4*>(bufA + ai_idx(row, lane * 8)) =
                *reinterpret_cast<const uint4*>(gy + row * Dm + lane * 8);
        }
    }
    __syncthreads();
    tm.mark(prof::kPrologue);
    if (debug_stage == 9) goto dump;      // the gather alone, nothing else run

    for (int layer = 0; layer < n_layers; ++layer) {
        const half* W = weights + (size_t)layer * Off::stride;
        const uint8_t* Wq = Q8 ? wq8 + (size_t)layer * Fp8Off::stride : nullptr;
        const float* Sq = Q8 ? sq8 + (size_t)layer * Fp8SOff::stride : nullptr;

        // The affine slots hold ones and zeros: gamma lives in w_qkv's input axis and
        // beta in b_qkv, folded there by `FusedEncoder._fold_norm`.
        layernorm<false>(bufB, bufA, W + Off::ln1_w, W + Off::ln1_b, eps, warp, lane);

        if (debug_stage == 1) { __syncthreads(); goto dump; }
        // Cross-warp handoff. layernorm writes bufB by row (warp w owns rows
        // w, w+8, w+16, w+24) and store_frags writes it by column (warp w owns
        // columns 32w..32w+31), while every GEMM below reads the whole buffer.
        // gemm_full used to open with a barrier of its own, which hid this
        // dependency; gemm_direct has none, so it has to be stated.
        __syncthreads();
        tm.mark(prof::kLn1);

        uint32_t q[2][4][2], k[2][4][2], v[2][4][2];
        zero_frags(q);
        zero_frags(k);
        zero_frags(v);
        // w_qkv is one packed [768][256]: Q occupies n-tiles 0..31, K 32..63,
        // V 64..95, and warp w takes four consecutive n-tiles of each.
        const half* wq = W + Off::w_qkv;
        constexpr int NK = Dm / 32;                  // 8 k-groups of 32
        gemm_direct<NK, NK>(q, bufB, AROW, wq, warp * 4,      0, lane);
        gemm_direct<NK, NK>(k, bufB, AROW, wq, 32 + warp * 4, 0, lane);
        gemm_direct<NK, NK>(v, bufB, AROW, wq, 64 + warp * 4, 0, lane);
        tm.mark(prof::kQkvGemm);
        add_bias(q, W + Off::b_qkv, warp * DH, lane);
        add_bias(k, W + Off::b_qkv + Dm, warp * DH, lane);
        add_bias(v, W + Off::b_qkv + 2 * Dm, warp * DH, lane);
        tm.mark(prof::kQkvBias);

        if (debug_stage >= 5 && debug_stage <= 7) {
            __syncthreads();
            store_frags(bufB, AROW, debug_stage == 5 ? q : (debug_stage == 6 ? k : v),
                        warp * DH, lane);
            __syncthreads();
            goto dump;
        }

        uint32_t o[2][4][2];
        attention(o, q, k, v, vbuf, warp * DH, alive, lane);
        tm.mark(prof::kAttention);

        __syncthreads();
        store_frags(bufB, AROW, o, warp * DH, lane);
        if (debug_stage == 4) { __syncthreads(); goto dump; }
        __syncthreads();
        tm.mark(prof::kAttnStore);

        uint32_t proj[2][4][2];
        zero_frags(proj);
        gemm_direct<NK, NK>(proj, bufB, AROW, W + Off::w_o, warp * 4, 0, lane);
        tm.mark(prof::kProjGemm);
        add_bias(proj, W + Off::b_o, warp * DH, lane);
        // scratch is dead here (it held the V transpose, and every warp passed
        // the barrier above), so it takes the staged projection.
        store_frags(scratch, AROW, proj, warp * DH, lane);
        __syncthreads();
        tm.mark(prof::kProjEpi);
        residual_rows(bufA, scratch, warp, lane);

        if (debug_stage == 2) goto dump;
        layernorm<false>(bufB, bufA, W + Off::ln2_w, W + Off::ln2_b, eps, warp, lane);
        if (debug_stage == 3) { __syncthreads(); goto dump; }
        __syncthreads();
        tm.mark(prof::kRes1Ln2);

        uint32_t ff[2][4][2];
        zero_frags(ff);
        if constexpr (Q8) {
            // Quantise the normed input **in place**, once for all four chunks: one
            // scale for the whole 256-wide row, which is what lets the row GEMM keep a
            // single accumulator. `quantise_row` owns the three aliasing orderings this
            // relies on; they are documented there and `quantise_row_int8` inherits
            // them verbatim, because they are a property of the buffer geometry rather
            // than of the format.
            uint8_t* aq = reinterpret_cast<uint8_t*>(bufB);
            float* as = reinterpret_cast<float*>(aq + fp8cfg::kScaleOff);
            const int r0 = warp * (T / NWARPS);
            if constexpr (INT8)
                int8q::quantise_row_int8<Dm, /*UNSIGNED=*/false>(
                    aq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                    as + (size_t)r0 * fp8cfg::kScaleStride, fp8cfg::kScaleStride,
                    bufB + (size_t)r0 * AROW, AROW, T / NWARPS, lane);
            else
                fp8::quantise_row<Dm>(aq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                                      as + (size_t)r0 * fp8cfg::kScaleStride,
                                      fp8cfg::kScaleStride,
                                      bufB + (size_t)r0 * AROW, AROW, T / NWARPS, lane);
            __syncthreads();
            tm.mark(prof::kQuantA);
        }
        for (int c = 0; c < DFF / HCHUNK; ++c) {
            uint32_t hacc[2][4][2];
            // ff1 is [1024][256]: chunk c starts at n-tile 32c.
            if constexpr (INT8) {
                // A is the normed input, which is signed.
                int8q::gemm_s8_row<NK, NK, /*ADD=*/false, /*UNSIGNED_A=*/false>(
                    hacc, reinterpret_cast<const uint8_t*>(bufB), fp8cfg::kAPitch,
                    reinterpret_cast<const float*>(
                        reinterpret_cast<const uint8_t*>(bufB) + fp8cfg::kScaleOff),
                    fp8cfg::kScaleStride,
                    reinterpret_cast<const uint2*>(Wq + Fp8Off::w_ff1),
                    Sq + Fp8SOff::s_ff1, c * 32 + warp * 4, 0, lane);
            } else if constexpr (FP8) {
                fp8::gemm_fp8_row<NK, NK, /*ADD=*/false>(
                    hacc, reinterpret_cast<const uint8_t*>(bufB), fp8cfg::kAPitch,
                    reinterpret_cast<const float*>(
                        reinterpret_cast<const uint8_t*>(bufB) + fp8cfg::kScaleOff),
                    fp8cfg::kScaleStride,
                    reinterpret_cast<const uint2*>(Wq + Fp8Off::w_ff1),
                    Sq + Fp8SOff::s_ff1, c * 32 + warp * 4, 0, lane);
            } else {
                zero_frags(hacc);
                gemm_direct<NK, NK>(hacc, bufB, AROW, W + Off::w_ff1, c * 32 + warp * 4, 0, lane);
            }
            tm.mark(prof::kFf1Gemm);
            add_bias(hacc, W + Off::b_ff1 + c * HCHUNK, warp * DH, lane);
            const __half2 zero2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    hacc[m][n][0] = u32(__hmax2(h2(hacc[m][n][0]), zero2));
                    hacc[m][n][1] = u32(__hmax2(h2(hacc[m][n][1]), zero2));
                }
            __syncthreads();
            tm.mark(prof::kFf1Epi);
            store_frags(hid, HROW, hacc, warp * DH, lane);
            // ff2 is [256][1024]: chunk c is the k-slice [256c, 256c+256), i.e.
            // k-groups 8c..8c+7 of a matrix whose packed row pitch is 32.
            __syncthreads();
            tm.mark(prof::kHidStore);
            if constexpr (Q8) {
                // Same in-place quantisation, on this chunk of the hidden. Its input
                // is post-ReLU and therefore non-negative, so its partial sums get no
                // sign cancellation -- this is where `q_max = 8`'s 4x accumulator
                // headroom is doing real work rather than being slack. ⚠️ Under int8
                // that non-negativity is spent differently and better: the operand goes
                // out **unsigned**, 256 levels instead of 127, because `mma...u8.s8`
                // exists and a float cannot make the same trade.
                //
                // Each chunk carries its **own** scale and is descaled before being
                // added into `ff`, so the four never have to agree on one -- which
                // matters because chunk 3's amax is not knowable when chunk 0 is
                // quantised, only 256 of the 1024 hidden units being resident.
                uint8_t* hq = reinterpret_cast<uint8_t*>(hid);
                float* hs = reinterpret_cast<float*>(hq + fp8cfg::kScaleOff);
                const int r0 = warp * (T / NWARPS);
                if constexpr (INT8)
                    int8q::quantise_row_int8<HCHUNK, /*UNSIGNED=*/true>(
                        hq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                        hs + (size_t)r0 * fp8cfg::kScaleStride, fp8cfg::kScaleStride,
                        hid + (size_t)r0 * HROW, HROW, T / NWARPS, lane);
                else
                    fp8::quantise_row<HCHUNK>(hq + (size_t)r0 * fp8cfg::kAPitch,
                                              fp8cfg::kAPitch,
                                              hs + (size_t)r0 * fp8cfg::kScaleStride,
                                              fp8cfg::kScaleStride,
                                              hid + (size_t)r0 * HROW, HROW,
                                              T / NWARPS, lane);
                __syncthreads();
                tm.mark(prof::kQuantH);
                // ⚠️ The weight is addressed globally from `k32_0 = 8c`; the activation
                // buffer holds only this chunk and is addressed from zero.
                // gemm_fp8_row keeps the two indices apart -- csrc/tests/tfp8.cu's
                // k-slice check is exactly this call shape, and two bugs lived in the
                // difference.
                if constexpr (INT8)
                    int8q::gemm_s8_row<HCHUNK / 32, DFF / 32, /*ADD=*/true,
                                       /*UNSIGNED_A=*/true>(
                        ff, hq, fp8cfg::kAPitch, hs, fp8cfg::kScaleStride,
                        reinterpret_cast<const uint2*>(Wq + Fp8Off::w_ff2),
                        Sq + Fp8SOff::s_ff2, warp * 4, c * (HCHUNK / 32), lane);
                else
                    fp8::gemm_fp8_row<HCHUNK / 32, DFF / 32, /*ADD=*/true>(
                        ff, hq, fp8cfg::kAPitch, hs, fp8cfg::kScaleStride,
                        reinterpret_cast<const uint2*>(Wq + Fp8Off::w_ff2),
                        Sq + Fp8SOff::s_ff2, warp * 4, c * (HCHUNK / 32), lane);
            } else {
                gemm_direct<8, DFF / 32>(ff, hid, HROW, W + Off::w_ff2, warp * 4, c * 8, lane);
            }
            tm.mark(prof::kFf2Gemm);
        }
        add_bias(ff, W + Off::b_ff2, warp * DH, lane);
        // bufB is dead: the first barrier of the last FFN chunk is already past
        // every warp's read of it, so the staged sum needs no barrier of its own
        // and the layer keeps its 14.
        store_frags(bufB, AROW, ff, warp * DH, lane);
        __syncthreads();
        tm.mark(prof::kFf2Epi);
        residual_rows(bufA, bufB, warp, lane);
        tm.mark(prof::kRes2);
    }

    // --- B2 epilogue: the final LayerNorm, then the three heads -------------
    if (policy || debug_stage == 8)
        tail_epilogue(bufA, bufB, scratch, tail, control, policy, promo, value, eps,
                      board, warp, lane, tid);
    tm.mark(prof::kEpilogue);
    if (debug_stage == 8) goto dump;
    if (policy) return;

dump:
    if (!y) return;
#pragma unroll
    for (int i = 0; i < T / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        // Stage 9 is the embedding gather, which lands in bufA; 8 is the final
        // norm, which lands in bufB like the other post-norm stages.
        bool fromB = (debug_stage == 1 || debug_stage == 3
                      || (debug_stage >= 4 && debug_stage <= 8));
        const half* srcbuf = fromB ? bufB + a_idx(row, lane * 8) : bufA + ai_idx(row, lane * 8);
        *reinterpret_cast<uint4*>(gy + row * Dm + lane * 8) = *reinterpret_cast<const uint4*>(srcbuf);
    }
}

}  // namespace brokefish

// ---------------------------------------------------------------------------

namespace {

// An empty tensor becomes a null pointer, which is how the kernel's mode switch
// is spelled from Python: no flags, just which arguments were given.
template <typename T>
T* ptr_or_null(const torch::Tensor& t) {
    return t.numel() ? reinterpret_cast<T*>(t.data_ptr()) : nullptr;
}

// ⚠️ Off by default and **compile-time** off, like `FP8` and for the same reason: a
// runtime branch would make ptxas allocate registers for the union of both paths and
// spill the one every number in perf.md was measured on. `PROF=false` is the shipped
// kernel; `csrc/tests/README.md` records the ptxas report that proves it.
bool g_profile = false;
template <bool FP8, bool INT8, bool PROF>
void launch_one(int n_boards, size_t smem, half* y, const half* weights,
                const uint8_t* wq8, const float* sq8, const int8_t* alive,
                const uint16_t* boards, const int16_t* control, const uint8_t* rep,
                const half* emb, const half* tail, half* policy, half* promo,
                float* value, int n_layers, float eps, int debug_stage) {
    constexpr bool Q8 = FP8 || INT8;
    cudaFuncSetAttribute(brokefish::encoder_kernel<FP8, INT8, PROF>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    brokefish::encoder_kernel<FP8, INT8, PROF><<<n_boards, brokefish::THREADS, smem>>>(
        y, weights, Q8 ? wq8 : nullptr, Q8 ? sq8 : nullptr, alive, boards, control,
        rep, emb, tail, policy, promo, value, n_layers, eps, debug_stage);
}

// `quant`: 0 = fp16, 1 = e4m3, 2 = int8. ⚠️ An explicit mode rather than a property of
// the pointers, because the int8 and e4m3 slabs have **identical** layout and size --
// int8 bytes read as e4m3 stay in range and produce plausible logits, so a
// pointer-derived switch would fail silently. The fp16/quantised distinction keeps its
// old all-or-nothing pointer check on top.
void launch(int n_boards, int quant, half* y, const half* weights, const uint8_t* wq8,
            const float* sq8, const int8_t* alive,
            const uint16_t* boards, const int16_t* control, const uint8_t* rep,
            const half* emb, const half* tail, half* policy, half* promo, float* value,
            int n_layers, float eps, int debug_stage) {
    size_t smem = brokefish::SMEM_HALVES * sizeof(half);
    const bool have_slab = (wq8 != nullptr) && (sq8 != nullptr);
    TORCH_CHECK(quant == 0 || have_slab,
                "quant mode ", quant, " needs both the byte slab and the scale slab; "
                "half a slab reads the wrong matrix and stays in range");
#define BROKEFISH_LAUNCH(F, I, P)                                                     \
    launch_one<F, I, P>(n_boards, smem, y, weights, wq8, sq8, alive, boards, control,  \
                        rep, emb, tail, policy, promo, value, n_layers, eps, debug_stage)
    if (quant == 2) {
        if (g_profile) BROKEFISH_LAUNCH(false, true, true);
        else           BROKEFISH_LAUNCH(false, true, false);
    } else if (quant == 1) {
        if (g_profile) BROKEFISH_LAUNCH(true, false, true);
        else           BROKEFISH_LAUNCH(true, false, false);
    } else {
        if (g_profile) BROKEFISH_LAUNCH(false, false, true);
        else           BROKEFISH_LAUNCH(false, false, false);
    }
#undef BROKEFISH_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The pre-B2 entry point, unchanged: activations in, activations out. It is the
// A/B control against Triton and the eleven tests of tests/test_model.py run
// through it, so it
// keeps its exact signature rather than growing optional arguments.
void encoder_forward(torch::Tensor y, torch::Tensor weights, torch::Tensor alive,
                     int64_t n_layers, double eps, int64_t debug_stage) {
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kHalf && y.is_contiguous());
    TORCH_CHECK(y.dim() == 3 && y.size(1) == brokefish::T && y.size(2) == brokefish::Dm,
                "y must be [boards, 32, 256] fp16 contiguous");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kHalf);
    // No fp8 here on purpose: this entry point is the A/B control against Triton and
    // the eleven tests of tests/test_model.py, so it stays the fp16 kernel exactly.
    launch((int)y.size(0), /*quant=*/0, ptr_or_null<half>(y),
           ptr_or_null<const half>(weights), nullptr, nullptr,
           ptr_or_null<const int8_t>(alive), nullptr, nullptr, nullptr, nullptr, nullptr,
           nullptr, nullptr, nullptr, (int)n_layers, (float)eps, (int)debug_stage);
}

// The B2 entry point: boards in, logits out, nothing in between touching HBM.
// `policy` empty with `y` given runs the prologue and the stack and dumps
// activations, which is how the embedding gather and the final norm get tested
// on their own.
void model_forward(torch::Tensor boards, torch::Tensor control, torch::Tensor rep,
                   torch::Tensor weights, torch::Tensor emb, torch::Tensor tail,
                   torch::Tensor policy, torch::Tensor promo, torch::Tensor value,
                   torch::Tensor y, int64_t n_layers, double eps, int64_t debug_stage,
                   torch::Tensor wq8, torch::Tensor sq8, int64_t quant) {
    TORCH_CHECK(boards.is_cuda() && boards.scalar_type() == torch::kShort
                && boards.is_contiguous() && boards.dim() == 2
                && boards.size(1) == brokefish::T,
                "boards must be [n, 32] int16 contiguous (spec 2.4)");
    const int64_t n = boards.size(0);
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == torch::kShort
                && control.is_contiguous() && control.numel() == n,
                "control must be [n] int16 contiguous");
    TORCH_CHECK(rep.is_cuda() && rep.scalar_type() == torch::kByte
                && rep.is_contiguous() && rep.numel() == n,
                "rep must be [n] uint8 contiguous");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kHalf);
    TORCH_CHECK(emb.is_cuda() && emb.scalar_type() == torch::kHalf
                && emb.numel() == brokefish::EmbOff::stride,
                "the embedding slab is the five tables of spec 7.2, concatenated");
    TORCH_CHECK(tail.is_cuda() && tail.scalar_type() == torch::kHalf
                && tail.numel() == brokefish::TailOff::stride,
                "the tail slab is norm_f's affine then the packed [96][256] head");
    if (policy.numel())
        TORCH_CHECK(policy.is_cuda() && policy.scalar_type() == torch::kHalf
                    && policy.is_contiguous()
                    && policy.numel() == n * brokefish::T * brokefish::NPOL,
                    "policy must be [n, 32, 64] fp16 contiguous");
    if (promo.numel())
        TORCH_CHECK(promo.is_cuda() && promo.scalar_type() == torch::kHalf
                    && promo.is_contiguous()
                    && promo.numel() == n * brokefish::T * brokefish::N_PROMO,
                    "promo must be [n, 32, 4] fp16 contiguous");
    if (value.numel())
        TORCH_CHECK(value.is_cuda() && value.scalar_type() == torch::kFloat
                    && value.is_contiguous() && value.numel() == n,
                    "value must be [n] fp32 contiguous");
    if (y.numel())
        TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kHalf && y.is_contiguous()
                    && y.numel() == n * brokefish::T * brokefish::Dm,
                    "y must be [n, 32, 256] fp16 contiguous");

    // §fp8. Empty tensors mean the fp16 path, matching every other optional argument
    // here. Both or neither: a slab without its scales would be read as fp16 weights
    // through an fp8 layout, in range and wrong.
    TORCH_CHECK(wq8.numel() == 0 || (wq8.is_cuda() && wq8.scalar_type() == torch::kByte
                                     && wq8.is_contiguous()),
                "wq8 must be empty or contiguous uint8 on CUDA");
    TORCH_CHECK(sq8.numel() == 0 || (sq8.is_cuda() && sq8.scalar_type() == torch::kFloat
                                     && sq8.is_contiguous()),
                "sq8 must be empty or contiguous float32 on CUDA");
    TORCH_CHECK((wq8.numel() == 0) == (sq8.numel() == 0),
                "pass both the fp8 weight slab and its scales, or neither");
    if (wq8.numel()) {
        TORCH_CHECK(wq8.numel() == n_layers * (int64_t)brokefish::Fp8Off::stride,
                    "wq8 must be [n_layers * ", brokefish::Fp8Off::stride, "] bytes, got ",
                    wq8.numel());
        TORCH_CHECK(sq8.numel() == n_layers * (int64_t)brokefish::Fp8SOff::stride,
                    "sq8 must be [n_layers * ", brokefish::Fp8SOff::stride, "] floats, got ",
                    sq8.numel());
    }

    TORCH_CHECK(quant >= 0 && quant <= 2, "quant is 0 fp16, 1 e4m3, 2 int8; got ", quant);
    launch((int)n, (int)quant, ptr_or_null<half>(y), ptr_or_null<const half>(weights),
           ptr_or_null<const uint8_t>(wq8), ptr_or_null<const float>(sq8), nullptr,
           ptr_or_null<const uint16_t>(boards), ptr_or_null<const int16_t>(control),
           ptr_or_null<const uint8_t>(rep), ptr_or_null<const half>(emb),
           ptr_or_null<const half>(tail), ptr_or_null<half>(policy),
           ptr_or_null<half>(promo), ptr_or_null<float>(value),
           (int)n_layers, (float)eps, (int)debug_stage);
}

// ---------------------------------------------------------------------------
// The profiling hook. Three calls, no state anywhere else: turn it on, run the
// kernel however you already run it, read the counters back.
//
// The returned tensor is `[2][kNumPhases]` of **cycles**, summed over every CTA:
// row 0 over all eight warps, row 1 over warp 0 alone. Cycles and not seconds on
// purpose -- this card's clock falls from 2055 to 1230-1290 MHz under a sustained
// load, so a wall-clock denominator would drift while the decomposition did not.
void set_profile(bool on) { g_profile = on; }

void reset_profile() {
    const std::vector<unsigned long long> zero(brokefish::prof::kSlots, 0ull);
    C10_CUDA_CHECK(cudaMemcpyToSymbol(brokefish::prof::g_prof, zero.data(),
                                      zero.size() * sizeof(unsigned long long)));
}

torch::Tensor read_profile() {
    auto raw = torch::empty({(int64_t)brokefish::prof::kShards,
                             (int64_t)brokefish::prof::kRows,
                             (int64_t)brokefish::prof::kNumPhases},
                            torch::dtype(torch::kInt64));
    C10_CUDA_CHECK(cudaMemcpyFromSymbol(raw.data_ptr<int64_t>(), brokefish::prof::g_prof,
                                        brokefish::prof::kSlots
                                        * sizeof(unsigned long long)));
    return raw.sum(0);          // [rows, phases]; the shards exist only to spread atomics
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_q_max", []() { return (double)brokefish::fp8::kQMax; });
    m.def("set_profile", &set_profile, "compile-time-gated phase accounting on/off");
    m.def("reset_profile", &reset_profile, "zero the cycle counters");
    m.def("read_profile", &read_profile, "[2][kNumPhases] cycles: all warps, warp 0");
    m.def("profile_phases", []() {
        return std::vector<std::string>{
            "prologue", "ln1", "qkv_gemm", "qkv_bias", "attention", "attn_store",
            "proj_gemm", "proj_epi", "res1_ln2", "quant_a", "ff1_gemm", "ff1_epi",
            "hid_store", "quant_h", "ff2_gemm", "ff2_epi", "res2", "epilogue"};
    }, "phase names, in slot order");
    m.def("encoder_forward", &encoder_forward, "fused encoder forward");
    m.def("model_forward", &model_forward, "boards in, policy/promo/value out",
          py::arg("boards"), py::arg("control"), py::arg("rep"), py::arg("weights"),
          py::arg("emb"), py::arg("tail"), py::arg("policy"), py::arg("promo"),
          py::arg("value"), py::arg("y"), py::arg("n_layers"), py::arg("eps"),
          py::arg("debug_stage"), py::arg("wq8"), py::arg("sq8"), py::arg("quant") = 0);
    m.def("int8_q_max", []() {
        return std::pair<double, double>{(double)brokefish::int8q::kQMaxS,
                                         (double)brokefish::int8q::kQMaxU};
    }, "the s8 and u8 quantisation maxima, pinned against nn/quant.py");
}
