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

// Boards per CTA. ⚠️ This is the whole of the two-board design: a CTA holding two
// boards is just `M = 64` for every weight matmul, because those matmuls are per-token
// and both boards read the *same* weights. One B-fragment load then feeds twice the
// arithmetic, which halves L2->SM weight traffic per board -- and that traffic is what
// `long_scoreboard` (3.62, the top stall since fp8 landed) is measuring. Attention is
// the exception and stays per board; it is 3.8 % of the budget.
// docs/journal/2026-08-14-int8-kernel-spec.md.
template <bool TWOB>
struct Lay {
    static constexpr int BPC = TWOB ? 2 : 1;      // boards per CTA
    static constexpr int TB = BPC * T;            // rows per CTA
    static constexpr int MT = TB / 16;            // m-tiles per weight GEMM
    static constexpr int SM_A = TB * AROWA;       // residual stream
    static constexpr int SM_B = TB * AROW;        // normed input / attention output
    static constexpr int SM_S = TB * AROW;        // V transpose, then FFN hidden
    static constexpr int HALVES = SM_A + SM_B + SM_S;
    static constexpr int BYTES = HALVES * (int)sizeof(half);
};
constexpr int SM_A = Lay<false>::SM_A;
constexpr int SM_B = Lay<false>::SM_B;
constexpr int SM_S = Lay<false>::SM_S;
constexpr int SMEM_HALVES = Lay<false>::HALVES;
static_assert(Lay<false>::BYTES <= 50176,
              "over 50,176 B the block stops fitting twice on an sm89 SM, which "
              "costs more than any use of the extra memory has been worth");
// ⚠️ Queried, not assumed: `cudaDevAttrMaxSharedMemoryPerBlockOptin` is 101,376 B on
// sm89 -- 99 KB, not the 102,400 of `MaxSharedMemoryPerMultiprocessor`. Two boards
// need 100,352, so they fit with 1,024 B to spare. The 2026-08-09 retrospective read
// that ceiling as the requirement and concluded this was blocked. It is not.
static_assert(Lay<true>::BYTES <= 101376,
              "two boards must fit one CTA's opt-in shared memory");

// §fp8. A parallel weight slab for the FFN only, in e4m3, plus its 128x128 block
// scales. docs/journal/2026-08-04-fp8-encoder.md measured 1.70x on the ff1 shape and
// 0.66 % max prior-space error for FFN-only quantisation; the other two matmuls stay
// fp16 because that is where three quarters of the error came from.
// ⚠️ `QATT` changes the **stride**, not just what is appended, so the host packer and
// the kernel must agree on it or layer 1's `w_ff1` lands on layer 0's tail -- bytes
// that are all in range and produce plausible logits. `cuda_impl._pack_int8` is the
// only writer and it takes the same flag.
template <bool QATT>
struct Fp8OffT {                                 // bytes, per layer
    static constexpr int w_ff1 = 0;              // [DFF][D], packed B-fragment order
    static constexpr int w_ff2 = w_ff1 + DFF * Dm;
    static constexpr int w_qkv = w_ff2 + Dm * DFF;   // [3D][D], QATT only
    static constexpr int w_o = w_qkv + 3 * Dm * Dm;  // [D][D],  QATT only
    static constexpr int stride = QATT ? w_o + Dm * Dm : w_qkv;
};
template <bool QATT>
struct Fp8SOffT {                                // floats, per layer
    // One scale per 128 output columns, covering the **whole** reduction -- the fp16
    // accumulator runs the full 256-deep GEMM, so every k-group must share a scale.
    static constexpr int s_ff1 = 0;              // [DFF/128]
    static constexpr int s_ff2 = s_ff1 + DFF / 128;
    static constexpr int s_qkv = s_ff2 + Dm / 128;
    static constexpr int s_o = s_qkv + 3 * Dm / 128;
    static constexpr int stride = QATT ? s_o + Dm / 128 : s_qkv;
};
using Fp8Off = Fp8OffT<false>;
using Fp8SOff = Fp8SOffT<false>;


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

/// Quantise a `[ROWS_TOTAL][WIDTH]` half buffer to int8 **in place**, one scale a row.
///
/// The three int8 GEMM inputs -- the normed layer input, the attention output and the
/// post-ReLU hidden chunk -- have identical `[rows][256]` geometry, so they share this
/// one call site rather than three transcriptions of it. `AROW`'s row padding holds the
/// scale; `quantise_row_int8` owns the in-place aliasing rules.
///
/// ⚠️ The caller supplies the `__syncthreads()` afterwards. Each warp quantises its own
/// rows, and every GEMM below reads *all* rows.
template <int WIDTH, bool UNSIGNED, int ROWS_TOTAL, int NW>
__device__ __forceinline__ void quantise_buf_int8(half* buf, int warp, int lane) {
    constexpr int ROWS = ROWS_TOTAL / NW;
    static_assert(ROWS * NW == ROWS_TOTAL, "the rows must divide over the warps");
    uint8_t* aq = reinterpret_cast<uint8_t*>(buf);
    float* as = reinterpret_cast<float*>(aq + fp8cfg::kScaleOff);
    const int r0 = warp * ROWS;
    int8q::quantise_row_int8<WIDTH, UNSIGNED, ROWS>(
        aq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
        as + (size_t)r0 * fp8cfg::kScaleStride, fp8cfg::kScaleStride,
        buf + (size_t)r0 * AROW, AROW, lane);
}

/// The `(bytes, scales)` pair `gemm_s8_row` wants, from a buffer `quantise_buf_int8`
/// has just written. Two casts that are easy to get subtly wrong and appear five times.
__device__ __forceinline__ const uint8_t* qbytes(const half* buf) {
    return reinterpret_cast<const uint8_t*>(buf);
}
__device__ __forceinline__ const float* qscales(const half* buf) {
    return reinterpret_cast<const float*>(qbytes(buf) + fp8cfg::kScaleOff);
}

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
// The value head starts here and is either one column (the scalar tanh head of spec
// §7.4) or three (win/draw/loss, in that order: 0 = loss, 1 = draw, 2 = win, from the
// side to move). ⚠️ **Three costs nothing.** The aux tile is padded from 5 to 32
// columns and warp 2 already computes all 32; the two extra columns come out of the
// 27 that were zeros, so there is no extra mma, no extra SMEM and no extra register.
constexpr int VALUE_COL = NPOL + 4;
constexpr int N_VALUE_MAX = 3;
static_assert(VALUE_COL + N_VALUE_MAX <= NHEAD, "the value rows have to fit the aux tile");
// 104 halves is 52 words, so eight rows sit 20 banks apart and 20 is coprime
// enough with 32 that all eight land in distinct banks -- same test as AROW.
constexpr int HDROW = NHEAD + 8;
static_assert(T * HDROW <= SM_S, "the head staging block has to fit inside scratch");

struct TailOff {
    static constexpr int lnf_w  = 0;
    static constexpr int lnf_b  = Dm;
    static constexpr int w_head = 2 * Dm;      // packed [NHEAD][Dm]
    // ⚠️ A second, **unpacked** `[N_VALUE_MAX][Dm]` copy of the value rows, 1.5 KB.
    // `w_head` is in `gemm_direct`'s fragment order, which is what makes the packed
    // head fast and what makes a plain dot product against three of its rows painful.
    // The prenorm path needs exactly that dot product, once per board, so it gets its
    // own row-major copy rather than a special case inside `pack_b`.
    static constexpr int w_val  = w_head + NHEAD * Dm;
    static constexpr int stride = w_val + N_VALUE_MAX * Dm;
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
// ⚠️ `MT` is **deduced from the accumulator** rather than passed, and that is safety
// rather than convenience: the same helpers serve a 32-row and a 64-row buffer in the
// same kernel, and a call looping four m-tiles over a two-tile array would read and
// write past it while staying inside shared memory and finite.
template <int K32N, int NK32, int MT>
__device__ __forceinline__ void gemm_direct(uint32_t (&acc)[MT][4][2], const half* a_base,
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
            uint32_t af[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; ++m)
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
                for (int m = 0; m < MT; ++m) {
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
// --------------------------------------------------------------------------
// Per-layer input re-injection (2026-08-28). A block's LayerNorm sees
// `h + sum_k c[site][k] * E_k` instead of `h`, where the five `E_k` are the embedding
// tables of spec 7.2 and `c` is a learned scalar per (site, source). The residual
// stream is untouched: this changes what the norm is *given*, not what the block adds.
//
// The whole cost question is which of the five tables is per token. Only three are:
// `square` (64 rows), `type_special` (12) and `color_turn` (4). `clock` and `rep` are
// indexed by the *board*, exactly as in `embed_board`, so they are the same vector for
// all 32 rows. And within one board `color_turn` takes only two of its four rows,
// because the side to move is fixed -- so it too can be read once per LayerNorm and
// selected per row by the token's colour bit.
//
// That is why the preamble below pre-mixes colour, clock and rep into **two slices per
// board**, `cs[colour]`, and the row loop gathers only `square` and `type_special`:
// two 16-byte loads per lane per row instead of five. Per board per site that is
// 32 rows x 2 x 512 B = 32 KB against 80 KB.
//
// ⚠️ The tables are read with `__ldg` and they are 40 KB in total for the three
// per-token ones -- shared by every CTA on the device and hit in L2 essentially always.
// That is the argument for gathering them again rather than materialising a per-board
// mix: the basis is smaller than any precomputed combination of it, and it is shared.
constexpr int kRjStride = 8;      // halves per site in the coefficient slab; 5 are used

struct RjCtx {
    const uint16_t* __restrict__ boards;
    const int16_t* __restrict__ control;
    const uint8_t* __restrict__ rep;
    const half* __restrict__ emb;
    //: `[site][kRjStride]` halves. **Null means this site does not inject**, which is
    //: how `--reinject ln1` skips `ln2` without a second kernel instantiation.
    const half* __restrict__ coef;
    int b0, nb;
};

__device__ __forceinline__ uint4 ldg16(const half* p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

template <bool AFFINE, int ROWS, bool RJ>
__device__ __forceinline__ void layernorm(half* dst, const half* src, const half* gamma,
                                          const half* beta, float eps, int warp, int lane,
                                          const RjCtx& rj, int site) {
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

    // -- re-injection preamble, loop-invariant over the rows below ---------
    constexpr int BPC = ROWS / T;
    static_assert(ROWS % T == 0, "a LayerNorm covers whole boards");
    static_assert(T % NWARPS == 0, "so that one unrolled iteration stays on one board");
    float c_sq = 0.f, c_ts = 0.f;
    uint32_t word0 = 0, word1 = 0;
    uint4 cs00 = {}, cs01 = {}, cs10 = {}, cs11 = {};
    const bool inject = RJ && rj.coef != nullptr && site >= 0;
    if constexpr (RJ) {
        if (inject) {
            const half* cc = rj.coef + (size_t)site * kRjStride;
            c_sq = __half2float(cc[0]);
            c_ts = __half2float(cc[1]);
            const float c_ct = __half2float(cc[2]);
            const float c_ck = __half2float(cc[3]);
            const float c_rp = __half2float(cc[4]);
#pragma unroll
            for (int bb = 0; bb < BPC; ++bb) {
                // Same clamps and the same duplicate-board rule as `embed_board`: an
                // odd batch's spare slot re-reads board `b0`, which is harmless
                // arithmetic whose outputs are never written.
                const int gb = rj.b0 + (bb < rj.nb ? bb : 0);
                const uint32_t w = rj.boards[(size_t)gb * T + lane];
                const int ctl = rj.control[gb];
                const int stm = ctl < 0;
                const int clk = min(max((ctl < 0 ? -ctl : ctl) - 1, 0), N_CLOCK - 1);
                const int rp = min((int)rj.rep[gb], N_REP - 1);
                const uint4 kc = ldg16(rj.emb + EmbOff::clock + clk * Dm + lane * 8);
                const uint4 kr = ldg16(rj.emb + EmbOff::rep + rp * Dm + lane * 8);
                const half* pc = reinterpret_cast<const half*>(&kc);
                const half* pr = reinterpret_cast<const half*>(&kr);
                uint4 mix[2];
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    const uint4 kt = ldg16(rj.emb + EmbOff::color_turn
                                           + (col * 2 + stm) * Dm + lane * 8);
                    const half* pt = reinterpret_cast<const half*>(&kt);
                    half o[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        o[j] = __float2half(c_ct * __half2float(pt[j])
                                            + (c_ck * __half2float(pc[j])
                                               + c_rp * __half2float(pr[j])));
                    mix[col] = *reinterpret_cast<const uint4*>(o);
                }
                // Named registers, not an array index: `bb` is a literal here because
                // BPC is, and a dynamically indexed register array is local memory.
                if (bb == 0) { word0 = w; cs00 = mix[0]; cs01 = mix[1]; }
                else         { word1 = w; cs10 = mix[0]; cs11 = mix[1]; }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < ROWS / NWARPS; ++i) {
        int row = warp + i * NWARPS;
        const half* p = src + ai_idx(row, lane * 8);
        uint4 raw = *reinterpret_cast<const uint4*>(p);
        const half* v = reinterpret_cast<const half*>(&raw);
        float sum = 0.f, sq = 0.f;
        // ⚠️ Held only on the re-injecting path. Without it the two passes below read
        // `v[j]` twice and the fp16 word is the only thing live across the shuffle
        // reduction; keeping eight floats there unconditionally would raise the whole
        // kernel's register demand for a feature that is off.
        float xm[RJ ? 8 : 1];
        if constexpr (RJ) {
            // `row` spans [i*NWARPS, i*NWARPS + NWARPS) and T is a multiple of NWARPS,
            // so every lane of this unrolled iteration is on the same board and the
            // compiler folds `bb` to a literal.
            const int bb = (i * NWARPS) / T;
            const uint32_t w = __shfl_sync(0xffffffffu, (BPC == 2 && bb) ? word1 : word0,
                                           row & (T - 1));
            // Exactly `embed_board`'s decode, including the clamp on `type`: 6 and 7
            // are reachable from a malformed word and unclamped would read a row of
            // `emb_clock`, in bounds by an accident of slab order.
            const int sq_i = w & 63;
            const int ts_i = min(((w >> 6) & 7) * 2 + ((w >> 9) & 1), N_TYPE_SPECIAL - 1);
            const int col = (w >> 10) & 1;
            const uint4 cs = (BPC == 2 && bb) ? (col ? cs11 : cs10)
                                              : (col ? cs01 : cs00);
            const half* pcs = reinterpret_cast<const half*>(&cs);
            // ⚠️ **Do not hoist this branch outward to skip the decode above.** Doing
            // exactly that, on the theory that a site which does not inject should not
            // pay for the shuffle, was measured on 2026-08-28 and **cost 0.7 % on `ln1`
            // and 3.0 % on `both`** (`logs/reinject-ab-branch-guarded.log`), while
            // taking the shipped arm from 250 to 254 registers. A branch around the two
            // loads is a scheduling barrier: ptxas can no longer lift them out of the
            // unrolled row loop, and their L2 latency stops hiding behind the
            // shuffle reduction of the previous row. The dead decode is cheaper than
            // the lost pipelining.
            uint4 es = {}, et = {};
            if (inject) {
                es = ldg16(rj.emb + EmbOff::square + sq_i * Dm + lane * 8);
                et = ldg16(rj.emb + EmbOff::type_special + ts_i * Dm + lane * 8);
            }
            const half* pe = reinterpret_cast<const half*>(&es);
            const half* pt = reinterpret_cast<const half*>(&et);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                // The grouping mirrors `BrokefishNet.mix`:
                // (square + type_special + color_turn) + (clock + rep), with the last
                // two folded into `cs` above. At c = 0 every added term is an exact
                // zero, so a re-injecting kernel with untrained coefficients is the
                // old kernel to the bit.
                xm[j] = __half2float(v[j])
                        + (c_sq * __half2float(pe[j]) + c_ts * __half2float(pt[j]))
                        + __half2float(pcs[j]);
                sum += xm[j];
                sq += xm[j] * xm[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float x = __half2float(v[j]);
                sum += x;
                sq += x * x;
            }
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
            const float xn = ((RJ ? xm[j] : __half2float(v[j])) - mean) * rstd;
            if constexpr (AFFINE)
                out[j] = __float2half(xn * __half2float(gv[j]) + __half2float(bv[j]));
            else
                out[j] = __float2half(xn);
        }
        *reinterpret_cast<uint4*>(dst + a_idx(row, lane * 8)) = *reinterpret_cast<uint4*>(out);
    }
}

/// The non-injecting form, so that `norm_f` and every pre-B2 caller keep the signature
/// they had. `RJ = false` deletes every reference to the context before ptxas sees it.
template <bool AFFINE, int ROWS>
__device__ __forceinline__ void layernorm(half* dst, const half* src, const half* gamma,
                                          const half* beta, float eps, int warp, int lane) {
    RjCtx none{};
    layernorm<AFFINE, ROWS, false>(dst, src, gamma, beta, eps, warp, lane, none, 0);
}

// Add a bias to a [2][4] block of D fragments. Element (m, n) of the fragment
// covers rows 16m + {g, g+8} and columns 8n + {2t, 2t+1}; the bias depends only
// on the column, so both halves of both registers take the same pair.
template <int MT>
__device__ __forceinline__ void add_bias(uint32_t (&acc)[MT][4][2], const half* bias,
                                         int col0, int lane) {
    int t = lane & 3;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        __half2 b = *reinterpret_cast<const __half2*>(bias + col0 + n * 8 + t * 2);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            acc[m][n][0] = u32(__hadd2(h2(acc[m][n][0]), b));
            acc[m][n][1] = u32(__hadd2(h2(acc[m][n][1]), b));
        }
    }
}

// Scatter a [2][4] D-fragment block into an SMEM activation buffer at column
// offset col0.
template <int MT>
__device__ __forceinline__ void store_frags(half* dst, int stride,
                                            const uint32_t (&acc)[MT][4][2],
                                            int col0, int lane) {
    int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < MT; ++m)
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
template <int ROWS>
__device__ __forceinline__ void residual_rows(half* x, const half* src, int warp, int lane) {
#pragma unroll
    for (int i = 0; i < ROWS / NWARPS; ++i) {
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

template <int MT>
__device__ __forceinline__ void zero_frags(uint32_t (&acc)[MT][4][2]) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
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
// One board's two m-tiles out of a CTA's `MT`. ⚠️ Attention mixes tokens *within* a
// board -- its score matrix is [32,32] per head -- so it is the one stage that cannot
// see 64 rows as one taller problem. The cast is on a pointer to the correct sub-block
// and yields a properly sized array reference, so the helpers inside `attention` still
// deduce two m-tiles and still cannot run off the end.
template <int MT>
__device__ __forceinline__ uint32_t (&board_tiles(uint32_t (&a)[MT][4][2], int bb))[2][4][2] {
    static_assert(MT % 2 == 0, "m-tiles come in pairs, one board each");
    return *reinterpret_cast<uint32_t (*)[2][4][2]>(&a[2 * bb]);
}

__device__ __forceinline__ void attention(uint32_t (&o)[2][4][2],
                                          const uint32_t (&q)[2][4][2],
                                          const uint32_t (&k)[2][4][2],
                                          const uint32_t (&v)[2][4][2],
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
                                           int n_value, int vmode,
                                           const uint16_t* __restrict__ boards,
                                           int board, int warp, int lane, int tid) {
    // Two independent reasons this norm is not optional. The stack is pre-norm,
    // so what falls out of it is a raw residual stream whose scale grows with
    // depth, in fp16, and a linear head on top of that is the classic pre-norm
    // mistake. And bufA is the one buffer carrying no row padding -- that missing
    // 512 B is what bought the second CTA per SM -- so a head GEMM reading it
    // through ldmatrix would collide eight ways. The norm lands the stream in
    // bufB, which is padded, and both problems go away at once.
    layernorm<true, T>(bufB, bufA, tail + TailOff::lnf_w, tail + TailOff::lnf_b, eps,
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
    // The value is a row select decided by the sign of the control word, then a
    // squash in fp32. Doing it here rather than shipping the whole [N,32,8] aux
    // tensor saves every consumer in the search a data-dependent gather of two
    // bytes out of every 512, on every backup.
    //
    // ⚠️ **Both heads leave through the same [N] fp32 in [-1, 1]**, which is why the
    // search, the terminal collapse and the whole Elo pipeline never learned that a
    // second head exists. `n_value == 1` is `tanh(x)`, unchanged instruction for
    // instruction. `n_value == 3` is a 3-way softmax collapsed to `p(win) - p(loss)`;
    // it must match `nn/model.py:wdl_to_scalar` and does, including the max
    // subtraction, which is what torch's softmax does too.
    //
    // `expf` and not `__expf`: this is one thread, once per board, in an epilogue that
    // is 0.3 % of the network, so the fast intrinsic would buy nothing measurable and
    // spend accuracy against the torch oracle that `tests/test_b2.py` compares to.
    //
    // ⚠️ **`vmode == 1` averages the head's own output over the live tokens, not a second
    // GEMM on a pooled residual.** `norm_f` is upstream of the pool and the value head
    // is biasless, so `W @ mean(hn) == mean(W @ hn)` exactly -- and the aux columns for
    // all 32 tokens are already sitting in `scratch`, computed by warp 2 on every
    // forward. So the pool is 32 reads by one thread in an epilogue that is 0.3 % of
    // the network, and costs no mma, no SMEM and no register. In fp16 the two orders
    // are different numbers, which is what `tests/test_b2.py`'s tolerance covers.
    //
    // ⚠️ **Dead slots are skipped.** A captured slot is `1 << 11` with colour, type and
    // square wiped, so it decodes as a live white pawn on a1 (CLAUDE.md) and its head
    // output is a real vector that means nothing. Averaging it in would make the value
    // track how many pieces have been taken, by an accident of the encoding.
    //
    // ⚠️ **The pooled head predicts White's frame**, because nothing about a symmetric
    // mean says whose turn it is. The mover's frame is one flip on the sign of the
    // control word (spec §2.4: magnitude is the clock, so the sign never vanishes),
    // and `search.cuh:1076` keeps consuming a mover-relative `[-1, 1]` either way.
    // ⚠️ **`vmode == 2` is the one that is not free.** `LN(mean(h))` is not a linear
    // function of the per-token value logits, so there is nothing in `scratch` to
    // average -- it pools `bufA`, the *raw* residual, norms that once, and takes three
    // dot products. One warp, ~300 ops a lane, in an epilogue that is 0.3 % of the
    // network. What it buys is a head input whose scale is fixed by the norm: measured
    // on `t12h-wdb`, `|mean(LN(h))|` drifts 15.47 -> 10.86 over a run and the head's
    // weights grow 42 % chasing it. Pooling first also lets a token with a larger
    // residual count for more, which an unweighted mean of normed tokens cannot.
    if (value && vmode == 2 && warp == 0) {
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
        int live = 0;
        for (int t = 0; t < T; ++t) {
            if ((boards[(size_t)board * T + t] >> 11) & 1) continue;   // captured
            const uint4 r4 = *reinterpret_cast<const uint4*>(bufA + ai_idx(t, lane * 8));
            const half* rv = reinterpret_cast<const half*>(&r4);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += __half2float(rv[j]);
            ++live;
        }
        const float inv = 1.0f / (float)(live > 0 ? live : 1);
        float sum = 0.0f, sq = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc[j] *= inv; sum += acc[j]; sq += acc[j] * acc[j]; }
#pragma unroll
        for (int off = 16; off; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off);
            sq += __shfl_xor_sync(0xffffffff, sq, off);
        }
        const float mu = sum / Dm;
        // Same clamped one-pass variance as `layernorm`, and for the same reason.
        const float rstd = rsqrtf(fmaxf(sq / Dm - mu * mu, 0.0f) + eps);
        const uint4 g4 = *reinterpret_cast<const uint4*>(tail + TailOff::lnf_w + lane * 8);
        const uint4 b4 = *reinterpret_cast<const uint4*>(tail + TailOff::lnf_b + lane * 8);
        const half* gv = reinterpret_cast<const half*>(&g4);
        const half* bv = reinterpret_cast<const half*>(&b4);
        float nv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            nv[j] = (acc[j] - mu) * rstd * __half2float(gv[j]) + __half2float(bv[j]);
        float o[N_VALUE_MAX] = {0.0f, 0.0f, 0.0f};
        for (int k = 0; k < n_value; ++k) {
            const uint4 w4 = *reinterpret_cast<const uint4*>(
                tail + TailOff::w_val + k * Dm + lane * 8);
            const half* wv = reinterpret_cast<const half*>(&w4);
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) d += nv[j] * __half2float(wv[j]);
#pragma unroll
            for (int off = 16; off; off >>= 1) d += __shfl_xor_sync(0xffffffff, d, off);
            o[k] = d;
        }
        if (lane == 0) {
            float v;
            if (n_value == 1) {
                v = tanhf(o[0]);
            } else {
                const float m = fmaxf(o[0], fmaxf(o[1], o[2]));
                const float ea = expf(o[0] - m), eb = expf(o[1] - m), ec = expf(o[2] - m);
                v = (ec - ea) / (ea + eb + ec);
            }
            if (control[board] < 0) v = -v;      // absolute frame -> the mover's
            value[board] = v;
        }
    }
    if (value && vmode != 2 && tid == 0) {
        float a = 0.0f, b = 0.0f, c = 0.0f;
        if (vmode == 0) {
            const half* row = scratch + (control[board] < 0 ? 31 : 15) * HDROW + VALUE_COL;
            a = __half2float(row[0]);                       // loss  (or the scalar)
            if (n_value != 1) {
                b = __half2float(row[1]);                   // draw
                c = __half2float(row[2]);                   // win
            }
        } else {
            int live = 0;
            for (int t = 0; t < T; ++t) {
                if ((boards[(size_t)board * T + t] >> 11) & 1) continue;   // captured
                const half* row = scratch + t * HDROW + VALUE_COL;
                a += __half2float(row[0]);                  // Black  (or the scalar)
                if (n_value != 1) {
                    b += __half2float(row[1]);              // draw
                    c += __half2float(row[2]);              // White
                }
                ++live;
            }
            // Two kings are never captured (spec §2.5), so `live >= 2`; the guard is
            // for the debug paths that hand this a zeroed board.
            const float inv = 1.0f / (float)(live > 0 ? live : 1);
            a *= inv; b *= inv; c *= inv;
        }
        float v;
        if (n_value == 1) {
            v = tanhf(a);
        } else {
            const float m = fmaxf(a, fmaxf(b, c));
            const float ea = expf(a - m), eb = expf(b - m), ec = expf(c - m);
            v = (ec - ea) / (ea + eb + ec);
        }
        if (vmode && control[board] < 0) v = -v;
        value[board] = v;
    }
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
    kQuantQ,        // quantise_row over the normed input          (QATT only)
    kQkvGemm,       // three [32,256]x[256,256]
    kQkvBias,       // three add_bias over the fragments
    kAttention,     // V transpose, scores, softmax, AV -- warp-local, no barrier
    kAttnStore,     // store_frags(o) between two barriers
    kQuantO,        // quantise_row over the attention output      (QATT only)
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
// ⚠️ `TWOB` carries its own `__launch_bounds__`: two boards need 100,352 B of shared
// memory, which is one CTA per SM, and at one CTA the register file divides by 256
// threads instead of 512 -- 255 per thread against 128. That is what pays for the s32
// accumulator int8 needs, and it is why the format change and the occupancy change are
// one design. Step 0 measured the combination at 223 registers with **zero spill**.
// `QATT` extends `INT8` from the FFN's two matmuls to all four, adding packed QKV and
// the attention output projection. Measured in emulation on three networks before a
// line of it was written (`nn/quant.py --targets attn --format e4m3 int8`): e4m3 costs
// 6.8e-2 max |dp| at that site against int8's 2.3e-2, so this is an integer path and
// there is no fp8 variant of it. All four matmuls in int8 measure 2.8e-2 / 3.1 % flip,
// **below** the e4m3 FFN-only path we trained `t24h-fp8` on (4.0e-2 / 4.8 %).
// The two extra quantisations are free of new shared memory: the normed input and the
// attention output have the same [TB][256] geometry the FFN's input already has, so
// `fp8cfg`'s row layout is reused verbatim and both quantise in place.
// ⚠️ `QOUT` is separate from `QATT` because the two matmuls are not the same trade.
// QKV is **25 %** of the body's FLOPs and out_proj is **8.3 %**, while out_proj costs
// its own quantisation pass of the attention output -- so the question "is out_proj
// worth its precision" is a real one and is answered by measuring both, not by
// assuming the FLOP share carries over. `QOUT` implies `QATT`: quantising the
// projection while leaving QKV in fp16 is the strictly worse half of the trade.
template <bool FP8, bool INT8, bool TWOB, bool PROF = false, bool QATT = false,
          bool QOUT = QATT, bool RJ = false>
__global__ __launch_bounds__(THREADS, TWOB ? 1 : 2) void encoder_kernel(
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
    int n_layers, float eps, int debug_stage, int n_boards, int n_value, int vmode,
    // Per-layer input re-injection. `rj_coef` is `[sites][kRjStride]` halves and
    // `rj_mode` is 1 for the attention norm alone, 2 for both norms of a block; the
    // template flag is what decides whether any of it is compiled in at all.
    const half* __restrict__ rj_coef = nullptr, int rj_mode = 0) {
    static_assert(!(FP8 && INT8), "the FFN is quantised once, in one format");
    static_assert(!QATT || INT8, "QKVO quantisation is int8 only; e4m3 was measured "
                                 "3x worse at that site before it was built");
    static_assert(!QOUT || QATT, "out_proj in int8 with QKV in fp16 is the worse half "
                                 "of the trade: a third of the FLOPs for its own "
                                 "quantisation pass");
    // Both quantised paths read the same byte slab and the same scale slab.
    constexpr bool Q8 = FP8 || INT8;
    using QOff = Fp8OffT<QATT>;
    using QSOff = Fp8SOffT<QATT>;
    using L = Lay<TWOB>;
    constexpr int BPC = L::BPC, TB = L::TB, MT = L::MT;
    extern __shared__ half smem[];
    half* bufA = smem;                    // residual stream, TB rows
    half* bufB = bufA + L::SM_A;          // normed input, then attention output
    // One scratch buffer serves both roles: the V transpose during attention
    // and the FFN hidden chunk afterwards. Their lifetimes do not overlap, and
    // sharing the allocation is what brings the block down to 49.5 KB.
    half* scratch = bufB + L::SM_B;
    half* hid = scratch;
    half* vbuf = scratch;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b0 = blockIdx.x * BPC;
    // ⚠️ An odd batch leaves the last CTA with one real board. The spare slot reads
    // board `b0` again -- a duplicate is harmless arithmetic -- and its outputs are
    // simply never written. Clamping the *input* rather than skipping the work keeps
    // the whole kernel branch-free below this point, which matters because every
    // barrier in it is a full-CTA barrier.
    const int nb = min(BPC, n_boards - b0);

    // Declared before the first goto: jumping over an initialised declaration in the
    // same scope does not compile, and the debug stages jump to `dump`.
    prof::Timer<PROF> tm(b0, warp, lane);

    // One re-injection context for the whole stack: the pointers and the CTA's board
    // window do not move between layers, only the site index does. Same reason as
    // `tm` for living up here.
    const RjCtx rj{boards, control, rep_ptr, emb, RJ ? rj_coef : nullptr, b0, nb};
    const bool rj_both = RJ && rj_mode == 2;

    // One 32-bit word per board says which slots hold a live piece (spec 7.3). Dead
    // keys leave the softmax as -inf; dead rows are computed and never read.
    uint32_t alive[BPC];
#pragma unroll
    for (int bb = 0; bb < BPC; ++bb) {
        const int gb = b0 + (bb < nb ? bb : 0);
        half* aB = bufA + (size_t)bb * T * AROWA;
        if (boards) {
            alive[bb] = embed_board(aB, boards, control, rep_ptr, emb, gb, warp, lane);
        } else {
            alive[bb] = 0xffffffffu;
            if (alive_ptr)
                // One coalesced byte per lane and a ballot, not 32 scalar loads per
                // thread. T == warpSize is what makes the ballot the whole reduction.
                alive[bb] = __ballot_sync(0xffffffffu,
                                          alive_ptr[(size_t)gb * T + lane] != 0);
            const half* g = y + (size_t)gb * T * Dm;
#pragma unroll
            for (int i = 0; i < T / NWARPS; ++i) {
                int row = warp + i * NWARPS;
                *reinterpret_cast<uint4*>(aB + ai_idx(row, lane * 8)) =
                    *reinterpret_cast<const uint4*>(g + row * Dm + lane * 8);
            }
        }
    }
    __syncthreads();
    tm.mark(prof::kPrologue);
    if (debug_stage == 9) goto dump;      // the gather alone, nothing else run

    // One context for the whole stack: the pointers and the CTA's board window do not
    // move between layers, only the site index does.
    for (int layer = 0; layer < n_layers; ++layer) {
        const half* W = weights + (size_t)layer * Off::stride;
        const uint8_t* Wq = Q8 ? wq8 + (size_t)layer * QOff::stride : nullptr;
        const float* Sq = Q8 ? sq8 + (size_t)layer * QSOff::stride : nullptr;

        // The affine slots hold ones and zeros: gamma lives in w_qkv's input axis and
        // beta in b_qkv, folded there by `FusedEncoder._fold_norm`.
        layernorm<false, TB, RJ>(bufB, bufA, W + Off::ln1_w, W + Off::ln1_b, eps,
                                 warp, lane, rj, rj_both ? layer * 2 : layer);

        if (debug_stage == 1) { __syncthreads(); goto dump; }
        // Cross-warp handoff. layernorm writes bufB by row (warp w owns rows
        // w, w+8, w+16, w+24) and store_frags writes it by column (warp w owns
        // columns 32w..32w+31), while every GEMM below reads the whole buffer.
        // gemm_full used to open with a barrier of its own, which hid this
        // dependency; gemm_direct has none, so it has to be stated.
        __syncthreads();
        tm.mark(prof::kLn1);

        uint32_t q[MT][4][2], k[MT][4][2], v[MT][4][2];
        zero_frags(q);
        zero_frags(k);
        zero_frags(v);
        // w_qkv is one packed [768][256]: Q occupies n-tiles 0..31, K 32..63,
        // V 64..95, and warp w takes four consecutive n-tiles of each.
        const half* wq = W + Off::w_qkv;
        constexpr int NK = Dm / 32;                  // 8 k-groups of 32
        if constexpr (QATT) {
            // The normed input is signed and is read by nothing else, so it is
            // quantised over the top of itself exactly as the FFN's input is. One
            // scale for the whole 256-wide row serves all three of Q, K and V --
            // they contract the same row, so there is one A operand, not three.
            quantise_buf_int8<Dm, /*UNSIGNED=*/false, TB, NWARPS>(bufB, warp, lane);
            __syncthreads();
            tm.mark(prof::kQuantQ);
            const uint2* wqi = reinterpret_cast<const uint2*>(Wq + QOff::w_qkv);
            const float* sqi = Sq + QSOff::s_qkv;
            int8q::gemm_s8_row<NK, NK, /*ADD=*/false, /*UNSIGNED_A=*/false>(
                q, qbytes(bufB), fp8cfg::kAPitch, qscales(bufB),
                fp8cfg::kScaleStride, wqi, sqi, warp * 4, 0, lane);
            int8q::gemm_s8_row<NK, NK, false, false>(
                k, qbytes(bufB), fp8cfg::kAPitch, qscales(bufB),
                fp8cfg::kScaleStride, wqi, sqi, 32 + warp * 4, 0, lane);
            int8q::gemm_s8_row<NK, NK, false, false>(
                v, qbytes(bufB), fp8cfg::kAPitch, qscales(bufB),
                fp8cfg::kScaleStride, wqi, sqi, 64 + warp * 4, 0, lane);
        } else {
            gemm_direct<NK, NK>(q, bufB, AROW, wq, warp * 4,      0, lane);
            gemm_direct<NK, NK>(k, bufB, AROW, wq, 32 + warp * 4, 0, lane);
            gemm_direct<NK, NK>(v, bufB, AROW, wq, 64 + warp * 4, 0, lane);
        }
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

        uint32_t o[MT][4][2];
#pragma unroll
        for (int bb = 0; bb < BPC; ++bb)
            attention(board_tiles(o, bb), board_tiles(q, bb), board_tiles(k, bb),
                      board_tiles(v, bb), vbuf + (size_t)bb * T * AROW, warp * DH,
                      alive[bb], lane);
        tm.mark(prof::kAttention);

        __syncthreads();
        store_frags(bufB, AROW, o, warp * DH, lane);
        if (debug_stage == 4) { __syncthreads(); goto dump; }
        __syncthreads();
        tm.mark(prof::kAttnStore);

        uint32_t proj[MT][4][2];
        zero_frags(proj);
        if constexpr (QOUT) {
            // ⚠️ The attention output is signed, unlike the FFN's post-ReLU hidden.
            // `dot(p, v)` is a convex combination of V rows and V is signed, so the
            // unsigned path would clamp every negative channel to zero -- a wrong
            // answer that stays finite and looks like a bad checkpoint.
            quantise_buf_int8<Dm, /*UNSIGNED=*/false, TB, NWARPS>(bufB, warp, lane);
            __syncthreads();
            tm.mark(prof::kQuantO);
            int8q::gemm_s8_row<NK, NK, /*ADD=*/false, /*UNSIGNED_A=*/false>(
                proj, qbytes(bufB), fp8cfg::kAPitch, qscales(bufB),
                fp8cfg::kScaleStride, reinterpret_cast<const uint2*>(Wq + QOff::w_o),
                Sq + QSOff::s_o, warp * 4, 0, lane);
        } else {
            gemm_direct<NK, NK>(proj, bufB, AROW, W + Off::w_o, warp * 4, 0, lane);
        }
        tm.mark(prof::kProjGemm);
        add_bias(proj, W + Off::b_o, warp * DH, lane);
        // scratch is dead here (it held the V transpose, and every warp passed
        // the barrier above), so it takes the staged projection.
        store_frags(scratch, AROW, proj, warp * DH, lane);
        __syncthreads();
        tm.mark(prof::kProjEpi);
        residual_rows<TB>(bufA, scratch, warp, lane);

        if (debug_stage == 2) goto dump;
        layernorm<false, TB, RJ>(bufB, bufA, W + Off::ln2_w, W + Off::ln2_b, eps,
                                 warp, lane, rj, rj_both ? layer * 2 + 1 : -1);
        if (debug_stage == 3) { __syncthreads(); goto dump; }
        __syncthreads();
        tm.mark(prof::kRes1Ln2);

        uint32_t ff[MT][4][2];
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
            const int r0 = warp * (TB / NWARPS);
            if constexpr (INT8)
                int8q::quantise_row_int8<Dm, /*UNSIGNED=*/false, TB / NWARPS>(
                    aq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                    as + (size_t)r0 * fp8cfg::kScaleStride, fp8cfg::kScaleStride,
                    bufB + (size_t)r0 * AROW, AROW, lane);
            else
                fp8::quantise_row<Dm>(aq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                                      as + (size_t)r0 * fp8cfg::kScaleStride,
                                      fp8cfg::kScaleStride,
                                      bufB + (size_t)r0 * AROW, AROW, TB / NWARPS, lane);
            __syncthreads();
            tm.mark(prof::kQuantA);
        }
        for (int c = 0; c < DFF / HCHUNK; ++c) {
            uint32_t hacc[MT][4][2];
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
                const int r0 = warp * (TB / NWARPS);
                if constexpr (INT8)
                    int8q::quantise_row_int8<HCHUNK, /*UNSIGNED=*/true, TB / NWARPS>(
                        hq + (size_t)r0 * fp8cfg::kAPitch, fp8cfg::kAPitch,
                        hs + (size_t)r0 * fp8cfg::kScaleStride, fp8cfg::kScaleStride,
                        hid + (size_t)r0 * HROW, HROW, lane);
                else
                    fp8::quantise_row<HCHUNK>(hq + (size_t)r0 * fp8cfg::kAPitch,
                                              fp8cfg::kAPitch,
                                              hs + (size_t)r0 * fp8cfg::kScaleStride,
                                              fp8cfg::kScaleStride,
                                              hid + (size_t)r0 * HROW, HROW,
                                              TB / NWARPS, lane);
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
        residual_rows<TB>(bufA, bufB, warp, lane);
        tm.mark(prof::kRes2);
    }

    // --- B2 epilogue: the final LayerNorm, then the three heads -------------
    if (policy || debug_stage == 8) {
#pragma unroll
        for (int bb = 0; bb < BPC; ++bb) {
            // ⚠️ Guarded, unlike the prologue: a duplicated board may be *computed*
            // but must never be *written*, or an odd batch would have the last board's
            // logits stored twice and the tail of `policy` would look plausible.
            if (bb < nb)
                tail_epilogue(bufA + (size_t)bb * T * AROWA, bufB + (size_t)bb * T * AROW,
                              scratch + (size_t)bb * T * AROW, tail, control, policy,
                              promo, value, eps, n_value, vmode, boards,
                              b0 + bb, warp, lane, tid);
            __syncthreads();
        }
    }
    tm.mark(prof::kEpilogue);
    if (debug_stage == 8) goto dump;
    if (policy) return;

dump:
    if (!y) return;
#pragma unroll
    for (int bb = 0; bb < BPC; ++bb) {
        if (bb >= nb) break;
        half* g = y + (size_t)(b0 + bb) * T * Dm;
        const half* aB = bufA + (size_t)bb * T * AROWA;
        const half* bB = bufB + (size_t)bb * T * AROW;
#pragma unroll
        for (int i = 0; i < T / NWARPS; ++i) {
            int row = warp + i * NWARPS;
            // Stage 9 is the embedding gather, which lands in bufA; 8 is the final
            // norm, which lands in bufB like the other post-norm stages.
            bool fromB = (debug_stage == 1 || debug_stage == 3
                          || (debug_stage >= 4 && debug_stage <= 8));
            const half* srcbuf = fromB ? bB + a_idx(row, lane * 8)
                                       : aB + ai_idx(row, lane * 8);
            *reinterpret_cast<uint4*>(g + row * Dm + lane * 8) =
                *reinterpret_cast<const uint4*>(srcbuf);
        }
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
template <bool FP8, bool INT8, bool TWOB, bool PROF, bool QATT = false,
          bool QOUT = QATT, bool RJ = false>
void launch_one(int n_boards, size_t smem, half* y, const half* weights,
                const uint8_t* wq8, const float* sq8, const int8_t* alive,
                const uint16_t* boards, const int16_t* control, const uint8_t* rep,
                const half* emb, const half* tail, half* policy, half* promo,
                float* value, int n_layers, float eps, int debug_stage, int n_value,
                int vmode, const half* rj_coef = nullptr, int rj_mode = 0) {
    constexpr bool Q8 = FP8 || INT8;
    constexpr int BPC = brokefish::Lay<TWOB>::BPC;
    const int grid = (n_boards + BPC - 1) / BPC;
    cudaFuncSetAttribute(brokefish::encoder_kernel<FP8, INT8, TWOB, PROF, QATT, QOUT, RJ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    brokefish::encoder_kernel<FP8, INT8, TWOB, PROF, QATT, QOUT, RJ>
        <<<grid, brokefish::THREADS, smem>>>(
        y, weights, Q8 ? wq8 : nullptr, Q8 ? sq8 : nullptr, alive, boards, control,
        rep, emb, tail, policy, promo, value, n_layers, eps, debug_stage, n_boards,
        n_value, vmode, RJ ? rj_coef : nullptr, RJ ? rj_mode : 0);
}

// `quant`: 0 = fp16, 1 = e4m3, 2 = int8. ⚠️ An explicit mode rather than a property of
// the pointers, because the int8 and e4m3 slabs have **identical** layout and size --
// int8 bytes read as e4m3 stay in range and produce plausible logits, so a
// pointer-derived switch would fail silently. The fp16/quantised distinction keeps its
// old all-or-nothing pointer check on top.
void launch(int n_boards, int quant, bool twob, half* y, const half* weights,
            const uint8_t* wq8,
            const float* sq8, const int8_t* alive,
            const uint16_t* boards, const int16_t* control, const uint8_t* rep,
            const half* emb, const half* tail, half* policy, half* promo, float* value,
            int n_layers, float eps, int debug_stage, int n_value = 1,
            int vmode = 0, const half* rj_coef = nullptr, int rj_mode = 0) {
    const size_t smem = twob ? brokefish::Lay<true>::BYTES : brokefish::Lay<false>::BYTES;
    const bool have_slab = (wq8 != nullptr) && (sq8 != nullptr);
    TORCH_CHECK(quant == 0 || have_slab,
                "quant mode ", quant, " needs both the byte slab and the scale slab; "
                "half a slab reads the wrong matrix and stays in range");
    // ⚠️ Only the combinations that exist are instantiated. e4m3 has no two-board
    // variant: int8 is better on both axes, so a two-board e4m3 kernel would be a
    // configuration nobody would run, paid for in compile time on every build.
#define BROKEFISH_LAUNCH(F, I, W, P)                                                   \
    launch_one<F, I, W, P>(n_boards, smem, y, weights, wq8, sq8, alive, boards,        \
                           control, rep, emb, tail, policy, promo, value, n_layers,    \
                           eps, debug_stage, n_value, vmode)
#define BROKEFISH_LAUNCH_Q(F, I, W, P, O)                                              \
    launch_one<F, I, W, P, true, O>(n_boards, smem, y, weights, wq8, sq8, alive,       \
                                    boards, control, rep, emb, tail, policy, promo,    \
                                    value, n_layers, eps, debug_stage, n_value, vmode)
    // quant 3 is quant 2 plus both attention matmuls, quant 4 is quant 2 plus QKV
    // alone. Both exist only at two boards per CTA, because that is the only occupancy
    // int8's s32 accumulators fit at and a one-board arm is a configuration nobody
    // would run. They share a slab: mode 4 carries `w_o`'s slot unused, 64 KB a layer,
    // which is the price of one stride instead of two.
    // ⚠️ **Re-injection gets its own arm and only two configurations.** It doubles
    // every instantiation it touches, and the two that matter are the one every number
    // is measured in (int8, two boards) and the fp16 one-board reference the
    // correctness tests and `nn/validate.py` run through. A third would cost build time
    // for a configuration nobody would run, which is the same rule the quant arms below
    // already follow.
    if (rj_coef != nullptr) {
        TORCH_CHECK(boards != nullptr,
                    "input re-injection needs the board words at every layer, so it "
                    "runs only on the boards-to-logits path, not on the activation-in "
                    "entry point");
        TORCH_CHECK(rj_mode == 1 || rj_mode == 2,
                    "rj_mode is 1 (the attention norm) or 2 (both norms of a block), "
                    "got ", rj_mode);
        if (quant == 3 && twob) {
            launch_one<false, true, true, false, true, true, true>(
                n_boards, smem, y, weights, wq8, sq8, alive, boards, control, rep, emb,
                tail, policy, promo, value, n_layers, eps, debug_stage, n_value, vmode,
                rj_coef, rj_mode);
        } else if (quant == 0 && !twob) {
            launch_one<false, false, false, false, false, false, true>(
                n_boards, smem, y, weights, wq8, sq8, alive, boards, control, rep, emb,
                tail, policy, promo, value, n_layers, eps, debug_stage, n_value, vmode,
                rj_coef, rj_mode);
        } else {
            TORCH_CHECK(false, "input re-injection is built for quant 3 with two "
                               "boards per CTA and for quant 0 with one; got quant ",
                        quant, ", twob ", (int)twob);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    if (quant == 3 || quant == 4) {
        TORCH_CHECK(twob, "quant ", quant, " is built for two boards per CTA");
        const bool qout = quant == 3;
        if (g_profile) { if (qout) BROKEFISH_LAUNCH_Q(false, true, true, true, true);
                         else      BROKEFISH_LAUNCH_Q(false, true, true, true, false); }
        else           { if (qout) BROKEFISH_LAUNCH_Q(false, true, true, false, true);
                         else      BROKEFISH_LAUNCH_Q(false, true, true, false, false); }
    } else if (quant == 2 && twob) {
        if (g_profile) BROKEFISH_LAUNCH(false, true, true, true);
        else           BROKEFISH_LAUNCH(false, true, true, false);
    } else if (quant == 2) {
        if (g_profile) BROKEFISH_LAUNCH(false, true, false, true);
        else           BROKEFISH_LAUNCH(false, true, false, false);
    } else if (quant == 1) {
        if (g_profile) BROKEFISH_LAUNCH(true, false, false, true);
        else           BROKEFISH_LAUNCH(true, false, false, false);
    } else if (twob) {
        if (g_profile) BROKEFISH_LAUNCH(false, false, true, true);
        else           BROKEFISH_LAUNCH(false, false, true, false);
    } else {
        if (g_profile) BROKEFISH_LAUNCH(false, false, false, true);
        else           BROKEFISH_LAUNCH(false, false, false, false);
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
    launch((int)y.size(0), /*quant=*/0, /*twob=*/false, ptr_or_null<half>(y),
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
                   torch::Tensor wq8, torch::Tensor sq8, int64_t quant, int64_t twob,
                   int64_t n_value, int64_t vmode, torch::Tensor rj_coef,
                   int64_t rj_mode) {
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
    // ⚠️ The output is [n] either way, so a wrong `n_value` reads two head columns
    // that exist and are in range and produces a plausible value. Nothing downstream
    // can catch it; this check is the only place it can be caught.
    TORCH_CHECK(n_value == 1 || n_value == brokefish::N_VALUE_MAX,
                "n_value is 1 for the scalar tanh head or 3 for win/draw/loss; got ",
                n_value);
    TORCH_CHECK(vmode >= 0 && vmode <= 2,
                "vmode is 0 for the king row select, 1 for the mean of the normed "
                "tokens, 2 for norm_f applied to the mean of the raw residual; got ",
                vmode);
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
    TORCH_CHECK(quant >= 0 && quant <= 4,
                "quant is 0 fp16, 1 e4m3, 2 int8 FFN, 3 int8 all four, 4 int8 FFN+QKV; "
                "got ", quant);
    // ⚠️ The stride is a function of the mode, and this check is the only thing that
    // catches a mismatch: mode 3 reading a mode-2 slab would find layer 1's weights a
    // third of a layer early, every byte in range, and produce plausible logits.
    if (wq8.numel()) {
        const int64_t wstride = quant >= 3 ? brokefish::Fp8OffT<true>::stride
                                           : brokefish::Fp8Off::stride;
        const int64_t sstride = quant >= 3 ? brokefish::Fp8SOffT<true>::stride
                                           : brokefish::Fp8SOff::stride;
        TORCH_CHECK(wq8.numel() == n_layers * wstride,
                    "at quant ", quant, " wq8 must be [n_layers * ", wstride,
                    "] bytes, got ", wq8.numel());
        TORCH_CHECK(sq8.numel() == n_layers * sstride,
                    "at quant ", quant, " sq8 must be [n_layers * ", sstride,
                    "] floats, got ", sq8.numel());
    }

    TORCH_CHECK(!twob || quant != 1,
                "two boards per CTA is built for int8 and fp16 only, got quant ", quant);
    if (rj_coef.numel()) {
        TORCH_CHECK(rj_coef.is_cuda() && rj_coef.scalar_type() == torch::kHalf
                    && rj_coef.is_contiguous(),
                    "the re-injection coefficients must be fp16 cuda contiguous");
        const int64_t sites = n_layers * (rj_mode == 2 ? 2 : 1);
        TORCH_CHECK(rj_coef.numel() == sites * brokefish::kRjStride,
                    "at rj_mode ", rj_mode, " the coefficient slab is [", sites, "][",
                    brokefish::kRjStride, "] halves, got ", rj_coef.numel());
    } else {
        TORCH_CHECK(rj_mode == 0, "rj_mode ", rj_mode, " without coefficients");
    }
    launch((int)n, (int)quant, twob != 0, ptr_or_null<half>(y),
           ptr_or_null<const half>(weights),
           ptr_or_null<const uint8_t>(wq8), ptr_or_null<const float>(sq8), nullptr,
           ptr_or_null<const uint16_t>(boards), ptr_or_null<const int16_t>(control),
           ptr_or_null<const uint8_t>(rep), ptr_or_null<const half>(emb),
           ptr_or_null<const half>(tail), ptr_or_null<half>(policy),
           ptr_or_null<half>(promo), ptr_or_null<float>(value),
           (int)n_layers, (float)eps, (int)debug_stage, (int)n_value, (int)vmode,
           ptr_or_null<const half>(rj_coef), (int)rj_mode);
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
            "prologue", "ln1", "quant_q", "qkv_gemm", "qkv_bias", "attention",
            "attn_store", "quant_o",
            "proj_gemm", "proj_epi", "res1_ln2", "quant_a", "ff1_gemm", "ff1_epi",
            "hid_store", "quant_h", "ff2_gemm", "ff2_epi", "res2", "epilogue"};
    }, "phase names, in slot order");
    m.def("encoder_forward", &encoder_forward, "fused encoder forward");
    m.def("model_forward", &model_forward, "boards in, policy/promo/value out",
          py::arg("boards"), py::arg("control"), py::arg("rep"), py::arg("weights"),
          py::arg("emb"), py::arg("tail"), py::arg("policy"), py::arg("promo"),
          py::arg("value"), py::arg("y"), py::arg("n_layers"), py::arg("eps"),
          py::arg("debug_stage"), py::arg("wq8"), py::arg("sq8"), py::arg("quant") = 0,
          py::arg("twob") = 0, py::arg("n_value") = 1, py::arg("vmode") = 0,
          py::arg("rj_coef") = torch::empty({0}, torch::dtype(torch::kHalf)),
          py::arg("rj_mode") = 0);
    m.def("int8_q_max", []() {
        return std::pair<double, double>{(double)brokefish::int8q::kQMaxS,
                                         (double)brokefish::int8q::kQMaxU};
    }, "the s8 and u8 quantisation maxima, pinned against nn/quant.py");
}
