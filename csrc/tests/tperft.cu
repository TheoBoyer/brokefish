// perft on the CUDA engine, against the published node counts of spec §10.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tperft.cu -o tperft && ./tperft
//
// This is the one test in the suite whose oracle is not this repository. The
// differential tests compare the kernel to the PyTorch engine, so a bug the two
// share is invisible to them; these counts are published values that the CUDA
// engine either reproduces or does not.
//
// Structure follows the reference harness in tests/test_env.py: depth-first over
// chunks, so what is resident is one expanded chunk per level rather than the
// whole frontier, and the last ply is counted rather than played. Startpos to
// depth 6 is 119M leaves and about 5.1M positions that actually get a movegen.
//
// The two helper kernels below are perft scaffolding, not engine code. C1 will
// expand nodes one at a time from inside the descent, with no host-side scan and
// no offsets array, which is why they live here and not in movegen.cuh.
//
// Pass a directory as argv[1] to point it at another dump; the default is
// ../../data/cuda_testset relative to this file.

#include <algorithm>
#include <cinttypes>

#include "harness.cuh"
#include "movegen.cuh"

using namespace brokefish;

// Positions per movegen launch. The frontier is chunked so depth 6 stays in tens
// of megabytes instead of the 7.6 GB its 119M boards would need.
static const size_t kChunk = 8192;

// Ranks 1 and 8. A pawn landing on either promotes, which spec §3 carries beside
// the move, so perft counts four edges where the mask holds one bit.
static __device__ const uint64_t kLastRanks = 0xFF000000000000FFULL;

__global__ void count_edges_kernel(const uint16_t* __restrict__ boards,
                                   const uint64_t* __restrict__ mask,
                                   int* __restrict__ counts, int n) {
    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const uint64_t m = mask[(size_t)board * 32 + lane];
    const uint16_t w = boards[(size_t)board * 32 + lane];
    unsigned edges = (unsigned)__popcll(m);
    if (((w >> 6) & 0b111) == kPawn) edges += 3u * (unsigned)__popcll(m & kLastRanks);
    edges = __reduce_add_sync(0xffffffff, edges);
    if (lane == 0) counts[board] = (int)edges;
}

// One warp per position, writing its children into [offsets[i], offsets[i+1]).
// The enumeration order matches the reference's: slot ascending, then target
// square ascending, then promotion type 0..3.
__global__ void expand_kernel(const uint16_t* __restrict__ boards,
                              const int16_t* __restrict__ control,
                              const uint64_t* __restrict__ mask,
                              const int* __restrict__ offsets,
                              uint16_t* __restrict__ out_boards,
                              int16_t* __restrict__ out_control, int n) {
    const unsigned kAll = 0xffffffff;
    const int board = blockIdx.x * kWarpsPerBlock + (int)(threadIdx.x >> 5);
    if (board >= n) return;
    const int lane = (int)(threadIdx.x & 31);

    const uint16_t word = boards[(size_t)board * 32 + lane];
    const int16_t c = control[board];
    const bool white = white_to_move(c);
    size_t out = (size_t)offsets[board];

    unsigned slots = __ballot_sync(kAll, mask[(size_t)board * 32 + lane] != 0);
    while (slots) {
        const int p = __ffs((int)slots) - 1;
        slots &= slots - 1;
        const uint16_t src = __shfl_sync(kAll, word, p);
        const bool src_is_pawn = ((src >> 6) & 0b111) == kPawn;
        // `p` is warp-uniform, so this is one broadcast load, not a shuffle.
        uint64_t targets = mask[(size_t)board * 32 + p];
        while (targets) {
            const int target = __ffsll((unsigned long long)targets) - 1;
            targets &= targets - 1;
            const int edges =
                (src_is_pawn && ((target >> 3) == 0 || (target >> 3) == 7)) ? 4 : 1;
            for (int k = 0; k < edges; ++k) {
                const StepResult sr = apply_move(word, lane, white, p * 64 + target, k);
                out_boards[out * 32 + lane] = sr.word;
                if (lane == 0) out_control[out] = apply_move_control(c, false, sr.clock_reset);
                ++out;
            }
        }
    }
}

// ---------------------------------------------------------------------------

struct Frontier {
    uint16_t* boards = nullptr;
    int16_t* control = nullptr;
    size_t cap = 0;

    void ensure(size_t n) {
        if (n <= cap) return;
        if (boards) CHECK(cudaFree(boards));
        if (control) CHECK(cudaFree(control));
        cap = n + n / 4;  // slack, so a growing level does not realloc every chunk
        CHECK(cudaMalloc(&boards, cap * 32 * sizeof(uint16_t)));
        CHECK(cudaMalloc(&control, cap * sizeof(int16_t)));
    }
};

static Luts g_luts;
static uint64_t* d_mask = nullptr;
static uint8_t* d_in_check = nullptr;
static int* d_counts = nullptr;
static int* d_offsets = nullptr;
static std::vector<int> h_counts, h_offsets;
static std::vector<Frontier> g_levels;
static long long g_movegen_positions = 0;

static int blocks_for(int n) { return (n + kWarpsPerBlock - 1) / kWarpsPerBlock; }

// The work buffers are shared across recursion levels on purpose: a level is
// finished with its mask, counts and offsets before it recurses, so only the
// expanded child boards have to be per-level.
static long long perft(const uint16_t* boards, const int16_t* control, size_t n, int depth,
                       int level) {
    if (depth == 0) return (long long)n;
    long long total = 0;
    for (size_t i = 0; i < n; i += kChunk) {
        const int m = (int)std::min(kChunk, n - i);
        const int blocks = blocks_for(m), threads = kWarpsPerBlock * 32;

        movegen_kernel<<<blocks, threads>>>(boards + i * 32, control + i, g_luts, d_mask,
                                            d_in_check, m);
        count_edges_kernel<<<blocks, threads>>>(boards + i * 32, d_mask, d_counts, m);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_counts.data(), d_counts, m * sizeof(int), cudaMemcpyDeviceToHost));
        g_movegen_positions += m;

        long long sum = 0;
        for (int k = 0; k < m; ++k) {
            h_offsets[k] = (int)sum;
            sum += h_counts[k];
        }
        if (depth == 1) {
            total += sum;
            continue;
        }
        if (sum > (long long)INT32_MAX) {
            printf("expanded chunk of %lld edges overflows the offsets array\n", sum);
            exit(1);
        }
        g_levels[level].ensure((size_t)sum);
        CHECK(cudaMemcpy(d_offsets, h_offsets.data(), m * sizeof(int), cudaMemcpyHostToDevice));
        expand_kernel<<<blocks, threads>>>(boards + i * 32, control + i, d_mask, d_offsets,
                                           g_levels[level].boards, g_levels[level].control, m);
        CHECK(cudaGetLastError());
        total += perft(g_levels[level].boards, g_levels[level].control, (size_t)sum, depth - 1,
                       level + 1);
    }
    return total;
}

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../../data/cuda_testset";

    auto h_move_bitsets = read_bin<uint64_t>("lut_move_bitsets.bin", 6 * 64);
    auto h_occl_offsets = read_bin<int16_t>("lut_occl_offsets.bin", 4 * 64 * 8);
    auto h_occl_masks = read_bin<uint8_t>("lut_occl_masks.bin", 4 * 64 * 8);
    auto h_filled_lines = read_bin<uint8_t>("lut_filled_lines.bin", 8 * 256);
    g_luts.move_bitsets = to_device(h_move_bitsets);
    g_luts.occl_offsets = to_device(h_occl_offsets);
    g_luts.occl_masks = to_device(h_occl_masks);
    g_luts.filled_lines = to_device(h_filled_lines);

    // Cases, and their start positions, both from the dump so that no FEN parser
    // has to be written and validated twice.
    FILE* f = fopen((g_dir + "/perft_cases.txt").c_str(), "r");
    if (!f) {
        printf("cannot open %s/perft_cases.txt\n", g_dir.c_str());
        regenerate_hint();
        return 1;
    }
    struct Case {
        char name[32];
        std::vector<long long> counts;
    };
    std::vector<Case> cases;
    char name[32];
    int ndepth;
    while (fscanf(f, "%31s %d", name, &ndepth) == 2) {
        Case cs;
        snprintf(cs.name, sizeof(cs.name), "%s", name);
        for (int i = 0; i < ndepth; ++i) {
            long long v = 0;
            if (fscanf(f, "%lld", &v) != 1) {
                printf("perft_cases.txt is truncated\n");
                return 1;
            }
            cs.counts.push_back(v);
        }
        cases.push_back(cs);
    }
    fclose(f);

    auto start_boards = read_bin<uint16_t>("perft_boards.bin", cases.size() * 32);
    auto start_control = read_bin<int16_t>("perft_control.bin", cases.size());

    CHECK(cudaMalloc(&d_mask, kChunk * 32 * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_in_check, kChunk * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_counts, kChunk * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, kChunk * sizeof(int)));
    h_counts.resize(kChunk);
    h_offsets.resize(kChunk);
    g_levels.resize(16);

    int failures = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        uint16_t* d_b = nullptr;
        int16_t* d_c = nullptr;
        CHECK(cudaMalloc(&d_b, 32 * sizeof(uint16_t)));
        CHECK(cudaMalloc(&d_c, sizeof(int16_t)));
        CHECK(cudaMemcpy(d_b, &start_boards[ci * 32], 32 * sizeof(uint16_t),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_c, &start_control[ci], sizeof(int16_t), cudaMemcpyHostToDevice));

        printf("%s\n", cases[ci].name);
        for (size_t d = 0; d < cases[ci].counts.size(); ++d) {
            g_movegen_positions = 0;
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            CHECK(cudaEventRecord(t0));
            const long long got = perft(d_b, d_c, 1, (int)d + 1, 0);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            const long long want = cases[ci].counts[d];
            if (got == want) {
                printf("  depth %zu  %12lld  OK    %8.1f ms", d + 1, got, ms);
                if (ms > 1.0f)
                    printf("   %.2fM movegen/s over %lld positions",
                           g_movegen_positions / (ms * 1e3), g_movegen_positions);
                printf("\n");
            } else {
                printf("  depth %zu  %12lld  FAIL  expected %lld (off by %+lld)\n", d + 1, got,
                       want, got - want);
                ++failures;
                break;  // deeper depths would only repeat the same divergence
            }
        }
        CHECK(cudaFree(d_b));
        CHECK(cudaFree(d_c));
    }

    printf(failures ? "\nFAILED (%d depths)\n" : "\nall perft counts match\n", failures);
    return failures ? 1 : 0;
}
