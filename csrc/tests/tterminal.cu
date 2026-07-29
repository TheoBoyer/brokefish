// Device-level test for spec §4.3, the terminal codes.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tterminal.cu -o tterminal && ./tterminal
//
// Repetition is not tested here and cannot be: it is a property of the game that
// reached a position, and a static dump holds positions, so `terminal_code.bin`
// can never carry code 4. The ring itself belongs to C1, which owns per-game
// state; `repetition_count` is implemented against spec §6.3 and is exercised
// below on a synthetic ring, which is a unit test rather than a differential one.
//
// Pass a directory as argv[1]; the default is ../../data/cuda_testset.

#include "harness.cuh"
#include "terminal.cuh"

using namespace brokefish;

// One warp, one synthetic ring. spec §6.3 wants at most four iterations over 100
// entries and the count to include the present position.
__global__ void repetition_probe_kernel(const uint64_t* ring, int len, uint64_t hash,
                                        int* out) {
    if (threadIdx.x >= 32) return;
    const int c = repetition_count(hash, ring, len, (int)threadIdx.x);
    if (threadIdx.x == 0) *out = c;
}

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../../data/cuda_testset";

    const int n = read_meta_n();
    printf("%d positions from %s\n", n, g_dir.c_str());

    auto boards = read_bin<uint16_t>("boards.bin", (size_t)n * 32);
    auto control = read_bin<int16_t>("control.bin", n);
    auto mask = read_bin<uint64_t>("masks.bin", (size_t)n * 32);
    auto in_check = read_bin<uint8_t>("in_check.bin", n);
    auto want_code = read_bin<uint8_t>("terminal_code.bin", n);
    auto want_result = read_bin<signed char>("terminal_result.bin", n);

    auto d_boards = to_device(boards);
    auto d_control = to_device(control);
    auto d_mask = to_device(mask);
    auto d_in_check = to_device(in_check);

    uint8_t* d_code = nullptr;
    signed char* d_result = nullptr;
    CHECK(cudaMalloc(&d_code, n));
    CHECK(cudaMalloc(&d_result, n));
    CHECK(cudaMemset(d_code, 0xEE, n));
    CHECK(cudaMemset(d_result, 0xEE, n));

    // No hash and no ring: repetition is a property of the game that reached a
    // position, and a static dump holds positions.
    terminal_kernel<<<(n + 7) / 8, 8 * 32>>>(d_boards, d_control, d_mask, d_in_check, nullptr,
                                             nullptr, nullptr, d_code, d_result, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> got_code(n);
    std::vector<signed char> got_result(n);
    CHECK(cudaMemcpy(got_code.data(), d_code, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_result.data(), d_result, n, cudaMemcpyDeviceToHost));

    int failures = 0;
    const char* names[] = {"none", "checkmate", "stalemate", "fifty-move", "repetition",
                           "insufficient"};

    int bad = 0, first = -1;
    for (int i = 0; i < n; ++i) {
        if (got_code[i] != want_code[i] || got_result[i] != want_result[i]) {
            if (bad == 0) first = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  %-30s OK   (%d positions)\n", "terminal code and result", n);
    } else {
        printf("  %-30s FAIL %d of %d differ, first at %d: want code %d (%s) result %d, "
               "got code %d (%s) result %d\n",
               "terminal code and result", bad, n, first, want_code[first],
               names[want_code[first] < 6 ? want_code[first] : 0], want_result[first],
               got_code[first], names[got_code[first] < 6 ? got_code[first] : 0],
               got_result[first]);
        print_board("position", &boards[(size_t)first * 32], control[first]);
        printf("    in_check %d\n", in_check[first]);
        ++failures;
    }

    // Coverage. Every code the dump can carry has to appear, or the branch that
    // produces it is untested.
    int hist[6] = {0};
    for (int i = 0; i < n; ++i)
        if (got_code[i] < 6) hist[got_code[i]]++;
    printf("\n  coverage by code:");
    for (int c = 0; c < 6; ++c) printf("  %s %d", names[c], hist[c]);
    printf("\n");
    for (int c = 0; c < 6; ++c) {
        if (c == kRepetition) continue;  // a static dump cannot carry one
        if (hist[c] == 0) {
            printf("  UNCOVERED: no %s in the dump, so that branch is untested\n", names[c]);
            ++failures;
        }
    }

    // repetition_count, on a synthetic ring, since the dump has no games in it.
    {
        std::vector<uint64_t> ring(kMaxHistory);
        for (int i = 0; i < kMaxHistory; ++i) ring[i] = 0x1000ULL + i;
        // Three occurrences of one hash, spread past the 32-lane stride so the
        // loop runs more than one iteration.
        ring[3] = 0xABCD;
        ring[40] = 0xABCD;
        ring[71] = 0xDEAD;
        auto d_ring = to_device(ring);
        int* d_out = nullptr;
        CHECK(cudaMalloc(&d_out, sizeof(int)));

        struct Probe {
            uint64_t hash;
            int len;
            int want;
            const char* what;
        };
        const Probe probes[] = {
            {0xABCD, kMaxHistory, 3, "two earlier occurrences over 100 entries"},
            {0xABCD, 40, 2, "one earlier, the second past the live length"},
            {0xDEAD, kMaxHistory, 2, "a single earlier occurrence"},
            {0xBEEF, kMaxHistory, 1, "never seen before"},
            {0xABCD, 0, 1, "an empty ring, as after an irreversible move"},
        };
        int bad_probe = 0;
        for (const Probe& p : probes) {
            repetition_probe_kernel<<<1, 32>>>(d_ring, p.len, p.hash, d_out);
            CHECK(cudaGetLastError());
            int got = 0;
            CHECK(cudaMemcpy(&got, d_out, sizeof(int), cudaMemcpyDeviceToHost));
            if (got != p.want) {
                printf("  %-30s FAIL %s: want %d got %d\n", "repetition_count", p.what,
                       p.want, got);
                ++bad_probe;
            }
        }
        failures += bad_probe;
        if (!bad_probe)
            printf("  %-30s OK   (%zu probes)\n", "repetition_count",
                   sizeof(probes) / sizeof(probes[0]));
    }

    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
