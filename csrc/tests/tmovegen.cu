// Device-level test for the first-order move generator.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tmovegen.cu -o tmovegen && ./tmovegen
//
// It reads the flat binaries scripts/dump_cuda_testset.py writes and compares
// the kernel's output to the PyTorch engine's, bit for bit, one rule at a time:
// base bitsets, then pawns, then sliders, then castling, then the friendly
// occupancy filter, then the control-mode attack map and in_check. Localising a
// failure to a stage is the point -- a wrong slider and a wrong pawn produce the
// same symptom in the final mask.
//
// Pass a directory as argv[1] to point it at another dump; the default is
// ../../data/cuda_testset relative to this file.

#include <cstring>

#include "harness.cuh"
#include "movegen.cuh"

using namespace brokefish;

static int compare(const char* label, const std::vector<uint64_t>& want,
                   const std::vector<uint64_t>& got, const std::vector<uint16_t>& boards,
                   const std::vector<int16_t>& control) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (want[i] != got[i]) {
            if (bad == 0) first = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  %-28s OK   (%zu masks)\n", label, want.size());
        return 0;
    }
    size_t pos = first / 32, slot = first % 32;
    printf("  %-28s FAIL %zu of %zu masks differ\n", label, bad, want.size());
    printf("  first at position %zu, slot %zu: want 0x%016llx got 0x%016llx\n", pos, slot,
           (unsigned long long)want[first], (unsigned long long)got[first]);
    printf("  slot %zu holds word 0x%03x: %s%s type %d square %d\n", slot,
           boards[pos * 32 + slot], (boards[pos * 32 + slot] >> 11) & 1 ? "captured " : "",
           (boards[pos * 32 + slot] >> 10) & 1 ? "black" : "white",
           (boards[pos * 32 + slot] >> 6) & 0b111, boards[pos * 32 + slot] & 63);
    print_board("position", &boards[pos * 32], control[pos]);
    print_diff(want[first], got[first]);
    return 1;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../../data/cuda_testset";

    const int n = read_meta_n();
    printf("%d positions from %s\n", n, g_dir.c_str());

    auto boards = read_bin<uint16_t>("boards.bin", (size_t)n * 32);
    auto control = read_bin<int16_t>("control.bin", n);

    Luts luts;
    auto h_move_bitsets = read_bin<uint64_t>("lut_move_bitsets.bin", 6 * 64);
    auto h_occl_offsets = read_bin<int16_t>("lut_occl_offsets.bin", 4 * 64 * 8);
    auto h_occl_masks = read_bin<uint8_t>("lut_occl_masks.bin", 4 * 64 * 8);
    auto h_filled_lines = read_bin<uint8_t>("lut_filled_lines.bin", 8 * 256);
    luts.move_bitsets = to_device(h_move_bitsets);
    luts.occl_offsets = to_device(h_occl_offsets);
    luts.occl_masks = to_device(h_occl_masks);
    luts.filled_lines = to_device(h_filled_lines);

    const uint16_t* d_boards = to_device(boards);
    const int16_t* d_control = to_device(control);
    uint64_t* d_out = nullptr;
    uint8_t* d_in_check = nullptr;
    CHECK(cudaMalloc(&d_out, (size_t)n * 32 * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_in_check, (size_t)n * sizeof(uint8_t)));

    const int blocks = (n + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int threads = kWarpsPerBlock * 32;
    std::vector<uint64_t> got((size_t)n * 32);

    auto run = [&](auto kernel_launch) {
        CHECK(cudaMemset(d_out, 0xEE, (size_t)n * 32 * sizeof(uint64_t)));
        kernel_launch();
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    };

    int failures = 0;

    run([&] {
        first_order_kernel<false, 1><<<blocks, threads>>>(d_boards, d_control, luts, d_out, n);
    });
    failures += compare("stage 1  base bitsets", read_bin<uint64_t>("fo_stage1.bin", got.size()),
                        got, boards, control);

    run([&] {
        first_order_kernel<false, 2><<<blocks, threads>>>(d_boards, d_control, luts, d_out, n);
    });
    failures += compare("stage 2  + pawns", read_bin<uint64_t>("fo_stage2.bin", got.size()), got,
                        boards, control);

    run([&] {
        first_order_kernel<false, 3><<<blocks, threads>>>(d_boards, d_control, luts, d_out, n);
    });
    failures += compare("stage 3  + sliders", read_bin<uint64_t>("fo_stage3.bin", got.size()), got,
                        boards, control);

    run([&] {
        first_order_kernel<false, 4><<<blocks, threads>>>(d_boards, d_control, luts, d_out, n);
    });
    failures += compare("stage 4  + castling", read_bin<uint64_t>("fo_stage4.bin", got.size()), got,
                        boards, control);

    run([&] {
        first_order_kernel<false, 5><<<blocks, threads>>>(d_boards, d_control, luts, d_out, n);
    });
    failures += compare("first order  + own filter", read_bin<uint64_t>("fo_masks.bin", got.size()),
                        got, boards, control);

    // Control mode has no separate stage snapshots: it is the same code path
    // with two switches flipped, so the stages above already cover it.
    run([&] {
        attack_map_kernel<<<blocks, threads>>>(d_boards, d_control, luts, d_out, d_in_check, n);
    });
    failures += compare("control mode  attack map",
                        read_bin<uint64_t>("fo_control.bin", got.size()), got, boards, control);

    auto want_check = read_bin<uint8_t>("in_check.bin", n);
    std::vector<uint8_t> got_check(n);

    auto compare_in_check = [&](const char* label) {
        CHECK(cudaMemcpy(got_check.data(), d_in_check, n, cudaMemcpyDeviceToHost));
        int bad = 0, first = -1;
        for (int i = 0; i < n; ++i) {
            if (got_check[i] != want_check[i]) {
                if (bad == 0) first = i;
                ++bad;
            }
        }
        if (bad == 0) {
            printf("  %-28s OK   (%d positions)\n", label, n);
            return 0;
        }
        printf("  %-28s FAIL %d of %d differ, first at %d: want %d got %d\n", label, bad, n,
               first, want_check[first], got_check[first]);
        print_board("position", &boards[(size_t)first * 32], control[first]);
        return 1;
    };
    failures += compare_in_check("in_check from the map");

    // Full legality: the second-order pass on top of everything above. This is
    // the one that gates the kernel, and the dump carries the positions the rest
    // of the suite would miss: 5 with no legal move at all, pins, castling
    // through check, and en passant that would expose the king.
    CHECK(cudaMemset(d_in_check, 0xEE, (size_t)n * sizeof(uint8_t)));
    run([&] {
        movegen_kernel<<<blocks, threads>>>(d_boards, d_control, luts, d_out, d_in_check, n);
    });
    failures += compare("movegen  full legality", read_bin<uint64_t>("masks.bin", got.size()),
                        got, boards, control);
    failures += compare_in_check("in_check from movegen");

    // Mate and stalemate are what `in_check` exists to separate (spec §4.1), and
    // an all-zero mask is also what a bug that drops every move looks like.
    int mates = 0, stalemates = 0;
    for (int i = 0; i < n; ++i) {
        bool none = true;
        for (int p = 0; p < 32; ++p) none &= got[(size_t)i * 32 + p] == 0;
        if (none) (got_check[i] ? mates : stalemates)++;
    }
    printf("  %-28s %d checkmate, %d stalemate\n", "no-legal-move positions", mates,
           stalemates);
    if (mates == 0 || stalemates == 0) {
        printf("  UNCOVERED: the dump has no %s, so spec 4.1's distinction is untested\n",
               mates == 0 ? "checkmate" : "stalemate");
        ++failures;
    }

    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
