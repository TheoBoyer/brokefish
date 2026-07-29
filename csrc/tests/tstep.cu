// Device-level test for the board mutation of spec §4.2.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tstep.cu -o tstep && ./tstep
//
// It applies every legal move of every position of the dump, with promoting
// moves expanded to all four choices and one null move per position, and asks
// whether the 32 slot words and the control word match the PyTorch engine's
// exactly. There is no tolerance to speak of here: a piece word is either right
// or it is a different position.
//
// The coverage counts at the end are part of the test. A `step` that never sees
// an en passant capture or a castle passes vacuously, and the dump is drawn from
// random playouts where both are rare.
//
// Pass a directory as argv[1] to point it at another dump; the default is
// ../../data/cuda_testset relative to this file.

#include <cstring>

#include "harness.cuh"
#include "step.cuh"

using namespace brokefish;

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../../data/cuda_testset";

    const int n = read_meta_n();
    const size_t m = bin_count("step_move.bin", sizeof(int16_t));
    printf("%d positions, %zu step cases from %s\n", n, m, g_dir.c_str());

    auto boards = read_bin<uint16_t>("boards.bin", (size_t)n * 32);
    auto control = read_bin<int16_t>("control.bin", n);
    auto index = read_bin<int32_t>("step_index.bin", m);
    auto move = read_bin<int16_t>("step_move.bin", m);
    auto promo = read_bin<uint8_t>("step_promo.bin", m);
    auto want_boards = read_bin<uint16_t>("step_boards.bin", m * 32);
    auto want_control = read_bin<int16_t>("step_control.bin", m);

    auto d_boards = to_device(boards);
    auto d_control = to_device(control);
    auto d_index = to_device(index);
    auto d_move = to_device(move);
    auto d_promo = to_device(promo);

    uint16_t* d_out_boards = nullptr;
    int16_t* d_out_control = nullptr;
    CHECK(cudaMalloc(&d_out_boards, m * 32 * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_out_control, m * sizeof(int16_t)));
    // Poison, so an unwritten entry fails loudly instead of matching a zero.
    CHECK(cudaMemset(d_out_boards, 0xEE, m * 32 * sizeof(uint16_t)));
    CHECK(cudaMemset(d_out_control, 0xEE, m * sizeof(int16_t)));

    const int blocks = (int)((m + kStepWarpsPerBlock - 1) / kStepWarpsPerBlock);
    apply_move_kernel<<<blocks, kStepWarpsPerBlock * 32>>>(
        d_boards, d_control, d_index, d_move, d_promo, d_out_boards, d_out_control, (int)m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got_boards(m * 32);
    std::vector<int16_t> got_control(m);
    CHECK(cudaMemcpy(got_boards.data(), d_out_boards, got_boards.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_control.data(), d_out_control, m * sizeof(int16_t),
                     cudaMemcpyDeviceToHost));

    int failures = 0;

    // Boards.
    size_t bad = 0, first = 0;
    for (size_t e = 0; e < m; ++e) {
        if (memcmp(&want_boards[e * 32], &got_boards[e * 32], 32 * sizeof(uint16_t)) != 0) {
            if (bad == 0) first = e;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  %-28s OK   (%zu positions)\n", "boards after the move", m);
    } else {
        const int pos = index[first];
        char from[3], to[3];
        square_name(boards[(size_t)pos * 32 + (move[first] >> 6)] & 63, from);
        square_name(move[first] & 63, to);
        printf("  %-28s FAIL %zu of %zu cases differ\n", "boards after the move", bad, m);
        printf("  first at case %zu: position %d, move %d = slot %d -> %s (from %s), promo %d\n",
               first, pos, move[first], move[first] >> 6, to, from, promo[first]);
        print_board("before", &boards[(size_t)pos * 32], control[pos]);
        print_board("want", &want_boards[first * 32], want_control[first]);
        print_board("got", &got_boards[first * 32], got_control[first]);
        print_slot_diff(&want_boards[first * 32], &got_boards[first * 32]);
        ++failures;
    }

    // Control word.
    bad = 0;
    for (size_t e = 0; e < m; ++e) {
        if (want_control[e] != got_control[e]) {
            if (bad == 0) first = e;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  %-28s OK   (%zu positions)\n", "control word", m);
    } else {
        printf("  %-28s FAIL %zu of %zu differ, first at case %zu: want %d got %d\n",
               "control word", bad, m, first, want_control[first], got_control[first]);
        ++failures;
    }

    // ---------------------------------------------------------------------
    // Coverage. Classified from the dump rather than from the kernel, so it
    // describes what was tested and not what the kernel thinks it did.
    // ---------------------------------------------------------------------
    size_t n_capture = 0, n_ep = 0, n_castle = 0, n_promo = 0, n_double = 0, n_null = 0;
    // En passant is the rarest rule and the one with the subtlest code, so its
    // four geometries are counted separately: 39 cases all pointing the same way
    // would leave three of them untested.
    size_t ep_geom[4] = {0, 0, 0, 0};  // (white, black) x (left, right)
    size_t n_short_castle = 0, n_long_castle = 0;
    for (size_t e = 0; e < m; ++e) {
        if (move[e] < 0) { ++n_null; continue; }
        const int pos = index[e];
        const uint16_t* before = &boards[(size_t)pos * 32];
        const int slot = move[e] >> 6, target = move[e] & 63;
        const uint16_t w = before[slot];
        const int type = (w >> 6) & 0b111, dx = target - (w & 63);
        const bool white = control[pos] > 0;

        size_t killed = 0, occupied = 0;
        for (int s = 0; s < 32; ++s) {
            if (!((before[s] >> 11) & 1) && ((got_boards[e * 32 + s] >> 11) & 1)) ++killed;
            if (!((before[s] >> 11) & 1) && (before[s] & 63) == target) ++occupied;
        }
        if (killed) ++n_capture;
        if (type == kPawn && killed && !occupied) {
            ++n_ep;
            // The capture is a diagonal, so dx is +-7 or +-9 from white's side
            // and mirrored for black; the file direction is what distinguishes
            // the two victim adjacencies.
            const int file_dx = (target & 7) - ((w & 63) & 7);
            ep_geom[(white ? 0 : 2) + (file_dx > 0 ? 1 : 0)]++;
        }
        if (type == kKing && (dx == 2 || dx == -2)) {
            ++n_castle;
            if (dx > 0) ++n_short_castle; else ++n_long_castle;
        }
        if (type == kPawn && (target >> 3) == (white ? 7 : 0)) ++n_promo;
        if (type == kPawn && (dx == 16 || dx == -16)) ++n_double;
    }
    printf("\n  coverage: %zu captures, %zu en passant, %zu castles, %zu promotions,\n"
           "            %zu double pushes, %zu null moves\n",
           n_capture, n_ep, n_castle, n_promo, n_double, n_null);
    printf("            en passant by geometry: white %zu left / %zu right,"
           " black %zu left / %zu right\n",
           ep_geom[0], ep_geom[1], ep_geom[2], ep_geom[3]);
    printf("            castling: %zu short, %zu long\n", n_short_castle, n_long_castle);
    const char* names[] = {"captures",      "en passant",      "castles",
                           "promotions",    "double pushes",   "null moves",
                           "white-left en passant",  "white-right en passant",
                           "black-left en passant",  "black-right en passant",
                           "short castles", "long castles"};
    const size_t counts[] = {n_capture,   n_ep,        n_castle,    n_promo,
                             n_double,    n_null,      ep_geom[0],  ep_geom[1],
                             ep_geom[2],  ep_geom[3],  n_short_castle, n_long_castle};
    for (int i = 0; i < 12; ++i) {
        if (counts[i] == 0) {
            printf("  UNCOVERED: no %s in the dump, so that rule is untested\n", names[i]);
            ++failures;
        }
    }

    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
