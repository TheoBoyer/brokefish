// Device-level test for hashing and the repetition window: spec §6.1, §6.2, and
// the tracked half of `step` in §4.2.
//
//     nvcc -arch=sm_89 -O3 -std=c++17 -I. -I.. tzobrist.cu -o tzobrist && ./tzobrist
//
// Six checks, in dependency order, because a wrong hash is one 64-bit number and
// says nothing about which of its inputs was wrong:
//
//   1. the key table, generated on device, against the one Python generated
//   2. castling rights
//   3. the legal en passant file
//   4. the full hash, which is 1 combined with 2 and 3
//   5. `step_full`: the incremental hash and `irreversible`
//   6. incremental against from-scratch, over every transition in the dump
//
// Check 6 is the one that cannot be faked. It needs no reference at all: it
// asserts that hashing a position from scratch and arriving at it incrementally
// agree, over 322 246 transitions. An incremental update that is wrong in a way
// the reference is also wrong in would still fail it.
//
// Pass a directory as argv[1]; the default is ../../data/cuda_testset.

#include <cstring>

#include "harness.cuh"
#include "zobrist.cuh"

using namespace brokefish;

__global__ void zobrist_keys_kernel(uint64_t* out) {
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < kZobristKeys;
         i += (int)(blockDim.x * gridDim.x))
        out[i] = zobrist_key(i);
}

template <typename T>
static int compare_scalar(const char* label, const std::vector<T>& want,
                          const std::vector<T>& got, size_t count) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < count; ++i) {
        if (want[i] != got[i]) {
            if (bad == 0) first = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("  %-34s OK   (%zu)\n", label, count);
        return 0;
    }
    printf("  %-34s FAIL %zu of %zu differ, first at %zu: want 0x%llx got 0x%llx\n", label,
           bad, count, first, (unsigned long long)(uint64_t)want[first],
           (unsigned long long)(uint64_t)got[first]);
    return 1;
}

int main(int argc, char** argv) {
    g_dir = argc > 1 ? argv[1] : "../../data/cuda_testset";

    const int n = read_meta_n();
    const size_t m = bin_count("step_move.bin", sizeof(int16_t));
    printf("%d positions, %zu step cases from %s\n", n, m, g_dir.c_str());

    int failures = 0;

    // ---- 1. the key table -------------------------------------------------
    // luts.py chose splitmix64 so that the CUDA side could regenerate the table
    // rather than load one, and the two could not drift. This is the check that
    // makes that claim true. The device form exploits splitmix64's state being a
    // plain Weyl sequence, so it computes key i directly instead of iterating.
    {
        uint64_t* d_keys = nullptr;
        CHECK(cudaMalloc(&d_keys, kZobristKeys * sizeof(uint64_t)));
        CHECK(cudaMemset(d_keys, 0xEE, kZobristKeys * sizeof(uint64_t)));
        zobrist_keys_kernel<<<4, 256>>>(d_keys);
        CHECK(cudaGetLastError());
        std::vector<uint64_t> got(kZobristKeys);
        CHECK(cudaMemcpy(got.data(), d_keys, kZobristKeys * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
        auto want = read_bin<uint64_t>("lut_zobrist.bin", kZobristKeys);
        failures += compare_scalar("zobrist keys, device-generated", want, got, kZobristKeys);
        CHECK(cudaFree(d_keys));
    }

    auto boards = read_bin<uint16_t>("boards.bin", (size_t)n * 32);
    auto control = read_bin<int16_t>("control.bin", n);
    auto hashes = read_bin<uint64_t>("hashes.bin", n);

    Luts luts;
    auto h_move_bitsets = read_bin<uint64_t>("lut_move_bitsets.bin", 6 * 64);
    auto h_occl_offsets = read_bin<int16_t>("lut_occl_offsets.bin", 4 * 64 * 8);
    auto h_occl_masks = read_bin<uint8_t>("lut_occl_masks.bin", 4 * 64 * 8);
    auto h_filled_lines = read_bin<uint8_t>("lut_filled_lines.bin", 8 * 256);
    luts.move_bitsets = to_device(h_move_bitsets);
    luts.occl_offsets = to_device(h_occl_offsets);
    luts.occl_masks = to_device(h_occl_masks);
    luts.filled_lines = to_device(h_filled_lines);

    auto d_boards = to_device(boards);
    auto d_control = to_device(control);
    auto d_hashes = to_device(hashes);

    // ---- 2, 3, 4. rights, en passant file, full hash ----------------------
    uint64_t* d_hash_out = nullptr;
    unsigned char* d_rights = nullptr;
    signed char* d_ep = nullptr;
    CHECK(cudaMalloc(&d_hash_out, (size_t)n * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_rights, (size_t)n));
    CHECK(cudaMalloc(&d_ep, (size_t)n));
    CHECK(cudaMemset(d_hash_out, 0xEE, (size_t)n * sizeof(uint64_t)));
    CHECK(cudaMemset(d_rights, 0xEE, (size_t)n));
    CHECK(cudaMemset(d_ep, 0xEE, (size_t)n));

    hash_position_kernel<<<(n + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(
        d_boards, d_control, luts, d_hash_out, d_rights, d_ep, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<unsigned char> got_rights(n);
    std::vector<signed char> got_ep(n);
    std::vector<uint64_t> got_hash(n);
    CHECK(cudaMemcpy(got_rights.data(), d_rights, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_ep.data(), d_ep, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_hash.data(), d_hash_out, (size_t)n * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));

    failures += compare_scalar("castling rights", read_bin<unsigned char>("castle_rights.bin", n),
                               got_rights, n);
    failures += compare_scalar("legal en passant file",
                               read_bin<signed char>("ep_file.bin", n), got_ep, n);
    failures += compare_scalar("hash_position, from scratch", hashes, got_hash, n);

    // ---- 5. step_full -----------------------------------------------------
    auto index = read_bin<int32_t>("step_index.bin", m);
    auto move = read_bin<int16_t>("step_move.bin", m);
    auto promo = read_bin<uint8_t>("step_promo.bin", m);
    auto d_index = to_device(index);
    auto d_move = to_device(move);
    auto d_promo = to_device(promo);

    uint16_t* d_child_boards = nullptr;
    int16_t* d_child_control = nullptr;
    uint64_t* d_child_hash = nullptr;
    uint8_t* d_irrev = nullptr;
    CHECK(cudaMalloc(&d_child_boards, m * 32 * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_child_control, m * sizeof(int16_t)));
    CHECK(cudaMalloc(&d_child_hash, m * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_irrev, m));
    CHECK(cudaMemset(d_child_boards, 0xEE, m * 32 * sizeof(uint16_t)));
    CHECK(cudaMemset(d_child_hash, 0xEE, m * sizeof(uint64_t)));
    CHECK(cudaMemset(d_irrev, 0xEE, m));

    const int step_blocks = (int)((m + kWarpsPerBlock - 1) / kWarpsPerBlock);
    step_full_kernel<<<step_blocks, kWarpsPerBlock * 32>>>(
        d_boards, d_control, d_hashes, d_index, d_move, d_promo, luts, d_child_boards,
        d_child_control, d_child_hash, d_irrev, (int)m);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> got_child_boards(m * 32);
    std::vector<int16_t> got_child_control(m);
    std::vector<uint64_t> got_child_hash(m);
    std::vector<uint8_t> got_irrev(m);
    CHECK(cudaMemcpy(got_child_boards.data(), d_child_boards,
                     got_child_boards.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_child_control.data(), d_child_control, m * sizeof(int16_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_child_hash.data(), d_child_hash, m * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_irrev.data(), d_irrev, m, cudaMemcpyDeviceToHost));

    // The mutation is re-checked here rather than trusted from tstep.cu, because
    // step_full re-derives it and a divergence would otherwise hide in the hash.
    {
        auto want = read_bin<uint16_t>("step_boards.bin", m * 32);
        size_t bad = 0, first = 0;
        for (size_t e = 0; e < m; ++e) {
            if (memcmp(&want[e * 32], &got_child_boards[e * 32], 32 * sizeof(uint16_t))) {
                if (bad == 0) first = e;
                ++bad;
            }
        }
        if (bad == 0) {
            printf("  %-34s OK   (%zu)\n", "step_full boards", m);
        } else {
            const int pos = index[first];
            printf("  %-34s FAIL %zu of %zu, first at case %zu (position %d)\n",
                   "step_full boards", bad, m, first, pos);
            print_board("before", &boards[(size_t)pos * 32], control[pos]);
            print_slot_diff(&want[first * 32], &got_child_boards[first * 32]);
            ++failures;
        }
    }
    failures += compare_scalar("step_full control", read_bin<int16_t>("step_control.bin", m),
                               got_child_control, m);
    failures += compare_scalar("step_full hash, incremental",
                               read_bin<uint64_t>("step_hash.bin", m), got_child_hash, m);
    failures += compare_scalar("step_full irreversible",
                               read_bin<uint8_t>("step_irrev.bin", m), got_irrev, m);

    // ---- 6. incremental against from-scratch ------------------------------
    // No reference involved: hashing the child position directly has to agree
    // with having arrived at it by an incremental update.
    {
        uint64_t* d_scratch = nullptr;
        CHECK(cudaMalloc(&d_scratch, m * sizeof(uint64_t)));
        CHECK(cudaMemset(d_scratch, 0xEE, m * sizeof(uint64_t)));
        hash_position_kernel<<<step_blocks, kWarpsPerBlock * 32>>>(
            d_child_boards, d_child_control, luts, d_scratch, nullptr, nullptr, (int)m);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<uint64_t> scratch(m);
        CHECK(cudaMemcpy(scratch.data(), d_scratch, m * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
        size_t bad = 0, first = 0;
        for (size_t e = 0; e < m; ++e) {
            if (scratch[e] != got_child_hash[e]) {
                if (bad == 0) first = e;
                ++bad;
            }
        }
        if (bad == 0) {
            printf("  %-34s OK   (%zu transitions)\n", "incremental == from scratch", m);
        } else {
            const int pos = index[first];
            char to[3];
            square_name(move[first] & 63, to);
            printf("  %-34s FAIL %zu of %zu, first at case %zu: position %d, "
                   "slot %d -> %s, promo %d\n",
                   "incremental == from scratch", bad, m, first, pos, move[first] >> 6, to,
                   promo[first]);
            printf("    incremental 0x%016llx  from scratch 0x%016llx  xor 0x%016llx\n",
                   (unsigned long long)got_child_hash[first],
                   (unsigned long long)scratch[first],
                   (unsigned long long)(got_child_hash[first] ^ scratch[first]));
            print_board("before", &boards[(size_t)pos * 32], control[pos]);
            print_board("after", &got_child_boards[first * 32], got_child_control[first]);
            ++failures;
        }
        CHECK(cudaFree(d_scratch));
    }

    // ---- coverage ---------------------------------------------------------
    size_t irrev = 0;
    for (size_t e = 0; e < m; ++e) irrev += got_irrev[e] != 0;
    int with_ep = 0, rights_hist[16] = {0};
    for (int i = 0; i < n; ++i) {
        with_ep += got_ep[i] >= 0;
        rights_hist[got_rights[i] & 15]++;
    }
    printf("\n  coverage: %zu irreversible of %zu step cases, %d positions with a legal\n"
           "            en passant, %d of 16 castling-rights combinations present\n",
           irrev, m, with_ep, [&] {
               int k = 0;
               for (int i = 0; i < 16; ++i) k += rights_hist[i] > 0;
               return k;
           }());
    if (with_ep == 0) {
        printf("  UNCOVERED: no legal en passant in the dump, so the ep key is untested\n");
        ++failures;
    }

    printf(failures ? "\nFAILED (%d checks)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
