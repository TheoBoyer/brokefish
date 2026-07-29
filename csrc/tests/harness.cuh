// Shared scaffolding for the device-level engine tests: reading the flat
// binaries scripts/dump_cuda_testset.py writes, and printing a failure in a form
// a human can act on. No Python, no torch, no test framework.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CHECK(expr)                                                                  \
    do {                                                                             \
        cudaError_t err_ = (expr);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(err_));                                         \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

inline std::string g_dir;

inline void regenerate_hint() {
    printf("regenerate the dump with:  .venv/bin/python scripts/dump_cuda_testset.py\n");
}

template <typename T>
std::vector<T> read_bin(const char* name, size_t count) {
    std::string path = g_dir + "/" + name;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("cannot open %s\n", path.c_str());
        regenerate_hint();
        exit(1);
    }
    std::vector<T> out(count);
    size_t got = fread(out.data(), sizeof(T), count, f);
    fclose(f);
    if (got != count) {
        printf("%s: expected %zu elements, read %zu\n", path.c_str(), count, got);
        regenerate_hint();
        exit(1);
    }
    return out;
}

// The element count of a file, for the dumps whose length is not implied by N.
inline size_t bin_count(const char* name, size_t elem_size) {
    std::string path = g_dir + "/" + name;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("cannot open %s\n", path.c_str());
        regenerate_hint();
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fclose(f);
    return (size_t)bytes / elem_size;
}

template <typename T>
T* to_device(const std::vector<T>& host) {
    T* p = nullptr;
    CHECK(cudaMalloc(&p, host.size() * sizeof(T)));
    CHECK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

inline int read_meta_n() {
    FILE* meta = fopen((g_dir + "/meta.txt").c_str(), "r");
    if (!meta) {
        printf("cannot open %s/meta.txt\n", g_dir.c_str());
        regenerate_hint();
        exit(1);
    }
    int n = 0;
    if (fscanf(meta, "%d", &n) != 1 || n <= 0) {
        printf("meta.txt does not hold a position count\n");
        exit(1);
    }
    fclose(meta);
    return n;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

inline void square_name(int square, char* out) {
    out[0] = (char)('a' + (square & 7));
    out[1] = (char)('1' + (square >> 3));
    out[2] = '\0';
}

// One position as eight ranks. `label` is printed above it, so a before/after
// pair reads as two labelled diagrams rather than two anonymous grids.
inline void print_board(const char* label, const uint16_t* b, int16_t control) {
    char sq[64];
    for (int i = 0; i < 64; ++i) sq[i] = '.';
    const char* names = "PNBRQK?";
    for (int s = 0; s < 32; ++s) {
        uint16_t w = b[s];
        if ((w >> 11) & 1) continue;
        int type = (w >> 6) & 0b111;
        char c = names[type < 6 ? type : 6];
        if ((w >> 10) & 1) c = (char)(c - 'A' + 'a');
        sq[w & 63] = c;
    }
    printf("    %s: %s to move, clock %d\n", label, control > 0 ? "white" : "black",
           (control > 0 ? control : -control) - 1);
    for (int r = 7; r >= 0; --r) {
        printf("    %d  ", r + 1);
        for (int c = 0; c < 8; ++c) printf("%c ", sq[r * 8 + c]);
        printf("\n");
    }
    printf("       a b c d e f g h\n");
}

// Two bitboards side by side plus their xor, so a missing square and a spurious
// one are distinguishable at a glance.
inline void print_diff(uint64_t want, uint64_t got) {
    printf("    want         got          diff\n");
    for (int r = 7; r >= 0; --r) {
        printf("    ");
        for (int pass = 0; pass < 3; ++pass) {
            uint64_t m = pass == 0 ? want : (pass == 1 ? got : (want ^ got));
            for (int c = 0; c < 8; ++c) printf("%c", ((m >> (r * 8 + c)) & 1) ? 'X' : '.');
            printf("     ");
        }
        printf("\n");
    }
}

// The 32 slot words of two positions, showing only the slots that differ. A
// wrong `special` bit or a wrong captured slot is invisible on a board diagram.
inline void print_slot_diff(const uint16_t* want, const uint16_t* got) {
    printf("    slot  want   got    (captured colour special type square)\n");
    for (int s = 0; s < 32; ++s) {
        if (want[s] == got[s]) continue;
        char wn[3], gn[3];
        square_name(want[s] & 63, wn);
        square_name(got[s] & 63, gn);
        printf("    %4d  0x%03x  0x%03x   want %d %s %d %d %s   got %d %s %d %d %s\n", s,
               want[s], got[s], (want[s] >> 11) & 1, (want[s] >> 10) & 1 ? "black" : "white",
               (want[s] >> 9) & 1, (want[s] >> 6) & 0b111, wn, (got[s] >> 11) & 1,
               (got[s] >> 10) & 1 ? "black" : "white", (got[s] >> 9) & 1,
               (got[s] >> 6) & 0b111, gn);
    }
}
