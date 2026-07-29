// Torch bindings for the CUDA engine: the four entry points of spec §4 plus the
// hash, exposed so that `brokefish/env/cuda_impl.py` can stand in for
// `torch_impl.py` behind the same signatures.
//
// The bindings are stateless. The four movegen tables are 11 KB of pure
// arithmetic over 64 squares, and the Python side already builds and caches them
// per device (`brokefish/env/luts.py`), so they arrive as tensors on every call
// rather than being uploaded into a global here. A cached device pointer would
// have to be invalidated on device change and would be wrong exactly once, at
// which point it reads someone else's memory and produces plausible numbers.
//
// Nothing here belongs in the self-play loop. The loop is device-side by
// construction (spec §4: none of these may synchronise with the host), and the
// launch-per-call shape below is a host-driven convenience for tests, benchmarks
// and data loading. C1 calls the device functions in the headers directly.
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "movegen.cuh"
#include "terminal.cuh"
#include "zobrist.cuh"

using namespace brokefish;

namespace {

void check_state(const torch::Tensor& boards, const torch::Tensor& control) {
    TORCH_CHECK(boards.is_cuda() && control.is_cuda(), "boards and control must be on CUDA");
    TORCH_CHECK(boards.dim() == 2 && boards.size(1) == 32, "boards must be [N, 32], got ",
                boards.sizes());
    TORCH_CHECK(control.dim() == 1 && control.size(0) == boards.size(0),
                "control must be [N] matching boards, got ", control.sizes());
    TORCH_CHECK(boards.scalar_type() == torch::kInt16, "boards must be int16 (spec §2.4)");
    TORCH_CHECK(control.scalar_type() == torch::kInt16, "control must be int16 (spec §2.2)");
    TORCH_CHECK(boards.is_contiguous() && control.is_contiguous(),
                "boards and control must be contiguous");
}

Luts make_luts(const torch::Tensor& move_bitsets, const torch::Tensor& occl_offsets,
               const torch::Tensor& occl_masks, const torch::Tensor& filled_lines) {
    TORCH_CHECK(move_bitsets.scalar_type() == torch::kInt64
                    && move_bitsets.numel() == 6 * 64,
                "move_bitsets must be [384] int64");
    TORCH_CHECK(occl_offsets.scalar_type() == torch::kInt16
                    && occl_offsets.numel() == 4 * 64 * 8,
                "occl_offsets must be [2048] int16");
    TORCH_CHECK(occl_masks.scalar_type() == torch::kUInt8 && occl_masks.numel() == 4 * 64 * 8,
                "occl_masks must be [2048] uint8");
    TORCH_CHECK(filled_lines.scalar_type() == torch::kUInt8 && filled_lines.numel() == 8 * 256,
                "filled_lines must be [2048] uint8");
    Luts luts;
    // int64 and uint64 hold the same 64 bits; torch has no usable uint64, which
    // is why the tables and the masks are signed on the Python side (docs/env.md).
    luts.move_bitsets = reinterpret_cast<const uint64_t*>(move_bitsets.data_ptr<int64_t>());
    luts.occl_offsets = occl_offsets.data_ptr<int16_t>();
    luts.occl_masks = occl_masks.data_ptr<uint8_t>();
    luts.filled_lines = filled_lines.data_ptr<uint8_t>();
    return luts;
}

inline const uint16_t* words(const torch::Tensor& boards) {
    return reinterpret_cast<const uint16_t*>(boards.data_ptr<int16_t>());
}
inline uint16_t* words_out(torch::Tensor& boards) {
    return reinterpret_cast<uint16_t*>(boards.data_ptr<int16_t>());
}
inline uint64_t* bits_out(torch::Tensor& t) {
    return reinterpret_cast<uint64_t*>(t.data_ptr<int64_t>());
}
inline const uint64_t* bits(const torch::Tensor& t) {
    return reinterpret_cast<const uint64_t*>(t.data_ptr<int64_t>());
}

int blocks_for(int64_t n) { return (int)((n + kWarpsPerBlock - 1) / kWarpsPerBlock); }

}  // namespace

// spec §4.1.
std::vector<torch::Tensor> movegen_cuda(torch::Tensor boards, torch::Tensor control,
                                        torch::Tensor move_bitsets,
                                        torch::Tensor occl_offsets, torch::Tensor occl_masks,
                                        torch::Tensor filled_lines) {
    check_state(boards, control);
    const int64_t n = boards.size(0);
    auto mask = torch::empty({n, 32}, boards.options().dtype(torch::kInt64));
    auto in_check = torch::empty({n}, boards.options().dtype(torch::kBool));
    if (n == 0) return {mask, in_check};

    movegen_kernel<<<blocks_for(n), kWarpsPerBlock * 32, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
        words(boards), control.data_ptr<int16_t>(),
        make_luts(move_bitsets, occl_offsets, occl_masks, filled_lines), bits_out(mask),
        reinterpret_cast<uint8_t*>(in_check.data_ptr<bool>()), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {mask, in_check};
}

// spec §4.2. `hash` empty means the untracked path, which skips the two en
// passant legality tests and returns the input hash unchanged; `irreversible`
// comes back either way, since it needs only the castling-rights delta.
std::vector<torch::Tensor> step_cuda(torch::Tensor boards, torch::Tensor control,
                                     torch::Tensor move, torch::Tensor promo,
                                     torch::Tensor hash, torch::Tensor move_bitsets,
                                     torch::Tensor occl_offsets, torch::Tensor occl_masks,
                                     torch::Tensor filled_lines) {
    check_state(boards, control);
    const int64_t n = boards.size(0);
    TORCH_CHECK(move.scalar_type() == torch::kInt16 && move.numel() == n,
                "move must be [N] int16 (slot * 64 + target, or -1)");
    TORCH_CHECK(promo.scalar_type() == torch::kUInt8 && promo.numel() == n,
                "promo must be [N] uint8");
    const bool track = hash.numel() > 0;
    TORCH_CHECK(!track || (hash.scalar_type() == torch::kInt64 && hash.numel() == n),
                "hash must be [N] int64 or empty");

    auto out_boards = torch::empty_like(boards);
    auto out_control = torch::empty_like(control);
    auto out_hash = track ? torch::empty({n}, boards.options().dtype(torch::kInt64))
                          : torch::empty({0}, boards.options().dtype(torch::kInt64));
    auto irreversible = torch::empty({n}, boards.options().dtype(torch::kBool));
    if (n == 0) return {out_boards, out_control, out_hash, irreversible};

    const Luts luts = make_luts(move_bitsets, occl_offsets, occl_masks, filled_lines);
    auto* irr = reinterpret_cast<uint8_t*>(irreversible.data_ptr<bool>());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (track) {
        step_full_kernel<true><<<blocks_for(n), kWarpsPerBlock * 32, 0, stream>>>(
            words(boards), control.data_ptr<int16_t>(), bits(hash), nullptr,
            move.data_ptr<int16_t>(), promo.data_ptr<uint8_t>(), luts, words_out(out_boards),
            out_control.data_ptr<int16_t>(), bits_out(out_hash), irr, (int)n);
    } else {
        step_full_kernel<false><<<blocks_for(n), kWarpsPerBlock * 32, 0, stream>>>(
            words(boards), control.data_ptr<int16_t>(), nullptr, nullptr,
            move.data_ptr<int16_t>(), promo.data_ptr<uint8_t>(), luts, words_out(out_boards),
            out_control.data_ptr<int16_t>(), nullptr, irr, (int)n);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_boards, out_control, out_hash, irreversible};
}

// spec §6.1, the full hash. Also returns the two intermediates, because a wrong
// hash is one 64-bit number and says nothing about which input was wrong.
std::vector<torch::Tensor> hash_position_cuda(torch::Tensor boards, torch::Tensor control,
                                              torch::Tensor move_bitsets,
                                              torch::Tensor occl_offsets,
                                              torch::Tensor occl_masks,
                                              torch::Tensor filled_lines) {
    check_state(boards, control);
    const int64_t n = boards.size(0);
    auto hash = torch::empty({n}, boards.options().dtype(torch::kInt64));
    auto rights = torch::empty({n}, boards.options().dtype(torch::kUInt8));
    auto ep = torch::empty({n}, boards.options().dtype(torch::kInt8));
    if (n == 0) return {hash, rights, ep};

    hash_position_kernel<<<blocks_for(n), kWarpsPerBlock * 32, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
        words(boards), control.data_ptr<int16_t>(),
        make_luts(move_bitsets, occl_offsets, occl_masks, filled_lines), bits_out(hash),
        rights.data_ptr<uint8_t>(), ep.data_ptr<int8_t>(), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {hash, rights, ep};
}

// spec §4.3. `ring` empty skips the repetition test, which is what a static
// position set needs: repetition is a property of the game that reached one.
std::vector<torch::Tensor> terminal_cuda(torch::Tensor boards, torch::Tensor control,
                                         torch::Tensor mask, torch::Tensor in_check,
                                         torch::Tensor hash, torch::Tensor ring,
                                         torch::Tensor length) {
    check_state(boards, control);
    const int64_t n = boards.size(0);
    TORCH_CHECK(mask.scalar_type() == torch::kInt64 && mask.dim() == 2 && mask.size(0) == n
                    && mask.size(1) == 32,
                "mask must be [N, 32] int64");
    TORCH_CHECK(in_check.scalar_type() == torch::kBool && in_check.numel() == n,
                "in_check must be [N] bool");
    const bool has_ring = ring.numel() > 0;
    TORCH_CHECK(!has_ring
                    || (ring.dim() == 2 && ring.size(0) == n && ring.size(1) == kMaxHistory
                        && ring.scalar_type() == torch::kInt64
                        && length.scalar_type() == torch::kInt64 && length.numel() == n
                        && hash.numel() == n),
                "ring must be [N, 100] int64 with a matching [N] int64 length and [N] hash");

    auto code = torch::empty({n}, boards.options().dtype(torch::kUInt8));
    auto result = torch::empty({n}, boards.options().dtype(torch::kInt8));
    if (n == 0) return {code, result};

    terminal_kernel<<<(int)((n + 7) / 8), 8 * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        words(boards), control.data_ptr<int16_t>(), bits(mask),
        reinterpret_cast<const uint8_t*>(in_check.data_ptr<bool>()),
        hash.numel() ? bits(hash) : nullptr, has_ring ? bits(ring) : nullptr,
        has_ring ? length.data_ptr<int64_t>() : nullptr, code.data_ptr<uint8_t>(),
        result.data_ptr<int8_t>(), (int)n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {code, result};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("movegen", &movegen_cuda, "fully legal moves and check status (spec §4.1)");
    m.def("step", &step_cuda, "apply one move per position (spec §4.2)");
    m.def("hash_position", &hash_position_cuda, "full Zobrist hash (spec §6.1)");
    m.def("terminal", &terminal_cuda, "terminal code and result (spec §4.3)");
}
