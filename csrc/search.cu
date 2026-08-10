// Torch bindings for the MCTS kernels of csrc/search.cuh.
//
// Four entry points, one per §9 kernel. The tree of §4.2 is 25 tensors and they
// arrive as a **dict keyed by name**, not as a positional list: two thirds of
// them share a dtype and a shape, so a permutation in the caller would pass every
// check and produce a plausible tree. The dict also means adding a field is one
// line on each side rather than a renumbering.
//
// The tensors are allocated and owned by brokefish/search/cuda_impl.py, and they
// are the same set brokefish/search/torch_impl.py allocates, so §12's harness can
// hand one tree to both implementations and swap a single kernel.
//
// Stateless for the same reason csrc/engine.cu is: a cached device pointer would
// be wrong exactly once, on a device change, and would then read someone else's
// memory and produce plausible numbers.
//
// Unlike engine.cu these *are* on the self-play path. Four launches per
// simulation is 3202 per move at n = 800, about 0.06 ms against a 65 ms
// simulation step, so the host-driven loop is not what will cost anything.
#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <map>
#include <string>
#include <vector>

#include "search.cuh"

using namespace brokefish;
using namespace brokefish::search;

namespace {

using TensorMap = std::map<std::string, torch::Tensor>;

const torch::Tensor& lookup(const TensorMap& m, const char* name, torch::ScalarType dtype,
                            int64_t dim) {
    auto it = m.find(name);
    TORCH_CHECK(it != m.end(), "tree is missing '", name, "'");
    const torch::Tensor& t = it->second;
    TORCH_CHECK(t.is_cuda(), "'", name, "' must be on CUDA");
    TORCH_CHECK(t.is_contiguous(), "'", name, "' must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, "'", name, "' must be ", dtype, ", got ",
                t.scalar_type());
    TORCH_CHECK(t.dim() == dim, "'", name, "' must have ", dim, " dimensions, got ", t.dim());
    return t;
}

template <typename T>
T* raw(const torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr());
}

Tree make_tree(const TensorMap& m) {
    Tree t;

    const auto& node_board = lookup(m, "node_board", torch::kInt16, 3);
    TORCH_CHECK(node_board.size(2) == 32, "node_board must be [B, Nmax, 32]");
    t.B = (int)node_board.size(0);
    t.N = (int)node_board.size(1);
    t.node_board = raw<uint16_t>(node_board);

    const int B = t.B, N = t.N;
    auto node_field = [&](const char* name, torch::ScalarType dt) -> const torch::Tensor& {
        const torch::Tensor& x = lookup(m, name, dt, 2);
        TORCH_CHECK(x.size(0) == B && x.size(1) == N, "'", name, "' must be [B, Nmax] = [", B,
                    ", ", N, "], got ", x.sizes());
        return x;
    };
    auto edge_field = [&](const char* name, torch::ScalarType dt) -> const torch::Tensor& {
        const torch::Tensor& x = lookup(m, name, dt, 3);
        TORCH_CHECK(x.size(0) == B && x.size(1) == N && x.size(2) == kE, "'", name,
                    "' must be [B, Nmax, ", kE, "], got ", x.sizes(),
                    ". E is a compile-time constant because §6.6's scan gives each "
                    "lane kE/32 edges with no predicate on the tail");
        return x;
    };
    auto game_field = [&](const char* name, torch::ScalarType dt) -> const torch::Tensor& {
        const torch::Tensor& x = lookup(m, name, dt, 1);
        TORCH_CHECK(x.size(0) == B, "'", name, "' must be [B] = [", B, "], got ", x.sizes());
        return x;
    };

    t.node_control = raw<int16_t>(node_field("node_control", torch::kInt16));
    t.node_hash = raw<uint64_t>(node_field("node_hash", torch::kInt64));
    t.node_value = raw<__half>(node_field("node_value", torch::kHalf));
    t.node_nedges = raw<uint8_t>(node_field("node_nedges", torch::kUInt8));
    t.node_flags = raw<uint8_t>(node_field("node_flags", torch::kUInt8));
    t.node_parent = raw<int16_t>(node_field("node_parent", torch::kInt16));
    t.node_pedge = raw<uint8_t>(node_field("node_pedge", torch::kUInt8));

    t.edge_move = raw<int16_t>(edge_field("edge_move", torch::kInt16));
    t.edge_prior = raw<__half>(edge_field("edge_prior", torch::kHalf));
    t.edge_child = raw<int16_t>(edge_field("edge_child", torch::kInt16));
    t.edge_N = raw<int16_t>(edge_field("edge_N", torch::kInt16));
    t.edge_Q = raw<float>(edge_field("edge_Q", torch::kFloat));

    // §6.6a's collapse mask. The one tree field that is allowed to be **empty**:
    // `torch_impl` allocates it only when `terminal_collapse` is on, because it is
    // 315 MB at the Gate 1a shape, and a null pointer is what the kernels read as
    // "off". Anything other than empty or the full edge shape is a caller bug and
    // must not be silently treated as off -- a [B, N] tensor here would mean the
    // collapse quietly stopped happening in a run that asked for it.
    {
        auto it = m.find("edge_win");
        TORCH_CHECK(it != m.end(), "tree is missing 'edge_win'");
        const torch::Tensor& x = it->second;
        TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kUInt8,
                    "'edge_win' must be a contiguous uint8 CUDA tensor");
        if (x.numel() == 0) {
            t.edge_win = nullptr;
        } else {
            TORCH_CHECK(x.dim() == 3 && x.size(0) == B && x.size(1) == N
                            && x.size(2) == kE,
                        "'edge_win' must be empty or [B, Nmax, ", kE, "], got ",
                        x.sizes());
            t.edge_win = raw<uint8_t>(x);
        }
    }

    const auto& path_node = lookup(m, "path_node", torch::kInt16, 2);
    TORCH_CHECK(path_node.size(0) == B, "path_node must be [B, Dmax]");
    t.D = (int)path_node.size(1);
    t.path_node = raw<int16_t>(path_node);
    const auto& path_edge = lookup(m, "path_edge", torch::kUInt8, 2);
    TORCH_CHECK(path_edge.size(0) == B && path_edge.size(1) == t.D,
                "path_edge must match path_node");
    t.path_edge = raw<uint8_t>(path_edge);
    t.path_len = raw<int32_t>(game_field("path_len", torch::kInt32));

    const auto& ring = lookup(m, "game_ring", torch::kInt64, 2);
    TORCH_CHECK(ring.size(0) == B && ring.size(1) == kMaxHistory, "game_ring must be [B, ",
                kMaxHistory, "], got ", ring.sizes());
    t.game_ring = raw<uint64_t>(ring);
    t.game_ring_len = raw<int64_t>(game_field("game_ring_len", torch::kInt64));
    t.node_count = raw<int32_t>(game_field("node_count", torch::kInt32));
    t.budget = raw<int32_t>(game_field("budget", torch::kInt32));

    const auto& leaf_board = lookup(m, "leaf_board", torch::kInt16, 2);
    TORCH_CHECK(leaf_board.size(0) == B && leaf_board.size(1) == 32,
                "leaf_board must be [B, 32], got ", leaf_board.sizes());
    t.leaf_board = raw<uint16_t>(leaf_board);
    t.leaf_control = raw<int16_t>(game_field("leaf_control", torch::kInt16));
    t.leaf_rep = raw<uint8_t>(game_field("leaf_rep", torch::kUInt8));
    t.leaf_node = raw<int16_t>(game_field("leaf_node", torch::kInt16));
    t.leaf_flags = raw<uint8_t>(game_field("leaf_flags", torch::kUInt8));
    return t;
}

Luts make_luts(const torch::Tensor& move_bitsets, const torch::Tensor& occl_offsets,
               const torch::Tensor& occl_masks, const torch::Tensor& filled_lines) {
    TORCH_CHECK(move_bitsets.scalar_type() == torch::kInt64 && move_bitsets.numel() == 6 * 64,
                "move_bitsets must be [384] int64");
    TORCH_CHECK(occl_offsets.scalar_type() == torch::kInt16
                    && occl_offsets.numel() == 4 * 64 * 8,
                "occl_offsets must be [2048] int16");
    TORCH_CHECK(occl_masks.scalar_type() == torch::kUInt8 && occl_masks.numel() == 4 * 64 * 8,
                "occl_masks must be [2048] uint8");
    TORCH_CHECK(filled_lines.scalar_type() == torch::kUInt8 && filled_lines.numel() == 8 * 256,
                "filled_lines must be [2048] uint8");
    Luts l;
    l.move_bitsets = raw<uint64_t>(move_bitsets);
    l.occl_offsets = raw<int16_t>(occl_offsets);
    l.occl_masks = raw<uint8_t>(occl_masks);
    l.filled_lines = raw<uint8_t>(filled_lines);
    return l;
}

// `counters` is an opaque byte buffer rather than a struct of tensors, so the
// kernel can atomicAdd into it with no packing. Python decodes it through
// `counter_format()` and `counter_names()` below, which come from this file so
// that the two cannot drift.
Counters* counter_ptr(const torch::Tensor& c) {
    if (c.numel() == 0) return nullptr;
    TORCH_CHECK(c.is_cuda() && c.is_contiguous() && c.scalar_type() == torch::kUInt8
                    && c.numel() == (int64_t)sizeof(Counters),
                "counters must be an empty tensor or [", sizeof(Counters), "] uint8 on CUDA");
    return reinterpret_cast<Counters*>(c.data_ptr());
}

int blocks_for(int b) { return (b + kWarps - 1) / kWarps; }

}  // namespace

void root_init_cuda(TensorMap tree, torch::Tensor game_board, torch::Tensor game_control,
                    torch::Tensor game_hash, torch::Tensor root_rep) {
    const Tree t = make_tree(tree);
    TORCH_CHECK(game_board.scalar_type() == torch::kInt16 && game_board.dim() == 2
                    && game_board.size(0) == t.B && game_board.size(1) == 32,
                "game_board must be [B, 32] int16");
    TORCH_CHECK(game_control.scalar_type() == torch::kInt16 && game_control.numel() == t.B,
                "game_control must be [B] int16");
    TORCH_CHECK(game_hash.scalar_type() == torch::kInt64 && game_hash.numel() == t.B,
                "game_hash must be [B] int64");
    TORCH_CHECK(root_rep.scalar_type() == torch::kUInt8 && root_rep.numel() == t.B,
                "root_rep must be [B] uint8");
    if (t.B == 0) return;

    root_init_kernel<<<blocks_for(t.B), kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        t, raw<uint16_t>(game_board), raw<int16_t>(game_control), raw<uint64_t>(game_hash),
        raw<uint8_t>(root_rep));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void descent_cuda(TensorMap tree, int64_t s, double pb_c_base, double pb_c_init,
                  torch::Tensor move_bitsets, torch::Tensor occl_offsets,
                  torch::Tensor occl_masks, torch::Tensor filled_lines,
                  torch::Tensor counters, torch::Tensor root_gumbel,
                  torch::Tensor visit_table, int64_t gumbel_m, double c_visit,
                  double c_scale, bool gumbel_interior) {
    const Tree t = make_tree(tree);
    if (t.B == 0) return;
    Params p{(float)pb_c_base, (float)pb_c_init};

    // §11's Gumbel. Empty tensors are how the host says "off", the same convention
    // `edge_win` and `counters` use, so there is no flag that can disagree with an
    // allocation. Checked here rather than trusted: the kernel indexes
    // `table[m * budget + idx]` with `m` up to `gumbel_m` and `idx` up to
    // `budget - 1`, and a table built for a different budget would read a valid
    // address holding the wrong schedule -- a tree that is wrong and not a crash.
    Gumbel gp{nullptr, nullptr, (int)gumbel_m, 1, (float)c_visit, (float)c_scale,
              gumbel_interior};
    if (root_gumbel.numel()) {
        TORCH_CHECK(root_gumbel.scalar_type() == torch::kFloat32
                        && root_gumbel.dim() == 2 && root_gumbel.size(0) == t.B
                        && root_gumbel.size(1) == kE,
                    "root_gumbel must be [B, ", kE, "] float32, got ",
                    root_gumbel.sizes());
        TORCH_CHECK(visit_table.scalar_type() == torch::kInt32 && visit_table.dim() == 2
                        && visit_table.size(0) == gumbel_m + 1,
                    "visit_table must be [gumbel_m + 1, budget] int32, got ",
                    visit_table.sizes());
        TORCH_CHECK(root_gumbel.is_contiguous() && visit_table.is_contiguous(),
                    "root_gumbel and visit_table must be contiguous");
        gp.root_gumbel = raw<float>(root_gumbel);
        gp.table = raw<int32_t>(visit_table);
        gp.budget = (int)visit_table.size(1);
    }

    descent_kernel<<<blocks_for(t.B), kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        t, make_luts(move_bitsets, occl_offsets, occl_masks, filled_lines), p, gp, (int)s,
        counter_ptr(counters));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand_cuda(TensorMap tree, torch::Tensor policy, torch::Tensor promo,
                 torch::Tensor value, torch::Tensor move_bitsets,
                 torch::Tensor occl_offsets, torch::Tensor occl_masks,
                 torch::Tensor filled_lines, torch::Tensor counters) {
    const Tree t = make_tree(tree);
    TORCH_CHECK(policy.scalar_type() == torch::kHalf && policy.dim() == 3
                    && policy.size(0) == t.B && policy.size(1) == 32 && policy.size(2) == 64,
                "policy must be [B, 32, 64] fp16, got ", policy.sizes());
    TORCH_CHECK(promo.scalar_type() == torch::kHalf && promo.dim() == 3
                    && promo.size(0) == t.B && promo.size(1) == 32 && promo.size(2) == 4,
                "promo must be [B, 32, 4] fp16, got ", promo.sizes());
    TORCH_CHECK(value.scalar_type() == torch::kFloat && value.numel() == t.B,
                "value must be [B] fp32");
    TORCH_CHECK(policy.is_contiguous() && promo.is_contiguous() && value.is_contiguous(),
                "the encoder outputs must be contiguous");
    if (t.B == 0) return;

    expand_kernel<<<blocks_for(t.B), kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        t, make_luts(move_bitsets, occl_offsets, occl_masks, filled_lines),
        raw<__half>(policy), raw<__half>(promo), raw<float>(value), counter_ptr(counters));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void backup_cuda(TensorMap tree) {
    const Tree t = make_tree(tree);
    if (t.B == 0) return;
    backup_kernel<<<blocks_for(t.B), kWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(t);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("root_init", &root_init_cuda, "§6.1, node 0 and the encoder staging");
    m.def("descent", &descent_cuda, "§6.2, select, step, repetition, terminal, allocate");
    m.def("expand", &expand_cuda, "§6.4, the legality mask into edges");
    m.def("backup", &backup_cuda, "§6.5, the leaf value up the path");
    m.def("counter_size", []() { return (int64_t)sizeof(Counters); });
    // The decode contract for the §15 block, from the file that owns the struct.
    m.def("counter_format", []() { return std::string("<8Q3If"); });
    m.def("counter_names", []() {
        return std::vector<std::string>{
            "simulations", "terminal_descents", "terminal_children", "truncated_nodes",
            "empty_mask_expansions", "pool_overflow", "depth_overflow", "depth_sum",
            "max_edges", "max_depth", "max_nodes", "truncated_mass"};
    });
    m.def("edge_cap", []() { return (int64_t)kE; });
}
