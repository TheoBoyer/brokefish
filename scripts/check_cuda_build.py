"""Prove the CUDA toolchain end to end, before any real kernel depends on it.

Compiles a three-line kernel through the same path `brokefish.nn.cuda_impl`
uses, launches it on a torch tensor, and checks the result. Run it first on any
new machine, and again after a toolkit change:

    .venv/bin/python scripts/check_cuda_build.py

It also prints the `-Xptxas -v` line, which is where registers, spills and
shared memory per kernel come from. Those three numbers are the occupancy story
on this card (docs/perf.md), so it is worth knowing they arrive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from brokefish.nn import _build

PROBE = r"""
#include <torch/extension.h>

// Deliberately trivial and deliberately not in csrc/: this file exists to test
// the compiler, not the network. If it stops working the toolchain broke, not
// the kernel.
__global__ void bump_kernel(__half* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __hadd(x[i], __float2half(1.0f));
}

void bump(torch::Tensor x) {
    int n = x.numel();
    bump_kernel<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<__half*>(x.data_ptr<at::Half>()), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("bump", &bump, "x += 1"); }
"""


def main() -> int:
    print(_build.toolchain_report())
    try:
        _build.cuda_home()
    except RuntimeError:
        return 1
    if not torch.cuda.is_available():
        print("no CUDA device visible to torch")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"sm{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    src = _build.build_dir("brokefish_probe") / "probe.cu"
    src.write_text(PROBE)

    print("\ncompiling (first build takes a few tens of seconds)...\n")
    try:
        mod = _build.load_extension("brokefish_probe", [src], verbose=True)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        print(f"\nBUILD FAILED: {type(exc).__name__}: {exc}")
        return 1

    x = torch.zeros(1024, device="cuda", dtype=torch.half)
    mod.bump(x)
    torch.cuda.synchronize()
    if not bool((x == 1).all()):
        print(f"\nkernel ran but produced {x[:4].tolist()}, expected ones")
        return 1

    print(f"\n[OK] compiled, launched and verified. Build tree: "
          f"{_build.build_dir('brokefish_probe')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
