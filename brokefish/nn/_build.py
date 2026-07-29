"""Compiling CUDA C++ against this venv's torch.

One function, :func:`load_extension`, plus the toolchain discovery it needs.
`scripts/check_cuda_build.py` exercises the whole path on a trivial kernel and
is the thing to run first on a new machine.

Why discovery is not trivial here. ``torch.utils.cpp_extension`` derives
``CUDA_HOME`` from whichever ``nvcc`` is first on ``PATH``, then refuses to
build if that toolkit's major version differs from the one torch was compiled
with. On this machine ``/usr/bin/nvcc`` is Ubuntu's ``nvidia-cuda-toolkit``
(12.0) and there is a second toolkit at ``/usr/local/cuda-12.8``, while torch is
built against 13.2 -- so the default path fails with a version mismatch and
neither installed toolkit can fix it. This module ignores ``PATH`` and looks for
a toolkit whose major matches ``torch.version.cuda``, including the pip wheel
form, and says exactly what it looked at when it finds none.

Build flags follow ``csrc/README.md``: ``-arch=sm_89 -O3 -lineinfo`` plus
``-Xptxas -v``, which prints registers, spills and shared memory per kernel.
Those three numbers are the occupancy story on this card, so the build log is
kept next to the object files rather than thrown away.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
CSRC = REPO / "csrc"
BUILD_ROOT = Path(os.environ.get("BROKEFISH_BUILD_DIR", Path.home() / ".cache" / "brokefish" / "ext"))

# -Xptxas -v is not optional: registers per thread are what caps occupancy on
# this kernel, and a silent build hides the one number that predicts the change.
CUDA_FLAGS = ["-arch=sm_89", "-O3", "-lineinfo", "-Xptxas", "-v"]
CXX_FLAGS = ["-O3"]


def _nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _candidate_homes() -> list[Path]:
    """Places a toolkit may live, most explicit first.

    The pip wheel form is included because ``nvidia-cuda-nvcc-cuXX`` installs a
    complete ``bin/nvcc`` inside site-packages and needs no root, which is the
    fallback when the system toolkit cannot be upgraded.
    """
    seen, out = set(), []
    for raw in (os.environ.get("BROKEFISH_CUDA_HOME"), os.environ.get("CUDA_HOME"),
                os.environ.get("CUDA_PATH")):
        if raw:
            out.append(Path(raw))
    out += sorted((p for p in Path("/usr/local").glob("cuda-*") if (p / "bin" / "nvcc").exists()),
                  reverse=True)
    out.append(Path("/usr/local/cuda"))
    site = Path(torch.__file__).resolve().parents[1]
    out += sorted(p.parent.parent for p in site.glob("nvidia/*/bin/nvcc"))
    out.append(Path("/usr"))
    return [p for p in out if not (p in seen or seen.add(p))]


@functools.lru_cache(maxsize=1)
def cuda_home() -> Path:
    """A toolkit whose major version matches the one torch was built against.

    Raises ``RuntimeError`` listing every path tried and the version found
    there, because "CUDA not found" on a machine holding three toolkits is not
    a useful error message.
    """
    want = torch.version.cuda
    if want is None:
        raise RuntimeError("this torch build has no CUDA support")
    want_major = int(want.split(".")[0])

    tried = []
    for home in _candidate_homes():
        nvcc = home / "bin" / "nvcc"
        if not nvcc.exists():
            tried.append(f"  {home}  (no bin/nvcc)")
            continue
        found = _nvcc_version(nvcc)
        if found is None:
            tried.append(f"  {home}  (nvcc will not run)")
        elif found[0] != want_major:
            tried.append(f"  {home}  (nvcc {found[0]}.{found[1]}, torch wants {want})")
        else:
            return home

    listing = "\n".join(tried) or "  nothing"
    raise RuntimeError(
        f"no CUDA toolkit with major version {want_major} (torch is built against {want}).\n"
        f"Tried:\n{listing}\n"
        f"Fix it with either:\n"
        f"  sudo apt install cuda-nvcc-13-2 cuda-cudart-dev-13-2 cuda-cccl-13-2   (needs NVIDIA's apt repo)\n"
        f"  uv pip install -p .venv/bin/python nvidia-cuda-nvcc-cu13 ninja        (no root)\n"
        f"or point BROKEFISH_CUDA_HOME at one that is already installed."
    )


def _require_ninja() -> None:
    try:
        import ninja  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "ninja is required to build CUDA extensions:\n"
            "  uv pip install -p .venv/bin/python ninja"
        ) from None


def build_dir(name: str) -> Path:
    d = BUILD_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_extension(name: str, sources, verbose: bool = False, extra_cuda_cflags=None):
    """Compile and import a CUDA extension out of ``csrc/``.

    ``sources`` are names relative to ``csrc/`` or absolute paths. The result is
    cached by ninja under :func:`build_dir`, so an unchanged source is a no-op
    and the first build of a kernel this size costs tens of seconds.

    Discovery runs before ``torch.utils.cpp_extension`` is imported, since that
    module resolves ``CUDA_HOME`` at import time from ``PATH``.
    """
    home = cuda_home()
    os.environ["CUDA_HOME"] = str(home)
    _require_ninja()

    from torch.utils import cpp_extension

    # Belt and braces: the module may already have been imported by something
    # else, in which case its CUDA_HOME global is the stale PATH-derived one.
    cpp_extension.CUDA_HOME = str(home)

    paths = [str(p if Path(p).is_absolute() else CSRC / p) for p in sources]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"no such source: {', '.join(missing)}")

    return cpp_extension.load(
        name=name,
        sources=paths,
        extra_cflags=CXX_FLAGS,
        extra_cuda_cflags=CUDA_FLAGS + list(extra_cuda_cflags or []),
        extra_include_paths=[str(CSRC)],
        build_directory=str(build_dir(name)),
        verbose=verbose,
    )


def toolchain_report() -> str:
    """One paragraph naming what would be used, for the harnesses to print."""
    try:
        home = cuda_home()
    except RuntimeError as exc:
        return str(exc)
    version = _nvcc_version(home / "bin" / "nvcc")
    return (f"nvcc {version[0]}.{version[1]} from {home}, torch {torch.__version__} "
            f"(cuda {torch.version.cuda}), building into {BUILD_ROOT}")
