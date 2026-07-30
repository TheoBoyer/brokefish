"""The search.

``torch_impl`` is the reference implementation of ``docs/mcts.md`` and the oracle
the CUDA search is written against; ``cuda_impl`` is the CUDA search, a subclass
of it with the four per-simulation steps replaced by kernels. The names
re-exported here are the ones both implementations provide, so a caller that
imports from ``brokefish.search`` can be switched between them.

Selection is by name, as in :mod:`brokefish.nn`, and for the same reason: both
live in one process so a benchmark can interleave them and §12's differential
harness can run one tree through each.
"""

from .torch_impl import (
    MoveRecord,
    Search,
    SearchConfig,
    SearchStats,
    check_invariants,
    make_evaluator,
)

IMPLEMENTATIONS = ("torch", "cuda")

_WHY: dict[str, str] = {}


def search_impl(name: str):
    """The ``Search`` class of one implementation.

    Raises ``ValueError`` for an unknown name and lets the implementation's own
    import error through otherwise: a CUDA kernel that fails to compile has to
    say so, not disappear.
    """
    if name == "torch":
        return Search
    if name == "cuda":
        from brokefish.search.cuda_impl import Search as CudaSearch
        return CudaSearch
    raise ValueError(f"unknown implementation {name!r}, expected one of {IMPLEMENTATIONS}")


def available() -> list[str]:
    """The implementations importable here, in registry order.

    A failure to import is recorded rather than fatal, since the CUDA one needs a
    toolchain a fresh machine may not have, and :func:`why_unavailable` prints the
    reason so a skip is never the end of the story.
    """
    found = []
    for name in IMPLEMENTATIONS:
        try:
            search_impl(name)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            _WHY[name] = f"{type(exc).__name__}: {exc}"
        else:
            found.append(name)
            _WHY.pop(name, None)
    return found


def why_unavailable() -> dict[str, str]:
    """Why each missing implementation is missing. Call :func:`available` first."""
    return dict(_WHY)


__all__ = [
    "Search", "SearchConfig", "SearchStats", "MoveRecord",
    "make_evaluator", "check_invariants",
    "IMPLEMENTATIONS", "search_impl", "available", "why_unavailable",
]
