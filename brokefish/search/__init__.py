"""The search.

``torch_impl`` is the reference implementation of ``docs/mcts.md`` and the oracle
the CUDA search will be written against. The names re-exported here are the ones
both implementations must provide, so a caller that imports from
``brokefish.search`` can be switched between them.
"""

from .torch_impl import (
    MoveRecord,
    Search,
    SearchConfig,
    SearchStats,
    check_invariants,
    make_evaluator,
)

__all__ = [
    "Search", "SearchConfig", "SearchStats", "MoveRecord",
    "make_evaluator", "check_invariants",
]
