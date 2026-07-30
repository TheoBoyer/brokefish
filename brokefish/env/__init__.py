"""The chess environment.

`torch_impl` is the reference implementation and the oracle the CUDA movegen is
written against; `cuda_impl` will land beside it with the same signatures. The
names re-exported here are the ones both implementations must provide, so a
caller that imports from `brokefish.env` can be switched between them.

python-chess conversions live in `interop`, and are imported explicitly rather
than re-exported: they are test scaffolding and drag in a dependency the engine
itself does not have.

`notation` is the output side — FEN, UCI, SAN and PGN — and is re-exported here
because it needs no dependency the engine does not already have. It binds to
`torch_impl` rather than to the selected implementation on purpose: it is
host-side string I/O for evaluation and debugging, never on the self-play path,
and making a PGN writer require a CUDA toolchain would be absurd.
"""

from .torch_impl import (
    castling_rights,
    empty_history,
    hash_position,
    insufficient_material,
    legal_ep_file,
    push_history,
    repetition_count,
    terminal,
    from_board,
    from_boards,
    from_fen,
    from_pgn,
    initial_boards,
    empty_boards,
    movegen,
    play,
    step,
    bitset_to_bool,
    bool_to_bitset,
    decode,
)
from .notation import (
    GameRecorder,
    STARTPOS_FEN,
    to_fen,
    to_pgn,
    to_san,
    to_uci,
)

__all__ = [
    "movegen", "step", "play", "terminal",
    "hash_position", "castling_rights", "legal_ep_file", "insufficient_material",
    "empty_history", "push_history", "repetition_count",
    "initial_boards", "empty_boards", "from_fen", "from_board", "from_boards", "from_pgn",
    "bitset_to_bool", "bool_to_bitset", "decode",
    "to_fen", "to_uci", "to_san", "to_pgn", "GameRecorder", "STARTPOS_FEN",
]
