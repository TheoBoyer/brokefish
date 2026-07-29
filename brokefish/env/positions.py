"""Random real positions, for testing the move generator only.

These come from human games and therefore may never reach training: the tabula
rasa boundary forbids human games as a source of *labels or supervision*, and
this module exists to give the move generator adversarial positions to be wrong
about. Nothing here is imported by the training loop, and the ancestor's
`IterableDataset`/`DataModule`, which existed to feed a network, were dropped on
purpose rather than ported.

The shards are parquet files of PGN movetext; `scripts/fetch_test_positions.py`
downloads one. They are not committed.
"""

import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple, Union

import chess

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_tls = threading.local()
_BRACES = re.compile(r"\{[^}]*\}")
_MOVENUM = re.compile(r"\d+\.(?:\.\.)?")
_ANNOTATION = re.compile(r"(?<=\S)[!?]+")


def parse_movetext(movetext: str) -> List[str]:
    s = _BRACES.sub("", movetext)
    s = _MOVENUM.sub("", s)
    s = _ANNOTATION.sub("", s)
    return s.split()


def _rng() -> random.Random:
    r = getattr(_tls, "r", None)
    if r is None:
        r = random.Random(random.getrandbits(64))
        _tls.r = r
    return r


def sample_shards(n: int = 32, path: Union[str, Path, None] = None) -> List[Tuple]:
    """`n` random (parquet file, row group, row count) triples."""
    import pyarrow.parquet as pq

    path = DATA_DIR if path is None else Path(path)
    shards = []
    for f in sorted(path.glob("**/*.parquet")):
        pf = pq.ParquetFile(f)
        shards += [(pf, i, pf.metadata.row_group(i).num_rows)
                   for i in range(pf.metadata.num_row_groups)]
    if not shards:
        raise FileNotFoundError(
            f"no parquet shard under {path}; run scripts/fetch_test_positions.py"
        )
    random.shuffle(shards)
    return (shards * (1 + n // len(shards)))[:n]


def sample_board(shard: Tuple, ret: str = "board") -> Union[chess.Board, str]:
    """One position from a random game of one row group, at a random ply."""
    r = _rng()
    pf, row_group, n_rows = shard
    moves = []
    while len(moves) < 2:
        table = pf.read_row_group(row_group, columns=["movetext"])
        moves = parse_movetext(table["movetext"][r.randrange(n_rows)].as_py())
    board = chess.Board()
    for san in moves[:r.randrange(len(moves) - 1)]:
        board.push_san(san)
    return board.fen() if ret == "fen" else board


def sample_board_batch(shards: List[Tuple], executor: Optional[ThreadPoolExecutor] = None,
                       ret: str = "board") -> List[Union[chess.Board, str]]:
    if executor is None:
        return [sample_board(s, ret) for s in shards]
    return list(executor.map(sample_board, shards, [ret] * len(shards)))
