"""Constant tables the move generator indexes, built in torch at first use.

Four lookup tables plus the starting position. They are pure arithmetic over
64 squares, so they are rebuilt in-process rather than loaded from a cache file:
the ancestor engine shelled out to a generator script at import time and read
``cache/buffers.pt`` relative to the current working directory, which made the
package importable from exactly one place.

``dump()`` writes the same tables in the flat layout the CUDA ``Luts`` struct
expects (``csrc/movegen.cuh``). The two implementations own their own copy of the
tables; only the layout below is shared.

    move_bitsets   [6*64]    int64   (type * 64 + square) -> destination bitset
    occl_offsets   [4,64,8]  int16   (direction, square, i) -> square offset
    occl_masks     [4,64,8]  bool    (direction, square, i) -> offset is on the board
    filled_lines   [8,256]   uint8   (position in line, occupancy) -> reachable
"""

from pathlib import Path
from typing import Dict, Optional

import torch

# Slot layout, spec §2.1: 0-7 white pawns, 8-9 knights, 10-11 bishops, 12-13
# rooks, 14 queen, 15 king, then 16-31 mirroring it for black. The colour bit is
# added by _build(); the words below are (type << 6) | square.
_START_SQUARES = [
    8, 9, 10, 11, 12, 13, 14, 15,       # white pawns a2..h2
    65, 70,                             # white knights b1, g1
    130, 133,                           # white bishops c1, f1
    192, 199,                           # white rooks a1, h1
    259, 324,                           # white queen d1, king e1
    1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079,   # black pawns a7..h7
    1145, 1150,                         # black knights b8, g8
    1210, 1213,                         # black bishops c8, f8
    1272, 1279,                         # black rooks a8, h8
    1339, 1404,                         # black queen d8, king e8
]


def _fill_from_msb(x: torch.Tensor) -> torch.Tensor:
    """Fill with 1s downwards from the highest set bit. uint8 range only."""
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    return x


def _fill_from_lsb(x: torch.Tensor) -> torch.Tensor:
    """Fill with 1s upwards from the lowest set bit. uint8 range only."""
    x |= x << 1
    x |= x << 2
    x |= x << 4
    return x & 0xFF


def _build() -> Dict[str, torch.Tensor]:
    r8 = torch.arange(8)
    r64 = torch.arange(64)
    row, col = r64 // 8, r64 % 8

    # Per-(type, square) destination sets, ignoring occupancy entirely. Pawns are
    # zero here: their moves depend on colour and occupancy, so they are built in
    # the generator instead.
    pawn = torch.zeros(64, dtype=torch.int64)
    knight = (
        (col < 7) << (r64 + 17) | (col < 6) << (r64 + 10)
        | (col < 6) << (r64 - 6) | (col < 7) << (r64 - 15)
        | (col > 0) << (r64 - 17) | (col > 1) << (r64 - 10)
        | (col > 1) << (r64 + 6) | (col > 0) << (r64 + 15)
    ).to(torch.int64)
    king = (
        (row < 7) << (r64 + 8) | ((row < 7) & (col < 7)) << (r64 + 9)
        | (col < 7) << (r64 + 1) | ((row > 0) & (col < 7)) << (r64 - 7)
        | (row > 0) << (r64 - 8) | ((row > 0) & (col > 0)) << (r64 - 9)
        | (col > 0) << (r64 - 1) | ((row < 7) & (col > 0)) << (r64 + 7)
    ).to(torch.int64)

    # The four slider directions, each described as the eight squares of the line
    # through a given square plus a validity mask for the ones that fall off it.
    pos_diag_offset = torch.where(row < col, col - row, 8 * (row - col))[:, None] + r8 * 9
    pos_diag_mask = r8 < 8 - torch.abs(row - col)[:, None]
    neg_diag_offset = torch.where(row <= 7 - col, 8 * (row + col), row + col + 49)[:, None] - r8 * 7
    neg_diag_mask = r8 < 8 - torch.abs(row + col - 7)[:, None]
    row_offset = (row * 8)[:, None] + r8
    row_mask = torch.ones(64, 8, dtype=torch.bool)
    col_offset = col[:, None] + r8 * 8
    col_mask = torch.ones(64, 8, dtype=torch.bool)

    bishop = ((pos_diag_mask << pos_diag_offset).sum(-1)
              | (neg_diag_mask << neg_diag_offset).sum(-1)).to(torch.int64)
    rook = ((row_mask << row_offset).sum(-1) | (col_mask << col_offset).sum(-1)).to(torch.int64)
    queen = bishop | rook

    # For every position in an 8-square line and every occupancy of that line,
    # which squares a slider standing there reaches: everything up to and
    # including the first blocker on each side, itself excluded.
    lines = torch.arange(256)[None].repeat(8, 1).view(8, -1)
    left = lines & ((1 << r8) - 1)[:, None]
    left = _fill_from_msb(left) >> 1
    right = lines >> (r8 + 1)[:, None]
    right = (_fill_from_lsb(right) << 1) & ((1 << (7 - r8)) - 1)[:, None]
    filled_lines = ~(left | (right << (r8 + 1)[:, None]))
    filled_lines ^= 1 << r8[:, None]

    start_board = torch.tensor(_START_SQUARES, dtype=torch.int16)
    start_board[16:] |= 1 << 10  # colour bit on the black half

    return {
        "start_board": start_board,
        "move_bitsets": torch.cat([pawn, knight, bishop, rook, queen, king], dim=0),
        "occl_offsets": torch.cat(
            [pos_diag_offset, neg_diag_offset, row_offset, col_offset], dim=0
        ).to(torch.int16).view(4, 64, 8),
        "occl_masks": torch.cat(
            [pos_diag_mask, neg_diag_mask, row_mask, col_mask], dim=0
        ).to(torch.bool).view(4, 64, 8),
        "filled_lines": filled_lines.to(torch.uint8),
        "range8": r8,
        "range64": r64,
        "zobrist": _zobrist_keys(),
    }


# spec §6.1: 781 keys, generated once from a fixed seed. Laid out as one flat
# int64 table so the CUDA side loads 6.25 KB into shared memory and indexes it
# with the offsets below.
ZOBRIST_PIECE = 0     # 768 entries, (colour * 6 + type) * 64 + square
ZOBRIST_SIDE = 768    # 1, xored in when black is to move
ZOBRIST_CASTLE = 769  # 4, in FEN order KQkq, xored in when the right is present
ZOBRIST_EP = 773      # 8, by file, xored in only when an ep capture is legal
ZOBRIST_KEYS = 781

_SPLITMIX64_SEED = 0x00B4C0FFEE12F00D
_M = (1 << 64) - 1


def _zobrist_keys() -> torch.Tensor:
    """The key table, as int64 carrying the uint64 bit patterns.

    splitmix64 rather than a shipped binary or `torch.randint`: it is ten lines
    of shift-multiply-xor that produce identical output in Python and in CUDA, so
    the CUDA engine can regenerate the table instead of loading one, and the two
    cannot drift. torch's own RNG would pin us to a torch version.
    """
    keys, x = [], _SPLITMIX64_SEED
    for _ in range(ZOBRIST_KEYS):
        x = (x + 0x9E3779B97F4A7C15) & _M
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
        z ^= z >> 31
        # int64 holds the same 64 bits; torch has no usable uint64 arithmetic.
        keys.append(z - (1 << 64) if z >> 63 else z)
    return torch.tensor(keys, dtype=torch.int64)


_bank: Dict[str, Dict[str, torch.Tensor]] = {}


def get(name: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """One table, on one device, built once and cached per device."""
    key = "cpu" if device is None else str(device)
    if "cpu" not in _bank:
        _bank["cpu"] = _build()
    if key not in _bank:
        _bank[key] = {}
    if name not in _bank[key]:
        _bank[key][name] = _bank["cpu"][name].to(device)
    return _bank[key][name]


def dump(out_dir: Path) -> None:
    """Write the flat binaries the CUDA movegen loads. Dtypes are normative."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    get("move_bitsets").numpy().tofile(out_dir / "lut_move_bitsets.bin")   # int64
    get("occl_offsets").numpy().tofile(out_dir / "lut_occl_offsets.bin")   # int16
    get("occl_masks").numpy().astype("uint8").tofile(out_dir / "lut_occl_masks.bin")
    get("filled_lines").numpy().tofile(out_dir / "lut_filled_lines.bin")   # uint8
    get("zobrist").numpy().tofile(out_dir / "lut_zobrist.bin")             # int64, 781 keys
