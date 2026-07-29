"""The chess environment on CUDA, behind `torch_impl`'s signatures.

Swapping ``from . import torch_impl as env`` for ``from . import cuda_impl as env``
changes the implementation and nothing else. The four device entry points of
[spec §4](../../docs/spec.md) plus the hash come from `csrc/engine.cu`; the
constructors, the python-chess conversions and the repetition ring stay in
`torch_impl`, because they are host-side helpers rather than kernels and there is
no second implementation of them to be wrong.

⚠️ **Nothing here is on the self-play path.** spec §4 forbids the engine from
synchronising with the host inside a step, and one kernel launch per call is a
host-driven shape. This module exists so the kernels can be tested against the
reference, benchmarked against the encoder, and driven from a notebook. C1's
descent calls the device functions in `csrc/*.cuh` directly, inside its own
kernel, with no launch between the movegen and the network.

Differences from `torch_impl`, both deliberate:

* ``in_place`` is accepted and ignored. It is documented as a hint in the
  reference too; here every call allocates.
* passing ``mask`` to :func:`step` turns on the legality assertion, which costs a
  device-to-host synchronisation. The reference pays the same price; it is just
  more obviously wrong to do in a loop here.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional, Tuple

import torch

from . import luts
from .torch_impl import (MAX_HISTORY, bitset_to_bool, bool_to_bitset,  # noqa: F401
                         castling_rights, decode, empty_boards, empty_history, from_board,
                         from_boards, from_fen, from_pgn, initial_boards,
                         insufficient_material, legal_ep_file, push_history,
                         repetition_count)

# The build helper lives under `nn/` because that is where the first CUDA kernel
# landed; it is the repository's single entry point for compiling against torch's
# own toolkit and has nothing network-specific in it.
from ..nn._build import load_extension


@functools.lru_cache(maxsize=1)
def _ext():
    return load_extension("brokefish_engine", ["engine.cu"])


_LUT_CACHE: Dict[str, Tuple[torch.Tensor, ...]] = {}


def _tables(device: torch.device) -> Tuple[torch.Tensor, ...]:
    """The four movegen tables on `device`, in the dtypes `engine.cu` checks for.

    `occl_masks` is bool in `luts` and `uint8` in the kernel, so the cast is done
    once here rather than per call.
    """
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = (
            luts.get("move_bitsets", device).contiguous(),
            luts.get("occl_offsets", device).reshape(-1).contiguous(),
            luts.get("occl_masks", device).reshape(-1).to(torch.uint8).contiguous(),
            luts.get("filled_lines", device).reshape(-1).contiguous(),
        )
    return _LUT_CACHE[key]


def _state(boards: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not boards.is_cuda:
        raise ValueError("cuda_impl needs CUDA tensors; use torch_impl on CPU")
    return boards.to(torch.int16).contiguous(), control.to(torch.int16).contiguous()


def movegen(boards: torch.Tensor, control: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully legal moves and check status, spec §4.1.

    ``mask [N,32] int64``, bit s of word p set iff p -> s is legal, and
    ``in_check [N] bool``. An all-zero mask is checkmate when ``in_check`` is set
    and stalemate otherwise.
    """
    boards, control = _state(boards, control)
    mask, in_check = _ext().movegen(boards, control, *_tables(boards.device))
    return mask, in_check


def step(boards: torch.Tensor, control: torch.Tensor, move: torch.Tensor,
         promo: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
         in_place: bool = False, hash: Optional[torch.Tensor] = None
         ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Apply one move per position: ``(boards, control, hash, irreversible)``.

    ``move`` is ``slot * 64 + target``, or ``-1`` for the null move of spec §9.
    ``promo`` is the spec §3 field (``0:N 1:B 2:R 3:Q``) and defaults to queen.

    The returned hash is ``None`` when ``hash`` is omitted, and omitting it is
    worth real time here: the incremental hash has to know whether an en passant
    capture was legal both before and after the move, which is two king-safety
    tests, and that is most of the cost of the call.
    """
    boards, control = _state(boards, control)
    n = boards.shape[0]
    move = move.reshape(-1).to(device=boards.device, dtype=torch.int16).contiguous()
    if move.numel() != n:
        raise ValueError(f"move must have one entry per position, got {move.numel()} for {n}")

    if mask is not None:
        # A host round-trip, on purpose and only when asked: this is the debug
        # assertion, and it is the one thing in the module that cannot be in a loop.
        legal = mask.reshape(n, 32)[torch.arange(n, device=boards.device),
                                    (move.long() // 64).clamp(min=0)]
        illegal = (((legal >> (move.long() % 64)) & 1) == 0) & (move >= 0)
        if bool(illegal.any()):
            bad = illegal.nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"illegal moves at batch indices {bad}")

    if promo is None:
        promo_t = torch.full((n,), 3, dtype=torch.uint8, device=boards.device)
    else:
        promo_t = promo.reshape(-1).to(device=boards.device, dtype=torch.uint8).contiguous()
    empty = torch.empty(0, dtype=torch.int64, device=boards.device)
    hash_t = empty if hash is None else hash.reshape(-1).to(torch.int64).contiguous()

    out_boards, out_control, out_hash, irreversible = _ext().step(
        boards, control, move, promo_t, hash_t, *_tables(boards.device))
    return out_boards, out_control, (out_hash if hash is not None else None), irreversible


def play(boards: torch.Tensor, control: torch.Tensor, move: torch.Tensor,
         promo: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
         in_place: bool = False) -> Tuple[torch.Tensor, ...]:
    """`step` then `movegen`: (boards, control, mask, in_check)."""
    boards, control, _, _ = step(boards, control, move, promo=promo, mask=mask)
    return (boards, control) + movegen(boards, control)


def hash_position(boards: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    """Full Zobrist hash of each position, ``[N] int64`` (spec §6.1)."""
    boards, control = _state(boards, control)
    return _ext().hash_position(boards, control, *_tables(boards.device))[0]


def hash_inputs(boards: torch.Tensor, control: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(hash, castling_rights_packed, legal_ep_file)``, for debugging a wrong hash.

    Not part of the engine contract; `torch_impl` exposes the same two inputs as
    separate functions and this returns them from the one launch that computed
    them.
    """
    boards, control = _state(boards, control)
    return tuple(_ext().hash_position(boards, control, *_tables(boards.device)))


def terminal(mask: torch.Tensor, in_check: torch.Tensor, control: torch.Tensor,
             boards: torch.Tensor, hash: Optional[torch.Tensor] = None,
             ring: Optional[torch.Tensor] = None, length: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(code [N] uint8, result [N] int8)`` per spec §4.3.

    Repetition is skipped when no history is supplied, which the fifty-move and
    material tests do not need.
    """
    boards, control = _state(boards, control)
    empty = torch.empty(0, dtype=torch.int64, device=boards.device)
    has_history = hash is not None and ring is not None and length is not None
    return tuple(_ext().terminal(
        boards, control, mask.to(torch.int64).contiguous(),
        in_check.to(torch.bool).contiguous(),
        empty if hash is None else hash.reshape(-1).to(torch.int64).contiguous(),
        empty.reshape(0, 0) if not has_history else ring.to(torch.int64).contiguous(),
        empty if not has_history else length.to(torch.int64).contiguous()))
