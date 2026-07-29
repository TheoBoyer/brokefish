"""Brokefish — the cheapest superhuman chessbot from scratch."""

from brokefish.nn import IMPLEMENTATIONS, available, encoder_impl, why_unavailable
from brokefish.nn.triton_impl import FusedEncoder

__all__ = [
    "FusedEncoder",       # the Triton implementation, still the default
    "IMPLEMENTATIONS",
    "encoder_impl",
    "available",
    "why_unavailable",
]
