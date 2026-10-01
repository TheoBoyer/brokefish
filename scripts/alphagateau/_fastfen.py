"""Batched `from_fen` for pgx chess: their functions, called once instead of N times.

⚠️ **Not a reimplementation.** The raw FEN parse is copied from
`pgx.experimental.chess.from_fen` verbatim, and every derived field is computed by
*their* function (`_possible_piece_positions`, `_legal_action_mask`, `_zobrist_hash`,
`_update_history`, `_check_termination`, `_observe`) under `jax.vmap`. Only the dispatch
shape changes: `from_fen` issues six `jax.jit` calls per position, so a batch of 64 pays
384 device round trips for work that fits in six.

Measured on this machine, CPU: `from_fen` is **34.9 ms a position**, against 0.6 ms for
the UCI decode and microseconds for the 5-layer forward. Their state constructor, not
their network, is the whole cost of a puzzle sweep.

`verify()` checks the batched states leaf-for-leaf against `from_fen`, which is what
makes this safe to use for a measurement.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pgx.chess import State
from pgx.experimental.chess import from_fen
import pgx.chess as C
from pgx.experimental.chess import _flip_pos


def _raw(fen: str):
    """The pure-Python half of `from_fen`, verbatim, returning numpy."""
    board, turn, castling, en_passant, hm, fm = fen.split()
    arr = []
    for line in board.split("/"):
        for c in line:
            if str.isnumeric(c):
                arr += [0] * int(c)
            else:
                ix = "pnbrqk".index(str.lower(c)) + 1
                arr.append(-ix if str.islower(c) else ix)
    ccq = np.zeros(2, dtype=bool)
    cck = np.zeros(2, dtype=bool)
    if "Q" in castling: ccq[0] = True
    if "q" in castling: ccq[1] = True
    if "K" in castling: cck[0] = True
    if "k" in castling: cck[1] = True
    if turn == "b":
        ccq, cck = ccq[::-1].copy(), cck[::-1].copy()
    mat = np.int32(arr).reshape(8, 8)
    if turn == "b":
        mat = -np.flip(mat, axis=0)
    ep = np.int32(-1) if en_passant == "-" else np.int32(
        "abcdefgh".index(en_passant[0]) * 8 + int(en_passant[1]) - 1)
    if turn == "b" and ep >= 0:
        ep = np.int32(int(_flip_pos(jnp.int32(ep))))
    return (np.rot90(mat, k=3).flatten(), np.int32(0 if turn == "w" else 1),
            ccq, cck, ep, np.int32(hm), np.int32(fm))


# ⚠️ **Hoisted and jitted once.** Writing `jax.vmap(C._legal_action_mask)(st)` inline
# creates a new wrapper object on every call, so nothing is ever cached: each request
# re-traces all six functions and the compiled executables accumulate. That is what
# leaked ~0.5 MB a position and killed the server twice on 2026-08-16. One module-level
# `jax.jit` means one compile per batch shape, reused forever.
@jax.jit
def _derive(st):
    st = st.replace(_possible_piece_positions=jax.vmap(C._possible_piece_positions)(st))
    st = st.replace(legal_action_mask=jax.vmap(C._legal_action_mask)(st))
    st = st.replace(_zobrist_hash=jax.vmap(C._zobrist_hash)(st))
    st = jax.vmap(C._update_history)(st)
    st = jax.vmap(C._check_termination)(st)
    return st.replace(observation=jax.vmap(C._observe)(st, st.current_player))


def batch_from_fen(fens):
    """Their `from_fen`, minus the six per-position `jax.jit` dispatches."""
    parts = [_raw(f) for f in fens]
    n = len(parts)
    tmpl = State(_board=jnp.asarray(parts[0][0]))
    st = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape).copy(), tmpl)
    st = st.replace(
        _board=jnp.asarray(np.stack([p[0] for p in parts])),
        _turn=jnp.asarray(np.stack([p[1] for p in parts])),
        _can_castle_queen_side=jnp.asarray(np.stack([p[2] for p in parts])),
        _can_castle_king_side=jnp.asarray(np.stack([p[3] for p in parts])),
        _en_passant=jnp.asarray(np.stack([p[4] for p in parts])),
        _halfmove_count=jnp.asarray(np.stack([p[5] for p in parts])),
        _fullmove_count=jnp.asarray(np.stack([p[6] for p in parts])),
    )
    return _derive(st)


def verify(fens):
    """Every leaf of the batched state, against `from_fen` position by position."""
    fast = batch_from_fen(fens)
    ref = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs),
                                 *[from_fen(f) for f in fens])
    fa, _ = jax.tree_util.tree_flatten(fast)
    fb, _ = jax.tree_util.tree_flatten(ref)
    bad = []
    for i, (x, y) in enumerate(zip(fa, fb)):
        if x.shape != y.shape or not bool(jnp.all(x == y)):
            bad.append((i, tuple(x.shape), tuple(y.shape)))
    return bad
