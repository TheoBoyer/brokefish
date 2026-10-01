"""AlphaGateau behind the same engine HTTP contract as brokefish.

    POST /move   {"fens": [...], "n": 128, "gumbel_scale": 0.0} -> {"moves": [...]}
    GET  /health

⚠️ **Their code, called rather than reimplemented.** `load_model` and `recurrent_fn`
are imported from their repo unmodified, and the search is their exact
`mctx.gumbel_muzero_policy` call from `mcts.py:120` with their `qtransform`. Nothing
about their network or their search is re-derived here, which is the entire reason
this runs in their venv instead of being ported.

The only original code is the two conversions the contract needs:
  * FEN -> pgx state, via `pgx.experimental.chess.from_fen`
  * pgx action -> UCI, by **stepping and diffing the placement**. pgx stores the
    board from the mover's point of view and flips it every ply, so decoding an
    action index by hand means reproducing that frame flip and its underpromotion
    encoding. Stepping the state and asking python-chess which of *its* legal moves
    reproduces the resulting placement needs none of that, and is self-checking:
    exactly one legal move can produce a given placement, so 0 or 2 matches is a
    loud failure rather than a plausible wrong move.

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
    .venv/bin/python serve_ag.py --ckpt models/chess_2024-08-20:00h13/000499.ckpt
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

import chess
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental.chess import from_fen, to_fen

# Their repository, checked out at AG_DIR (default: the working directory), supplies
# `mcts` and `models`. This file lives in brokefish; see README.md beside it.
sys.path.insert(0, os.environ.get("AG_DIR", os.getcwd()))
from mcts import recurrent_fn          # theirs, unmodified
from models import load_model          # theirs, unmodified

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class AGEngine:
    def __init__(self, ckpt: str, pad: int = 64):
        self.ckpt, self.pad = ckpt, pad
        # ⚠️ Hoisted, and it is the same bug `_batch` documents. `partial` defines no
        # `__eq__`, so it hashes by identity: a fresh one per request gives mctx a new
        # cache key every call, re-traces the whole search and accumulates one compiled
        # executable per request. Measured 2026-08-19: ~2 MB/s of host RAM, an
        # 8 GiB cap reached after 99 games, and the match oom-killed mid-way.
        self.env = pgx.make("chess")
        self.model, self.params = load_model(self.env, ckpt, "alphagateau")
        self._recurrent = partial(recurrent_fn, env=self.env, model=self.model)
        self._searches = {}
        with open(ckpt, "rb") as fh:
            d = pickle.load(fh)
        self.meta = {"iteration": int(d["iteration"]), "frames": int(d["frames"]),
                     "layers": d["config"].get("n_gnn_layers"),
                     "inner": d["config"].get("inner_size")}
        self._step = jax.jit(jax.vmap(self.env.step))
        # ⚠️ **Split placement.** The two halves of a request want different devices.
        # Building the pgx state is `_legal_action_mask` over 4672 actions -- many small
        # ops with control flow, launch-overhead-bound, measured 43 ms a position on the
        # GPU against 35 on... both are bad, and neither is the network. The 5-layer GNN
        # forward is the opposite: dense, batched, and what a GPU is for. So the state is
        # built on CPU and only the model runs on the accelerator.
        self._cpu = jax.devices("cpu")[0]
        try:
            self._acc = jax.devices("gpu")[0]
        except RuntimeError:
            self._acc = self._cpu
        self.split = self._acc is not self._cpu
        self._lock = threading.Lock()
        self._key = jax.random.PRNGKey(0)

    def _batch(self, fens):
        """FEN -> batched pgx state.

        ⚠️ `from_fen` issues six `jax.jit` calls on a *single* unbatched state, so a
        batch of 64 pays 384 device round trips: 28-35 ms a position. `_fastfen` runs
        the same six functions -- theirs, not reimplemented -- once under one hoisted
        `jax.jit(vmap(...))`, verified leaf-for-leaf by `_fastfen.verify`, at 0.37 ms.

        ⚠️ The hoisting is the whole point. An earlier version called `jax.vmap(f)(st)`
        inline, which builds a new wrapper object per call, caches nothing, re-traces
        every request and accumulates executables -- ~0.5 MB a position, which killed
        this server twice on 2026-08-16.
        """
        try:
            # ⚠️ `AG_FASTFEN=0` forces the pre-2026-08-17 path. The -70 Elo match of
            # 2026-08-13 ran before `_fastfen` existed and did not OOM; this switch is
            # what makes the two comparable in a leak bisect.
            import os as _os
            if _os.environ.get("AG_FASTFEN", "1") == "0":
                raise ImportError("disabled by AG_FASTFEN=0")
            from _fastfen import batch_from_fen
            return batch_from_fen(fens)
        except Exception as exc:  # noqa: BLE001
            print(f"fastfen unavailable ({exc}); using from_fen", flush=True)
            states = [from_fen(f) for f in fens]
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    def _search(self, n_sim: int, gumbel_scale: float):
        """The search, jitted. One compile per `(n_sim, gumbel_scale)`, reused forever.

        ⚠️ **This is the leak, and it was not `_fastfen`.** `mctx.gumbel_muzero_policy`
        was called eagerly from the request handler, so every request re-traced and
        re-compiled an `n_sim`-deep search. The compiled executables are not
        `jax.Array`s, so nothing in `jax.live_arrays()` shows them and the process just
        grows. Measured 2026-08-19, 20 requests of 64 positions at n = 128:

            RSS +81.2 MiB/request, live arrays +0, `_fastfen._derive` cache constant at 1

        which is why the dates were misleading -- `_fastfen` landed 2026-08-17, four days
        after the match that did not OOM, and had nothing to do with it.

        AlphaGateau's own `mcts.py:69` makes the identical call and is fine, because
        `play_ply` runs inside a `lax.scan` under a jitted training step: theirs is
        traced once. A server has to supply that jit boundary itself.

        `n_sim` and `gumbel_scale` are closed over rather than passed, so they are
        compile-time constants; `params`, `state` and `rng_key` are arguments, so one
        executable serves every request.
        """
        f = self._searches.get((n_sim, gumbel_scale))
        if f is None:
            def run(params, state, rng):
                logits, value = self.model(
                    self.model.format_data(state=state),
                    legal_action_mask=state.legal_action_mask, params=params)
                root = mctx.RootFnOutput(prior_logits=logits, value=value,
                                         embedding=state)
                return mctx.gumbel_muzero_policy(
                    params=params, rng_key=rng, root=root,
                    recurrent_fn=self._recurrent,
                    num_simulations=n_sim,
                    invalid_actions=~state.legal_action_mask,
                    qtransform=mctx.qtransform_completed_by_mix_value,
                    gumbel_scale=gumbel_scale)
            f = jax.jit(run)
            self._searches[(n_sim, gumbel_scale)] = f
        return f

    def _policy(self, state, n_sim: int, gumbel_scale: float):
        self._key, sub = jax.random.split(self._key)
        return self._search(n_sim, gumbel_scale)(self.params, state, sub)

    def _policy_only(self, state, k: int = 1):
        """Raw policy argmax over legal actions -- no search at all.

        ⚠️ **This is what a puzzle probe should ask an engine.** `move_pass@1` grades a
        single move choice, and running 128 simulations to make it costs 369 ms a
        position here against under a millisecond for the forward pass. It also removes
        `max_num_considered_actions = 16` from the comparison: the search can only ever
        consider its top-16 prior moves, so a search-based probe measures the cap as
        much as the network. Both engines are then compared on the same object -- the
        policy head -- exactly as brokefish's own probe does at `n_sims = 0`.
        """
        st = jax.device_put(state, self._acc) if self.split else state
        logits, _ = self.model(
            self.model.format_data(state=st),
            legal_action_mask=st.legal_action_mask, params=self.params)
        masked = jnp.where(st.legal_action_mask, logits, -jnp.inf)
        act = (jnp.argmax(masked, axis=-1) if k <= 1
               else jnp.argsort(-masked, axis=-1)[:, :k])
        # Back to CPU: the caller steps the state to decode the action, and stepping is
        # the same pgx legality work the construction was.
        return jax.device_put(act, self._cpu) if self.split else act

    def moves(self, fens, n_sim: int, gumbel_scale: float = 0.0, k: int = 1):
        with self._lock:
            real = len(fens)
            for i, f in enumerate(fens):
                b = chess.Board(f)
                if b.is_game_over():
                    raise ValueError(f"position {i} is already over "
                                     f"({b.result()}): {f}")
            # ⚠️ Pad to a fixed width. JAX retraces and recompiles the whole search
            # for every new batch shape, and the arbiter's batches shrink as games
            # finish -- without this, most of the match is spent compiling.
            padded = list(fens) + [START] * (-real % self.pad or 0)
            if len(padded) % self.pad:
                padded += [START] * (self.pad - len(padded) % self.pad)
            out = []
            for lo in range(0, len(padded), self.pad):
                chunk = padded[lo:lo + self.pad]
                state = self._batch(chunk)
                # n = 0 is the raw policy, matching brokefish's `n_sims = 0` probe.
                action = (self._policy_only(state, k) if n_sim == 0
                          else self._policy(state, n_sim, gumbel_scale).action)
                if action.ndim == 1:
                    after = self._step(state, action)
                    for j in range(len(chunk)):
                        out.append(self._to_uci(chunk[j], jax.tree_util.tree_map(
                            lambda x, j=j: x[j], after)))
                else:
                    # ⚠️ One step per rank, so a top-k answer costs k steps. The step is
                    # pgx legality work, which is why `k` is opt-in rather than always on.
                    per = []
                    for r in range(action.shape[1]):
                        aft = self._step(state, action[:, r])
                        row = []
                        for j in range(len(chunk)):
                            try:
                                row.append(self._to_uci(chunk[j], jax.tree_util.tree_map(
                                    lambda x, j=j: x[j], aft)))
                            except RuntimeError:
                                # ⚠️ Ranks past the legal-move count decode to nothing.
                                # Tolerated **only** beyond rank 0: an ambiguous decode at
                                # rank 0 is still a hard failure, which is what keeps the
                                # action mapping self-checking.
                                if r == 0:
                                    raise
                                row.append(None)
                        per.append(row)
                    for j in range(len(chunk)):
                        out.append([per[r][j] for r in range(action.shape[1])])
            return out[:real]

    @staticmethod
    def _to_uci(fen: str, after_state) -> str:
        """Which legal move reproduces the placement pgx reached? Exactly one does."""
        want = to_fen(after_state).split(" ")[0]
        board = chess.Board(fen)
        hits = []
        for m in board.legal_moves:
            board.push(m)
            if board.board_fen() == want:
                hits.append(m.uci())
            board.pop()
        if len(hits) != 1:
            raise RuntimeError(
                f"action decode is ambiguous ({len(hits)} matches) from {fen}: "
                f"pgx reached placement {want}")
        return hits[0]


def make_handler(engine: AGEngine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, {"engine": "alphagateau", "ckpt": engine.ckpt,
                                 "device": str(jax.devices()[0]), **engine.meta})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/move":
                return self._send(404, {"error": "not found"})
            req = {}
            try:
                req = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))) or b"{}")
                moves = engine.moves(req["fens"], int(req.get("n", 128)),
                                     float(req.get("gumbel_scale", 0.0)),
                                     int(req.get("k", 1)))
                self._send(200, {"moves": moves})
            except Exception as exc:  # noqa: BLE001
                sent = req.get("fens", [])
                print(f"FAILED on {len(sent)} fens: {type(exc).__name__}: {exc}",
                      flush=True)
                for f in sent:
                    print(f"   {f}", flush=True)
                self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                                 "fens": sent})

        def log_message(self, *a):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/chess_2024-08-20:00h13/000499.ckpt")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--warm-n", type=int, default=128,
                   help="budget for the warm-up call; 0 warms the policy path "
                        "only and allocates no search tree")
    a = p.parse_args()
    eng = AGEngine(a.ckpt, pad=a.pad)
    import time
    t = time.perf_counter()
    # ⚠️ `--warm-n 0` warms the policy path only. At `--pad 256` the mctx warmup
    # allocates several [pad, n+1, 4672] fp32 trees -- 588 MiB each -- and OOMs an
    # 8 GB card that also drives the display. A puzzle sweep never calls the search,
    # so it wants a large pad and no tree; a match wants pad 64 and the tree.
    eng.moves([START], a.warm_n)                 # compile + warm
    print(f"alphagateau [{'split cpu-state/gpu-model' if eng.split else 'single device'}] ready on :{a.port}  ckpt={a.ckpt}  {eng.meta}  "
          f"(warm {time.perf_counter()-t:.1f}s, device {jax.devices()[0]})", flush=True)
    HTTPServer(("127.0.0.1", a.port), make_handler(eng)).serve_forever()


if __name__ == "__main__":
    main()
