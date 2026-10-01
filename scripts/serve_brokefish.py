"""brokefish behind the engine HTTP contract.

    POST /move   {"fens": [...], "n": 128}  ->  {"moves": ["e2e4", ...]}
    GET  /health                            ->  {"engine": ..., "ckpt": ...}

Stateless and batched: the arbiter owns every board, this only ever answers "given
these positions, your moves", in order. That is the whole contract, and it is what
lets any two engines meet regardless of stack.

⚠️ **`http.server`, not fastapi.** `.venv` was cloned with `cp -al` and must never
re-resolve (`CLAUDE.md`), so the serving side of this cannot introduce a dependency.
The stdlib handler is sufficient: one request at a time, already batched.

⚠️ **A FEN carries no history, so the search runs repetition-blind** — spec §6.3's
ring is empty on every request. The arbiter still adjudicates threefold correctly, so
games end properly; neither engine can see a repetition coming inside its own search.
Symmetric across engines, so it does not bias a score.

    uv run --no-project --python .venv/bin/python serve_brokefish.py \
        --ckpt checkpoints/t12h-gumbel-004009.pt --port 8081
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from brokefish.env import cuda_impl as cenv  # noqa: E402
from brokefish.env import notation, torch_impl as tenv  # noqa: E402
from brokefish.eval.layer0 import load_net_state  # noqa: E402
from brokefish.eval.runner import eval_config, evaluator  # noqa: E402
from brokefish.nn.model import net_for_state  # noqa: E402
from brokefish.search import search_impl  # noqa: E402
from brokefish.search.torch_impl import MOVE_BITS, PROMO_SHIFT  # noqa: E402

MOVE_MASK = (1 << MOVE_BITS) - 1


class Engine:
    """One loaded network, with a Search cached per (batch, budget).

    The tree is allocated per shape, so rebuilding it on every request would cost
    more than the search itself. The arbiter sends a constant batch, so the cache
    holds one entry in practice.
    """

    def __init__(self, ckpt: str, device: str = "cuda", impl: str = "cuda",
                 trace_path: str = None, gumbel: bool = True, gumbel_m: int = 16,
                 terminal_collapse: bool = True, c_scale: float = None):
        self.ckpt, self.device, self.impl = ckpt, device, impl
        # `None` is `SearchConfig`'s default, mctx's 0.1. The 2026-09-09 dissection
        # found the paper's 1.0 removes a quarter of the fixable hanging moves at
        # n = 128, and this is how the net effect in play is measured.
        self.c_scale = c_scale
        # ⚠️ The defaults are the **training** configuration -- every run in this line
        # trained with `--gumbel --gumbel-m 16 --sims 128 --terminal-collapse` -- so a
        # match here plays the engine the network was trained for. `eval_config(n, B)`
        # leaves `gumbel=False`, which is why `eval/league.py` rates every checkpoint
        # under PUCT instead. These flags exist so that difference can be measured.
        self.gumbel, self.gumbel_m = gumbel, gumbel_m
        self.terminal_collapse = terminal_collapse
        self.trace = open(trace_path, "w") if trace_path else None
        self._last_n = 0
        state = load_net_state(ckpt, device="cpu")
        net = net_for_state(state)          # scalar or win/draw/loss, read off the file
        net.load_state_dict(state)
        self.net = net.to(device).eval()
        self.evaluate = evaluator(self.net, impl if impl == "cuda" else None)
        self._searches: dict = {}
        self._lock = threading.Lock()

    def _search(self, b: int, n: int):
        key = (b, n)
        s = self._searches.get(key)
        if s is None:
            # `eval_config` is the protocol, not a preference: eps = 0, tau_plies = 0
            # and gumbel_scale = 0, so play is deterministic and all diversity comes
            # from the arbiter's openings (`evaluation.md` §5.4, `match.py`'s header).
            extra = {} if self.c_scale is None else {"c_scale": self.c_scale}
            cfg = eval_config(n, b, E=96, gumbel=self.gumbel,
                              gumbel_m=self.gumbel_m,
                              terminal_collapse=self.terminal_collapse, **extra)
            s = search_impl(self.impl)(cfg, self.evaluate, env=cenv,
                                       device=self.device, seed=0)
            self._searches[key] = s
        return s

    @torch.no_grad()
    def moves(self, fens, n: int, k: int = 1):
        with self._lock:
            b = len(fens)
            boards = torch.cat([tenv.from_fen(f)[0] for f in fens]).to(torch.int16)
            control = torch.cat([tenv.from_fen(f)[1] for f in fens]).to(torch.int16)
            boards, control = boards.to(self.device), control.to(self.device)
            # ⚠️ Reject a finished position here, by name. `Search.reset` does not run
            # terminal detection, so `game_done` stays False and invariant 8 ("a
            # finished game was searched") cannot fire; the search then builds a root
            # with zero edges and the failure surfaces as invariant 6, three layers
            # from the cause. A contract that says "given these positions, your move"
            # owes a clear answer when a position has no move.
            mask, in_check = cenv.movegen(boards, control)
            code, _ = cenv.terminal(mask, in_check, control, boards)
            if bool((code != 0).any()):
                i = int((code != 0).nonzero()[0, 0])
                raise ValueError(
                    f"position {i} is already over "
                    f"({tenv.TERMINAL_NAMES[int(code[i])]}): {fens[i]}")

            s = self._search(b, n)
            s.reset(boards, control)
            rec = s.self_play_move(sims=n)
            label = rec.played.to(torch.int64).cpu()
            uci = notation.to_uci(boards, label & MOVE_MASK,
                                  (label >> PROMO_SHIFT) & 0b11)
            if k > 1:
                # ⚠️ Ranked by the **target** `pi`, which is the search's own ordering
                # (visit counts at n > 0, the prior at n = 0). Ranking by raw policy
                # logits instead would answer a different question at every n but the
                # one where they coincide, and would silently disagree with `uci[0]`.
                pi, mv = rec.policy_prob.float().cpu(), rec.policy_move.cpu()
                plen = rec.policy_len.cpu()
                topk = []
                for i in range(b):
                    m = int(plen[i])
                    order = torch.argsort(pi[i, :m], descending=True)[:k]
                    lab = mv[i, :m][order].to(torch.int64)
                    topk.append(notation.to_uci(
                        boards[i:i + 1].expand(len(order), -1).cpu(),
                        lab & MOVE_MASK, (lab >> PROMO_SHIFT) & 0b11))
                return topk
            if self.trace is not None:
                self._dump_roots(fens, uci, rec, boards, s)
            return uci

    def _dump_roots(self, fens, uci, rec, boards, s):
        """One JSONL line per position: why this move, from the root's own numbers.

        ⚠️ The **root table**, not the tree. A full `TracingSearch` dump is the
        reference search at B = 1 syncing to the host every simulation
        (`trace.py`'s header: "speed is explicitly not part of the contract"), which
        cannot ride a 200-game match. What it would add is the subtree under each
        edge; what decides the move is the root, and that is here.

        Nothing is lost by the trim: the engines are deterministic and the contract
        is stateless, so **any position in any game can be re-searched afterwards**
        from its FEN with the full tracer and opened in the debugger. This file is
        what tells you *which* positions are worth that.
        """
        pi = rec.policy_prob.float().cpu()
        mv = rec.policy_move.cpu()
        plen = rec.policy_len.cpu()
        visits = s.edge_N[:, 0].cpu()
        prior = s.edge_prior[:, 0].float().cpu()
        q = s.edge_Q[:, 0].cpu()
        for i, fen in enumerate(fens):
            k = int(plen[i])
            labels = mv[i, :k].to(torch.int64)
            ucis = notation.to_uci(boards[i:i + 1].expand(k, -1).cpu(),
                                   labels & MOVE_MASK, (labels >> PROMO_SHIFT) & 0b11)
            order = torch.argsort(visits[i, :k].to(torch.int32), descending=True)[:12]
            edges = [{"uci": ucis[int(j)], "n": int(visits[i, int(j)]),
                      "p": round(float(prior[i, int(j)]), 5),
                      "q": round(float(q[i, int(j)]), 4),
                      "pi": round(float(pi[i, int(j)]), 5)} for j in order]
            self.trace.write(json.dumps({
                "fen": fen, "played": uci[i], "edges_total": k,
                "root_value": round(float(rec.root_value[i]), 4),
                "top": edges}) + "\n")
        self.trace.flush()


def make_handler(engine: Engine, name: str):
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
                self._send(200, {"engine": name, "ckpt": engine.ckpt,
                                 "device": engine.device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/move":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))) or b"{}")
                fens, n = req["fens"], int(req.get("n", 128))
                if not fens:
                    return self._send(400, {"error": "empty fens"})
                moves = engine.moves(fens, n, int(req.get("k", 1)))
                if len(moves) != len(fens):
                    raise RuntimeError(f"{len(moves)} moves for {len(fens)} fens")
                self._send(200, {"moves": moves})
            except Exception as exc:  # noqa: BLE001
                # Loud rather than a wrong move: the arbiter aborts the match. The
                # request is echoed back because a failure here is always about
                # *which positions* were sent, and they are otherwise unrecoverable.
                try:
                    sent = req.get("fens", [])
                except Exception:  # noqa: BLE001
                    sent = []
                print(f"FAILED on {len(sent)} fens, n={req.get('n')}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                for f in sent:
                    print(f"   {f}", flush=True)
                self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                                 "fens": sent})

        def log_message(self, *a):    # keep the console for the match, not for GETs
            pass
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--name", default="brokefish")
    p.add_argument("--device", default="cuda")
    p.add_argument("--impl", default="cuda", choices=("cuda", "torch"))
    p.add_argument("--puct", action="store_true",
                   help="PUCT over all root moves instead of Gumbel m=16 -- which is "
                        "what eval/league.py rates every checkpoint under")
    p.add_argument("--gumbel-m", type=int, default=16)
    p.add_argument("--no-collapse", action="store_true")
    p.add_argument("--c-scale", type=float, default=None,
                   help="Gumbel's interior sigma scale; mctx's 0.1 by default, the paper's 1.0")
    p.add_argument("--trace", default=None,
                   help="JSONL: one root table per position per move")
    a = p.parse_args()

    engine = Engine(a.ckpt, device=a.device, impl=a.impl, trace_path=a.trace,
                    gumbel=not a.puct, gumbel_m=a.gumbel_m,
                    terminal_collapse=not a.no_collapse, c_scale=a.c_scale)
    # One warm request: the fused encoder and the tree allocate on first use, and
    # the arbiter's timeout should not have to cover a build.
    engine.moves(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"], 8)
    mode = "PUCT" if a.puct else f"gumbel m={a.gumbel_m}"
    print(f"{a.name} ready on :{a.port}  ckpt={a.ckpt}  search={mode}", flush=True)
    HTTPServer(("127.0.0.1", a.port), make_handler(engine, a.name)).serve_forever()


if __name__ == "__main__":
    main()
