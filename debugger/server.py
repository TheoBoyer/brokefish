"""The debugger's server, to ``docs/debugger.md`` §9.1.

Static files, a trace listing, board rendering, and the live half a recorded
trace cannot provide: starting a search, evaluating a position, and playing a
game. Every search it runs emits a trace of §4, which the viewer then reads like
any other.

The dependency direction is one-way (§8): this imports :mod:`brokefish`, and
nothing in the package imports this. FastAPI and uvicorn stay out of
``requirements.lock``'s runtime set.

Run it::

    PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m uvicorn debugger.server:app

⚠️ Searches are synchronous and a ``n = 800`` move on the reference takes as long
as it takes (§11). The endpoint returns when the search finishes.

⚠️ Games live in this process and traces live on disk, so ``--reload`` loses every
game in progress and no trace. It also fires on any ``.py`` under the repository,
which includes work happening in another session.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional

import chess
import chess.svg
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from brokefish.env import torch_impl as env
from brokefish.env.interop import label_to_move, list_legal_moves, to_chess_board
from brokefish.nn.model import BrokefishNet, net_for_state
from brokefish.search import Search, SearchConfig, make_evaluator
from brokefish.search.trace import (
    MOVE_BITS,
    TracingSearch,
    checkpoint_info,
    read_trace,
    write_trace,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TRACES = os.environ.get("BROKEFISH_TRACES", os.path.join(ROOT, "traces"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="brokefish debugger")

# One search at a time. Two concurrent GPU searches would interleave their
# allocations for no benefit, and the viewer only ever asks for one.
_lock = threading.Lock()
_nets: Dict[str, BrokefishNet] = {}
_meta_cache: Dict[str, tuple] = {}
_games: Dict[str, "GameSession"] = {}


# -- the network ------------------------------------------------------------- #

def load_net(path: Optional[str]) -> BrokefishNet:
    """A checkpoint, or an untrained network when ``path`` is ``None``.

    An untrained network is a legitimate subject: most of what the debugger is
    for shows up at generation 0, where the priors are flat and the search is the
    only thing doing any work.
    """
    key = path or ""
    if key not in _nets:
        if path:
            state = torch.load(path, map_location=DEVICE)
            state = state.get("model", state)
            net = net_for_state(state).to(DEVICE).eval()
            net.load_state_dict(state)
        else:
            net = BrokefishNet().to(DEVICE).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _nets[key] = net
    return _nets[key]


def evaluator(path: Optional[str]):
    return make_evaluator(load_net(path))


# -- positions --------------------------------------------------------------- #

def from_fen(fen: Optional[str]):
    if not fen:
        return env.initial_boards(1, device=DEVICE)
    try:
        b, c = env.from_fen(fen)
    except Exception as exc:  # noqa: BLE001 - the message is the useful part
        raise HTTPException(400, f"bad FEN: {exc}") from exc
    return b.to(DEVICE).reshape(1, 32), c.to(DEVICE).reshape(1)


def legal_labels(board: torch.Tensor, control: torch.Tensor) -> Dict[str, int]:
    """``{uci: edge label}`` for one position, through the engine's own mask."""
    mask, _ = env.movegen(board, control)
    out = {}
    for move in list_legal_moves(board[0], mask[0]):
        promo = 0 if move.promotion is None else move.promotion - 2
        slot = int(((board[0] & 0b100000111111) == move.from_square).int().argmax())
        out[move.uci()] = (slot * 64 + move.to_square) | (promo << MOVE_BITS)
    return out


# -- traces ------------------------------------------------------------------ #

def trace_path(trace_id: str) -> str:
    if not trace_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "bad trace id")
    path = os.path.join(TRACES, trace_id + ".json")
    if not os.path.exists(path):
        raise HTTPException(404, f"no trace {trace_id}")
    return path


def new_trace_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def emit(search: TracingSearch) -> str:
    trace_id = new_trace_id()
    write_trace(search.trace(), os.path.join(TRACES, trace_id + ".json"))
    return trace_id


def trace_meta(path: str) -> dict:
    """The listing row for one trace, cached on mtime since a trace is ~2 MB."""
    mtime = os.path.getmtime(path)
    hit = _meta_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    doc = read_trace(path)
    meta = {
        "id": os.path.basename(path)[:-len(".json")],
        "created": doc["created"],
        "kind": doc["kind"],
        "n": doc["config"]["n"],
        "checkpoint": (doc.get("checkpoint") or {}).get("path"),
        "fen": doc["root"]["fen"],
        "played": doc["root"]["edges"]["san"][doc["played"]],
    }
    _meta_cache[path] = (mtime, meta)
    return meta


# -- games ------------------------------------------------------------------- #

class GameSession:
    """One game against a checkpoint.

    The engine's game state (position, hash, repetition ring, ply) lives here and
    is copied into a fresh tree for every move, which is §2's fresh-tree-per-move
    with the game outliving the search. Takeback replays from the start rather
    than undoing, for the same reason §5's scrubber does.
    """

    def __init__(self, checkpoint: Optional[str], n: int, human_color: str,
                 fen: Optional[str] = None) -> None:
        self.checkpoint = checkpoint
        self.n = n
        self.human_white = human_color != "black"
        self.start_fen = fen
        self.labels: List[int] = []
        self.traces: List[Optional[str]] = []
        self.search = TracingSearch(
            SearchConfig(n=n, B=1), evaluate=evaluator(checkpoint), device=DEVICE,
            kind="move", checkpoint=checkpoint_info(checkpoint, load_net(checkpoint)),
            impl={"env": "torch", "encoder": "torch"})
        self._rewind()

    def _rewind(self) -> None:
        self.board, self.control = from_fen(self.start_fen)
        self.hash = env.hash_position(self.board, self.control)
        self.ring, self.ring_len = env.empty_history(1, device=DEVICE)
        self.ply = 0
        self.code = 0
        for label in self.labels:
            self._advance(label)

    def _advance(self, label: int) -> None:
        move = torch.tensor([label & ((1 << MOVE_BITS) - 1)], device=DEVICE)
        promo = torch.tensor([(label >> MOVE_BITS) & 0b11], device=DEVICE)
        board, control, hash_, irrev = env.step(
            self.board, self.control, move, promo=promo, hash=self.hash)
        # The ring records the position being left, and the move's own
        # irreversibility empties it (spec §6.2), so this precedes the terminal
        # test of the new position. Same order as `select_and_advance`.
        self.ring, self.ring_len = env.push_history(
            self.ring, self.ring_len, self.hash, irrev)
        mask, in_check = env.movegen(board, control)
        code, _ = env.terminal(mask, in_check, control, board, hash_,
                               self.ring, self.ring_len)
        self.board, self.control, self.hash = board, control, hash_
        self.ply += 1
        self.code = int(code[0])

    # -- moves ---------------------------------------------------------------- #

    def fen(self) -> str:
        return to_chess_board(self.board[0].cpu(), self.control[0].cpu()).fen()

    def legal(self) -> Dict[str, int]:
        return {} if self.code else legal_labels(self.board, self.control)

    def push_uci(self, uci: str) -> None:
        label = self.legal().get(uci)
        if label is None:
            raise HTTPException(400, f"{uci} is not legal here")
        self.labels.append(label)
        self.traces.append(None)
        self._advance(label)

    def reply(self) -> tuple:
        """One engine move. Returns ``(uci, trace_id)``, or ``(None, None)``."""
        if self.code:
            return None, None
        s = self.search
        s.game_board.copy_(self.board)
        s.game_control.copy_(self.control)
        s.game_hash.copy_(self.hash)
        s.game_ring.copy_(self.ring)
        s.game_ring_len.copy_(self.ring_len)
        s.game_ply.fill_(self.ply)
        s.game_done.fill_(False)
        s.game_result.fill_(0)
        record = s.self_play_move()
        label = int(record.played[0])
        uci = label_to_move(self.board[0], label).uci()
        self.labels.append(label)
        self._advance(label)
        trace_id = emit(s)
        self.traces.append(trace_id)
        return uci, trace_id

    def takeback(self) -> None:
        """Undo back to the human's turn, which is one ply or two."""
        drop = 2 if len(self.labels) >= 2 else len(self.labels)
        self.labels = self.labels[:-drop] if drop else self.labels
        self.traces = self.traces[:-drop] if drop else self.traces
        self._rewind()

    def state(self) -> dict:
        return {"fen": self.fen(), "code": self.code, "ply": self.ply,
                "legal": sorted(self.legal()),
                "human_white": self.human_white,
                "last": (label_to_move(self._prev_board(), self.labels[-1]).uci()
                         if self.labels else None),
                "traces": self.traces}

    def _prev_board(self) -> torch.Tensor:
        """The position the last move was made from, for naming that move."""
        board, control = from_fen(self.start_fen)
        hash_ = env.hash_position(board, control)
        for label in self.labels[:-1]:
            move = torch.tensor([label & ((1 << MOVE_BITS) - 1)], device=DEVICE)
            promo = torch.tensor([(label >> MOVE_BITS) & 0b11], device=DEVICE)
            board, control, hash_, _ = env.step(board, control, move, promo=promo,
                                                hash=hash_)
        return board[0]


# -- API --------------------------------------------------------------------- #

class SearchRequest(BaseModel):
    fen: Optional[str] = None
    n: int = 128
    eps: float = 0.25
    seed: int = 0
    checkpoint: Optional[str] = None


class EvalRequest(BaseModel):
    fen: Optional[str] = None
    checkpoint: Optional[str] = None
    top: int = 12


class GameRequest(BaseModel):
    checkpoint: Optional[str] = None
    n: int = 128
    human_color: str = "white"
    fen: Optional[str] = None


class MoveRequest(BaseModel):
    uci: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/traces")
def list_traces() -> List[dict]:
    if not os.path.isdir(TRACES):
        return []
    rows = []
    for name in sorted(os.listdir(TRACES), reverse=True):
        if name.endswith(".json"):
            rows.append(trace_meta(os.path.join(TRACES, name)))
    return rows


@app.get("/api/trace/{trace_id}")
def get_trace(trace_id: str) -> FileResponse:
    # Served as a file rather than re-serialised: it is already the §4 document,
    # and at a few megabytes the round trip through Python is pure latency.
    return FileResponse(trace_path(trace_id), media_type="application/json")


@app.delete("/api/trace/{trace_id}")
def delete_trace(trace_id: str) -> dict:
    os.remove(trace_path(trace_id))
    return {"deleted": trace_id}


@app.get("/api/board.svg")
def board_svg(fen: str, lastmove: Optional[str] = None, check: Optional[str] = None,
              flipped: bool = False, size: int = 480, coordinates: bool = True) -> Response:
    """One position as an SVG. ``coordinates`` is load-bearing, not cosmetic.

    python-chess draws the file and rank labels in a margin *inside* the image,
    so a coordinate board's 8x8 area is not the image's own box. The play board
    asks for ``coordinates=false`` so that the 64 click rectangles overlaid on it
    line up by construction rather than by a magic inset.
    """
    board = chess.Board(fen)
    kw = {}
    if lastmove:
        kw["lastmove"] = chess.Move.from_uci(lastmove)
    if check:
        kw["check"] = chess.parse_square(check)
    elif board.is_check():
        kw["check"] = board.king(board.turn)
    svg = chess.svg.board(board, size=size, orientation=not flipped,
                          coordinates=coordinates, **kw)
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "max-age=3600"})


@app.post("/api/search")
def run_search(req: SearchRequest) -> dict:
    board, control = from_fen(req.fen)
    with _lock:
        search = TracingSearch(
            SearchConfig(n=req.n, B=1, eps=req.eps), evaluate=evaluator(req.checkpoint),
            device=DEVICE, seed=req.seed, kind="search",
            checkpoint=checkpoint_info(req.checkpoint, load_net(req.checkpoint)),
            impl={"env": "torch", "encoder": "torch"})
        search.reset(board, control)
        t0 = time.time()
        search.self_play_move()
        elapsed = time.time() - t0
        trace_id = emit(search)
    return {"trace_id": trace_id, "seconds": elapsed}


@app.post("/api/eval")
def evaluate(req: EvalRequest) -> dict:
    """The network alone, with no search and no noise.

    The priors come out of §6.4's expansion rather than out of a second reading
    of the policy head, so what is displayed is what a search would start from,
    promotion split and truncation included.
    """
    board, control = from_fen(req.fen)
    with _lock:
        s = Search(SearchConfig(n=1, B=1, eps=0.0), evaluate=evaluator(req.checkpoint),
                   device=DEVICE)
        s.reset(board, control)
        s.root_init()
        ne = int(s.node_nedges[0, 0])
        labels = s.edge_move[0, 0, :ne].tolist()
        priors = s.edge_prior[0, 0, :ne].float().tolist()
        value = float(s.node_value[0, 0])
    position = to_chess_board(board[0].cpu(), control[0].cpu())
    top = sorted(
        ({"san": position.san(label_to_move(board[0], label)),
          "uci": label_to_move(board[0], label).uci(), "p": p}
         for label, p in zip(labels, priors)),
        key=lambda r: -r["p"])[:req.top]
    # §3.5: the tree works in [0,1] and the head is tanh, so both are shown.
    return {"value": value, "value_tanh": 2.0 * value - 1.0, "n_edges": ne, "top": top}


@app.post("/api/game")
def new_game(req: GameRequest) -> dict:
    game_id = new_trace_id()
    with _lock:
        game = GameSession(req.checkpoint, req.n, req.human_color, req.fen)
        _games[game_id] = game
        reply = {}
        if not game.human_white:
            uci, trace_id = game.reply()
            reply = {"reply_uci": uci, "trace_id": trace_id}
    return {"game_id": game_id, **game.state(), **reply}


@app.get("/api/game/{game_id}")
def get_game(game_id: str) -> dict:
    return _game(game_id).state()


@app.post("/api/game/{game_id}/move")
def play_move(game_id: str, req: MoveRequest) -> dict:
    game = _game(game_id)
    with _lock:
        game.push_uci(req.uci)
        uci, trace_id = game.reply()
    return {**game.state(), "reply_uci": uci, "trace_id": trace_id}


@app.post("/api/game/{game_id}/takeback")
def takeback(game_id: str) -> dict:
    game = _game(game_id)
    with _lock:
        game.takeback()
    return game.state()


def _game(game_id: str) -> GameSession:
    if game_id not in _games:
        raise HTTPException(404, f"no game {game_id}")
    return _games[game_id]


@app.exception_handler(HTTPException)
def http_error(request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
