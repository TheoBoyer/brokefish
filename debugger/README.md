# The search debugger

A viewer over recorded searches, plus the live half that records them. The
contract is [`debugger.md`](../docs/reference/debugger.md); this file is how to run it.

## Install

Two packages, neither of which is in `requirements.lock`'s runtime set, because
nothing in `brokefish/` imports them:

```
.venv/bin/pip install fastapi uvicorn
```

## Run

```
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m uvicorn debugger.server:app
```

Then <http://127.0.0.1:8000>. Traces are written to `traces/` (gitignored);
`BROKEFISH_TRACES` moves that elsewhere.

⚠️ **`--reload` drops every game in progress**, and it fires on any `.py` in the
repository, so a session editing `brokefish/` elsewhere restarts the server under you.
Games live in memory (traces do not, they are files), so a takeback after a restart
gets a 404 in the banner and the game is gone. Add `--reload` when editing
`server.py` itself, not when playing.

## What is where

| file | what |
|---|---|
| `server.py` | the API of `docs/reference/debugger.md` §9.1, and the game sessions |
| `static/trace.js` | §5's replay and §3's derived PUCT scores |
| `static/app.js` | the views of §7 |
| `../brokefish/search/trace.py` | `TracingSearch`, the recorder |

The recorder lives in the package rather than here because it subclasses the
reference search and is tested with the rest of the suite (`tests/test_trace.py`).
The import direction is one-way: this depends on `brokefish`, and the reverse is a
defect.

## What it is not evidence about

The debugger drives the reference search in `brokefish/search/torch_impl.py` and
not the kernels. The two are validated tree for tree, so a trace of the reference
is a trace of what the kernel does wherever `tests/test_search_cuda.py` covers,
and a CUDA-only bug outside that coverage is invisible here by construction
(`docs/reference/debugger.md` §10).

Searches are synchronous. A move at `n = 800` on the reference takes as long as it
takes, and the tab waits for it.

## No external engine

There is no Stockfish button and there will not be one. The network's own value
and policy are the only evaluations displayed, for the reason `docs/reference/debugger.md`
§8 gives: an operator who tunes against an engine's opinion contaminates the curve
by a route no test covers.
