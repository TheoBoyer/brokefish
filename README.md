# Brokefish

> **GPU poor. Elo rich.**

Brokefish is an attempt to build the cheapest superhuman chess engine that learns
entirely from scratch. Getting there on one consumer GPU means treating the budget
as a design constraint, so the run also produces a curve of Elo against euros
spent. No such curve has been published for chess.

The hardware is one RTX 4060 Laptop with 8 GB.

## The training boundary

Training data comes from games Brokefish plays against itself. Excluded as training
input: human game databases, games or evaluations produced by any other engine,
pretrained networks, distillation from a stronger teacher, opening books, and
hand-written chess heuristics such as piece-square tables.

Chess rules are allowed. Chess opinions are not.

Randomised openings, temperature sampling and Dirichlet noise diversify self-play.
They generate positions without importing knowledge, so they stay inside the
boundary. Standard opening suites such as UHO and TCEC appear during evaluation,
which is measurement and sits outside the boundary.

## How it is built

The environment is a chess engine on the GPU. A position is 32 `uint16` words, one
per piece slot, plus a control word carrying side to move and the halfmove clock.
Slots are stable for a whole game, so the network's 32 piece tokens are
index-aligned with the engine's 32 legality masks and the policy is masked by a
device-side AND.

The network is a transformer over those 32 tokens: `d=256`, 8 layers, 8 heads,
6,383,360 parameters, around 400 MFLOPs per evaluation. Search is AlphaZero PUCT at
800 simulations a move, following AlphaGo Zero rather than the released
`pseudocode.py`; Gumbel is a seam left open for the throughput work, not what v0
runs.

## Success

Brokefish succeeds when it reaches clearly measured superhuman strength at a cost
low enough for the result to be reproduced. Every strength claim carries the
evaluation conditions and the training budget that produced it.

## Layout

```
brokefish/nn/        the network: one implementation per fused kernel, same contract
brokefish/env/       the PyTorch engine, reference implementation and oracle
brokefish/search/    the MCTS, torch reference and CUDA loop
brokefish/train/     self-play, replay buffer and the gradient phases
brokefish/eval/      the evaluation harness
csrc/                CUDA C++: the representation, the encoder and the search
tests/               correctness; torch and python-chess are the oracles
bench/               throughput, order-balanced interleaved A/B
docs/                the specification, the roadmap and the performance ledger
```

Everything runs from the repository root through `uv`, against the self-contained
`.venv`. The `--no-project` is load-bearing: without it a run can turn into a
re-resolve, and a re-resolve re-downloads roughly 3 GB of CUDA wheels.

```
alias py='uv run --no-project --python .venv/bin/python'

py -m pytest tests/                    the suite
py -m pytest tests/ --slow --mutation  and the two opt-in ones
py -m bench.bench_model                encoder throughput
py -m bench.bench_search               the in-loop number, Gate 1a
py -m bench.bench_env                  environment throughput
py scripts/check_cuda_build.py         the CUDA toolchain, end to end
```

## The documentation

`docs/` is split by what a document owes the reader, and the directory is the kind.

```
docs/reference/   what the code must do. spec.md is the frozen engine-network
                  contract; search.md, training.md, evaluation.md, environment.md
                  and debugger.md are normative for their component
docs/ledger/      the numbers. state.md is one bullet per landed component,
                  perf.md one row per measurement. Cited, never restated
docs/journal/     dated and append-only: experiments, audits, dead ends, and the
                  reasons behind decisions. Never edited to stay true
docs/roadmap.md   what is next, and the decisions still open
```

Where a journal entry and a reference page disagree, the reference page is current
and the entry records what was believed on its date. That is the arrangement, not a
defect.
