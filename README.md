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
6.32M parameters, around 400 MFLOPs per evaluation. Search is Gumbel MCTS.

## Success

Brokefish succeeds when it reaches clearly measured superhuman strength at a cost
low enough for the result to be reproduced. Every strength claim carries the
evaluation conditions and the training budget that produced it.

## Layout

```
brokefish/nn/        the encoder: one implementation per file, same contract
brokefish/env/       the PyTorch engine, reference implementation and oracle
csrc/                CUDA C++: the representation, and the two kernels to come
tests/               correctness; torch and python-chess are the oracles
bench/               throughput, order-balanced interleaved A/B
docs/                the specification, the roadmap and the performance ledger
```

```
pytest tests/                    the suite
pytest tests/ --slow --mutation  and the two opt-in ones
python -m bench.bench_model      encoder throughput
python -m bench.bench_env        environment throughput
python scripts/check_cuda_build.py   the CUDA toolchain, end to end
```

`docs/spec.md` is the normative contract between engine and network, and it is
frozen. `docs/roadmap.md` says what is built and what is next. `docs/perf.md`
records every measured number, including the optimisations that returned nothing.
