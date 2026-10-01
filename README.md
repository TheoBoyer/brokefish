# Brokefish

Brokefish is a chess engine that learns only from games it plays against itself, in
the style of AlphaZero, built to train on a single consumer GPU. The rules engine, the
tree search and the neural network all run on the GPU, so self-play never leaves the
device.

The aim is to find out how cheaply a superhuman engine can be trained from zero, and
to plot playing strength against the money spent on compute. That point has not been
reached. After 24 hours of training on an RTX 4060 Laptop, the current network scores
0.4175 over 200 games (13 wins, 141 draws, 46 losses) against the published
[AlphaGateau](https://arxiv.org/abs/2410.23753) model, which trained for 13.7 days on
eight RTX A5000s.

Training uses no human games, no games or evaluations from other engines, no
pretrained weights and no opening books. The code knows the rules of chess and
nothing about which positions are good.

The project is also practice at running a whole project with a coding agent: Claude
Code wrote most of the code and documentation under my direction.

## Requirements

- An NVIDIA GPU of the Ada generation (sm89). The kernels use `mma.sync` and
  `cp.async` and were only tested on an RTX 4060 Laptop with 8 GB.
- The CUDA 13.2 toolkit (`nvcc`), matching the CUDA version of the torch build below.
  It can sit beside an older system toolkit; `csrc/README.md` explains how the build
  finds it.
- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- Linux. Everything runs on one GPU; there is no multi-GPU or distributed training.

## Installation

```
uv sync
uv run scripts/check_cuda_build.py
```

`uv sync` creates `.venv` from `uv.lock`, with torch and triton from the CUDA 13.2
wheel index and every other package pinned to the versions the project was developed
and measured with. The CUDA extensions compile on first use; the second command checks
the toolchain before that.

## Usage

### Training

```
uv run -m brokefish.train.loop \
    --run my-run --minutes 60 --games 1024 \
    --gumbel --gumbel-m 16 --sims 128 --int8 --terminal-collapse \
    --optimizer adamw --lr 0.001 --decay cosine --warmup 30 \
    --keep-checkpoints --checkpoint-every 200
```

The loop alternates between playing a batch of games in parallel (`--games`) and
taking gradient steps on a replay buffer. Everything a run produces goes to
`runs/<run>/`: the log, a JSONL record of every step, checkpoints and the replay
buffer. Metrics are also sent to Weights & Biases; add `--wandb-mode offline` on a
machine without credentials. `--help` lists all options.

The 24-hour network above was trained with the command in
`scripts/chain-t24h-adamw-int8.sh`.

### Rating checkpoints

```
uv run -m brokefish.eval.league --run my-run
uv run -m brokefish.eval.curve runs/my-run/league-my-run.json
```

The league plays the run's checkpoints against each other and against a fixed
randomly initialised network, then fits Elo ratings to the results. Ratings are only
comparable within one league. Pass `--run` several times to rate several runs
together. `curve` turns the ratings into a plot of Elo against training time and cost.

### Matches

`scripts/arbiter.py` plays two engines against each other over HTTP, with
python-chess deciding legality and results. Each opening is played twice, once with
each engine as White.

```
scripts/h2h-bf.sh <checkpoint-a> <checkpoint-b> <tag>      # two Brokefish checkpoints
scripts/h2h-vs-ag.sh <checkpoint> <tag>                     # against AlphaGateau
```

Matches against AlphaGateau need their code and environment next to this repository;
`scripts/alphagateau/README.md` gives the setup.

### Search debugger

A web viewer for playing against a checkpoint and inspecting its search tree. See
`debugger/README.md`.

## How it works

**Board.** A position is 32 16-bit words, one per piece, plus a control word for side
to move and the move counters. A piece keeps the same slot for the whole game. Move
generation, move application and repetition detection are written in CUDA and run one
warp per position.

**Network.** A transformer whose 32 input tokens are the 32 pieces: 8 layers, width
256, 8 heads, 6.4 M parameters, about 400 MFLOPs per evaluation. For each piece it
outputs 64 logits, one per destination square, so the legal-move mask from the board
lines up with the policy output directly. It also outputs a value and promotion
logits.

**Search.** Monte Carlo tree search running on the GPU, with two selection rules:
PUCT as in AlphaGo Zero, and Gumbel MuZero. Self-play uses Gumbel at 128 simulations
per move.

**Inference kernel.** The whole network runs in one CUDA kernel launch per batch, from
board words to logits, with weights quantised to int8. Inside the search it reaches
78,000 network evaluations per second on the RTX 4060 Laptop (search at 800
simulations, 4096 games in parallel).

**Correctness.** Each CUDA component has a PyTorch reference implementation, and the
tests check the two against each other. The move generator is also checked against
python-chess on random positions and against published perft counts.

## Tests and benchmarks

```
# the test suite, about 8 minutes
uv run -m pytest tests/
# adds deep perft and mutation checks
uv run -m pytest tests/ --slow --mutation

# network throughput, throughput inside the search, move generation throughput
uv run -m bench.bench_model
uv run -m bench.bench_search
uv run -m bench.bench_env
```

The GPU's clock varies with temperature by a few percent, so benchmarks compare
variants by interleaving them in one process rather than running one after the other.

## Repository layout

```
brokefish/env/       chess rules, PyTorch reference and CUDA bindings
brokefish/nn/        the network and its Triton and CUDA implementations
brokefish/search/    tree search, PyTorch reference and CUDA
brokefish/train/     self-play loop, replay buffer, losses, optimisers
brokefish/eval/      league, Elo fit, matches, puzzle and rule test suites
csrc/                CUDA sources, with device-side tests in csrc/tests/
tests/               pytest suite
bench/               benchmarks
scripts/             match tools, the AlphaGateau bridge, the commands of past runs
debugger/            the search viewer
docs/                specifications, measurements and experiment notes
```

## Documentation

`docs/` holds the specifications each component is built to (`docs/reference/`), the
measured numbers (`docs/ledger/`) and dated notes on every experiment, including the
ones that failed (`docs/journal/`). It builds into a site with
`uvx --with mkdocs-material mkdocs build`.
