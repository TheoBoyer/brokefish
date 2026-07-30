# Brokefish

Brokefish is an attempt to build the cheapest superhuman chess engine that learns
entirely from scratch.

Getting there on consumer hardware means treating the budget as the design
constraint, so the run also produces a curve of Elo against euros spent. No such
curve has been published for chess.

## The training boundary

Training data comes from games Brokefish plays against itself. Excluded as training
input: human game databases, games or evaluations produced by any other engine,
pretrained networks, distillation from a stronger model, opening books, and
hand-written chess heuristics such as piece-square tables.

The environment encodes the rules of chess. Anything that encodes a judgment about
which positions are better has to be learned.

Randomised openings, temperature sampling and Dirichlet noise diversify self-play.
They generate positions without importing knowledge, so they stay inside the
boundary.

Standard opening suites such as UHO and TCEC are used during evaluation to reduce
variance between paired games. Evaluation is measurement and sits outside the
training boundary.

## Where this sits

lc0 is the only project that has reached superhuman play from scratch, at roughly
eight GPU-years per run on volunteer compute that was never costed. Solo attempts
published since then plateau between 1200 and 2000 Elo, or start from a supervised
network. The nearest reference point with a full accounting is
[AlphaGateau](https://arxiv.org/abs/2410.23753), which reached 1830-2100 Elo in
13.7 days on eight A5000s with a 1M-parameter network and 128k games.

Brokefish targets that range first, then measures Elo gained per additional order of
magnitude of compute.

## How it is built

The environment is a chess engine in CUDA. A position is 32 `uint16` words, one per
piece slot, plus a control word carrying side to move and the halfmove clock. Move
generation runs entirely on the GPU. Piece slots are stable for the whole game, so
the network's 32 piece tokens are index-aligned with the engine's 32 legality masks
and the policy is masked by a device-side AND inside the search.

The network is a transformer over those 32 tokens: `d=256`, 8 layers, 8 heads,
6.38M parameters, roughly 400 MFLOPs per evaluation. Search is AlphaZero PUCT at
800 simulations a move, specified in [`search.md`](reference/search.md).

The hardware is one RTX 4060 Laptop with 8 GB.

The contract between engine and network is frozen in the
[specification](reference/spec.md). Measured inference throughput, including the
optimisations that returned nothing, is in the [performance ledger](ledger/perf.md).

## Reporting

Every strength claim carries the evaluation conditions and the training budget that
produced it.

## How this documentation is organised

Three kinds of page, and the section it sits under is the kind.

**Reference** is what the code must do. It has no dates in the body and no
measurements, and it is rewritten whenever the code moves.
[`spec.md`](reference/spec.md) is the frozen engine-network contract; the others are
normative for their component.

**Ledger** is the numbers, each in one place. [`state.md`](ledger/state.md) is one
bullet per landed component, [`perf.md`](ledger/perf.md) one row per measurement.
Everything else cites them rather than restating them.

**[Journal](journal/index.md)** is dated and append-only: the experiments, the
audits, the dead ends, and the reasoning behind decisions. An entry is never edited
to stay true — it has to be right as of its date and nothing more. Where an entry
and a reference page disagree, the reference page is current.

[`roadmap.md`](roadmap.md) sits on top of all three and says what is next.
