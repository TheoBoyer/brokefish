# Prior art

Survey run on 2026-07-28 across four research agents. The question was whether
anyone has already built the cheapest possible superhuman chess engine from scratch.
Nobody has. The survey also produced one correction to the network sizing, in the
section below on parameter count.

## AlphaZero replications in chess

lc0 is the only genuinely from-scratch superhuman replication. It gained about
2900 Elo over its first ten months across roughly 10M games, which is the anchor
used for our own sizing. Its distributed compute was never accounted for, at around
8 GPU-years per run, and no cost-versus-Elo curve was ever extracted from it.

Every known solo attempt stops between 1200 and 2000 Elo, or bootstraps from
supervision. CrazyAra, BadGyal and the 270M searchless models fall in the second
category.

CF-6M (arXiv:2409.12272) trains 6M parameters on a single A100 from public lc0 data
and beats AlphaZero's policy network. Accepting distillation therefore means the
flag is already planted, and NNUE has made superhuman play a consumer commodity
since 2020, with "download Stockfish, it's free" as the degenerate case. The claim
Brokefish makes has to be strictly tabula rasa, and has to say so in its title. One
adjacent sub-case remains open and poorly defined: training on human games only.

## The budget framing

No cost-versus-Elo curve for chess has been published. The closest published figure
is a 2018 lczero forum estimate of about $15k.

The search-contempt paper (April 2025, arXiv:2504.07757) proposes consumer-GPU
feasibility without executing it, which suggests the idea is in the air and the
window is finite.

Andy Jones (arXiv:2104.03113) provides the methodology to replicate: Hex, around
500 GPU-hours, and the empirical law of +500 Elo per 10× compute. That is the
template for the write-up.

## Full-chess GPU environments

pgx (JAX) is the only mature full-chess GPU environment, reaching about 2×10⁵
steps/s on an A100. Our target of 50-100M boards/s on a 4060 sits two to three
orders of magnitude above that on hardware roughly 20× cheaper, which is where the
project's technical edge lies.

torchess (CUDA) is a broken proof of concept, missing mate detection and handling
repetition incorrectly.

Ankan Banerjee's perft_gpu is prior art for pure GPU move generation. Its published
figures of 12-21 Gnps come from a 2013 GTX 780 and use bulk counting, so they are
not directly comparable and the code has to be recompiled locally for an honest
benchmark.

## The bar to beat

AlphaGateau (NeurIPS 2024, arXiv:2410.23753) combines pgx, Gumbel search at 128
simulations and a 1M-parameter GNN, reaching 1830-2100 Elo from 128k games in
13.7 days on eight A5000s. It is the best available reference for small-budget
self-play on full chess, and beating it is Gate 2.

## Correction to the sizing

The original bet of 1M parameters is contradicted by three independent points.
Scaling laws (Neumann and Gros, arXiv:2210.00849) show an Elo plateau in parameter
count. The smallest roughly superhuman networks known are around 5-6M parameters
(lc0 T1-distilled, CF-6M), and both reach that through distillation plus search.
AlphaGateau, at exactly 1M parameters, plateaus around 2100.

The network was therefore revised to 5-10M parameters, and d=256 with 8 layers gives
6.32M. Recomputing the budget: 2.6×10¹⁰ evaluations at 400 MFLOPs each is about
10¹⁹ FLOPs, which is a few days of 4090 or H100 time at realistic utilisation, or
two to four weeks on the 4060. The hobbyist budget of roughly 100-500 € spot still
holds, on the condition that inference stays above 30 % of tensor peak, which is
what Gate 1 measures.

One extrapolation remains unvalidated and worth watching: Gumbel at n=32 is
validated on 9×9 Go, where it costs about 100 Elo at convergence, but has never been
demonstrated on full chess. The chess-side Gumbel paper uses n=400.

## Independent convergence

Applying Jones' law to AlphaGateau, starting from 2100 Elo at 128k games and adding
500 Elo per 10× compute, predicts 2850-2900 Elo somewhere around 5-50M games. That
agrees with the lc0 anchor by a completely different route.

## On pufferlib

pufferlib was the original inspiration for the project, and reading its code shows
its gains come from a C environment, shared and pinned memory, networks of around
150k parameters, `torch.compile` for about +20 %, and a single custom kernel for
GAE.

Their bottleneck is the CPU-to-GPU bridge, which does not exist for Brokefish since
everything is device-side, so their recipe solves a different problem and cannot be
copied. Their network recipe is also insufficient for MCTS, where forwards are
sequential and dependent, which is what motivates CUDA graphs or a persistent
kernel.

## Position diversity

During training, diversity is self-generated: temperature over the first 30 or so
plies, Dirichlet noise at the root, uniform random openings in the style of NNUE
data generation, and KataGo-style branching. A human game database is forbidden
because it leaks opening theory, and unnecessary because AZ and lc0 are existence
proofs.

During evaluation, standard books (UHO, TCEC) are mandatory, with games played in
pairs on both colours. Evaluation is measurement and sits outside the tabula rasa
boundary. The AlphaZero-Stockfish match in the Science paper set the precedent by
starting from TCEC openings.
