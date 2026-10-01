# The AlphaGateau bridge

AlphaGateau ([Rigaux & Kashima, arXiv:2410.23753](https://arxiv.org/abs/2410.23753),
code at [Akulen/AlphaGateau](https://github.com/Akulen/AlphaGateau)) is written in JAX on
pgx and mctx. Brokefish is PyTorch and CUDA. The two engines meet over HTTP: each runs
behind the same `/move` contract and `scripts/arbiter.py` referees, with python-chess as
the only authority on legality and result.

The files here run inside a checkout of their repository and import their code
unmodified:

| file | what it does |
|---|---|
| `serve_ag.py` | their network and their exact `mctx.gumbel_muzero_policy` call behind `/move`. The only original code is FEN to pgx state and pgx action to UCI, the latter by stepping the state and matching the resulting placement against python-chess's legal moves |
| `_fastfen.py` | a batched `from_fen`: their functions under `jax.vmap`, verified leaf for leaf by `verify()` |
| `_leaktest.py` | the RSS / live-array / jit-cache harness that found the server's memory leak (`docs/journal/2026-08-19-the-alphagateau-server-leak.md`) |
| `_eqtest.py` | checks that the jitted search chooses the same moves as the eager one it replaced |
| `requirements-ag.lock` | their side's environment, Python 3.11.13, unchanged since 2026-08-13, before the first match |

The Brokefish side is `scripts/serve_brokefish.py`, and the driver is
`scripts/h2h-vs-ag.sh`.

## Reproducing the matches

1. Clone their repository beside this one, at the commit the matches ran against. Their
   checkpoints are tracked in it; the one played is
   `models/chess_2024-08-20:00h13/000499.ckpt`, the final iteration of the run their
   `models_description.json` labels "AlphaGateau 5 layers 500 iterations".

   ```
   git clone https://github.com/Akulen/AlphaGateau.git ../alphagateau
   git -C ../alphagateau checkout 7ee3a05
   ```

   Another location works if `AG_DIR` points to it.

2. Build their environment:

   ```
   cd ../alphagateau
   uv venv --python 3.11 .venv
   uv pip install --python .venv/bin/python -r ../brokefish/scripts/alphagateau/requirements-ag.lock
   ```

3. Build ours from `requirements.lock` (see its header), and put the Brokefish
   checkpoint to be played under `runs/`. The 24 h network of the headline match is
   `runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt`, which is not in this
   repository.

4. Run the match from the Brokefish root:

   ```
   scripts/h2h-vs-ag.sh runs/t24h-adamw-int8/checkpoints/t24h-adamw-int8.pt t24hfull
   ```

   The defaults are the protocol of every AlphaGateau match in `docs/journal/`: 200 games
   over 100 openings of 8 uniformly random plies (seed 0), each opening played once with
   each colour, 128 simulations a move on both sides, Gumbel over the top 16 moves on
   both sides, a 300-ply cap scored as a draw. The script starts both servers under
   `systemd-run --user` scopes with memory caps, so it expects a systemd user session.
   It writes `logs/h2h-<tag>.{log,json,pgn}`.

The recorded results are in `logs/`:

| match | file | W-D-L | score |
|---|---|---|---|
| `t12h-gumbel-004009`, 2026-08-13 | `logs/gate2-h2h-alphagateau.json`, `logs/gate2-h2h.pgn` | 20-120-60 | 0.4000 |
| `t24h-adamw-int8`, 2026-08-19 | `logs/h2h-t24hfull.json`, `logs/h2h-t24hfull.pgn` | 13-141-46 | 0.4175 |

## What a rerun will and will not reproduce

- The games will differ. `serve_ag.py` seeds `PRNGKey(0)` once and splits the key per
  request, so AlphaGateau's move depends on how many requests preceded it. Our side is
  deterministic. What carries over between runs is the protocol, and the score up to its
  sampling error.
- The 2026-08-13 match ran an earlier `serve_ag.py`: eager search, before `_fastfen.py`
  and the cached `jax.jit` that fixed the leak. The file here is the one that played the
  2026-08-19 match. `_eqtest.py` is the check that the two make the same choices; the
  journal does not record a run of it.
- `t12h-gumbel-004009.pt`, the 12 h network of the first match, no longer exists.
