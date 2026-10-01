"""Is the jitted search the same computation as the eager one it replaced?

Not "bit exact" -- `jax.jit` lets XLA fuse across ops that eager execution ran as
separate kernels, so the last bits may differ. This asks the question that matters:
**does it choose the same moves**, and how far apart are the visit distributions.

Both paths are built here from the same engine instance, so params, state, key,
`num_simulations`, `gumbel_scale`, `recurrent_fn` and `qtransform` are literally the same
objects. The only difference under test is the jit boundary.

⚠️ Runs on CPU (`JAX_PLATFORMS=cpu`) so it cannot disturb a live match on the GPU. Fusion
decisions are backend-specific, so this establishes *semantic* equivalence -- same
algorithm, same key handling, same choices -- not that the GPU backend agrees to the bit.

    JAX_PLATFORMS=cpu python _eqtest.py [batch] [n_sim] [rounds]
"""
import os
import random
import sys

import chess
import jax
import jax.numpy as jnp
import mctx

sys.path.insert(0, os.environ.get("AG_DIR", os.getcwd()))  # their repo: mcts, models
from serve_ag import AGEngine  # noqa: E402

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SIM = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def eager(eng, state, rng, n_sim, gumbel_scale):
    """`_policy` exactly as it was before 2026-08-19, inlined."""
    logits, value = eng.model(
        eng.model.format_data(state=state),
        legal_action_mask=state.legal_action_mask, params=eng.params)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    return mctx.gumbel_muzero_policy(
        params=eng.params, rng_key=rng, root=root,
        recurrent_fn=eng._recurrent,
        num_simulations=n_sim,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=gumbel_scale)


rng_py = random.Random(0)
fens = []
while len(fens) < B * ROUNDS:
    b = chess.Board()
    for _ in range(rng_py.randint(6, 40)):
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(rng_py.choice(ms))
    if not b.is_game_over():
        fens.append(b.fen())

eng = AGEngine("models/chess_2024-08-20:00h13/000499.ckpt", pad=B)
print(f"  batch {B}, n_sim {SIM}, {ROUNDS} rounds, backend {jax.devices()[0].platform}",
      flush=True)

same = total = 0
worst_w = 0.0
worst_q = 0.0
for r in range(ROUNDS):
    state = eng._batch(fens[r * B:(r + 1) * B])
    key = jax.random.PRNGKey(1234 + r)
    a = eager(eng, state, key, SIM, 0.0)
    b_ = eng._search(SIM, 0.0)(eng.params, state, key)
    eq = int(jnp.sum(a.action == b_.action))
    same += eq
    total += a.action.shape[0]
    dw = float(jnp.max(jnp.abs(a.action_weights - b_.action_weights)))
    worst_w = max(worst_w, dw)
    dq = float(jnp.max(jnp.abs(a.search_tree.node_values
                               - b_.search_tree.node_values)))
    worst_q = max(worst_q, dq)
    print(f"    round {r}: actions equal {eq}/{a.action.shape[0]}  "
          f"max|d action_weights| {dw:.3e}  max|d node_values| {dq:.3e}", flush=True)

print(f"\n  ACTIONS IDENTICAL: {same}/{total}"
      f"   max|d action_weights| {worst_w:.3e}   max|d node_values| {worst_q:.3e}")
print("  -> same moves" if same == total else "  -> MOVES DIFFER, investigate")
