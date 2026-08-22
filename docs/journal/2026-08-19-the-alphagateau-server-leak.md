# 2026-08-19 — a jit boundary the server never had, and 81 MiB a request

The AlphaGateau head to head is `chain-t24h-adamw-int8.sh`'s step 4 and it failed three
times before it ran. The failures were all the same failure and none of my first three
explanations for it was right. The measurement that ended it is small enough to state in
one line, so it goes first:

    RSS +81.2 MiB/request, live arrays +0, `_fastfen._derive` cache constant at 1

Growth with no retained buffers and no re-tracing means the memory is not `jax.Array`s at
all. It is **compiled executables**, which nothing in `jax.live_arrays()` counts.

## What was actually wrong

`serve_ag.py` called `mctx.gumbel_muzero_policy` **eagerly, from inside the HTTP request
handler**, with no `jax.jit` above it. Every request therefore re-traced and re-compiled
an `n_sim`-deep search, and every executable stayed resident. A 200-game match at n = 128
is roughly 280 requests, so ~23 GiB on a 15 GiB machine.

⚠️ **AlphaGateau's own code makes the identical call and is fine.** `mcts.py:69` builds
the same `root` and calls the same policy — but from inside `play_ply`, which their
training loop runs under a `lax.scan` inside a jitted step. Theirs is traced once. **A
server has to supply that jit boundary itself**, and a straight lift of a training-loop
function into a request handler silently loses it. That is the transferable lesson and it
is not specific to mctx or to JAX: any framework that caches compilation per call site
will behave this way when the call site moves from "once, under a scan" to "once per
request".

The fix is `_search`, a per-`(n_sim, gumbel_scale)` cached `jax.jit` with `params`,
`state` and `rng_key` as *arguments* and the two scalars closed over as compile-time
constants, so one executable serves the life of the server. Measured on the same 20
requests of 64 positions:

| | before | after |
|---|---|---|
| RSS growth | **81.2 MiB/request** | **0.2 MiB/request** |
| `jax.live_arrays()` | +0 | +0 |
| `_fastfen._derive` cache size | 1 | 1 |
| warm RSS | 1.333 GiB | 1.193 GiB |

400x. RSS plateaus after the first request and does not move again.

## Three wrong diagnoses, and why each was tempting

⚠️ **This is the part worth keeping.** The bug took four hours, and none of that was the
fix, which is nine lines.

**1. "It is the retry loop."** The first crash left the match dead for 43 minutes and the
retry did not recover it, because a systemd scope killed by the OOM killer stays *loaded*
in the `failed` state and a second `systemd-run --unit` with the same name refuses with
"already loaded". `systemctl --user stop` does not clear that; only `reset-failed` does.
Real bug, fixed, irrelevant to the leak.

**2. "It is the `partial`."** `_policy` built
`partial(recurrent_fn, env=..., model=...)` fresh on every call. `functools.partial`
defines no `__eq__`, so it hashes by identity and gives any cache keyed on it a new key
every time. That is a genuine bug and hoisting it **halved** the rate, ~3 MB/s to
~1.75 MB/s — which is exactly why it was convincing. A fix that improves the symptom by
2x is the most dangerous kind of wrong answer, and I announced it as "found the leak"
before measuring whether the leak had stopped. It had not.

**3. "It is `_fastfen`, look at the dates."** The match that did *not* OOM ran
2026-08-13; `_fastfen.py` was created 2026-08-17. That is a clean discriminator, it is the
kind of evidence that normally settles things, and it was **wrong**. `_fastfen` already
had the correct hoisted module-level `@jax.jit _derive`, and its cache size stayed at 1
through the whole test. The date coincidence was a coincidence: the *real* cause had been
there since before the baseline and the baseline simply finished — 7553 s — a few minutes
before it would have hit the wall.

The thing that broke the tie was not more reading. It was instrumenting three counters
that fail differently:

| counter | what its growth means |
|---|---|
| RSS from `/proc/self/status` | the symptom systemd kills on |
| `len(jax.live_arrays())` | buffers are **retained** |
| `f._cache_size()` on the suspect jit | that function **re-traces** |

⚠️ `resource.ru_maxrss` is a high-water mark and never falls, so it cannot tell growth
from a transient peak. My first harness used it and would not have distinguished the fixed
build from the broken one. Current RSS or nothing.

`_leaktest.py` in the alphagateau tree is that harness, with an `AG_FASTFEN=0` switch that
forces the pre-2026-08-17 state constructor so the two eras can be compared in one
process. It is worth keeping for the next JAX server that grows.

## What is still open

⚠️ **The jitted search is not known to be bit-identical to the eager one.** `jax.jit`
lets XLA fuse across operations that eager execution ran as separate kernels, which can
change accumulation order and therefore the last bits. The search ends in an `argmax`
over completed-Q, so a 1e-7 perturbation changes a move only when two candidates are
within 1e-7 — rare over ~28 000 plies, but not impossible, and unmeasured. The test is
cheap and specified: same state, same `rng_key`, same `n_sim`, compare `action` for exact
integer equality and report `max |Δ action_weights|`. It has not been run, because doing
it during the live match would need a second engine on the same 8 GiB card.

⚠️ **No AlphaGateau match in this project has ever been bit-reproducible**, including the
−70 Elo baseline. `serve_ag.py` seeds `PRNGKey(0)` once and splits per request, so AG's
move depends on how many requests preceded it. Our side is deterministic
(`eval_config` sets `eps = 0`, `tau_plies = 0`, `gumbel_scale = 0`). What is comparable
between two matches is the protocol, not the games.

`--opening-skip` was added to `scripts/arbiter.py` on the way through. It is unused by the
run that finally went, but it is what lets a segmented match keep a seed's exact opening
set — `openings()` is deterministic in `(seed, plies)` and appends in order, so
`skip + count` minus the first `skip` gives disjoint slices whose union is the whole list.
Verified: the four 25-slices of seed 0 union to exactly the seed-0 hundred.

## Cost

Four hours of wall clock, two aborted matches (one reaching 99 games, one 58), and a
desktop that got to 1 GiB free before I killed the second one. The leak had been present
for at least six days and the only reason it had not bitten was that the one match that
mattered finished 3 % inside the wall.
