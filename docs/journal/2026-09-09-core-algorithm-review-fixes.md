# 2026-09-09 — the four findings of the core-algorithm review, fixed

`docs/core-algorithm-review.md` (reviewed commit `6989a97`) confirmed four defects in
search, position import and the training loss, plus a stale test configuration. This
entry records what changed and what did not. No number on the ledger moves: none of the
four was measured for its effect on playing strength, and this entry does not claim one.

## 1 + 4. The Gumbel prior floor, `2^-14` → `2^-24`

`gumbel.edge_logits` clamped the fp16 prior at `torch.finfo(torch.float16).tiny`, the
smallest *normal* fp16, and `search.cuh:kHalfTiny` copied the number. fp16 goes on down
to the subnormal `2^-24`, so every representable prior in `[2^-24, 2^-14)` collapsed onto
one logit. The review's start-position case: a `1e-5` edge and eighteen `1e-6` edges tie,
and sequential halving considers the wrong one.

Now `gumbel.PRIOR_FLOOR = 2^-24` and `search.cuh:kPriorFloor` is the same number. Every
positive fp16 is at or above it, so the clamp moves only the hard zero — which still
needs moving, because `log 0` on a legal move is the absorbing state the FPU bug was.

Finding 4 is the same floor applied one function earlier: `completed_q` weighted the
visited Q values by the *raw* prior, so an edge visited at a hard-zero prior counted in
`sum_visits` and not in the weighted mean, pulling `v_mix` toward a loss whatever its Q
said. `mctx._compute_mixed_value` guards the priors first; both implementations now do,
and in the kernel the guard sits on `pr[j]` at load, so the weighted mean and the logit
recovery see one prior. Tests: `test_gumbel.py::test_small_priors_keep_their_order`,
`::test_a_visited_edge_at_zero_prior_still_weighs_in_the_completion`; the Torch/CUDA
agreement suite in `test_search_cuda.py` holds the two implementations together.

## 2. A third rook could castle

Surplus rooks live in pawn slots. `from_board` set the castling `special` bit on slots 12,
13, 28, 29 only; `from_fen` rewrote `char` to the pawn letter before the rook branch ran.
Either way a promoted rook on h1 kept `special` clear, and `movegen.cuh:unmoved_rook_word`
compares the whole word, so `RR2K2R w Q` generated e1g1. Both importers now decide the
bit from the rook's own square and the supplied rights, whatever the slot: `from_board`
over every live rook by type bits against python-chess's rights bitboard, `from_fen` by
"on its corner and the letter is in the FEN". A rook anywhere else gets `special`, which
is also what `step` leaves on a rook that has moved, so the two importers now agree on
every live word. Self-play from the initial position was never affected: a promoted rook
is born on the far rank and any move sets the bit. Test:
`test_env.py::test_a_third_rook_does_not_grant_a_castle`, both importers.

## 3. Micro-batch accumulation was not exact under a value mask

`train_step` scaled each micro-batch's `total` by its row share. The policy term is a
mean over rows, so that is exact; the value term is a mean over the *masked* rows, so a
micro-batch with one supervised row weighed as much as one with two. The review's
four-row case cancels a `−2/3` value gradient to zero. `az_loss` now reports
`value_rows`, `loss.micro_batch_weights` returns one scale per term, and `train_step`
backpropagates `policy * w_p + value * w_v`; the logged `value` and `total` use the same
weights. A batch with no supervised row gets `w_v = 0`. `training.md` §7.2 says so.
Test: `test_train.py::test_gradient_accumulation_is_exact_under_a_value_mask`, at the
identity's tolerance in double, with a check that the old weighting fails it.

⚠️ One run on the ledger trained under the old weighting: `t12h-muon-wdl-vs4`
(`--value-subsample 4`, the only chain that ever set it). Measured the same day on
1024 live records of `t24h-reinject-lr6` through `t12h-wdl`'s final network in double,
the old weighting against the exact gradient, with a Bernoulli mask at the run's rate:

| mask | micro-batches | whole gradient | value-head gradient | head cosine / norm ratio |
|---|---|---:|---:|---|
| 1 in 4 | 4 × 256 | 2.9 % | 5.4 % | 0.999 / 0.97 |
| 1 in 4 | 16 × 64 | 13.5 % | 20.6 % | 0.983 / 1.07 |
| 1 in 8 | 4 × 256 | 8.1 % | 8.5 % | 0.997 / 0.97 |

The new weighting sits at 1e-8 of the exact gradient in the same table. Production is
4 × 1024, where the supervised count fluctuates half as much in relative terms as at
4 × 256, so `t12h-muon-wdl-vs4` trained on a value gradient a few percent off in scale
and within a cosine of 0.999 of the right direction. Not a contamination worth a
re-run; the run's own noise is larger.

## Test maintenance

`test_quant.py`'s two-board comparisons asked for `int8=True, two_boards=False` under
the default scheme `"all"`, which is two-boards-only by construction. They now name
`"ffn"` on both sides; the claim is about the CTA split, not the scheme.

Run: `test_env.py`, `test_gumbel.py`, `test_train.py` (targeted), `test_search_cuda.py`
(43 passed, 2 slow skipped), `test_cuda_env.py`, `test_quant.py -k two_boards` — green.

## The slow tests, and one stale depth bound

The slow opt-in tests were run too (`--slow`; 171 in `test_train`, `test_search`,
`test_search_cuda`). One failed, and it failed on untouched `HEAD` as well:
`test_search_cuda.py::test_agrees_at_the_full_budget` asserted the deepest path at
n = 800 over 64 random positions exceeds 8, and it is exactly 8. The kernel and the
reference agreed; only the bound was off.

Predicted cause: the first-play urgency moved from 0 to 0.5 on 2026-08-02
(`journal/2026-07-31-value-collapse.md`), so an untried edge scores a draw instead of a
loss, and the tree widens rather than deepens. Measured on the reference over the same
positions and seed, `FPU_DRAW` monkeypatched:

| `FPU_DRAW` | max depth | leaves at depth 7 / 8 / 9+ |
|---|---|---|
| 0.0 (search of 2026-07-30) | 13 | 4622 / 2238 / 1193 |
| 0.5 (since 2026-08-02) | 8 | 63 / 2 / 0 |

The bound was the old search's number and the test had not run since. It is now
`>= 7`, one below the current measurement, with both numbers in the test.
