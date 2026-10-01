# Core algorithm review

Date: 2026-09-09  
Reviewed commit: `f7443f9bc3efc8da21fb005a2b26a9d706f2482a`

Scope: search, chess-state conversion and legal moves, and training-loss mathematics. Run resumption and checkpoint recovery are intentionally excluded. This document records confirmed findings; it does not implement fixes or claim that the remaining code is bug-free.

CUDA became accessible after the session permissions changed. GPU checks ran on an NVIDIA GeForce RTX 4060 Laptop GPU.

## 1. Gumbel search flattens distinct small priors

Locations: `brokefish/search/gumbel.py`, line 160; `csrc/search.cuh`, lines 338, 458, and 477.

### Problem

The log-probability calculation clamps stored priors to `torch.finfo(torch.float16).tiny`, or `6.103515625e-5`. That is the smallest **normal** float16 value, not the smallest representable positive value. Float16 subnormals extend down to `5.960464477539063e-8`.

Representable priors such as `1e-5` and `1e-6` therefore become identical before selection. The CUDA implementation duplicates this floor through `kHalfTiny`.

### Confirmed reproduction

Use the initial chess position's 20 legal root edges, disable Gumbel noise, and consider two candidates. Set the priors to:

- Edge 0: the remaining probability mass, approximately `0.999972`.
- Edge 19: `1e-5`.
- The other 18 edges: `1e-6` each.

The two highest-prior edges are `[0, 19]`. Both the Torch search and CUDA search instead selected `[0, 1]` in the reproduction. A diagnostic change preserving representable small priors restored edge 19 as the second candidate in the Torch implementation.

### Impact and correction direction

With equal initial Q scores, the candidate set should preserve the prior ordering when noise is disabled. The current floor can exclude a stronger-prior move and replace it with a weaker one. The same logit recovery also affects interior selection.

Handle hard-zero priors without flattening positive, representable probabilities. Keep the Torch and CUDA implementations consistent. The frequency and playing-strength impact on trained checkpoints have not been measured.

## 2. Imported positions can acquire illegal castling rights

Locations: `brokefish/env/torch_impl.py`, `from_board` around lines 705–740 and `from_fen` around lines 775–789.

### Problem

Extra rooks are stored in pawn slots. Their piece type remains rook, but the importers fail to initialize their lost-castling-right flag correctly:

- `from_board` initializes rook rights for the normal rook slots, missing surplus rooks stored elsewhere.
- `from_fen` changes the parsing character to a pawn character when assigning an overflow piece to a pawn slot. This skips the later rook-rights branch.

### Confirmed reproduction

Import this valid position through either `from_fen` or `from_board`:

```text
4k3/8/8/8/8/8/8/RR2K2R w Q - 0 1
```

White has queenside castling rights only. Queenside castling is currently blocked by the rook on b1; kingside castling is forbidden by the position's rights.

Both importers nevertheless produce kingside rights as well. Both the Torch and CUDA move generators then include `e1g1`, an illegal kingside castle. The python-chess comparison recognizes the FEN as valid and permits no castling move.

### Impact and correction direction

Search can choose an illegal move from an imported position. Incorrect rights can also affect position identity and subsequent search state.

Initialize rights for every live rook using its actual piece type, square, and the supplied rights, independently of its storage slot. This reproduction concerns imported positions; it does not establish the same failure in uninterrupted self-play from the initial board.

## 3. Microbatching changes the masked value-loss gradient

Locations: `brokefish/train/loss.py`, lines 263–294; `brokefish/train/loop.py`, lines 859–871.

### Problem

Within each microbatch, value loss is divided by the number of value-supervised rows. Gradient accumulation then scales the entire loss by the microbatch's total row count divided by the full batch's row count.

This weighting is correct for the policy mean, but not for the masked value mean when the fraction of supervised rows differs between microbatches. It affects both scalar MSE and the masked win/draw/loss cross-entropy path.

The loss comments acknowledge this approximation; the training-loop comment still describes accumulation as exact. This finding is a demonstrated consequence of that approximation, not an assertion that it was undocumented everywhere.

### Confirmed reproduction

Using the actual `az_loss` function with a shared scalar value prediction initialized to zero:

```text
Targets:          [-1, 0, +1, +1]
Value mask:       [ 1, 0,  1,  1]
Microbatch split: [first two] [last two]

Full-batch value gradient:       -0.6666666865
Two accumulated microbatches:     0.0
```

The first microbatch's single supervised loss receives the same aggregate weight as the second microbatch's two supervised losses, canceling a gradient that should remain nonzero.

### Impact and correction direction

Changing `micro_batch` changes the training objective when value supervision is subsampled. Small or sparse microbatches can show a large discrepancy; the reproduction does not quantify the effect in a production-size batch. With all rows supervised, this particular mismatch disappears.

Accumulate policy and value terms separately: weight policy means by row count and value means by supervised count, each relative to its corresponding full-batch count. Handle a full batch with no value-supervised rows explicitly, and apply consistent weighting to reported losses.

## 4. Zero-prior visited moves corrupt Gumbel's completion value

Locations: `brokefish/search/gumbel.py`, lines 127–136; `csrc/search.cuh`, lines 386–407.

### Problem

Selection assigns a finite logit to a legal move even when its stored prior has rounded to zero. However, completed-Q calculation uses the raw prior when computing the weighted mean of visited Q values.

If all visited edges have zero stored prior, the denominator fallback avoids division by zero but leaves the weighted Q equal to zero, regardless of the observed outcomes. Search evidence then pulls the completion value toward a loss in the tree's `[0, 1]` convention.

### Confirmed reproduction

For two valid edges, pass the following to `completed_q`:

```text
Priors:           [0, 1]
Visits:           [1, 0]
Q values:         [1, 0]
Network value:    0.5

Actual completed Q:   [1, 0.25]
With guarded priors:  [1, 0.75]
```

The only visited edge is a win. With a positive prior guard, its weighted Q is 1 and the unvisited completion is `(0.5 + 1 * 1) / 2 = 0.75`. The current calculation instead produces `0.5 / 2 = 0.25`. Increasing the winning edge's visit count drives the unvisited completion toward zero rather than one.

This numerical reproduction was run against the Torch helper on GPU. The CUDA source contains the same raw-prior weighting and denominator fallback; a separate end-to-end CUDA reproduction of this fourth finding was not performed.

### Impact and correction direction

This is conditional on all visited priors being zero, not a general failure of every completion calculation. It can distort exploration and completed-value estimates in that case.

Use a positive-prior guard or another explicitly defined fallback before forming the visited weighted mean. The [mctx reference implementation](https://raw.githubusercontent.com/google-deepmind/mctx/main/mctx/_src/qtransforms.py), `_compute_mixed_value`, guards priors before this calculation. Coordinate the correction with finding 1, without reintroducing its excessively large floor.

## Test results and separate test-maintenance issue

The two GPU-enabled test batches completed with **429 passed, 18 skipped, and 11 failed**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_search.py tests/test_search_cuda.py tests/test_gumbel.py \
  tests/test_train.py tests/test_muon.py tests/test_cuda_env.py \
  tests/test_b2.py tests/test_quant.py
# 351 passed, 10 skipped, 11 failed

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_env.py tests/test_notation.py tests/test_model.py \
  tests/test_oracle.py tests/test_validate.py --tb=short
# 78 passed, 8 skipped
```

All 11 failures occurred during encoder construction in `tests/test_quant.py`, principally lines 406 and 428. These tests request `int8=True, two_boards=False`, while the default quantization scheme is now `all`, which the encoder rejects with `two_boards=False`.

The failures concern `test_two_boards_per_cta_is_bit_identical_to_one` (10 parameterizations) and `test_two_boards_runs_the_debug_stages`. They indicate stale test/configuration assumptions, not demonstrated incorrect kernel output. The comparison tests should explicitly select a scheme supporting both configurations, or be updated to cover the currently supported combinations.

The four findings above were established through targeted reproductions and source inspection, separately from these existing-suite failures. No algorithm fixes or regression tests were committed as part of this review.
