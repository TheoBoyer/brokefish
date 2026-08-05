# 2026-08-04 — QA for a trained checkpoint, and three things it found

`brokefish/nn/validate.py`, `tests/test_validate.py`.

## The gap

Every encoder correctness test in the repository ran on an **untrained** net, and two
of them did not run a net at all:

* `tests/test_model.py` drives the megakernel with `torch.randn(n, 32, 256)`
  activations — synthetic noise, no board, no network.
* `tests/test_b2.py` does exercise the full boards-to-logits path, on
  `BrokefishNet()` at random init.

Both then measure in **logit space**, relative to the largest magnitude, at
`REL_TOL = 5e-3`. That bar loosens as training sharpens the policy, exponentially:
`_rel` permits an absolute logit error of `5e-3 * max|logit|`, and the softmax turns
an absolute logit error `d` into a ratio error up to `exp(2d)`.

| `max|logit|` | error the bar permits in probability space |
|---|---|
| 3 (random init) | ~3 % |
| 10 | ~10 % |
| 30 | ~35 % |

A green `tests/` was therefore consistent with a third of the prior mass being wrong
on a trained checkpoint. The suite measures what the search consumes instead: the
**prior over the node's legal edges**, taken out of `Search._expand` by swapping the
evaluator under `root_init` rather than re-derived. Re-deriving that expression is how
this repository ended up with four copies of the first-play-urgency constant.

## The measurement

`t7h-n128-collapse@2006`, 4,111 non-terminal positions from that run's own replay
buffer plus five adversarial FENs. Against an fp32 oracle carrying the same
fp16-rounded weights, so only the arithmetic differs:

| | max \|Δp\| | p95 | max rel | top-1 moved | flip risk | flushed |
|---|---|---|---|---|---|---|
| `store` (fp16 storage floor) | 7.32e-4 | 3.05e-5 | 0.25 % | 0.024 % | 0.243 % | **0** |
| `torch16` | 9.77e-4 | 1.22e-4 | 0.63 % | 0.219 % | 0.316 % | **0** |
| `triton` | 1.22e-3 | 1.22e-4 | 0.75 % | 0.195 % | 0.365 % | **0** |
| `cuda` | **2.20e-3** | 1.22e-4 | 1.36 % | 0.219 % | 0.486 % | **0** |

`max |Δv| = 1.6e-3`. `max|logit| = 8.2`, which is **0.013 % of fp16's 65504** — four
orders of headroom, so overflow is not a live risk at this stage of training. **Zero
flushed priors**, which answers `search.md` §13's open question at this sharpness: no
legal move's prior reaches fp16's subnormal floor yet.

### Predictions, scored

| predicted | measured | |
|---|---|---|
| kernel max \|Δp\| ≈ 1e-3 | 2.2e-3 | 2× optimistic |
| the fp16 **storage floor would exceed** the kernel error | floor 7.3e-4 **<** kernel 2.2e-3 | **wrong** |
| top-1 disagreement < 0.1 % | 0.219 % | 2× |

The floor prediction assumed logits near 20, where an fp16 ULP is worth ~1.6 % of an
unnormalised softmax weight. Real logits are 8.2, so the ULP is worth ~0.8 % and
mostly cancels in the ratio, while the kernel's error accumulates over eight layers
and dominates. The crossover has not happened yet; the suite can now watch for it.

## Three findings

### 1. `cuda.forward_full` returns views into buffers it reuses. `triton` does not.

A caller that keeps outputs across calls gets the last batch repeated. This cost the
first measurement: chunking 4,111 positions into five calls and concatenating gave
four copies of chunk four, which read as a **950× kernel error, 59 % top-1
disagreement, and a value head off by 1.811 on a [-1, 1] output**. Every one of those
numbers was the harness.

⚠️ `.contiguous()` does not save you — on an already-contiguous tensor it returns the
same object, and `cuda_impl._evaluate_staged` calls exactly that.

Not a defect: the search hands the output straight to `expand` in the same step, and
re-allocating on a path that runs 800 times a move would cost. It is a **contract
difference between the two implementations that nothing stated**, which makes it a
trap for the next caller. Now detected by `aliases_across_calls`, reported on every
run, and pinned by a test so a change to it has to be deliberate.

### 2. §4.3 truncation is precision-dependent on a near-uniform policy.

On the 218-move position at random init, fp16 and fp32 keep a **different 96 of 218**
candidates — the 96th and 97th priors are close enough that the two precisions order
them differently. Zero on the trained checkpoint, where the policy is not tied.

A different edge set is a different move list, which is worse than a shifted
probability, so `report` treats it as a hard stop. It was also silently making one
test flaky, because `BrokefishNet()` draws from the global generator and nothing
seeded it.

### 3. The bite tests passed for the wrong reason. Twice.

A validator nobody has watched fail is an assertion, not a measurement — so the suite
was pointed at deliberately corrupted encoders. The first two attempts were both
broken, and both were green:

* **Perturbing `policy[:, 0, 0]`** measured nothing at a bump of **0.5**. Slot 0 is a
  white pawn and slot 0 → a1 is never legal, so §6.4's mask deletes that logit before
  the softmax ever sees it. A test that perturbs an unreachable input always passes.
* **`validate(checkpoint=None, forwards=...)`** built its *own* fresh random
  `BrokefishNet` for the oracle, so the reference was a different network from the one
  the injected encoder wrapped. It reported a 0.85 prior delta for a **0.002** logit
  bump, non-monotone in the bump, and asserted successfully. `validate` now refuses
  `forwards` without an explicit `net`.

The working version perturbs either every logit by a scale factor — what a wrong
normalisation looks like, and something no softmax cancels — or a checkerboard of the
*legal* entries, and calibrates its gate from the clean encoder in the same run rather
than hard-coding a number.

## One judgement call, flagged

The `2.0×`-versus-torch-fp16 gate applies to **p95, not max**. Both sides of a
max-ratio are the single largest of ~10⁵ samples, so the ratio is built from one
observation each: measured here, all three fp16 paths share a p95 of 1.22e-4 while
their maxima spread over 2.25×. The max is still gated absolutely by `--max-abs-dp`
(default 0.01) and the ratio is still printed on every run.

⚠️ That bar was moved **after** it tripped, which is exactly when to be suspicious.
The evidence is printed so it can be overruled.
