"""Mutation testing for the dead-token mask: do the tests in test_model.py bite?

A passing test suite says nothing until something has tried to defeat it. This
one edits ``brokefish/nn/triton_impl.py`` textually, one plausible mistake at a time,
writes each variant to a real file (Triton reads source off disk, so exec'ing a
string does not work), imports it, and points the production tests at it. A
mutant that survives every test is a hole in the suite.

M9 is a control that changes only dead rows, which nothing downstream reads. It
is expected to survive. A suite that also kills M9 is reacting to noise.

Opt-in: it compiles nine kernels and takes a few minutes, so ``pytest tests/``
skips it. Run it after touching the mask.

    pytest tests/ --mutation
    python -m tests.test_mutation_mask     # same thing, with the full report
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

import torch

import tests.test_model as tm

try:  # the module has to stay runnable without pytest installed
    import pytest
    mutation = pytest.mark.mutation
except ImportError:  # pragma: no cover
    def mutation(fn):
        return fn

MODEL = pathlib.Path(__file__).resolve().parent.parent / "brokefish" / "nn" / "triton_impl.py"
OUT = pathlib.Path(__file__).resolve().parent / "_mutants"

# Anchors are exact lines of triton_impl.py. If one stops matching, the harness is
# stale and says so rather than silently testing nothing.
MASK_LINE = "        keep = (board[:, None] == board[None, :]) & alive[None, :]"
LOAD_LINE = "        alive = tl.load(alive_ptr + rows) != 0"
WHERE_LINE = ('            s = tl.where(keep, tl.dot(q, tl.trans(k), out_dtype=ACC).to(tl.float32), -float("inf"))')
V_LINE = ("            v = tl.load(qkv_scratch + rows[:, None] * (3 * D) + 2 * D "
          "+ h * DH + head_col[None, :])")

MUTANTS = {
    "M1 mask rows instead of columns": [
        (MASK_LINE, "        keep = (board[:, None] == board[None, :]) & alive[:, None]"),
    ],
    "M2 mask rows as well as columns": [
        (MASK_LINE,
         "        keep = (board[:, None] == board[None, :]) & alive[None, :] & alive[:, None]"),
    ],
    "M3 predicate inverted": [
        (LOAD_LINE, "        alive = tl.load(alive_ptr + rows) == 0"),
    ],
    "M4 alive read one slot off": [
        (LOAD_LINE, "        alive = tl.load(alive_ptr + rows + 1) != 0"),
    ],
    "M5 every CTA reads board 0": [
        (LOAD_LINE, "        alive = tl.load(alive_ptr + tl.arange(0, BM)) != 0"),
    ],
    "M6 mask applied on layer 0 only": [
        (WHERE_LINE,
         "            s = tl.where(keep | (layer > 0), "
         'tl.dot(q, tl.trans(k), out_dtype=ACC).to(tl.float32), -float("inf"))'),
    ],
    "M7 mask applied on head 0 only": [
        (WHERE_LINE,
         "            s = tl.where(keep | (h > 0), "
         'tl.dot(q, tl.trans(k), out_dtype=ACC).to(tl.float32), -float("inf"))'),
    ],
    "M8 dead values zeroed instead of dead keys masked": [
        (MASK_LINE, "        keep = board[:, None] == board[None, :]"),
        (V_LINE, V_LINE + "\n            if HAS_MASK:\n"
                 "                v = tl.where(alive[:, None], v, 0.0).to(tl.float16)"),
    ],
    "M9 dead tokens attend to themselves (control, must survive)": [
        (MASK_LINE,
         "        _diag = tl.arange(0, BM)[:, None] == tl.arange(0, BM)[None, :]\n"
         "        keep = (board[:, None] == board[None, :]) & (alive[None, :] | _diag)"),
    ],
}

EXPECTED_SURVIVORS = {"M9"}

TESTS = [
    "test_matches_torch_unmasked",
    "test_masked_matches_torch_across_occupancies",
    "test_kernel_is_no_further_from_fp32_than_torch_is",
    "test_dead_tokens_are_inert",
    "test_dead_tokens_must_be_finite",
    "test_all_alive_reproduces_the_unmasked_kernel",
    "test_board_counts",
]


def load_mutant(tag: str, edits, source: str):
    src = source
    for old, new in edits:
        if old not in src:
            raise RuntimeError(
                f"{tag}: anchor no longer present in {MODEL.name}, harness is stale:\n{old}")
        src = src.replace(old, new, 1)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{tag}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


def run_suite():
    """(names that failed, first line of each failure). A failure is a catch."""
    caught, why = [], {}
    for name in TESTS:
        tm._CACHE.clear()
        torch.cuda.empty_cache()
        try:
            getattr(tm, name)()
        except AssertionError as e:
            caught.append(name)
            why[name] = (str(e).splitlines() or [""])[0][:70]
        except Exception:
            caught.append(name)
            why[name] = "raised " + traceback.format_exc().splitlines()[-1][:60]
    return caught, why


def per_occupancy():
    tm._CACHE.clear()
    return [(r["name"], r["rel"] >= tm.REL_TOL or r["ulps"] >= tm.ULP_TOL, r["rel"])
            for r in tm.compare()]


# Mutants to break down per occupancy in the report, because which patterns catch
# them is the interesting part.
DETAIL = ("M5",)


@mutation
def test_mask_mutants_are_all_killed() -> None:
    """Every plausible mask defect has to fail at least one test, and the control
    has to pass all of them."""
    source = MODEL.read_text()
    already_red, _ = run_suite()
    assert not already_red, f"suite is red before mutating: {already_red}"
    print(f"unmutated suite: {len(TESTS)} tests, all green\n")

    # This suite is Triton-only: it works by rewriting that file's source.
    impls, survivors = tm.IMPLS, []
    tm.IMPLS = ["triton"]
    for tag, edits in MUTANTS.items():
        short = tag.split(" ", 1)[0]
        tm.IMPL_OVERRIDE["triton"] = load_mutant(short, edits, source).FusedEncoder
        caught, why = run_suite()
        print(tag)
        for name in caught:
            print(f"   caught by {name:50s} {why[name]}")
        if not caught:
            survivors.append(short)
            print("   SURVIVED every test")
        if short in DETAIL:
            print("   per occupancy:")
            for name, bad, rel in per_occupancy():
                print(f"     {'catch ' if bad else '  --  '} {name:26s} rel {rel:.2e}")
        print()
        tm.IMPL_OVERRIDE.pop("triton", None)
        tm._CACHE.clear()
        torch.cuda.empty_cache()

    tm.IMPLS = impls
    unexpected_survivors = sorted(set(survivors) - EXPECTED_SURVIVORS)
    missing = sorted(EXPECTED_SURVIVORS - set(survivors))
    killed = len(MUTANTS) - len(survivors)
    print(f"{killed}/{len(MUTANTS) - len(EXPECTED_SURVIVORS)} defects killed, "
          f"{len(EXPECTED_SURVIVORS)} controls expected to survive")
    assert not unexpected_survivors, f"hole in the suite, these survived: {unexpected_survivors}"
    assert not missing, f"controls were killed, the suite is over-sensitive: {missing}"
    print("[OK] suite is both sensitive and specific")


if __name__ == "__main__":
    try:
        test_mask_mutants_are_all_killed()
    except AssertionError as e:
        raise SystemExit(f"[FAIL] {e}")
