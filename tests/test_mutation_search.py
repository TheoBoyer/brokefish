"""Mutation testing for the MCTS reference: do the tests in test_search.py bite?

A passing test suite says nothing until something has tried to defeat it, and the
search has no whole-search oracle, so this is the closest thing to one available
before the kernel exists. It edits ``brokefish/search/torch_impl.py`` textually,
one plausible mistake at a time, writes each variant to a real file, imports it,
and points the production tests at it. A mutant that survives every test is a hole
in the suite.

The mistakes are not invented: every one of them is either a documented ⚠️ in
``docs/mcts.md`` or something that went wrong while writing the reference.

M13 is a control and is expected to survive. Replacing the lowest-index tie-break
with ``torch.argmax`` changes nothing measurable, because torch happens to return
the lowest index on both CPU and CUDA. ``_lowest_argmax`` stays because torch does
not *promise* that and §12 compares trees with the kernel edge for edge, so the
tie-break is a contract rather than an observed accident. A suite that kills M13 is
reacting to an implementation detail of torch.

Opt-in: it runs the whole search suite sixteen times and takes about seven
minutes, so ``pytest tests/`` skips it. Run it after touching the search.

    pytest tests/ --mutation
    python -m tests.test_mutation_search     # same thing, with the full report
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

try:  # the module has to stay runnable without pytest installed
    import pytest
    mutation = pytest.mark.mutation
except ImportError:  # pragma: no cover
    def mutation(fn):
        return fn

SEARCH = pathlib.Path(__file__).resolve().parent.parent / "brokefish" / "search" / "torch_impl.py"
OUT = pathlib.Path(__file__).resolve().parent / "_mutants"

# Anchors are exact fragments of torch_impl.py. If one stops matching, the harness
# is stale and says so rather than silently testing nothing.
MUTANTS = {
    "M1  parity (L-d)%2 -> d%2": (
        "flip = ((length[rows] - d) % 2) == 1",
        "flip = (d % 2) == 1"),
    "M2  parity: never flip": (
        "q = torch.where(flip, 1.0 - q_leaf[rows], q_leaf[rows])",
        "q = q_leaf[rows]"),
    "M3  pb_c_init multiplied, not added": (
        "pb_c = torch.log((n_v + c.pb_c_base + 1.0) / c.pb_c_base) + c.pb_c_init",
        "pb_c = torch.log((n_v + c.pb_c_base + 1.0) / c.pb_c_base) * c.pb_c_init"),
    "M4  N_v instead of sqrt(N_v)": (
        "pb_c = pb_c * n_v.sqrt() / (nvis + 1.0)",
        "pb_c = pb_c * n_v / (nvis + 1.0)"),
    "M5  denominator N instead of N+1": (
        "pb_c = pb_c * n_v.sqrt() / (nvis + 1.0)",
        "pb_c = pb_c * n_v.sqrt() / (nvis + 1e-6)"),
    "M6  first-play urgency 0 -> 0.5": (
        "score = pb_c * prior + torch.where(nvis > 0, q, torch.zeros_like(q))",
        "score = pb_c * prior + torch.where(nvis > 0, q, torch.full_like(q, 0.5))"),
    "M7  terminal value (result+1)/2 -> (1-result)/2": (
        "(result[terminal_rows].float() + 1.0) / 2.0",
        "(1.0 - result[terminal_rows].float()) / 2.0"),
    "M8  value head (v+1)/2 -> (1-v)/2": (
        "(value[touched].float() + 1.0) / 2.0",
        "(1.0 - value[touched].float()) / 2.0"),
    "M9  truncation keeps the lowest priors": (
        'key = torch.where(cand, -logit, torch.full_like(logit, float("inf")))',
        'key = torch.where(cand, logit, torch.full_like(logit, float("inf")))'),
    "M10 repetition ignores the irreversible cut": (
        "cut = torch.where(child_irrev, length, deepest.clamp(min=0))",
        "cut = torch.zeros_like(length)"),
    "M11 repetition drops the game ring": (
        "ring_len = torch.where(keep_ring, self.game_ring_len, "
        "torch.zeros_like(self.game_ring_len))",
        "ring_len = torch.zeros_like(self.game_ring_len)"),
    "M12 repetition drops the in-tree half": (
        "on_tree = ((hashes == child_hash[:, None]) & window).sum(-1)",
        "on_tree = torch.zeros_like(length)"),
    "M13 tie-break by argmax (control, expected to survive)": (
        "return torch.where(is_max, idx, torch.full_like(idx, e)).min(dim=-1).values",
        "return x.argmax(dim=-1)"),
    "M14 promotion order reversed": (
        "field | ((cols % 4) << PROMO_SHIFT)",
        "field | ((3 - cols % 4) << PROMO_SHIFT)"),
    "M15 Gamma(alpha<1) without the U^(1/alpha) boost": (
        "        if boost:\n",
        "        if False:\n"),
    "M16 argmax before tau instead of sampling": (
        "e = torch.where(self.game_ply < c.tau_plies, sampled, best)",
        "e = best"),
}

EXPECTED_SURVIVORS = {"M13"}


def build(name: str, old: str, new: str) -> pathlib.Path:
    source = SEARCH.read_text()
    if old not in source:
        raise AssertionError(f"{name}: anchor no longer in torch_impl.py, the harness is stale:\n"
                             f"  {old!r}")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"search_{name.split()[0]}.py"
    path.write_text(source.replace(old, new, 1))
    return path


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"mutant_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_suite_against(module) -> list[str]:
    """Every test in test_search.py, with `Search` swapped for the mutant's.

    The tests build their search through the module-global names, so rebinding
    three attributes redirects all of them, including the subclass in
    `test_dirichlet_mixes_at_eps`.
    """
    import tests.test_search as ts

    saved = (ts.Search, ts.SearchConfig, ts.check_invariants)
    ts.Search, ts.SearchConfig = module.Search, module.SearchConfig
    ts.check_invariants = module.check_invariants
    failed = []
    try:
        for name, fn in ts._tests(slow=False):
            try:
                fn()
            except Exception:  # noqa: BLE001 - a failure is the point
                failed.append(name)
    finally:
        ts.Search, ts.SearchConfig, ts.check_invariants = saved
    return failed


@mutation
def test_every_mutant_is_killed():
    survivors = []
    for name, (old, new) in MUTANTS.items():
        tag = name.split()[0]
        try:
            module = load(build(name, old, new))
        except Exception:
            print(f"  {name:52s} BUILD FAILED")
            traceback.print_exc()
            survivors.append(tag)
            continue
        failed = run_suite_against(module)
        first = ", ".join(failed[:2]) if failed else "NOTHING"
        print(f"  {name:52s} {len(failed):3d} killed by  {first}", flush=True)
        if not failed:
            survivors.append(tag)

    unexpected = set(survivors) - EXPECTED_SURVIVORS
    missing = EXPECTED_SURVIVORS - set(survivors)
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} killed, "
          f"survivors {sorted(survivors)}")
    assert not unexpected, f"holes in the suite: {sorted(unexpected)}"
    assert not missing, (f"{sorted(missing)} was expected to survive and did not; "
                         "the suite is reacting to an implementation detail of torch")


if __name__ == "__main__":
    test_every_mutant_is_killed()
