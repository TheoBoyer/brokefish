# Journal

Dated entries, newest first. **Append-only: an entry is never edited to stay true.**

That is the whole point of the split. A reference document has to be correct today,
so it gets rewritten whenever the code moves, and its history is destroyed by
design. A journal entry has to be correct *as of its date* and nothing more, so it
costs nothing to keep and can say things a reference page cannot: what was believed,
what was tried, what the measurement actually was, and what turned out to be wrong.

Where an entry and a reference page disagree, the reference page is current and the
entry records what was true when it was written. Neither is a bug.

| date | entry | what it settles |
|---|---|---|
| 2026-07-31 | [The collapse of `c2-8h`](2026-07-31-value-collapse.md) | the first long training run fell into an absorbing state: a first-play-urgency constant that did not ride the `[0,1]` remap made low-prior moves unreachable at any budget |
| 2026-07-31 | [Build log, A0 through D1](2026-07-31-build-log.md) | how the engine, the network, the search, the training loop and the diagnostics were built, and where each estimate was wrong |
| 2026-07-30 | [Eval prior art](2026-07-30-eval-prior-art.md) | AlphaZero, KataGo, lc0 and SAI read against the papers rather than from memory; four claims in `evaluation.md` were wrong |
| 2026-07-30 | [Fidelity audit](2026-07-30-fidelity.md) | where the reproduction of FIDE and of AlphaZero rests on a reading rather than on a test. A betting sheet, not a bug list |
| 2026-07-29 | [Writing the encoder kernel](2026-07-29-encoder-kernel.md) | how `csrc/encoder.cu` was built, including the version slower than the Triton kernel it replaced |
| 2026-07-28 | [Prior art survey](2026-07-28-prior-art.md) | who has done this before, at what cost, and the scaling-law correction that killed the 1M-parameter sizing |

## Where the other kinds of writing live

- **[`reference/`](../reference/spec.md)** — what the code must do. No dates in the
  body, no measurements, no history except a changelog at the foot of the page.
- **[`ledger/`](../ledger/state.md)** — the numbers. `state.md` is one bullet per
  landed component; `perf.md` is one row per measurement. Every number in the
  repository cites the ledger rather than restating it, so a kernel getting faster
  is one edit and not twelve.
- **[`roadmap.md`](../roadmap.md)** — what is next, and the decisions still open.
