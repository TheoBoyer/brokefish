# Evaluation prior art — verification pass

**2026-07-30.** Written to check the claims in [`docs/evals.md`](docs/evals.md) §12
against the papers rather than against memory, and to look at what the other
from-scratch projects actually do. Deliberately **not** in the MkDocs nav (`mkdocs.yml`):
working material, not part of the published site.

Everything below was read this session. Where a claim could not be sourced it says so.

---

## 1. Scorecard: what `evals.md` §12 claimed, and what is true

| claim as written | verdict |
|---|---|
| AZ drops AGZ's gating/evaluator, maintains one continuously-updated net | ✅ **confirmed**, and the 55 % threshold is named explicitly |
| AZ uses 800 simulations per move in training self-play | ✅ **confirmed** |
| AZ-vs-Stockfish preprint: SF8, 64 threads, 1 GB hash, 1 min/move, 100 games | ✅ **confirmed**, all five numbers |
| the *Science* version widens to 1000 games, TCEC openings, time odds | ✅ **confirmed**, plus details I did not have |
| AZ's training curve is self-anchored | ⚠️ **half wrong** — see §2.4 |
| AZ's training curve uses BayesElo | ✅ **confirmed** (marked "medium confidence"; it is stated outright) |
| KataGo fits BT/BayesElo over a graph rather than chaining | ✅ **confirmed**, with a better pairing rule than I proposed |
| **KataGo does not gate** — implied by `evals.md` §11's "AZ says no" | ❌ **wrong**. KataGo gates: 100 wins out of 200 |
| lc0 gated early and dropped it later | ❌ **not verified, and probably backwards** — see §4 |
| lc0 has visibly hit self-anchored Elo inflation | ✅ **confirmed from lc0's own FAQ** |
| Lichess puzzles: CC0, ~4M, Glicko-rated | ⚠️ **stale** — CC0 ✅, but 6.01M and Glicko-2 with a deviation field |

Two of these change the design, not just the footnote: §2.4 and §3.2.

---

## 2. AlphaZero

Sources: Silver et al., *Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm*, [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
(read via [ar5iv](https://ar5iv.labs.arxiv.org/html/1712.01815)); and
[Science 362:1140-1144](https://www.science.org/doi/10.1126/science.aar6404), whose
match conditions are collated on the
[Chessprogramming wiki](https://www.chessprogramming.org/AlphaZero).

### 2.1 No gating — confirmed, and stronger than I said

> AlphaZero "simply maintains a single neural network that is updated continually,
> rather than waiting for an iteration to complete"

and it omits the AGZ evaluation step that required a 55 % win margin before
replacement. So the removal is explicit, not inferred.

For contrast, the thing that was removed
([AGZ, Nature 550:354-359](https://www.nature.com/articles/nature24270), Methods,
[free PDF](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)):

> Each evaluation consists of 400 games, using an MCTS with 1,600 simulations to
> select each move, using an infinitesimal temperature […] If the new player wins by
> a margin of >55 % (to avoid selecting on noise alone) then it becomes the best
> player.

400 games at 55 % is a ~35 Elo detection threshold — consistent with the §9
arithmetic in `evals.md`, and worth noting that AGZ chose a *margin*, not a
significance test.

### 2.2 800 simulations — confirmed

> "During training, each MCTS used 800 simulations."

### 2.3 The matches — confirmed, with additions

**Preprint (2017).** Stockfish 8 (official Linux release), 64 threads, 1 GB hash,
1 minute per move, 100 games. Chess result from Table 1: AZ as White 25 W / 25 D / 0 L,
as Black 3 W / 47 D / 0 L → **28 W / 72 D / 0 L**. No opening book.

**Science (2018).** 1000 games; Stockfish under its 2016 TCEC Season 9 superfinal
settings — 44 threads on 44 cores, 32 GiB hash, **6-man Syzygy tablebases**, time
control **3 h + 15 s increment**. Openings: 12 common human openings plus the TCEC
S9 superfinal positions. Result **155 W / 6 L** (so ~839 draws). Additional matches
at time odds of 1/3 and 1/10.

⚠️ **AZ gave Stockfish tablebases.** `docs/roadmap.md` and `evals.md` §7 say "no
tablebases on either side or the same on both". The precedent we cite is the third
option: tablebases for the opponent only. That is defensible (it strengthens the
opponent) but it is not what our sentence says, and it should be a deliberate choice.

### 2.4 The training curve — this is the one that was half wrong

> "Elo ratings were computed from the results of a 1 second per move tournament
> between iterations of AlphaZero during training, and also a baseline player" using
> "BayesElo" with "standard constant c_elo = 1/400".

Two corrections to `evals.md` §12:

1. **It is not purely self-anchored.** The tournament includes a baseline player
   alongside the AZ iterations. The self-anchored part is right — most games are
   between checkpoints — but there is an external anchor *in the fit*, which is
   closer to what §6 (calibration) proposes than to a pure checkpoint league.
2. **It is a time control, not a simulation count.** One second per move.
   `evals.md` §3 recommends fixed simulations per move and does not cite AZ for
   it — correctly, as it turns out, but the doc should say so explicitly so nobody
   later assumes AZ backs it. The reasons in §3 (thermal drift, hardware
   independence, `n` as an axis) stand on their own; AZ is simply silent, or mildly
   against.

Also worth having: at evaluation AZ "selects moves greedily with respect to the root
visit count" — temperature 0, no Dirichlet. That is a protocol detail `evals.md`
does not currently state and should.

---

## 3. KataGo

Source: David J. Wu, *Accelerating Self-Play Learning in Go*,
[arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
([ar5iv](https://ar5iv.labs.arxiv.org/html/1902.10565)).

### 3.1 The rating method — confirmed

> "a global Bayesian maximum-likelihood Elo based on all game results so far", via
> "a custom implementation of BayesElo"

A global fit over all games, not a chain of pairwise deltas. This is what `evals.md`
§5.2 claims and it holds.

**The pairing rule is better than what I proposed.** Games are played

> "with frequency proportional to the predicted variance p(1−p) of the game result"

where `p` is the win probability predicted by the current global fit. That is an
information-maximising schedule: it spends games on pairings whose outcome is
genuinely uncertain and stops burning games on pairings the fit already resolves.
`evals.md` §5.2 currently says "predecessor plus several much older checkpoints",
which is a hand-specified approximation of the same idea. Variance-proportional
sampling is strictly better and is not harder to implement — it needs the fit to be
online, which it has to be anyway.

Scale: ~21 000 games for the main comparison, ~147 000 for the ablation ladder.

**The anchor is external:** "Elo values are versus a mix of various Leela Zero
versions and ELF, anchored so that ELF is about Elo 0." Not a random-init net.

### 3.2 KataGo *does* gate — I had this wrong

> "candidate neural nets must pass a *gating* test by winning at least 100 out of
> 200 test games against the current net to become the new net for self-play"

So the two references we follow disagree: **AZ removed gating, KataGo kept it**, and
KataGo is the one that is explicitly optimising for wall-clock efficiency on a small
budget — which is our situation, not AZ's. That makes the open question in
`evals.md` §11 sharper rather than settled, and it means the eval harness may well be
a C2 dependency.

200 games at a 50 % bar is a much weaker filter than AGZ's 400 at 55 %; it is closer
to "reject clear regressions" than "require improvement".

---

## 4. Leela Chess Zero

Sources: the [lc0 FAQ](https://lczero.org/dev/wiki/faq/),
[project history](https://lczero.org/dev/wiki/project-history/), and the
[Chessprogramming wiki](https://www.chessprogramming.org/Leela_Chess_Zero).

### 4.1 Elo inflation — confirmed from the primary source

lc0's own FAQ, on its training chart:

> "The chart is not calibrated to CCRL or any other common list. It sets 'the first
> net' to Elo 0, so it is not comparable, even between different training runs."

and

> "Self-play tends to exaggerate gains in Elo compared to gains when playing other
> chess engines."

Two things follow for us. The inflation warning in `evals.md` §5.2 is well founded
and comes from the project that lived it. And **lc0 anchors its scale at the first
net = Elo 0**, which is exactly the frozen-random-init anchor of §5.1 — that
proposal now has a precedent rather than being invented here.

### 4.2 "lc0 gated early and dropped it later" — withdraw the claim

I could not source it. What I *did* find points the other way: promotion gating
existed and was being *tuned* (a threshold changed from −50 to −150 in April 2018),
and secondary material describes gating as an ongoing part of Leela Zero's procedure,
including as a defence against value-head overfitting. There is no document I found
saying it was removed.

⚠️ **This claim should come out of `evals.md` §12 rather than be softened.** It is
the kind of half-remembered project-lore that is worse than saying nothing, because
it reads as evidence for dropping gating when it is not.

---

## 5. SAI — the precise source for "do not chain"

Source: Morandin et al., *SAI: a Sensible Artificial Intelligence that plays with
handicap and targets high scores in 9×9 Go*,
[arXiv:1905.10863](https://arxiv.org/abs/1905.10863)
([ar5iv](https://ar5iv.labs.arxiv.org/html/1905.10863)), §3.5 "Elo evaluation".

> "When this is limited to matches between newly promoted networks and their
> predecessor, as in Leela Zero, the resulting estimates are believed to give an Elo
> rating inflation, in particular in combination with gating."

and their fix, credited to CloudyGo:

> "we reverted to the Elo rating, but to get better global estimates of the networks
> strengths, following [CloudyGo], we confronted every network against several
> others, of comparable ability, obtaining a graph of pairings with about 1,000 nodes
> and 13,000 edges."

Their concrete schedule: each promoted network plays generations at offsets
**±1, ±2, ±3, ±6, ±8, ±12**, plus periodic games against a slowly-changing reference
panel and across the two training runs, fitted by maximum likelihood with a
draw-aware Bayesian model.

This is the citation `evals.md` §5.2 should carry — it states the failure mode, names
gating as an aggravating factor, and gives a ready-made pairing schedule. Note the
graph density: **13 000 edges over 1 000 nodes**, i.e. ~13 pairings per network, which
is a useful sizing number for our own league.

Worth flagging: SAI says inflation is *"believed to"* happen. It is the community's
consensus explanation, not a measured result, and nobody in this literature seems to
have isolated it experimentally.

---

## 6. ELF OpenGo

Source: Tian et al., [arXiv:1902.04522](https://arxiv.org/abs/1902.04522).

Evaluation is against a **fixed external opponent** — Leela Zero, the strongest
open-source Go AI at the time — both sides at 50 s per move, final record 980:18
(98.2 %). No self-anchored ladder is used for the headline claim.

The relevance is negative and useful: ELF is the pattern where the whole evaluation
rests on one external opponent, and the resulting number ("98.2 % against LZ") does
not place the engine on any transferable scale. That is what `evals.md` §7's
"basket, not one opponent" is trying to avoid.

---

## 7. CCRL, since §7 of `evals.md` proposes anchoring to it

Source: [CCRL 40/15 about page](https://computerchess.org.uk/4040/about.html).

- **Time control:** 40 moves in 15 minutes, referenced to an Intel i7-4770k, with
  Stockfish 10 used as the benchmark to derive the equivalent control on other
  hardware. Repeating (another 15 min per subsequent 40 moves).
- **Pondering:** off.
- **Tablebases:** 4-, 5- or 6-man permitted.
- **Opening book:** generic books, maximum 12 moves; engines' own books disabled.
- **Hash:** 256 or 512 MB, identical for all engines in a match.
- **Format:** any (match, round-robin, gauntlet, Swiss).

⚠️ Two consequences for us. The time control is **CPU-referenced**, which is
awkward for a GPU engine — the equivalence is defined by a Stockfish benchmark, so
our own hardware normalisation is undefined and would have to be argued. And their
12-move generic book is not UHO or TCEC, so "UHO/TCEC books" and "CCRL conditions"
are not the same protocol; `evals.md` currently implies they compose.

---

## 8. Puzzles

Source: [database.lichess.org](https://database.lichess.org/),
[Lichess/chess-puzzles on Hugging Face](https://huggingface.co/datasets/Lichess/chess-puzzles).

- **6 014 381** puzzles as of the June 2026 export, not "roughly 4M".
- **CC0**, downloadable and redistributable without permission. Updated monthly.
- Each puzzle is rated by treating **every solve attempt as a Glicko-2 game between
  the player and the puzzle**, and the export carries a `RatingDeviation` field.

The rating-deviation field is worth using rather than ignoring: filtering to puzzles
with low deviation gives a much cleaner difficulty axis for the solve-rate curve of
`evals.md` §8.2.

---

## 9. What this changes in `docs/evals.md`

Applied in the same pass as this report:

1. §3 gains an explicit note that **AZ rated at 1 s/move, not fixed simulations**, so
   the fixed-simulation recommendation is ours and not inherited.
2. §3 gains AZ's evaluation-time protocol: greedy on root visit count, no
   temperature, no Dirichlet.
3. §5.1 cites lc0's first-net-at-zero as precedent for the frozen anchor.
4. §5.2 replaces the KataGo attribution with SAI §3.5 for the claim itself, keeps
   KataGo for the global fit, and adopts **variance-proportional opponent sampling**
   over the hand-specified "predecessor plus older" schedule. SAI's ±1,2,3,6,8,12 and
   ~13 pairings per node go in as the fallback if the online fit is not ready.
5. §7 records that AZ gave Stockfish 6-man tablebases, and that CCRL's book is a
   12-move generic one rather than UHO/TCEC.
6. §8.2 corrects 4M → 6.01M and adds the rating-deviation filter.
7. §11's gating row records that **AZ says no and KataGo says yes**, with both sets
   of numbers, instead of implying consensus.
8. §12 becomes a pointer to this file, with the lc0 gating claim withdrawn.

## 10. What is still not verified

- Whether lc0 ever ran without gating. §4.2 — the claim is withdrawn, not resolved.
- Whether Elo inflation from chained pairings has ever been *measured* rather than
  asserted. SAI says "believed to"; I found no experiment.
- Whether the *Science* AZ paper describes its figure-1 rating procedure differently
  from the preprint. I read the preprint's Methods; the Science version is paywalled
  and the match conditions above come from a secondary collation.
- CCRL's exact hardware-normalisation procedure for a GPU engine. Their rule is
  defined by a Stockfish CPU benchmark and I did not find a statement covering
  engines whose strength scales with a GPU.
