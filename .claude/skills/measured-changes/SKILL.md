---
name: measured-changes
description: Working rules for this repo — how to validate a change, and how to write a guard that actually guards. Use before changing model maths in model.py, adding or tuning a feature or constant, adding or editing a check in test_model.py, quoting or updating a measured figure in CLAUDE.md, or after running refresh_data.py.
---

# Measured changes

Every rule below is a bug that shipped or nearly shipped **in this repo**. They are
written concretely because the generic version ("write good tests") prevented none of them.

`CLAUDE.md` holds the settled decisions and the numbers. This file holds the method.

## The bar for a model change

A change is not done when the tests pass. It is done when:

1. it is measured on the chronological 80/20 split, against the current pool;
2. the comparison is **paired, with a confidence interval** — not a bare delta;
3. a check in `test_model.py` fails if the feature is unwired;
4. that check has been **mutation-tested** (rule 7);
5. `CLAUDE.md` carries the measurement and the date.

Calibration from this repo: a feature measuring **+0.0002 ± 0.0011 was refused** as an
accuracy win, and one measuring **+0.0048 ± 0.0028 was accepted**. If a result does not
clear noise, say so and do not ship it.

Several decisions here deliberately score worse than the obvious alternative. Chasing the
metric is the failure mode, not the goal — read "Locked decisions" in `CLAUDE.md` before
changing a feature set, a constant or the blend.

## The eight rules

**1. Assert against the wired object, not just the pure function.**
`check_elo_is_causal` proved `_elo_ratings` causal while the realistic leak — wiring the
*final* rating table into the fit inside `Model.__init__` — passed every check in the
suite. Ask where a careless edit would actually put the bug, and assert there.

**2. Work out what cancels before trusting a numeric assert.**
`check_home_advantage_is_per_team` compared `median(adv)` against `exp(coef_[-1])`. Since
`adv[t] == exp(coef_[-1] + 0.4*coef_team)`, that term cancels — the ratio is `1.000000`
whichever column the index points at. It looked like an index guard and never was. Prefer
comparing against an independently computed quantity.

**3. Derive any threshold on a sampled metric from a bootstrapped noise band.**
`check_backtest_not_regressed` asserted `< 0.86` on a 400-match sample whose 99% range was
0.808–0.946. It passed on `random_state=42`, not on the model, and tripped the moment the
data refreshed. Put the sample size and the band in the docstring.

**4. Order asserts most-specific-first, and confirm the one you added is the one that fires.**
A swapped design-matrix column surfaced as `Bolivia home advantage 2.72 is out of range` —
an unrelated range check firing first, by luck, with ~10% headroom. The precise guard never
ran.

**5. Enforce invariants at public boundaries. A docstring is not a guarantee.**
`_elo_ratings` documented "assumes date-sorted, which `load_matches` guarantees", but
`Model(matches=...)` is public and bypasses `load_matches`. A shuffled frame raised nothing
and silently flipped thousands of pairwise strength orderings.

**6. Every number carries its measurement date — and when a stale figure is justifying a
locked decision, replace the argument, not the number.**
The geometric pool was justified by "XGBoost gives San Marino ~20%", which Elo had cut to
0.59%; `model.py` ended up contradicting itself 35 lines apart. Patching the digit would
have left a correct decision resting on dead evidence. Mark anything not re-run as not
re-run, the way `CLAUDE.md` already does.

**7. Mutation-test every new guard before describing what it does.**
The Elo commit claimed its check prevents "a large, entirely fake improvement". It did not
— that was written from intent, not evidence. Break the thing deliberately, confirm the
assert fires, and quote the failure message in the commit. Doing this caught three real
gaps in one session, plus the surprise in rule 4.

**8. Re-measure after the data moves.**
One `refresh_data.py` run invalidated several recorded figures and relabelled ~420 rows
into seven new competitions. `check_tournament_groups` failing is the cue to classify them
— never to loosen the check.

## The thread

Six of the eight are the same mistake: **a check that looks like it guards something is
worse than no check**, because it buys confidence without providing cover. Before trusting
any guard — including one already in the file — ask what it would take for it to pass while
the thing it names is broken.

## Before merging anything non-trivial

Run an adversarial review. In the session these rules came from, the author believed the
work was finished and had written confident commit messages; a reviewer with no stake in
that reasoning found rules 1, 2 and 4. Brief the reviewer with the locked-decision context
from `CLAUDE.md`, or it will propose reverting deliberate choices to improve a metric.

## Known traps in the current code

- `home_advantage()` reads `self._poisson.coef_[-1]` positionally, so the `home` column
  must stay last in `_poisson_features`. `check_elo_reaches_both_models` pins the column
  count; if you add a column, update that width deliberately rather than loosening it.
- Anything added to `_long_form`'s output must be supplied at **both** call sites
  (`_fit_poisson` and `_lambdas`), or prediction breaks while fitting still works.
- Virtualenv is `.env/`, not `.venv/`. There is no CI, so a green PR means nothing was run.
