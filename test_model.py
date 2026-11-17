"""
Sanity checks for model.py. Run directly: `python test_model.py`.

Deliberately assert-based with no test framework -- the point is one runnable
thing that fails loudly if the prediction maths breaks.
"""

import numpy as np
import pandas as pd

from model import (
    AWAY, DRAW, ELO_START, HOME, MAX_GOALS, Model, _alpha_key, _elo_ratings, _markets,
    _outcome_from_grid, _rescale_grid, load_matches,
)


def check_probabilities_are_coherent(m):
    p = m.predict("England", "Germany")
    o = p["outcome"]
    total = o["home"] + o["draw"] + o["away"]
    assert abs(total - 1) < 1e-3, f"outcome must sum to 1, got {total}"
    assert all(0 <= v <= 1 for v in o.values()), o

    mk = p["markets"]
    assert mk["over_1_5"] >= mk["over_2_5"] >= mk["over_3_5"], f"over/under not monotonic: {mk}"
    assert 0 <= mk["btts"] <= 1 and 0 <= mk["home_clean_sheet"] <= 1, mk

    scorelines = p["scorelines"]
    probs = [s["p"] for s in scorelines]
    assert probs == sorted(probs, reverse=True), "scorelines must be ranked"
    assert sum(probs) < 1.001, "top scorelines cannot exceed total probability"


def check_grid_matches_headline(m):
    """The rescale exists so these two cannot disagree. Verify they don't."""
    for home, away in [("Brazil", "France"), ("England", "Germany"), ("Japan", "Australia")]:
        p = m.predict(home, away)
        lam_h, lam_a = m._lambdas(home, away, False)
        grid = _rescale_grid(
            m._score_grid(lam_h, lam_a),
            _outcome_from_grid(m._score_grid(lam_h, lam_a)),
            np.array([p["outcome"]["away"], p["outcome"]["draw"], p["outcome"]["home"]]),
        )
        assert abs(grid.sum() - 1) < 1e-9, grid.sum()
        derived = _outcome_from_grid(grid)
        assert abs(derived[2] - p["outcome"]["home"]) < 1e-3, (
            f"{home} v {away}: grid says home {derived[2]:.4f}, headline says {p['outcome']['home']}"
        )


def check_expected_goals_are_plausible(m):
    """The uncapped Poisson solved Brazil v San Marino to 10.6 goals. Guard that."""
    for home, away in [("Brazil", "San Marino"), ("Brazil", "France"), ("San Marino", "Brazil")]:
        xg = m.predict(home, away)["expected_goals"]
        for side, v in xg.items():
            assert 0.0 <= v <= 6.5, f"{home} v {away}: implausible {side} xG {v}"


def check_lopsided_fixtures(m):
    """XGBoost alone gave San Marino a 16.8% chance of beating Brazil. The blend must not."""
    strong_home = m.predict("Brazil", "San Marino")["outcome"]
    assert strong_home["home"] > 0.85, f"Brazil at home vs San Marino: {strong_home}"

    strong_away = m.predict("San Marino", "Brazil")["outcome"]
    assert strong_away["away"] > 0.80, f"San Marino at home vs Brazil: {strong_away}"
    assert strong_away["home"] < 0.10, f"San Marino should not be favoured: {strong_away}"


def check_home_advantage(m):
    at_home = m.predict("Japan", "Australia", neutral=False)["outcome"]["home"]
    on_neutral = m.predict("Japan", "Australia", neutral=True)["outcome"]["home"]
    assert at_home > on_neutral, f"home venue must help: {at_home} vs neutral {on_neutral}"


def check_home_advantage_is_per_team(m):
    """Guards HOME_BLOCK_SCALE: unshrunk, teams with five home matches went wild.

    At scale 1.0 Tonga came out at 0.54x -- i.e. actively worse at home -- off
    five home matches, and the block spanned 0.54-2.31. Shrinkage is what keeps
    the range defensible while leaving Bolivia's altitude edge intact.

    The 0.8 floor is deliberately below the current worst (Samoa 0.92, twenty-nine
    home matches). Being mildly worse at home is noise in a 322-team tail;
    0.54 was not.
    """
    adv = {t: m.home_advantage(t) for t in m.teams}
    values = np.array(list(adv.values()))
    globally = float(np.exp(m._poisson.coef_[-1]))

    for team, v in adv.items():
        assert 0.8 <= v <= 2.5, f"{team} home advantage {v:.2f} is out of range"

    # Shrinkage pulls the typical team onto the global term. If this drifts, the
    # per-team block has started explaining things the global one should.
    assert abs(np.median(values) - globally) < 0.05 * globally, (
        f"median {np.median(values):.2f} should sit on the global term {globally:.2f}"
    )

    # Altitude is the effect most obviously real, so it is the one to insist on.
    assert adv["Bolivia"] > 1.5, f"altitude should show up: Bolivia {adv['Bolivia']:.2f}"
    assert adv["Bolivia"] >= values.max(), f"Bolivia should top the table, got {adv['Bolivia']:.2f}"


def check_tournament_groups(m):
    """Every competition must land in a region, sorted, exactly once.

    Fails when the dataset gains a tournament no keyword matches -- that is the
    signal to classify it rather than let it fall into an "Other" bucket.
    """
    groups = m.tournament_groups()
    grouped = [t for g in groups for t in g["tournaments"]]

    assert sorted(grouped) == sorted(m.tournaments), "grouping must not add or drop tournaments"
    assert len(grouped) == len(set(grouped)), "a tournament cannot be in two regions"

    unclassified = [g for g in groups if g["region"] == "Other"]
    assert not unclassified, f"unclassified tournaments: {unclassified}"

    for g in groups:
        assert g["tournaments"] == sorted(g["tournaments"], key=_alpha_key), (
            f"{g['region']} is not alphabetical: {g['tournaments'][:5]}"
        )

    assert groups[0]["region"] == "Friendly", "Friendly is the default, so it goes first"

    region = {t: g["region"] for g in groups for t in g["tournaments"]}
    assert region["FIFA World Cup"] == "World", region["FIFA World Cup"]
    assert region["Copa América"] == "South America", region["Copa América"]

    # Upstream names competitions by acronym, so neither of these carries its
    # confederation in the text the old keywords looked for.
    assert region["UEFA Euro qualification"] == "Europe", region["UEFA Euro qualification"]
    assert region["Gold Cup"] == "North & Central America", region["Gold Cup"]

    # Order-sensitivity: cross-confederation entries are matched before the
    # continental keywords that would otherwise swallow them.
    assert region["Afro-Asian Games"] == "World", "cross-confederation beats the 'asia' keyword"
    assert region["CONIFA World Football Cup"] == "World"
    assert region["CONIFA Asia Cup"] == "Asia", "continental CONIFA events follow their continent"

    # Substring collisions an exact-name override exists to prevent.
    assert region["Nations Cup"] == "Europe", "the British Isles four, not a confederation event"
    assert region["Oceania Nations Cup"] == "Oceania"
    assert region["Prime Minister's Cup"] == "World", "Bahrain to Kenya"
    assert region["MSG Prime Minister's Cup"] == "Oceania", "Melanesian Spearhead Group"
    assert region["Tri-Nation Series"] == "Asia", "India's; one hyphen and an s apart from Fiji's"
    assert region["Tri-Nations Series"] == "Oceania"


def check_input_validation(m):
    for args, expected in [
        (("Atlantis", "France"), "unknown team"),
        (("France", "France"), "cannot play itself"),
    ]:
        try:
            m.predict(*args)
        except ValueError as e:
            assert expected in str(e), f"wrong error for {args}: {e}"
        else:
            raise AssertionError(f"{args} should have raised")

    try:
        m.predict("France", "Brazil", tournament="Kickabout")
    except ValueError as e:
        assert "unknown tournament" in str(e)
    else:
        raise AssertionError("bad tournament should have raised")


def check_history(m):
    h = m.h2h("England", "Germany")
    assert h["played"] == h["home_wins"] + h["draws"] + h["away_wins"], h
    assert len(h["recent"]) <= 5

    form = m.form("Brazil")
    assert len(form) == 5 and set(form) <= {"W", "D", "L"}, form

    teams = m.team_list()
    assert len(teams) > 200 and all(t["color"].startswith("#") for t in teams)


def check_elo_is_causal(m):
    """The one feature here that must not see the future.

    `_elo_ratings` returns row i's ratings as they stood *before* match i. If
    that ever slipped to post-match, every backtest would improve and the
    improvement would be entirely fake, so this is worth asserting directly
    rather than inferring from a log loss that got better.
    """
    beat = pd.DataFrame({
        "home_team": ["A"] * 4, "away_team": ["B"] * 4,
        "outcome": [HOME] * 4, "neutral": [False] * 4,
    })
    pre_home, pre_away, final = _elo_ratings(beat)

    assert pre_home[0] == ELO_START and pre_away[0] == ELO_START, (
        f"the first match must be seen on fresh ratings, got {pre_home[0]}/{pre_away[0]}"
    )
    assert (np.diff(pre_home) > 0).all(), "A wins every time, so its pre-match rating must climb"
    assert pre_home[-1] < final["A"], "the last result must still move the final table"
    assert abs((final["A"] - ELO_START) + (final["B"] - ELO_START)) < 1e-9, (
        "Elo is zero-sum: what A gains, B loses"
    )

    # A draw between equals moves nobody, which is the sign convention holding.
    level = pd.DataFrame({
        "home_team": ["C"], "away_team": ["D"], "outcome": [DRAW], "neutral": [True],
    })
    _, _, drawn = _elo_ratings(level)
    assert abs(drawn["C"] - ELO_START) < 1e-9, f"a neutral draw between equals moved C to {drawn['C']}"

    # And on the real table, the ordering has to be sane.
    assert m.elo("Brazil") > m.elo("San Marino") + 400, (
        f"Brazil {m.elo('Brazil'):.0f} vs San Marino {m.elo('San Marino'):.0f}"
    )
    assert m.elo("Nowhere Land") == ELO_START, "an unseen team starts at the default"


def check_markets_math():
    """Hand-checkable: all mass on 2-1 means over 2.5, BTTS, no clean sheets."""
    grid = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    grid[2, 1] = 1.0
    mk = _markets(grid)
    assert mk["over_1_5"] == 1.0 and mk["over_2_5"] == 1.0 and mk["over_3_5"] == 0.0, mk
    assert mk["btts"] == 1.0 and mk["home_clean_sheet"] == 0.0, mk

    assert _outcome_from_grid(grid).tolist() == [0.0, 0.0, 1.0], "2-1 is a home win"


def check_backtest_not_regressed():
    """Refit on the first 80% and score a sample of the rest. Catches gross regression.

    Measured on the full held-out set (4,860 scored of 5,174) with Elo in both
    models: pool 0.8711, Poisson alone 0.8741, XGBoost alone 0.9064.

    This docstring used to claim the threshold sat below both parents, so that
    dropping either model would fail. It cannot: bootstrapped over the full set,
    a 1,000-match subsample of the pool spans 0.8270-0.9143 at 99% while
    XGBoost alone spans 0.8552-0.9572, and those overlap. 0.93 is therefore a
    gross-regression guard, not a blend guard -- it clears the pool's own noise
    by 0.016 and still fails a model that has genuinely broken. The separation
    job belongs to check_elo_is_causal and check_lopsided_fixtures, which test
    behaviour rather than an aggregate.

    1,000 matches rather than all 5,174 keeps this check at ~15s. The 400-match
    sample used before ranged 0.808-0.946 against a 0.86 threshold -- it passed
    in August on the seed, not on the model, and tripped on the next refresh.
    """
    from sklearn.metrics import accuracy_score, log_loss

    matches = load_matches()
    cutoff = matches["date"].quantile(0.8)
    train = matches[matches["date"] <= cutoff]
    test = matches[matches["date"] > cutoff].sample(1000, random_state=42)

    fitted = Model(matches=train.reset_index(drop=True))
    probs, actual = [], []
    for _, row in test.iterrows():
        try:
            p = fitted.predict(row["home_team"], row["away_team"], row["neutral"], row["tournament"])
        except ValueError:
            continue  # team or tournament unseen before the cutoff
        probs.append([p["outcome"]["away"], p["outcome"]["draw"], p["outcome"]["home"]])
        actual.append(row["outcome"])

    ll = log_loss(actual, probs, labels=[0, 1, 2])
    acc = accuracy_score(actual, np.argmax(probs, axis=1))
    print(f"  backtest on {len(actual)} held-out matches: logloss {ll:.4f}  acc {acc:.4f}")
    assert ll < 0.93, f"blend log loss regressed to {ll:.4f}"


def main():
    print("fitting model...")
    m = Model()
    checks = [
        check_probabilities_are_coherent, check_grid_matches_headline,
        check_expected_goals_are_plausible, check_lopsided_fixtures,
        check_home_advantage, check_home_advantage_is_per_team,
        check_tournament_groups, check_input_validation, check_history,
        check_elo_is_causal,
    ]
    for fn in checks:
        fn(m)
        print(f"  ok  {fn.__name__}")
    for fn in [check_markets_math, check_backtest_not_regressed]:
        fn()
        print(f"  ok  {fn.__name__}")
    print("all checks passed")


if __name__ == "__main__":
    main()
