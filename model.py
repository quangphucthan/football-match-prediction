"""
Match prediction model for international football.

Two models, blended, both reading an Elo rating built from the same matches:
  - XGBoost classifier  -> outcome probabilities from team/tournament identity
  - Poisson regression  -> goal rates, which give the full scoreline distribution

On a chronological 80/20 split of dataset/results.csv (5,174 held out, 4,860 of
them scorable -- the rest involve a team or competition unseen before the
cutoff). Re-measured 2026-11-17 on a refreshed copy:

                            without Elo        with Elo
    class-prior baseline    0.477 / 1.051      0.477 / 1.051
    XGBoost alone           0.535 / 0.989      0.587 / 0.906
    Poisson alone           0.601 / 0.877      0.601 / 0.874
    geometric pool          0.601 / 0.877      0.603 / 0.872   <- what we serve
                            (at w=0.9)         (at w=0.8)

Elo is worth 0.0048 +/- 0.0028 log loss on the pool (95% CI, paired over the
4,860 scorable matches, t = 3.4, better on 64.4% of them). That is a real win,
unlike the per-team home-advantage term, which measured +0.0002 +/- 0.0011 --
compare the two before citing either.

Those are worse than the numbers this file used to quote (0.630 / 0.838), and
the reason is the data, not the model. The previous dataset listed the stronger
side in `home_team`: over the shared window (2000-01-04 to 2026-03-31) it
showed home/draw/away of 0.575/0.236/0.189 where upstream shows
0.481/0.233/0.286 -- a 9-point home/away swing with the draw rate untouched.
The old split was flattering an artifact. These numbers are honest ones
against a harder, correctly-ordered problem.

The blend weight has moved twice. On the old dataset it was 0.5. After the
migration XGBoost, having only team identity to go on, dropped to barely above
the prior and the weight went to 0.9 -- at which point XGBoost was worth 0.0004
log loss and was kept only because it is the only model that reads
`tournament`, without which the competition selector stops changing anything.

Elo changes that. Giving XGBoost a strength feature takes it from 0.989 to
0.906, and the sweep puts the optimum at 0.75-0.80 (flat: 0.8711 at both, 
against 0.8721 at 0.9). It ships at 0.8, the Poisson-leaning end of that flat
region, because the Poisson is the half that behaves on extreme fixtures.

Elo also fixed the two symptoms that made XGBoost untrustworthy. It used to put
San Marino around 20% against Brazil, which is what the geometric pool exists to
suppress; with a rating to read it says 0.59%. And the neutral-venue toggle,
which ran backwards on some fixtures, now moves the right way on Brazil v France
(46.8% -> 38.7%), Japan v Australia and England v Germany.

Only the Poisson model can produce a scoreline distribution, so the displayed
grid comes from it, rescaled so its win/draw/loss margins match the blend. That
keeps every number on screen derived from one consistent object.
"""

import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

import refresh_data

MATCHES_CSV = "dataset/results.csv"
COUNTRIES_CSV = "dataset/countries_names.csv"

FROM_YEAR = 2000       # soccer pre-2000 is a different game; also where the data gets dense
MAX_GOALS = 10         # scoreline grid is (MAX_GOALS+1)^2; tail beyond this is negligible
HALF_LIFE_DAYS = 2920  # 8y. Tuned on the split above: shorter half-lives scored worse.
BLEND = 0.8            # weight on Poisson in the pool. See the docstring: 0.5 on
                       # the old dataset, 0.9 after the migration, 0.8 once Elo
                       # made XGBoost worth listening to again.
BLEND_FLOOR = 1e-4     # keeps log(0) out of the pool

ELO_START = 1500.0     # every team's rating before its first match in the window
ELO_K = 20.0           # update step. 20 is the usual international-football value
ELO_HOME = 65.0        # home edge, in rating points
ELO_SCALE = 400.0      # points per 10x odds, and the divisor that keeps the
                       # Poisson's elo column on the same scale as its one-hots

# ponytail: multiplicative Poisson explodes when two extremes meet (Brazil vs San
# Marino solved to 10.6 expected goals). Capping costs zero measured log loss --
# no test-set fixture is that lopsided -- and keeps the UI sane. Revisit if a
# team-strength prior replaces the raw one-hot ratings.
LAMBDA_CAP = 6.0

# One-hot level for "this row is not the home side", so the per-team home block
# only ever fires on actual home rows.
NOT_HOME = "(not home)"

# Shrinkage on the per-team home block. PoissonRegressor takes a single alpha
# for every coefficient, so the block is scaled instead: at scale s the same
# effect needs a coefficient 1/s as large, which L2 penalises 1/s^2 harder.
# Measured on the chronological split -- 1.0 gave Tonga a 0.54x home advantage
# off five home matches, 0.4 pulls that to 1.04x and holds Bolivia at 1.97x,
# for identical log loss (0.8361 vs 0.8360). Tuned for sane coefficients, not
# for the metric, which cannot separate them.
HOME_BLOCK_SCALE = 0.4

# Competition dropdown grouping. Friendly first because it is the default and a
# third of the dataset, then World, then the confederations alphabetically.
REGION_ORDER = (
    "Friendly", "World", "Africa", "Asia", "Europe",
    "North & Central America", "Oceania", "South America",
)

# Checked against who actually played in them, not guessed from the name. Two
# kinds land here: competitions that span confederations (Island Games runs
# Bermuda to Greenland, Coupe de l'Outre-Mer spans CONCACAF/CAF/OFC, Millennium
# Cup put Bosnia against India), and host-nation invitationals that are really
# friendlies dressed as tournaments (US Cup, Kirin Cup).
#
# An exact-name override also sidesteps substring collisions a keyword cannot:
# "Nations Cup" is the British Isles four, but "Oceania Nations Cup" and
# "CONCACAF Nations League" are not; "MSG Prime Minister's Cup" is Melanesian
# while a bare "Prime Minister's Cup" ran Bahrain against Kenya.
TOURNAMENT_REGION_OVERRIDES = {
    # cross-confederation
    "French Territory Cup": "World",
    "Island Games": "World",
    "Islamic Games": "World",
    "Inter Games": "World",
    "Millenium Cup": "World",       # local spelling
    "Millennium Cup": "World",      # upstream spelling
    "Coupe de l'Outre-Mer": "World",
    "Unity Cup": "World",
    "Canadian Shield": "World",
    "Corsica Cup": "World",
    "Al Ain International Cup": "World",
    "Prime Minister's Cup": "World",
    "ELF Cup": "World",
    "Tynwald Hill Tournament": "World",
    "International Tournament of Peoples, Cultures and Tribes": "World",
    # A sponsor badge and a generic name, each stuck on unrelated events in
    # different confederations -- not one host's series. LG Cup ran across nine
    # host nations (Egypt to Russia) with 31 teams from five confederations;
    # Four Nations Tournament is two separate events, China's and Zambia's.
    "LG Cup": "World",
    "Four Nations Tournament": "World",
    # invitationals
    "US Cup": "Friendly",
    "USA Cup": "Friendly",
    "Kirin Cup": "Friendly",
    "Kirin Challenge Cup": "Friendly",
    "China Cup": "Friendly",        # the Kirin pattern: one host, guests from everywhere
    # regional, but named in a way no keyword catches
    "Nations Cup": "Europe",
    "Cyprus International Tournament": "Europe",
    "Malta International Tournament": "Europe",
    "Jordan International Tournament": "Asia",
    "Merdeka Tournament": "Asia",
    "Lunar New Year Cup": "Asia",
    "Dragon Cup": "Asia",
    "VFF Cup": "Asia",
    "Tri Nation Tournament": "Asia",
    "Tri-Nation Series": "Asia",            # India's, unlike the Fijian "Tri-Nations Series"
    "FAS Tri-Nations Series": "Asia",       # Singapore, Malaysia, Philippines
    "CTFA International Tournament": "Asia",
    "Fuchs International Tournament": "Asia",
    "Navruz Cup": "Asia",
    "OSN Cup": "Asia",
    "Mahinda Rajapaksa Cup": "Asia",
    "King Hassan II Tournament": "Africa",
    "Mauritius Four Nations Cup": "Africa",
    "Nile Basin Tournament": "Africa",
    "Mapinduzi Cup": "Africa",
    "SKN Football Festival": "North & Central America",
    "ABCS Tournament": "North & Central America",
    # one-off invitationals, classified from the team sheet rather than the name
    "Cup of Ancient Civilizations": "World",        # China, Iran, Egypt
    "The Other Final": "World",                     # Bhutan v Montserrat
    "World Unity Cup": "World",                     # Barawa, Chagos Islands, Tamil Eelam
    "Niamh Challenge Cup": "World",                 # Ellan Vannin to Panjab
    "Atlantic Heritage Cup": "World",               # Jersey parishes, Yorkshire, Chagos
    "ConIFA Challenger Cup": "World",               # lowercase n, so no keyword catches it
    "Hungary Heritage Cup": "Europe",               # Hungarian-diaspora sides
    "KTFF 50th Anniversary Cup": "Europe",          # Northern Cyprus, Sapmi, Kosovo
    "Benedikt Fontana Cup": "Europe",               # Raetia v Franconia
    "Diamond Jubilee International Football Tournament": "Asia",
    "International Friendship Championship": "Asia",  # Iraq, Qatar, Syria
    "Three Nations Cup": "Asia",                    # Kyrgyzstan, Bangladesh, Palestine
    "Tri-Nations Cup": "Asia",                      # Myanmar, Philippines, Guam
    "Mukuru 4 Nations": "Africa",                   # Botswana, Zimbabwe, Malawi
    "TIFOCO Tournament": "Africa",                  # Benin v Niger
    "Soccer Ashes": "Oceania",                      # the Australia-New Zealand rivalry
    "Tri-Nations Series": "Oceania",                # Fiji, Solomon Islands, Hong Kong
}

# First matching keyword wins, so order matters. The cross-confederation block
# has to precede the continental ones -- "Afro-Asian Games" must not be caught
# by "asian" -- and plain "world cup" has to come last, so that the combined
# qualifiers ("World Cup and African Cup qual", "WC q and Oce Cup") group under
# the confederation that actually plays them.
#
# Upstream names competitions by confederation acronym where the older local
# dataset spelled them out, so both spellings are matched: "UEFA Euro" and
# "European Championship", "Gold Cup" and "CONCACAF".
TOURNAMENT_REGIONS = (
    ("World", ("afro-asian", "afc-ofc", "intercontinental", "confederations",
               "fifa series", "conifa world", "viva world", "fifi")),
    ("Friendly", ("friendly", "independence")),
    ("Europe", ("uefa", "european", "baltic", "nordic", "muratti")),
    ("Africa", ("africa", "cosafa", "cecafa", "cemac", "comesa", "cabral",
                "indian ocean")),
    ("Asia", ("asia", "afc ", "aff ", "saff", "eaff", "waff", "cafa", "asean",
              "gulf cup", "arab", "king's cup", "nehru", "bangabandhu",
              "philippine", "palestine international")),
    ("North & Central America", ("concacaf", "uncaf", "caribbean", "n am ch",
                                 "central american", "windward", "leeward",
                                 "gold cup")),
    ("South America", ("copa", "superclásico", "superclasico", "south america")),
    ("Oceania", ("oce cup", "oceania", "pacific", "melanesia", "polynesian",
                 "marianas", "outrigger", "msg ")),
    ("World", ("world cup", "wc ", "w cup")),
)


def _alpha_key(name):
    """Accent-blind sort key, so 'Copa América' files beside 'Copa America'."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()


def tournament_region(name):
    """Which dropdown group a competition belongs in."""
    if name in TOURNAMENT_REGION_OVERRIDES:
        return TOURNAMENT_REGION_OVERRIDES[name]
    lowered = name.lower()
    for region, keywords in TOURNAMENT_REGIONS:
        if any(k in lowered for k in keywords):
            return region
    return "Other"


XGB_PARAMS = dict(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="mlogloss",
)

# XGBoost class order, fixed by the outcome encoding below.
AWAY, DRAW, HOME = 0, 1, 2


def load_matches(path=MATCHES_CSV):
    """Matches from FROM_YEAR on.

    No name normalisation step: upstream already writes every row under the
    team's current name (its 1882 fixtures say "Northern Ireland", not
    "Ireland"), so the old original->current map from countries_names.csv was
    measured to rewrite 0 of 223 matching names and was deleted. Genuinely
    distinct historical nations -- German DR, Vietnam Republic -- stay separate
    teams, which is correct. countries_names.csv is now only a colour lookup.
    """
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    df = df[df["date"].dt.year >= FROM_YEAR].sort_values("date").reset_index(drop=True)
    df["outcome"] = np.where(
        df["home_score"] > df["away_score"], HOME,
        np.where(df["home_score"] == df["away_score"], DRAW, AWAY),
    )
    return df


def _elo_ratings(matches):
    """Pre-match Elo for every row, plus the final table.

    Causal by construction: row i records the ratings as they stood *before*
    match i, so a model fit on these columns cannot see the result that moved
    them. That is the whole reason this is a sequential loop and not a groupby
    -- it is the one feature here that must not be computed from the future.

    Assumes `matches` is sorted by date, which load_matches guarantees.
    """
    rating = {}
    pre_home = np.empty(len(matches))
    pre_away = np.empty(len(matches))
    rows = zip(matches["home_team"], matches["away_team"],
               matches["outcome"], matches["neutral"])
    for i, (home, away, outcome, neutral) in enumerate(rows):
        r_home = rating.get(home, ELO_START)
        r_away = rating.get(away, ELO_START)
        pre_home[i], pre_away[i] = r_home, r_away

        edge = 0.0 if neutral else ELO_HOME
        expected = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - edge) / ELO_SCALE))
        actual = 1.0 if outcome == HOME else 0.5 if outcome == DRAW else 0.0
        delta = ELO_K * (actual - expected)
        rating[home] = r_home + delta
        rating[away] = r_away - delta
    return pre_home, pre_away, rating


def _long_form(df):
    """One row per team per match: what this team scored, against whom, at home or not.

    Fitting a single Poisson on this gives every team an attack rating (as `team`)
    and a defence rating (as `opp`) from one regression.
    """
    return pd.concat([
        pd.DataFrame({
            "team": df["home_team"], "opp": df["away_team"],
            "home": np.where(df["neutral"], 0.0, 1.0),
            "goals": df["home_score"], "date": df["date"],
            "rating": df["home_elo"], "opp_rating": df["away_elo"],
        }),
        pd.DataFrame({
            "team": df["away_team"], "opp": df["home_team"],
            "home": 0.0,
            "goals": df["away_score"], "date": df["date"],
            "rating": df["away_elo"], "opp_rating": df["home_elo"],
        }),
    ], ignore_index=True)


class Model:
    def __init__(self, matches=None):
        self.matches = load_matches() if matches is None else matches
        self.teams = sorted(set(self.matches["home_team"]) | set(self.matches["away_team"]))
        # Frequency-ordered because only membership matters here -- _validate
        # checks against it. The dropdown gets tournament_groups() instead.
        self.tournaments = self.matches["tournament"].value_counts().index.tolist()
        self._elo_home, self._elo_away, self._elo = _elo_ratings(self.matches)
        self._fit_xgb()
        self._fit_poisson()

    # -- fitting ---------------------------------------------------------------

    def _fit_xgb(self):
        m = self.matches
        self._team_enc = LabelEncoder().fit(pd.concat([m["home_team"], m["away_team"]]))
        self._trn_enc = LabelEncoder().fit(m["tournament"])
        self._xgb = XGBClassifier(**XGB_PARAMS).fit(
            self._xgb_features(m, self._elo_home - self._elo_away), m["outcome"]
        )

    def _xgb_features(self, df, elo_diff):
        # `is_neutral` is still deliberately absent, but the reason changed with
        # the dataset. On the old file the flag was a strength proxy -- neutral
        # fixtures had a 0.747 home win rate against 0.508 otherwise, because the
        # stronger side was listed at home -- and XGBoost latched on hard enough
        # to put Japan at 0.987 against Australia on neutral ground. Upstream does
        # not have that defect: its neutral home rate is 0.416 against 0.508
        # non-neutral, which is what real home advantage looks like.
        #
        # Re-measured on upstream anyway: adding the flag moves the pooled log
        # loss from 0.8969 to 0.8968, and the venue toggle still runs backwards on
        # some fixtures (Brazil v France rises from 0.391 to 0.416 on neutral). It
        # is no longer harmful, just worthless. Venue stays the Poisson `home`
        # term's job.
        # np.asarray is for the type checker, not the runtime: LabelEncoder carries
        # no annotations, so the stubs widen transform() to numpy's ArrayLike, which
        # pandas will not take. It is already an ndarray.
        return pd.DataFrame({
            "home_team": np.asarray(self._team_enc.transform(df["home_team"])),
            "away_team": np.asarray(self._team_enc.transform(df["away_team"])),
            "tournament": np.asarray(self._trn_enc.transform(df["tournament"])),
            "is_friendly": (df["tournament"] == "Friendly").astype(int).values,
            "elo_diff": elo_diff,
        })

    def _fit_poisson(self):
        long = _long_form(self.matches.assign(
            home_elo=self._elo_home, away_elo=self._elo_away
        ))
        self._ohe = OneHotEncoder(handle_unknown="ignore")
        X = self._poisson_features(long, fit=True)
        # Recent matches count for more; exponential decay on match age.
        age_days = (self.matches["date"].max() - long["date"]).dt.days.values
        weights = np.exp(-np.log(2) / HALF_LIFE_DAYS * age_days)
        self._poisson = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(
            X, long["goals"].values, sample_weight=weights
        )

    def _poisson_features(self, long, fit=False):
        # Home advantage is per team, not one number for everybody: altitude,
        # travel and crowd are not shared evenly. The plain `home` column keeps
        # carrying the average effect and the one-hot `home_team` block carries
        # each side's deviation from it, so L2 pulls a team with few home
        # matches back toward the global term instead of inventing an edge for
        # it. Sides that are never the home team collapse into NOT_HOME.
        cats = pd.DataFrame({
            "team": long["team"],
            "opp": long["opp"],
            "home_team": np.where(long["home"] > 0, long["team"], NOT_HOME),
        })
        encoded = self._ohe.fit_transform(cats) if fit else self._ohe.transform(cats)
        # Shrink the home_team block, which sits after the team and opp blocks.
        split = len(self._ohe.categories_[0]) + len(self._ohe.categories_[1])
        encoded = encoded.tocsc()
        encoded = sparse.hstack([encoded[:, :split], encoded[:, split:] * HOME_BLOCK_SCALE])
        # Divided by ELO_SCALE so the column lands on roughly the same scale as
        # the one-hots: PoissonRegressor applies one alpha to every coefficient,
        # and a raw +/-400 diff would be penalised into irrelevance.
        elo = (long["rating"] - long["opp_rating"]).values / ELO_SCALE
        elo = sparse.csr_matrix(elo.reshape(-1, 1))
        # `home` stays last: home_advantage() reads coef_[-1].
        home = sparse.csr_matrix(long[["home"]].values.astype(float))
        return sparse.hstack([encoded, elo, home]).tocsr()

    def elo(self, team):
        """Current rating: where the team stood after the last match in the window."""
        return self._elo.get(team, ELO_START)

    def home_advantage(self, team):
        """What playing at home multiplies this team's expected goals by.

        The global term times the team's own deviation. Bolivia comes out near
        2x (altitude); most sides land between 1.0 and 1.3.
        """
        names = self._ohe.get_feature_names_out(["team", "opp", "home_team"])
        idx = {n: i for i, n in enumerate(names)}
        key = f"home_team_{team}"
        own = HOME_BLOCK_SCALE * self._poisson.coef_[idx[key]] if key in idx else 0.0
        return float(np.exp(self._poisson.coef_[-1] + own))

    # -- prediction ------------------------------------------------------------

    def _lambdas(self, home_team, away_team, neutral):
        """Expected goals for each side, capped."""
        rows = _long_form(pd.DataFrame([{
            "home_team": home_team, "away_team": away_team, "neutral": neutral,
            "home_score": 0, "away_score": 0, "date": self.matches["date"].max(),
            "home_elo": self.elo(home_team), "away_elo": self.elo(away_team),
        }]))
        lam = self._poisson.predict(self._poisson_features(rows))
        return float(min(lam[0], LAMBDA_CAP)), float(min(lam[1], LAMBDA_CAP))

    def _xgb_probs(self, home_team, away_team, neutral, tournament):
        row = pd.DataFrame([{
            "home_team": home_team, "away_team": away_team,
            "tournament": tournament, "neutral": neutral,
        }])
        diff = np.array([self.elo(home_team) - self.elo(away_team)])
        return self._xgb.predict_proba(self._xgb_features(row, diff))[0]

    def predict(self, home_team, away_team, neutral=False, tournament="Friendly"):
        self._validate(home_team, away_team, tournament)

        lam_home, lam_away = self._lambdas(home_team, away_team, neutral)
        grid = self._score_grid(lam_home, lam_away)
        p_poisson = _outcome_from_grid(grid)
        p_xgb = self._xgb_probs(home_team, away_team, neutral, tournament)
        blended = _pool(p_poisson, p_xgb)

        # Pull the scoreline grid onto the blended margins so the correct-score
        # panel and the headline probabilities cannot contradict each other.
        grid = _rescale_grid(grid, p_poisson, blended)

        goals = np.arange(MAX_GOALS + 1)
        return {
            "home": home_team,
            "away": away_team,
            "neutral": bool(neutral),
            "tournament": tournament,
            "outcome": {
                "home": round(float(blended[HOME]), 4),
                "draw": round(float(blended[DRAW]), 4),
                "away": round(float(blended[AWAY]), 4),
            },
            "expected_goals": {
                "home": round(float((grid.sum(axis=1) * goals).sum()), 2),
                "away": round(float((grid.sum(axis=0) * goals).sum()), 2),
            },
            "scorelines": _top_scorelines(grid),
            "markets": _markets(grid),
            # Full grid, not a 0-5 window: lopsided fixtures put their mode well
            # outside it (Brazil v San Marino peaks at 6-0). The client picks a
            # display window. 121 floats is not worth trimming.
            "grid": [[round(float(p), 5) for p in row] for row in grid],
            "h2h": self.h2h(home_team, away_team),
            "form": {"home": self.form(home_team), "away": self.form(away_team)},
        }

    def _validate(self, home_team, away_team, tournament):
        unknown = {home_team, away_team} - set(self.teams)
        if unknown:
            raise ValueError(f"unknown team(s): {', '.join(sorted(unknown))}")
        if home_team == away_team:
            raise ValueError("a team cannot play itself")
        if tournament not in self.tournaments:
            raise ValueError(f"unknown tournament: {tournament}")

    def _score_grid(self, lam_home, lam_away):
        goals = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
        return grid / grid.sum()  # renormalise the truncated tail

    # -- history (no model involved) -------------------------------------------

    def h2h(self, team_a, team_b, recent=5):
        m = self.matches
        played = m[
            ((m["home_team"] == team_a) & (m["away_team"] == team_b))
            | ((m["home_team"] == team_b) & (m["away_team"] == team_a))
        ]
        # Results are reported from team_a's perspective.
        a_scored = np.where(played["home_team"] == team_a, played["home_score"], played["away_score"])
        b_scored = np.where(played["home_team"] == team_a, played["away_score"], played["home_score"])
        return {
            "played": int(len(played)),
            "home_wins": int((a_scored > b_scored).sum()),
            "draws": int((a_scored == b_scored).sum()),
            "away_wins": int((a_scored < b_scored).sum()),
            "recent": [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "home": row["home_team"], "away": row["away_team"],
                    "score": f"{int(row['home_score'])}-{int(row['away_score'])}",
                    "neutral": bool(row["neutral"]),
                }
                for _, row in played.tail(recent).iloc[::-1].iterrows()
            ],
        }

    def form(self, team, last=5):
        """Most recent results first, as W/D/L from `team`'s point of view."""
        m = self.matches
        played = m[(m["home_team"] == team) | (m["away_team"] == team)].tail(last)
        scored = np.where(played["home_team"] == team, played["home_score"], played["away_score"])
        conceded = np.where(played["home_team"] == team, played["away_score"], played["home_score"])
        return [
            "W" if s > c else ("D" if s == c else "L")
            for s, c in zip(scored[::-1], conceded[::-1])
        ]

    def tournament_groups(self):
        """Tournaments as <optgroup>s: regions in REGION_ORDER, names A-Z inside.

        The internal `tournaments` list stays frequency-ordered because
        `_validate` only needs membership, but a 106-entry dropdown is only
        navigable grouped and sorted.
        """
        groups = {}
        for name in self.tournaments:
            groups.setdefault(tournament_region(name), []).append(name)
        rank = {region: i for i, region in enumerate(REGION_ORDER)}
        return [
            {"region": region, "tournaments": sorted(names, key=_alpha_key)}
            for region, names in sorted(
                groups.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])
            )
        ]

    def team_list(self, countries=COUNTRIES_CSV):
        """Teams with a colour swatch and match count, for the picker."""
        colours = pd.read_csv(countries).set_index("current_name")["color_code"].to_dict()
        counts = pd.concat([self.matches["home_team"], self.matches["away_team"]]).value_counts()
        return [
            {"name": t, "color": colours.get(t, "#888888"), "played": int(counts.get(t, 0))}
            for t in self.teams
        ]


def _pool(p_poisson, p_xgb, weight=BLEND):
    """Geometric blend (log-opinion pool) of two probability vectors.

    Chosen over an arithmetic mean because a near-zero from either model survives
    the pool instead of being averaged away. That is what stops XGBoost's poor
    behaviour on never-played fixtures from leaking through: it rates San Marino
    at ~20% to beat Brazil at home, which an arithmetic mean would carry to 10%.
    Still the better of the two on upstream at the shipped weight -- 0.8777
    geometric against 0.8772 arithmetic is a wash on log loss, but the arithmetic
    mean reintroduces the extreme-fixture leak the geometric pool suppresses.
    """
    logs = (
        weight * np.log(np.clip(p_poisson, BLEND_FLOOR, 1.0))
        + (1 - weight) * np.log(np.clip(p_xgb, BLEND_FLOOR, 1.0))
    )
    pooled = np.exp(logs)
    return pooled / pooled.sum()


def _outcome_from_grid(grid):
    """[away, draw, home] probabilities. Row index = home goals, column = away goals."""
    rows, cols = np.triu_indices(grid.shape[0], k=1)  # rows < cols -> home scored fewer
    return np.array([grid[rows, cols].sum(), np.trace(grid), grid[cols, rows].sum()])


def _rescale_grid(grid, p_from, p_to):
    """Reweight each of the three outcome regions so the grid's margins become p_to."""
    scale = p_to / np.maximum(p_from, 1e-12)
    i, j = np.indices(grid.shape)
    out = grid * np.select([i > j, i == j], [scale[HOME], scale[DRAW]], default=scale[AWAY])
    return out / out.sum()


def _top_scorelines(grid, n=6):
    flat = grid.ravel()
    top = np.argsort(flat)[::-1][:n]
    return [
        {"score": f"{i}-{j}", "p": round(float(flat[k]), 4)}
        for k, (i, j) in ((k, divmod(k, grid.shape[1])) for k in top)
    ]


def _markets(grid):
    i, j = np.indices(grid.shape)
    total = i + j
    return {
        "over_1_5": round(float(grid[total > 1].sum()), 4),
        "over_2_5": round(float(grid[total > 2].sum()), 4),
        "over_3_5": round(float(grid[total > 3].sum()), 4),
        "btts": round(float(grid[1:, 1:].sum()), 4),
        "home_clean_sheet": round(float(grid[:, 0].sum()), 4),
        "away_clean_sheet": round(float(grid[0, :].sum()), 4),
    }


@lru_cache(maxsize=1)
def get_model():
    """Fitted model, built once per process (~4s).

    Refreshes the dataset first, because this is the only point where a refresh
    can change anything -- the fit is cached for the process lifetime, so a
    newer CSV written afterwards would not be seen until a restart.
    """
    refresh_data.refresh()
    return Model()
