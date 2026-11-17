"""
Match prediction model for international football.

Two models, blended:
  - XGBoost classifier  -> outcome probabilities from team/tournament identity
  - Poisson regression  -> goal rates, which give the full scoreline distribution

On a chronological 80/20 split of dataset/results.csv (5,174 held out, 4,860 of
them scorable -- the rest involve a team or competition unseen before the
cutoff). Re-measured 2026-11-17 on a refreshed copy:

    class-prior baseline    acc 0.477   logloss 1.051
    XGBoost alone           acc 0.535   logloss 0.989
    Poisson alone           acc 0.601   logloss 0.877
    geometric pool @ 0.9    acc 0.601   logloss 0.877   <- what we serve

Those are worse than the numbers this file used to quote (0.630 / 0.838), and
the reason is the data, not the model. The previous dataset listed the stronger
side in `home_team`: over the shared window (2000-01-04 to 2026-03-31) it
showed home/draw/away of 0.575/0.236/0.189 where upstream shows
0.481/0.233/0.286 -- a 9-point home/away swing with the draw rate untouched.
The old split was flattering an artifact. These numbers are honest ones
against a harder, correctly-ordered problem.

That shift also moved the blend. XGBoost leant on the artifact through team
identity, and without it drops to barely above the prior; Poisson carries the
model now, and the pool weight went 0.5 -> 0.9. At that weight XGBoost is worth
0.0004 log loss, which would not justify a second model on its own. It stays
because it is the only model that reads `tournament`: drop it and the
competition selector in the UI stops changing the prediction at all.

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
BLEND = 0.9            # weight on Poisson in the pool. See the docstring: 0.5 was
                       # right on the old dataset, 0.9 on this one.
BLEND_FLOOR = 1e-4     # keeps log(0) out of the pool

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
        }),
        pd.DataFrame({
            "team": df["away_team"], "opp": df["home_team"],
            "home": 0.0,
            "goals": df["away_score"], "date": df["date"],
        }),
    ], ignore_index=True)


class Model:
    def __init__(self, matches=None):
        self.matches = load_matches() if matches is None else matches
        self.teams = sorted(set(self.matches["home_team"]) | set(self.matches["away_team"]))
        # Frequency-ordered because only membership matters here -- _validate
        # checks against it. The dropdown gets tournament_groups() instead.
        self.tournaments = self.matches["tournament"].value_counts().index.tolist()
        self._fit_xgb()
        self._fit_poisson()

    # -- fitting ---------------------------------------------------------------

    def _fit_xgb(self):
        m = self.matches
        self._team_enc = LabelEncoder().fit(pd.concat([m["home_team"], m["away_team"]]))
        self._trn_enc = LabelEncoder().fit(m["tournament"])
        self._xgb = XGBClassifier(**XGB_PARAMS).fit(self._xgb_features(m), m["outcome"])

    def _xgb_features(self, df):
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
        })

    def _fit_poisson(self):
        long = _long_form(self.matches)
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
        home = sparse.csr_matrix(long[["home"]].values.astype(float))
        return sparse.hstack([encoded, home]).tocsr()

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
        }]))
        lam = self._poisson.predict(self._poisson_features(rows))
        return float(min(lam[0], LAMBDA_CAP)), float(min(lam[1], LAMBDA_CAP))

    def _xgb_probs(self, home_team, away_team, neutral, tournament):
        row = pd.DataFrame([{
            "home_team": home_team, "away_team": away_team,
            "tournament": tournament, "neutral": neutral,
        }])
        return self._xgb.predict_proba(self._xgb_features(row))[0]

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
