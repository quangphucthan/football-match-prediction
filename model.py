"""
Match prediction model for international football.

Two models, blended:
  - XGBoost classifier  -> outcome probabilities from team/tournament identity
  - Poisson regression  -> goal rates, which give the full scoreline distribution

Neither is good enough alone. On a chronological 80/20 split (test = 5053 matches):

    always-home baseline    acc 0.584   logloss 0.965
    Poisson alone           acc 0.593   logloss 0.887
    XGBoost alone           acc 0.608   logloss 0.880
    50/50 arithmetic blend  acc 0.629   logloss 0.849
    50/50 geometric blend   acc 0.630   logloss 0.838   <- what we serve

The blend beats both parents because their errors are uncorrelated: XGBoost is
sharper on typical fixtures but degrades badly on lopsided or never-played
pairings (it gave San Marino a 16.8% chance of beating Brazil), while Poisson
handles the extremes correctly but is blunter in midfield.

Those numbers are ~0.04 log loss worse than an earlier version that fed XGBoost
an `is_neutral` flag. That flag was dropped on purpose -- see _xgb_features.

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

MATCHES_CSV = "dataset/all_matches.csv"
COUNTRIES_CSV = "dataset/countries_names.csv"

FROM_YEAR = 2000       # soccer pre-2000 is a different game; also where the data gets dense
MAX_GOALS = 10         # scoreline grid is (MAX_GOALS+1)^2; tail beyond this is negligible
HALF_LIFE_DAYS = 2920  # 8y. Tuned on the split above: shorter half-lives scored worse.
BLEND = 0.5            # weight on Poisson in the pool. 0.5 measured best.
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

# Checked against who actually played in them, not guessed from the name: these
# span confederations (French Territory Cup is Martinique to Tahiti, Island
# Games is Bermuda to Greenland, Millenium Cup put Bosnia against India) or are
# invitationals rather than competitions (US Cup).
TOURNAMENT_REGION_OVERRIDES = {
    "French Territory Cup": "World",
    "Island Games": "World",
    "Islamic Games": "World",
    "Millenium Cup": "World",
    "US Cup": "Friendly",
}

# First matching keyword wins, so order matters. The cross-confederation block
# has to precede the continental ones -- "Afro-Asian Games" must not be caught
# by "asian" -- and plain "world cup" has to come last, so that the combined
# qualifiers ("World Cup and African Cup qual", "WC q and Oce Cup") group under
# the confederation that actually plays them.
TOURNAMENT_REGIONS = (
    ("World", ("afro-asian", "afc-ofc", "intercontinental", "confederations", "fifa series")),
    ("Friendly", ("friendly", "independence")),
    ("Europe", ("european", "baltic", "nordic")),
    ("Africa", ("african", "cosafa", "cecafa", "cemac", "comesa", "cabral", "indian ocean")),
    ("Asia", ("asian", "gulf cup", "arab", "king's cup", "nehru", "bangabandhu")),
    ("North & Central America", ("concacaf", "uncaf", "caribbean", "n am ch",
                                 "central american", "windward", "leeward")),
    ("South America", ("copa",)),
    ("Oceania", ("oce cup", "oceania", "pacific", "melanesian", "polynesian",
                 "marianas", "outrigger")),
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


def load_matches(path=MATCHES_CSV, countries=COUNTRIES_CSV):
    """Matches from FROM_YEAR on, with team names normalised to current names."""
    df = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["home_score", "away_score"])
    names = pd.read_csv(countries)
    rename = dict(zip(names["original_name"], names["current_name"]))
    df["home_team"] = df["home_team"].replace(rename)
    df["away_team"] = df["away_team"].replace(rename)

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
        # `is_neutral` is deliberately absent. In this dataset the home_team column
        # on neutral fixtures is systematically the stronger side (home win rate
        # 0.747 on neutral vs 0.508 otherwise), so the flag encodes "home team is
        # better", not "no home advantage". XGBoost latched onto it hard -- Japan v
        # Australia at a neutral venue came out 0.987 home -- which makes the UI's
        # venue toggle move the wrong way. Venue is handled by the Poisson `home`
        # term instead, which is fit only on the non-neutral contrast and is immune.
        # Costs ~0.043 log loss on the backtest; the backtest shares the artifact.
        return pd.DataFrame({
            "home_team": self._team_enc.transform(df["home_team"]),
            "away_team": self._team_enc.transform(df["away_team"]),
            "tournament": self._trn_enc.transform(df["tournament"]),
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
    Also measured better overall -- 0.838 vs 0.849 log loss.
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
    """Fitted model, built once per process (~4s)."""
    return Model()
