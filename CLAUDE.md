# Football Match Prediction

Predict international football fixtures: pick two national teams, get outcome
probabilities, expected goals and a scoreline distribution.

## Agreed plan

Full plan: `~/.claude/plans/let-s-develop-a-front-end-streamed-spark.md`.
The decisions below are settled — follow them rather than re-deriving.

**Done**
- `model.py` — the served model. `get_model().predict(home, away, neutral, tournament)` returns the whole API payload.
- `test_model.py` — assert-based checks, no framework. Run before touching model maths.
- `api.py` — FastAPI, `GET /api/teams` and `POST /api/predict`.
- `prototype.html` — approved layout, built by `scratchpad/build_mockup.py`. Port markup/CSS from here.
- `predict_matches.py` — benchmark only, not the serving path. Chronological split.

**Next: `web/`** — Vite + React + TS, porting `prototype.html` into components:
`TeamPicker`, `OutcomeBar`, `ScoreGrid`, `Markets`, `H2H`, `Form`.

## Locked decisions

**Model — geometric blend, not arithmetic.** `_pool()` in `model.py` combines
Poisson and XGBoost via a log-opinion pool. An arithmetic mean let XGBoost's bad
extreme-fixture behaviour through (San Marino at 10.2% to beat Brazil). Measured
on a chronological 80/20 split: blend 0.630 acc / 0.838 log loss, vs XGBoost
0.608 / 0.880 and Poisson 0.593 / 0.887. Always-home baseline is 0.584 / 0.965.

**`is_neutral` stays out of the XGBoost features.** In this dataset the
`home_team` column on neutral fixtures is systematically the stronger side (home
win rate 0.747 neutral vs 0.508 otherwise), so the flag means "home team is
better", not "no home advantage". Including it costs ~0.043 log loss but makes
the venue toggle move the wrong way on screen. Venue is the Poisson `home` term's
job. Do not re-add it for the metric.

**Team flag colours are never used for data encoding.** `color_code` in
`countries_names.csv` is mostly `#FFFFFF` / `#000000` / red — illegible on one
ground or the other and not distinguishable as pairs. Home/away use the fixed
`--home` / `--away` theme tokens. The real colour appears only as a bordered
identity swatch beside the team name.

**λ is capped at 6.0.** Multiplicative Poisson explodes on extreme mismatches
(Brazil v San Marino solved to 10.6 expected goals). Costs zero measured log loss.

**Scoreline grid is sent whole (11×11).** The client windows it to the smallest
square covering 95% of the mass — lopsided fixtures peak well outside 0–5.

## Front-end conventions

- **No state library.** Server state is everything and there are two endpoints;
  `useState` plus one `usePrediction` hook covers it. Reach for TanStack Query
  when endpoints pass ~4 or cross-fixture caching starts to matter — not before.
- **No chart library.** Every visual is CSS: flex bars, a CSS-grid heatmap.
- **Fonts: Roboto + Roboto Mono.** Sans for prose, mono for every number
  (`font-variant-numeric: tabular-nums`). **Icons: Font Awesome**, used only where
  they carry function, never as section decoration.
- **Self-host fonts, never CDN-link them.** The Artifact CSP blocks external
  hosts, so `<link>` to Google Fonts or the Font Awesome CDN silently falls back.
  In `prototype.html` they are inlined as base64 `@font-face` data URIs; Font
  Awesome is subsetted with `pyftsubset` to the glyphs used (204 KB → 1.4 KB).
  In `web/`, use the `@fontsource/roboto` and `@fortawesome` npm packages.
- **Both themes.** Tokens on `:root`, redefined under
  `@media (prefers-color-scheme: dark)` and both `:root[data-theme="..."]` blocks.

## Data constraints

`dataset/all_matches.csv` has eight columns: `date, home_team, away_team,
home_score, away_score, tournament, country, neutral`. 25,328 matches since 2000,
237 teams, through 2026-03-31.

**No cards, shots, possession or xG exist.** Do not design UI for them and do not
claim to predict them. Adding them means sourcing a club-league dataset;
international data does not carry them. Club/league teams are likewise out of
scope — this dataset is internationals only.

## Commands

```bash
.env/bin/python test_model.py          # model sanity checks
.env/bin/uvicorn api:app --reload --port 8000
.env/bin/python predict_matches.py     # benchmark, writes results/
```

Virtualenv is `.env/` (not `.venv/`).

## Deferred, in priority order

1. **Elo or rolling-form features** — the biggest accuracy win still available,
   ~20 lines, feeds both models.
2. Model persistence via joblib — currently fits on first request (~5s).
3. URL search params for shareable predictions (needs a router).
4. Club/league support — blocked on a dataset.
