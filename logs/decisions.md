# Research log — decisions & progress

One dated entry per decision. This file feeds the methodology chapter.
Weekly footer: pages banked vs. quota | results table updated? | backup done?

---

## Week 1

### 2026-07-06 — Scope locked
- Primary dataset: EPEX-DE from the Lago et al. (2021) open benchmark via
  epftoolbox (keyless auto-download). Benchmark era = DE-AT-LU joint zone.
- Live data (tool only): Energy-Charts API (Fraunhofer ISE), zone DE-LU,
  keyless, CC BY 4.0 attribution required.
- Stretch market (week-7 decision): France (FR) — same loaders, config change.
- Models (5): naive, SARIMAX, LEAR-LASSO, LightGBM, LSTM (+ weighted ensemble).
- Targets: hourly (24 day-ahead prices, D+1) and daily (baseload average,
  direct vs. aggregated).
- Metrics: MAE, RMSE, sMAPE, rMAE + Diebold–Mariano tests. No plain MAPE
  (negative prices).
- Validation: rolling-origin walk-forward per Lago et al. protocol.
- Random seed: 42 everywhere.

### 2026-07-06 — Dataset verification
- [ ] `python scripts/verify_dataset.py` outcome: Dataset: DE  (test years: 2)
Downloading / reading from cache ...
=== TRAIN ===
Shape:            (34944, 3)
Date range:       2012-01-09 00:00:00  ->  2016-01-03 23:00:00
Columns:          ['price', 'exog_1', 'exog_2']
Missing hours:    0
NaNs per column:  {'price': 0, 'exog_1': 0, 'exog_2': 0}
Price stats:      mean=36.20  std=15.95  min=-221.99  max=210.00
Negative prices:  297 hours (0.85%)
=== TEST ===
Shape:            (17472, 3)
Date range:       2016-01-04 00:00:00  ->  2017-12-31 23:00:00
Columns:          ['price', 'exog_1', 'exog_2']
Missing hours:    0
NaNs per column:  {'price': 0, 'exog_1': 0, 'exog_2': 0}
Price stats:      mean=31.64  std=15.49  min=-130.09  max=163.52
Negative prices:  241 hours (1.38%)
- [ ] `python scripts/smoke_test_energycharts.py` outcome: - [x] `python scripts/smoke_test_energycharts.py` outcome: PASS — 192 hourly
      rows for DE-LU, 2026-06-01 to 2026-06-08, mean=99.20 min=-44.74
      max=355.25 EUR/MWh. Note: ~3x higher mean than 2012-2017 benchmark
      period — flag as a limitation/discussion point (regime shift).

### 2026-07-06 — Admin
- [x] Supervisor scope sign-off sent: received
- [x] Week-9 partial review slot booked: booked
- [x] Week-10 full-draft review slot booked: booked
- [x] Faculty formalities checked (page limit, deadlines, plagiarism scan): checked
      (page limits, plagiarism procedure, progress-report requirements under
      the 12-month formal registration)
- [x] Defense format confirmed (duration, demo allowed?): confirmed with supervisor

---

## Week 2

### 2026-07-05 — EDA complete (notebooks/01_eda.ipynb)
- Spike threshold: train mean + 3*std = 84.04 EUR/MWh (train-only, no test
  leakage). Spike rate: train 180 hrs (0.52%), test 145 hrs (0.83%).
- ADF: statistic=-16.693, p≈0.0000 -> rejects unit-root null (stationary).
- KPSS: statistic=8.721, p=0.01 (capped, actual p smaller) -> rejects
  level-stationarity null (not stationary).
- Reading: classic ADF/KPSS contradiction for electricity prices — series
  is mean-reverting (no unit root) but has strong deterministic seasonal
  structure (daily/weekly harmonics, confirmed by ACF/PACF peaks at lag 24
  and 168) that KPSS picks up. Decision: model seasonality explicitly
  (seasonal terms/dummies) rather than treating the series as I(1).
- Figures exported to reports/figures/ (01-09, feeds thesis 3-3-3 and 5-2).

---

## Week 3

### 2026-07-11 — Feature pipeline built (src/features/pipeline.py)
- Fixed epftoolbox not being installed in .venv (was breaking
  test_benchmark_download) and restored notebooks/01_eda.ipynb after an
  accidental output-clearing left it uncommitted; both now clean.
- Feature set copied exactly from epftoolbox's own LEAR implementation
  (epftoolbox/models/_lear.py::LEAR._build_and_split_XYs), read directly
  from the installed package rather than reconstructed from memory:
  price lags D-1/D-2/D-3/D-7 (24h vectors), exogenous lags D-1/D-7 plus
  exog for the target day itself (D0 — legal, exog_1/exog_2 are
  day-ahead load/generation forecasts known before the forecast origin,
  not realized prices), weekday one-hot dummies. Config in
  configs/features.yaml.
- Rationale: reusing the exact published-benchmark feature convention
  (rather than inventing a parallel one) keeps all 5 models on one
  consistent, comparable feature set and avoids a second, undocumented
  feature-engineering path for LEAR-LASSO vs. the rest.
- Implementation: hourly df pivoted to one row per calendar day
  (`_pivot_to_daily_wide`), lag columns built via `.shift(n)` on the
  day-indexed frame (vectorized, avoids a slow per-row Python loop).
  Any day with an incomplete lag window or a missing source hour is
  dropped via `dropna`, never filled/interpolated — no risk of an
  interpolated value crossing the forecast origin.
- Tests: tests/test_features.py — deterministic synthetic price/exog
  series (value encodes day-number + hour) so every lag column's
  expected value is checked exactly; explicit leakage guards assert no
  X column reads the target day's own price, and exog columns are
  restricted to the D0/D-1/D-7 tags. 9/9 pass; full suite 13/13 pass.
- Reviewed by the leakage-reviewer agent: **no origin-crossing leakage
  found** (label never appears in X, no exog read past the target day,
  "price_D-1 = origin day itself" confirmed correct — day-ahead prices
  for day O are published the day before O, so they're already known by
  origin day O's own gate closure). One real bug found and fixed:
  `_pivot_to_daily_wide` built lag columns with positional `.shift()`,
  which would silently mislabel a farther day as a closer lag if a whole
  calendar day were missing from the source (never a future-leak, since
  shift always looks backward, but a silent mislabeling of lag distance).
  Fixed by reindexing to a contiguous daily calendar before shifting, so
  a missing day now becomes an explicit NaN row that gets dropped instead.
  Added regression test `test_full_day_gap_does_not_misalign_lags`.
  Also dropped the unused `min_history_days` config key (dead — trimming
  already happens via shift-induced NaN + dropna).
  Deferred (logged, not blocking): (a) `exog_current_day` applies to any
  `exog_*` column with no schema guard — fine today since exog_1/exog_2
  are both day-ahead forecasts, but would silently leak if a realized
  (non-forecast) exog column were ever added; (b) DST fall-back hour
  collision in the pivot (`groupby(day, hour).first()` merges the
  repeated nominal hour) — not addressed this week, note as a data-
  quality item under assumption (3) if DE/DE-LU DST edges matter later.
- Open item for week 4: decide whether LEAR-LASSO consumes this shared
  X/Y directly (feeding epftoolbox's LEAR.recalibrate(Xtrain, Ytrain)
  with our arrays) or keeps epftoolbox's internal builder — functionally
  identical, deferred since it doesn't block feature-pipeline work now.

### 2026-07-11 — Gameplan decision: Plan A / Plan B
Plan A = match/beat Lago et al.'s published LEAR/DNN numbers on EPEX-DE
(the only fair "beat" claim). Plan B (built regardless, weeks 5-8) =
innovation-led defense: regime-aware ensemble weighting (calm/spike weight
sets switched on the 84.04 EUR/MWh threshold), calm-vs-spike +
hourly-vs-daily SHAP comparison, OOD stress test of frozen models on live
2026 Energy-Charts data. Week-5 checkpoint: LightGBM walk-forward results
compared against Lago et al. published numbers decides which plan leads.
Week-7 priority: static ensemble → regime-aware ensemble → France (only
if slack).

### 2026-07-13 — Data source testing schedule

Principle: every source is tested BEFORE anything downstream depends on it.
The live API (the only external dependency) gets re-touched at three points
rather than trusted from one early smoke test.

| Week | What gets tested | Status |
|---|---|---|
| 1 | Initial verification: epftoolbox full download (gaps/NaNs/stats) + Energy-Charts /price smoke test | DONE — both passed (see week 1 entries) |
| 3 | New Energy-Charts endpoints (load + renewables): JSON parsing, 15-min→hourly resampling, schema match vs BenchmarkLoader, unit test on a sample month. Plus leakage assertion test on the feature pipeline | DONE — see 2026-07-13 entry below |
| 4 | Indirect re-test: walk-forward framework consumes processed benchmark data end-to-end; LEAR sanity check vs published Lago et al. numbers doubles as a silent-data-bug detector | Scheduled |
| 7 | Pre-freeze reproducibility check: fresh environment, one model end-to-end from config — re-verifies benchmark download path from scratch | Scheduled |
| 8 or 11 | Live pipeline under real load: OOD stress test pulls a large 2026 window through EnergyChartsLoader (much bigger than week-1 smoke test) | Scheduled |
| 11 | Full live path inside PriceCast: date picker → API fetch → forecast → chart, plus CSV-upload fallback path | Scheduled |

Mitigation note: on the first successful large 2026 pull (week 8 or 11),
cache the window to data/processed/live_2026_cache.csv so the OOD test and
defense demo can run from the cached copy if the API hiccups on defense day.

### 2026-07-13 — Week 3 closed: Energy-Charts load/renewables endpoints

- Real endpoint names/params pulled from the live openapi.json spec
  (https://api.energy-charts.info/openapi.json), not guessed. Key finding:
  parameter naming is NOT uniform across the API — `/price` takes `bzn`
  (bidding zone, e.g. `DE-LU`) but `/public_power_forecast` takes `country`
  (e.g. `de`). Both kept in configs/data.yaml (`live.bzn` / `live.country`)
  and must be updated together if the market changes (France stretch goal).
- No dedicated `/load` or `/total_load` endpoint exists. Load is only
  available via `/public_power_forecast?production_type=load&forecast_type=
  day-ahead` — confirmed this is the correct exog_1 equivalent (day-ahead
  load forecast, matching epftoolbox's DE dataset convention).
- Implemented in `src/data/loader.py`: `EnergyChartsLoader.fetch_load`
  (day-ahead load forecast, exog_1), `.fetch_renewables` (solar +
  wind_onshore + wind_offshore day-ahead forecast summed, exog_2), and
  `.fetch_exog` (joins price+load+renewables into the same
  `['price','exog_1','exog_2']` schema as BenchmarkLoader, so the live
  loader is a drop-in equivalent).
- Reviewed by the leakage-reviewer agent: **no origin-crossing leakage** —
  `_fetch_forecast` hardcodes `forecast_type="day-ahead"` on every call
  (no code path can substitute realized/actuals data into exog_1/exog_2),
  and `fetch_exog`'s joins are all `how="inner"` with no
  backfill/interpolation, so no future value can bleed into an earlier row.
  `pipeline.py` does not import/call `EnergyChartsLoader` yet, so no
  benchmark train/eval path is contaminated (tool-only rule intact).
  Two real (non-leakage) bugs found and fixed: (a) `fetch_renewables`
  summed the three renewable components with default `skipna=True`,
  silently treating a missing component (e.g. wind_offshore gap) as 0 and
  underestimating exog_2 — fixed with `min_count=len(components)` so a
  missing component now surfaces as NaN; (b) `fetch_prices` used a blanket
  `dropna` intended only to trim not-yet-published trailing nulls, which
  would have also silently deleted interior API gaps and misaligned
  downstream lag features — fixed to trim only the trailing NaN run via
  `last_valid_index`. Added two offline (non-network) regression tests:
  one asserting every load/renewables request hardcodes
  `forecast_type="day-ahead"` (the single most safety-critical invariant
  in this file), one asserting a missing renewable component yields NaN
  not a silently-low sum.
  Deferred (logged, not blocking): whether `/public_power_forecast?
  forecast_type=day-ahead` values are genuinely published before the D
  noon origin for day D+1, vs. continuously re-issued — verified operationally
  by construction (values are stable once past the query date; no
  re-fetch-and-diff verification run yet). Revisit if the week-8/11 OOD
  test surfaces any origin-timing anomaly.
- Also discovered: the API 429s on a burst of ~4-8 sequential requests
  well under any documented quota (hit during test runs, not just at
  week-8/11 "real load" scale). Added retry-with-backoff (honors
  `Retry-After`, else exponential, 3 retries) to `_get` rather than
  deferring the fix to week 8 — this would have blocked even a single
  `fetch_exog` call in normal use.
- Tests: tests/test_loaders.py — 19/19 pass (was 6 before this entry).
  Network-marked tests include a load fetch and a renewables fetch over a
  full sample month (2026-06-01 to 2026-07-01, ~700 hourly rows) plus a
  schema-match test confirming `fetch_exog` output has exactly
  `['price', 'exog_1', 'exog_2']` columns matching BenchmarkLoader.
- Full leakage-assertion suite (tests/test_features.py, from the earlier
  week-3 feature pipeline entry) re-run alongside these changes: still
  10/10 pass; combined with loader tests, full suite is 19/19 green.

---

## Week 4

### 2026-07-13 — Evaluation framework built (src/evaluation/)

- `src/evaluation/metrics.py`: thin wrappers around `epftoolbox.evaluation`
  (MAE, RMSE, sMAPE, rMAE, DM) rather than reimplemented — same convention
  as the week-3 feature pipeline, keeps results numerically comparable to
  the published Lago et al. benchmark. No plain MAPE exposed (CLAUDE.md —
  negative prices make percentage error undefined).
- `src/evaluation/walk_forward.py`: rolling-origin (walk-forward) split
  generator implementing epftoolbox's own LEAR daily-recalibration protocol
  exactly (cross-checked against
  `epftoolbox.models._lear.LEAR.recalibrate_and_forecast_next_day`): each
  forecast origin trains on a fixed trailing calibration window, predicts
  one day ahead, origin advances by `step_days`. Config in
  `configs/evaluation.yaml`: `calibration_window_days: 1092` (= 364*3,
  epftoolbox/Lago et al.'s own LEAR default, not invented), `validation_days:
  364`, `optuna.n_trials: 50` (per CLAUDE.md), `random_seed: 42`.
- Also added `carve_validation_from_train` (trailing-slice split of the
  train calendar for Optuna tuning) and `assert_validation_before_test`
  (hard assertion that the validation window ends strictly before the test
  window starts — the project's non-negotiable ordering rule).
- Reviewed by the leakage-reviewer agent: **no actual leakage** with the
  shipped default config values — the train/origin boundary assertion
  (`train_days.max() < origin`) is reachable and sufficient, splits are
  emitted in plain chronological order (no shuffling/KFold), and
  `first_origin` restricts which days are used as origins without ever
  truncating or leaking into the training-history slice. Four latent
  edge-case issues found and fixed before treating this as the frozen
  harness models will be built on: (a) `assert_validation_before_test`
  silently no-op'd (returned instead of raising) on an empty
  validation/test window — since this function is the last line of
  defense against tuning/test overlap, a silent pass-through defeated its
  purpose; now raises `ValueError`. (b) `carve_validation_from_train` hit
  Python's negative-zero slicing trap when `validation_days: 0`
  (`train_days[:-0]` is empty, `train_days[-0:]` is everything) — would
  have silently inverted fit/validation instead of erroring; now raises
  `ValueError` for `validation_days <= 0`. (c) Removed one line of dead
  code (`if pos - window < 0: continue`) that could never trigger given
  `start_pos = max(window, ...)`. (d) Added test coverage for the
  documented-but-previously-untested case where a requested `first_origin`
  sits before a full calibration window of history exists — behavior
  (silently pushed forward to the first day with full history) was
  already safe, just unverified.
  Also added one integration test using the REAL `configs/evaluation.yaml`
  values (not the shrunk config used in the rest of the test file) tying
  `carve_validation_from_train` + `assert_validation_before_test` +
  `walk_forward_splits` together end-to-end, confirming the first
  test-period origin's training window contains zero test-period days —
  this is the concrete check backing the thesis's "validation strictly
  before test" leakage claim, previously only exercised function-by-function.
- Tests: tests/test_evaluation.py — 18 tests, all passing; full offline
  suite (features + loaders + evaluation) is 32/32 green.
- Open item carried from week 3, still open: decide whether LEAR-LASSO
  consumes the shared X/Y from `src/features/pipeline.py` directly or
  keeps epftoolbox's internal builder. Next up: wire naive, SARIMAX, and
  LEAR-LASSO baselines onto this evaluation harness.

---

## Week 5

### 2026-07-27 — Environment fix: epftoolbox/numpy-2.x incompatibility

- Discovered the active shell had no working Python environment for this
  project at all: no `conda` binary anywhere on the machine (bash or
  PowerShell), and `epftoolbox` was not importable in the system Python
  3.11 install — meaning even the existing week-4 evaluation suite
  (`tests/test_evaluation.py`, which imports `src/evaluation/metrics.py`
  -> `epftoolbox.evaluation`) was silently unrunnable before today.
- Fix: created a project-local `.venv` (Python 3.11 — the only interpreter
  available on this machine; `environment.yml`'s Python 3.10 pin could not
  be honored since no 3.10 install or conda exists here) and
  `pip install -r requirements.txt`. This resolved cleanly, including
  `epftoolbox` + its `tensorflow`/`keras` dependency chain.
- Re-ran the full pre-existing suite as a smoke test immediately after
  install: 32/32 still green, confirming no regression from the new
  environment.
- Found (and worked around, not by downgrading numpy) a real
  epftoolbox/numpy-2.x incompatibility: `epftoolbox.models.LEAR.predict()`
  (`_lear.py:109`) does `Yp[h] = self.models[h].predict(X)`, relying on
  numpy's old implicit array-to-scalar coercion for the length-1 array
  each per-hour Lasso model returns. numpy>=1.25 deprecated this and
  numpy 2.x (2.4.6, installed here) removed it outright, raising
  "setting an array element with a sequence" on every `LEAR.predict()`
  call. epftoolbox's own metadata declares only `numpy>=1` with no upper
  bound and was never updated for this — not a bug in our code. Downgrading
  numpy was rejected: the installed `tensorflow==2.21.0` (an epftoolbox
  dependency via its DNN model) requires numpy 2.x, so pinning numpy<2
  would trade one broken dependency for another. Resolution: `src/models/
  lear_lasso.py`'s `LEARLassoModel.predict()` reproduces `LEAR.predict()`'s
  exact operation order (scale non-dummy columns via the already-fitted
  `scalerX`, per-hour `models[h].predict()`, `scalerY.inverse_transform`
  on the assembled 24-vector) using epftoolbox's own fitted state, with
  only the final scalar extraction made numpy-2.x-safe (`.predict(Xtest)[0]`
  instead of relying on removed coercion). `LEAR.recalibrate()` (the
  numerically significant LassoLarsIC + Lasso fitting step) is called
  as-is, untouched — this is a compatibility shim for one broken line,
  not a reimplementation of LEAR.

### 2026-07-27 — Naive / SARIMAX / LEAR-LASSO model wrappers built

- Resolves the week-3/4 open item: **LEAR-LASSO consumes the shared
  `build_features()` X/Y directly**, via epftoolbox's low-level
  `LEAR.recalibrate(Xtrain, Ytrain)` / `LEAR.predict(X)` API, not the
  high-level `recalibrate_and_forecast_next_day` (which reruns its own
  internal df -> X/Y builder). Confirmed by reading the installed
  `epftoolbox/models/_lear.py` source: `recalibrate()` accepts pre-built
  `[n_days, n_features]` / `[n_days, 24]` arrays directly and expects the
  last 7 columns to be day-of-week dummies — exactly `build_features()`'s
  existing column order (247 features for the shipped 2-exog config:
  96 price-lag + 144 exog-lag/current-day + 7 dow, matching epftoolbox's
  own `n_features = 96 + 7 + n_exogenous*72` formula exactly). This keeps
  one leakage-audited feature source of truth instead of trusting a
  second internal builder path, and gives LEAR the same
  `fit(X,Y)`/`predict(X)` call shape as every other model wrapper.
- `src/models/base.py`: `BaseModel` ABC — `fit(X, Y) -> self`,
  `predict(X) -> DataFrame[y_h00..y_h23]`, `save(path)`, `load(path)`,
  with a shared pickle-based default `save`/`load` every wrapper can use
  or override. All fit/predict calls take/return DataFrames in
  `build_features()`'s exact shape so `src/evaluation/run_baselines.py`
  never branches on model type.
- `src/models/naive.py`: the standard Lago et al. day-of-week "similar
  day" naive — Monday -> `price_D-3` (last Friday; D-1/D-2 are the
  weekend), Saturday/Sunday -> `price_D-7`, Tuesday-Friday -> `price_D-1`.
  Pure column selection over existing `build_features()` lag columns, no
  fitting, no new numeric logic.
- `src/models/sarimax.py`: 24 independent per-hour `statsmodels` SARIMAX
  models (mirroring LEAR-LASSO's own per-hour independence), exogenous =
  the target day's own day-ahead `exog_1_D0`/`exog_2_D0` forecasts
  (legal — known before the forecast origin, never realized price).
  **Logged deviation from strict daily recalibration**: fully refitting
  24 seasonal SARIMAX models across the full ~730-origin walk-forward
  test period (~17.5k fits) is not practical, and — unlike LEAR-LASSO —
  there is no upstream Lago et al./epftoolbox precedent constraining
  SARIMAX's cadence specifically (SARIMAX is this thesis's own addition
  as the classical/interpretable baseline). SARIMAX fully refits every
  `refit_every_n_days` (default 7, `configs/models.yaml`) and uses
  statsmodels' cheap `.append(refit=False)` state-space update in
  between; the harness still *forecasts* every origin day
  (`walk_forward.step_days` stays 1), so the daily model-comparison stays
  fair — only SARIMAX's own fitted parameters are held between full
  refits. Disclosed here rather than silently baked in, since CLAUDE.md's
  daily-recalibration walk-forward protocol was written with LEAR's fast
  Lasso refit as the reference case.
- `configs/models.yaml` (new): `naive.rule`, `sarimax.{order,
  seasonal_order, exog_columns_prefix, refit_every_n_days}`,
  `lear_lasso.calibration_window_days` (null -> falls back to
  `evaluation.yaml`'s single source of truth), `artifact_dir`.
- `src/evaluation/run_baselines.py` (new): wires
  `BenchmarkLoader` -> `build_features` -> `walk_forward_splits` ->
  `model.fit`/`predict` -> a long `[origin, hour, y_true, y_pred, model]`
  DataFrame per model. Stops there deliberately — the metrics/results-
  table export (model x target x metric, per `.claude/skills/
  export-results/SKILL.md`) is a separate, not-yet-built task.
- **Daily-target modeling (RQ4) explicitly deferred**: land hourly for
  all three models first; the long-format results frame already contains
  everything needed for daily-aggregated metrics later
  (`groupby('origin').mean()`). Daily-*direct* fitting is its own later
  task, once hourly is validated against Lago et al.'s numbers (week-5
  checkpoint).
- Leakage-reviewer review of `src/models/{base,naive,sarimax,
  lear_lasso}.py` and `src/evaluation/run_baselines.py`: **no leakage**
  found (no feature/model path reads same-day or future-of-origin data;
  naive never touches a same-day price column since `build_features()`
  doesn't even produce one; SARIMAX's exog columns are confirmed
  day-ahead-legal; LEAR's scaler fit/transform separation and the
  numpy-2.x predict() shim were both confirmed faithful to epftoolbox's
  original by reading its installed source). The review did catch one
  real (non-leakage) correctness bug, fixed before treating this as done:
  `SARIMAXModel.fit()` was updating `_last_refit_end` unconditionally on
  every call instead of only on an actual full refit — since
  `walk_forward_splits` advances the origin by 1 day per call, this made
  `(train_end - _last_refit_end).days` always equal 1, so
  `needs_full_refit` could never re-trigger after the very first cycle
  (SARIMAX would have silently refit once and appended forever, growing
  its effective training history unboundedly instead of holding a fixed
  rolling window — not a future-information leak, since every appended
  day was still strictly pre-origin, but it would have invalidated the
  "fixed window, refit every N days" methodology this section describes).
  Fixed by only stamping `_last_refit_end` inside the `needs_full_refit`
  branch. Also hardened `LEARLassoModel.predict()` with the same
  exactly-one-row guard `SARIMAXModel.predict()` already had (the
  reviewer flagged that without it, a future multi-row caller would
  silently get row-0's prediction repeated for every row instead of a
  loud error).
- Tests: `tests/test_models.py` — 16 new tests (interface conformance,
  naive's 4 weekday-rule branches + idempotency + schema, SARIMAX's
  single-row guard / exog-column contract / refit-cadence regression test
  (7 sequential calls, long enough to distinguish "cadence from last full
  refit" from the caught bug's "cadence from last call") / end-to-end
  forecast shape, LEAR-LASSO's single-row guard / column-order guard /
  real-epftoolbox recalibrate+predict shape check — the last two marked
  `epftoolbox` (new `pytest.ini` marker, deselect with
  `-m "not epftoolbox"`) since they exercise the real package). Full
  offline suite: 48/48 green (32 prior + 16 new), 45/48 with
  `-m "not epftoolbox"`.
- Manually sanity-ran `run_baselines.run_model()` for all three models on
  real `BenchmarkLoader` DE data over a handful of origins (not the full
  2-year test period — expensive and unnecessary just to validate
  wiring): naive and SARIMAX each produced `n_origins * 24` rows in
  seconds; LEAR-LASSO likewise, ~15s for 3 origins with a 300-day
  window. Output shapes and column names matched `Y`'s schema in all
  three cases.
- Open item for next session: the full walk-forward run over the real
  2-year test period (needed before the week-5 LightGBM-vs-Lago-et-al.
  checkpoint) hasn't been executed yet — today's runs were small
  wiring-validation slices only.

### 2026-07-28 — Full 2-year baseline walk-forward run complete

- Ran all three baselines over the complete benchmark test period via the
  new `scripts/run_full_baselines.py` (same wiring as
  `src/evaluation/run_baselines.py`, plus what a multi-hour run needs:
  per-origin CSV checkpointing with crash-safe resume — only origins with
  all 24 hours written count as done — progress/ETA logging, one output
  file per model under `data/processed/baselines/`).
- Coverage: 728 origins (2016-01-04 -> 2017-12-31), 17,472 predictions
  per model, zero NaN predictions in any of the three files.
- Headline hourly metrics (informal, computed directly from the long
  frames — the formal metrics layer / results-table export is still a
  separate task):

  | model | MAE | RMSE |
  |---|---|---|
  | naive | 7.750 | 13.257 |
  | SARIMAX | 4.351 | 7.117 |
  | LEAR-LASSO | 3.899 | 6.475 |

- Reading: ordering is exactly as expected (naive >> SARIMAX >
  LEAR-LASSO), and LEAR-LASSO's 3.90 MAE sits in the right neighborhood
  of Lago et al.'s published DE LEAR results — strong evidence the whole
  loader -> features -> walk-forward path reproduces the benchmark
  protocol faithfully. This satisfies the week-4 "indirect re-test" row
  of the 2026-07-13 data source testing schedule (LEAR sanity check as
  silent-data-bug detector). The formal side-by-side against the
  published table (exact numbers from the paper, not recollection) is
  the week-5 checkpoint and still to come, together with LightGBM.
- SARIMAX emitted statsmodels ConvergenceWarnings on some weekly refits
  (expected for seasonal orders on real price data; forecasts were still
  produced for every origin). Runtime: ~3.4 s/origin (~41 min total) —
  the weekly-refit cadence logged on 2026-07-27 made the full run
  practical. LEAR-LASSO and naive were faster still.
- Decision: committed the three result CSVs to git via a narrow
  `.gitignore` exception (`!data/processed/baselines/`), deviating from
  the "processed is regenerable" default. Rationale: the v1.0-results
  freeze rule makes these exact numbers load-bearing for the thesis
  text, and SARIMAX refits are not bit-reproducible across
  BLAS/statsmodels builds even at seed 42 — versioning ~4 MB of CSV is
  cheaper than discovering a silent drift after the freeze.

## Week 6

### 2026-07-30 — Machinery built ahead of the checkpoint (backfilled 2026-07-31)

Backfilled: twelve commits landed on 07-30 without a log entry. No new
results were produced — this was all machinery, built so that whichever
gameplan the week-5 checkpoint selects can start immediately.

- LSTM wrapper (`src/models/lstm.py`, TDD) plus its Optuna tuning script
  (`scripts/tune_lstm.py`). The tuning run has NOT been executed — LSTM
  has no tuned params and no walk-forward results yet.
- Results-analysis layer (`src/evaluation/results.py`): daily-baseload
  aggregation from hourly predictions and the pairwise Diebold-Mariano
  matrix. The hourly-aggregated daily path is one half of the RQ4
  comparison; the direct-daily half still needs its own runs.
- Ensemble machinery (`src/evaluation/ensemble.py`), in the week-7
  priority order from CLAUDE.md: static weighted combiner + weight
  fitting first, then regime-aware weighting switching calm/spike weight
  sets on the 84.04 EUR/MWh threshold from the week-2 EDA.
- `run_full_baselines.py` gained `--first-origin/--last-origin/--out-dir`
  so validation-window predictions can be generated into
  `data/processed/validation_preds/` without touching the test-period
  files. This is what keeps the ensemble weight fit from ever seeing
  test-period predictions. Only `naive.csv` exists there so far, so
  weight fitting is still blocked on validation runs for the other
  models.
- `week5_checkpoint.py` now skips models whose CSV is partial rather than
  failing. Convenient during a long run, but it means the script exits 0
  while silently omitting a model — the checkpoint is only meaningful
  once every model file is complete. Worth re-checking the model list in
  its output, not just its exit status.
- Farsi drafts for thesis sections 3-5 and 3-6. Writing is otherwise
  parked until the code/results are done (decision 2026-07-30).
- Tooling: `scripts/task_monitor.py` (renamed) gained watch-mode input,
  phase-aware recommendations, pause/resume, and keep-awake.

### 2026-07-31 — LightGBM walk-forward resumed

- The LightGBM full run started 07-30 13:10 and stopped 07-30 17:16 with
  364 of 728 origins written (through 2017-01-01). The file ended on an
  exact origin boundary (364 x 24 rows, no partial origin), so it was an
  interruption between origins, not a crash mid-write — most likely the
  machine sleeping or the terminal closing. No stdout had been captured,
  so no traceback existed to confirm this. Runs now log to
  `logs/runs/`; the earlier absence of a log is the only reason the
  cause is inferred rather than known.
- First resume attempt failed immediately: the shell's default
  interpreter is the system Python 3.11, which has no `lightgbm`. The
  project `.venv` (lightgbm 4.6.0, epftoolbox importable under numpy
  2.4.6) is the correct one. Decision: invoke `.venv/Scripts/python.exe`
  explicitly for long runs rather than relying on an activated shell —
  an unactivated shell fails at import, which is loud, but only after
  the operator assumes the run is under way.
- Resumed from origin 365 via the existing per-origin checkpointing
  (`completed_origins()` counts only origins with all 24 hours present).

### 2026-07-31 — ROOT CAUSE: connected standby kills unattended runs

- Both jobs running today (LightGBM test period, and the validation-window
  runs) died simultaneously at ~15:21 with no traceback in either log —
  they simply stop mid-progress. Simultaneous death of two unrelated
  Python processes is not a model bug.
- Windows System log, same minute: `Kernel-Power 506, "The system is
  entering connected standby. Reason: Idle Timeout"` at 15:21:45,
  followed by 507 (exit, "Input Keyboard") at 15:27:59. The machine idles
  into connected standby and the long runs do not survive it.
- This is almost certainly the same cause as the 07-30 17:16 stop, which
  had the identical signature (clean stop on an origin boundary, no
  traceback). The 07-30 event log shows the machine slept that afternoon
  too. Recorded then as "most likely machine sleep"; now confirmed by
  event-log correlation rather than inferred from the artifact.
- Why the existing mitigation did not fire: `task_monitor.py` already
  implements keep-awake (`SetThreadExecutionState` with
  `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`), but it asserts it only while the
  monitor itself believes a job is running. Today's runs were launched
  directly, outside the monitor, so nothing held the machine awake. The
  keep-awake is coupled to the monitor's job registry, not to the actual
  presence of a long-running process.
- Power settings, confirmed with `powercfg`: this machine is Modern
  Standby (S0 Low Power Idle; S1/S2/S3 all unavailable in firmware), with
  "Sleep after" = 0 (never) on AC but 300 s (5 min) on battery. So an
  unattended run on battery dies about five minutes after the last
  keypress, which matches the repeated short-interval 506 events today.
- Decision, implemented same session: `run_full_baselines.py` now wraps
  its run loop in a `keep_awake()` context manager asserting
  `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` and releasing it on exit
  (including on exception). A mitigation that lives in a different
  process from the job it protects is a mitigation that silently does
  not apply — so the job now asserts it directly rather than depending
  on the monitor being open and tracking that job.
- Belt and braces for long runs: stay on AC. On Modern Standby an
  execution-state request is not an absolute guarantee against every
  standby path (a lid close still forces it — see the 10:00 "Reason:
  Lid" event), whereas AC power already sets the idle timeout to never.
- Also noticed: two `task_monitor.py --watch` instances are running, one
  under the project `.venv` and one under the system Python 3.11.
  Duplicate watchers should be reconciled.

### 2026-07-31 — Checkpoint script: partial input now fails by default

- `week5_checkpoint.py` had been changed on 07-30 to skip models whose
  CSV was still partial. That made it exit 0 while printing a
  well-formed table with LightGBM silently absent — i.e. the script that
  decides Plan A vs Plan B could report success without evaluating the
  model the checkpoint exists for.
- Decision: partial input now aborts with exit 1, naming each unready
  model. Previewing mid-run requires `--allow-partial`, which prints the
  table under a "PARTIAL PREVIEW - NOT THE CHECKPOINT" banner listing
  what was omitted. Rationale: tolerance of incomplete input is safe
  only while output is advisory; once output drives a decision, a silent
  omission is worse than the crash it replaced.
- Preview run against the current (still partial) state, for the record —
  full test period, our metric code, published forecasts side by side:
  DNN Ensemble 3.413 < LEAR Ensemble 3.609 < **our LEAR-LASSO 3.899** <
  LEAR 1092 3.930 < LEAR 84 4.180 < LEAR 56 4.283, then our SARIMAX
  4.351 and naive 7.750 (MAE, EUR/MWh). So our LEAR-LASSO already beats
  three of Lago et al.'s four individual LEAR calibration windows on
  identical data. The data cross-check is exact: max |our y_true -
  published Real price| = 0.000000 over all 17,472 hours.
- Plan A therefore hinges on LightGBM. The bar for a clean "beat" claim
  against the best published model is 3.413 (DNN Ensemble).

### 2026-07-31 — Ensemble weight fitting: leakage contract made checkable

- `fit_weights()` documented "pass validation-period frames only" but
  enforced nothing. Added an optional `test_days` argument that routes
  through the existing `assert_validation_before_test()`, so the
  contract is a check rather than a comment. Kept optional so unit tests
  can still fit on synthetic frames with no test window; production
  callers (the week-7 runner) must pass it. Two tests added.
- Reviewed the rest of `ensemble.py` for leakage: `regime_labels()` is
  correct — it decides day D's regime from day D-1's realized prices,
  which are known at the D-1 noon auction before D's forecast origin,
  and gaps in the origin sequence only make it look further back, never
  forward. Already covered by a test.

### 2026-07-31 — Validation-window predictions started (week-7 prerequisite)

- `data/processed/validation_preds/` held only `naive.csv` with 3
  origins — a wiring test, not a run — so ensemble weight fitting had no
  data. Started full validation-window runs (2015-01-05 -> 2016-01-03,
  357 origins, strictly before the 2016-01-04 test start) for naive,
  LEAR-LASSO and SARIMAX. naive complete; the other two in progress.
- LightGBM's validation run is deliberately deferred until the
  test-period run finishes, to avoid two ~4-hour LightGBM jobs
  contending. LightGBM is pinned to `n_jobs=4` of 12 logical processors,
  which is why the cheap models could run alongside it without slowing
  it (38.8 s/origin before and during).

### 2026-07-31 — WEEK-5 CHECKPOINT DECIDED: Plan B leads

- LightGBM's walk-forward completed (728 origins, 17,472 rows, 0 NaN, no
  duplicate origin/hour). `scripts/week5_checkpoint.py` then ran against
  Lago et al.'s own published DE forecasts on the identical test period
  with our metric code. Data identity is exact: max |our y_true -
  published Real price| = 0.000000 over all 17,472 hours.
- Pooled MAE, 2016-01-04 -> 2017-12-31: DNN Ensemble 3.413, DNN 4 3.592,
  LEAR Ensemble 3.609, DNN 3 3.706, DNN 1 3.716, DNN 2 3.850,
  **our LEAR-LASSO 3.899**, LEAR 1092 3.930, **our LightGBM 3.968**,
  LEAR 1456 3.988, LEAR 84 4.180, LEAR 56 4.283, our SARIMAX 4.351,
  our naive 7.750.
- **Decision: Plan A is not achievable with the sanctioned model list;
  Plan B (innovation-led defense) becomes the primary line.** Rejected
  alternative: claiming a partial-window win. The mid-run signal
  (LightGBM 3.212 over 2016 alone) did not survive year two.
- Per-year MAE, published models recomputed per-year so the comparison is
  like-for-like rather than year-vs-pooled:

  | model | 2016 | 2017 | change |
  |---|---|---|---|
  | DNN Ensemble (Lago) | 2.935 | 3.889 | +0.954 |
  | LEAR Ensemble (Lago) | 2.960 | 4.254 | +1.294 |
  | LightGBM (ours) | 3.204 | 4.727 | +1.523 |
  | LEAR-LASSO (ours) | 3.452 | 4.343 | +0.891 |
  | LEAR 1092 (Lago) | 3.474 | 4.383 | +0.909 |
  | SARIMAX (ours) | 3.883 | 4.815 | +0.932 |

- **No "beat the benchmark" claim is licensed in either year.** LightGBM's
  2016 figure beats all four published LEAR variants but loses to the DNN
  Ensemble's 2.935. Comparing our 2016 number against a published pooled
  number would be comparing different windows; the per-year table exists
  to stop that error reaching the thesis. For chapter 4, the defensible
  claim is reproduction quality: our LEAR-LASSO 3.452 vs their LEAR 1092
  3.474 in 2016, same protocol, identical data.
- **The checkpoint hands Plan B its central finding.** LightGBM has the
  steepest calm->volatile degradation of any serious model (+1.523),
  LEAR-LASSO among the flattest (+0.891); 2017 is the more volatile year
  (price std 17.62 vs 12.48, max 163.52 vs 104.96) and LightGBM's five
  worst days are all 2017, led by 2017-10-29 (58.16 MAE — Storm Herwart).
  Different models win in different regimes, which is precisely the
  premise of the regime-aware ensemble sanctioned on 2026-07-11. The
  innovation is now empirically motivated rather than assumed.

### 2026-07-31 — Ensemble membership, freeze date, validation_preds versioning

- Ensemble members are SARIMAX, LEAR-LASSO, LightGBM and LSTM. naive is a
  reference model only (rMAE denominator, baseline row), not a weighted
  member. Rejected alternative: include naive and let `fit_weights` drive
  its weight to ~0 as a reportable finding — declined because at 7.750
  pooled MAE it contributes nothing and its near-zero weight would be a
  foregone conclusion rather than a result.
- `data/processed/validation_preds/` joins `baselines/` as a `.gitignore`
  exception. Same rationale as 2026-07-28: these files determine the
  ensemble weights, so they are load-bearing for the freeze, and SARIMAX
  is not bit-reproducible across BLAS/statsmodels builds even at seed 42.
- **The `v1.0-results` tag slips past the end-of-week-7 target in
  CLAUDE.md.** An audit of what the tag must cover found two required
  deliverables entirely unbuilt: the daily-direct target (section 4-4 =
  RQ4; `build_features()` returns a 24-column hourly Y only, so this is
  new code plus another full pass of runs, not a config change) and the
  OOD stress test on live Energy-Charts data. Both produce model results,
  so both must precede the freeze. Rejected alternative: a staged
  v1.0/v1.1 freeze — declined because chapter 4 would then cite two
  frozen sets and the "never rerun" rule would need restating per tag.
  One freeze, covering hourly + both daily routes + both ensembles + OOD.
- Also noted for scheduling: SHAP (section 4-6, 8 pages, the largest
  results section) has no implementation at all — nothing imports `shap`.
  It explains already-fitted models and creates no new forecasts, so it
  correctly sits after the freeze, as the outline's week-8 slot assumes.
- The canonical results table has never been exported; `reports/tables/`
  does not exist. Both it and the DM table must be exported *before* the
  tag, because the `export-results` skill's PreToolUse hook blocks
  results exports once `v1.0-results` exists.

### 2026-07-31 — LSTM tuned and run; it is our best single model

- Optuna tuning complete, 50 trials, TPE seeded 42, validation window
  2015-01-05 -> 2016-01-03 asserted strictly before the test period.
  Best validation MAE 3.6960 (units 29, epochs 47, batch 32, lr 8.317e-4)
  -> `configs/tuned/lstm_params.yaml`. For comparison LightGBM's tuned
  validation MAE on the same window was 3.7316. Trials cost 4-10 s each;
  the whole search took minutes, not the hours budgeted.
- Bug found while wiring the run: LSTM had a wrapper, a tuning script and
  a `configs/models.yaml` entry but was never registered in
  `run_full_baselines.py`'s model dict, so `run_full_baselines.py LSTM`
  exited with "no matching models". Added there and to
  `week5_checkpoint.py`'s `OUR_MODELS` — the checkpoint now aborts until
  `lstm.csv` is complete instead of silently comparing four models.
- Full test-period walk-forward complete and accepted: 728 origins,
  17,472 rows, 0 NaN, no duplicate origin/hour. Cost ~0.9 s/origin
  (~11 min total) because `refit_every_n_days: 7` means ~104 network
  trainings rather than 728.
- **LSTM is our best single model: 3.8734 pooled MAE** (2016: 3.210,
  2017: 4.533), ahead of LEAR-LASSO 3.899 and LightGBM 3.968. It beats
  every published individual LEAR variant (1092 3.930, 1456 3.988, 84
  4.180, 56 4.283) but loses to all four published DNNs and both
  published ensembles. The Plan B decision above is unchanged: our best
  single model still sits 0.46 MAE behind the DNN Ensemble's 3.413.
- Note for chapter 4: LSTM's calm->volatile degradation (+1.323) sits
  between LightGBM's +1.523 and LEAR-LASSO's +0.891, so the regime story
  holds across all three ML models rather than being a LightGBM quirk.

## Week 7

### 2026-08-02 — The "offline" test command was never offline

- The suite flaked twice on 07-31, on two different Energy-Charts tests
  (`test_energycharts_fetch_renewables_month`,
  `test_energycharts_fetch_exog_schema_matches_benchmark`), each passing
  in isolation. Diagnosis: not flaky tests — the wrong deselect.
- `pytest.ini` already defines a `network` marker and all five live-API
  loader tests already carry `@pytest.mark.network`. But the command in
  use everywhere, `-m "not epftoolbox"`, does not deselect it, so every
  "offline" run was hitting api.energy-charts.info.
- **Correct offline invocation is `-m "not epftoolbox and not network"`**
  — 81 passed, 8 deselected, deterministic. Use this one from now on;
  earlier entries in this file quoting `-m "not epftoolbox"` predate the
  correction.
- No code change was needed. Worth recording because the failure mode was
  self-concealing: intermittent failures in a suite believed to be
  offline get written off as flakes, which is exactly how a real
  regression would slip through.

### 2026-08-02 — keep_awake() is NOT sufficient on Modern Standby

- The LightGBM/LSTM validation run launched 07-31 ~23:40 died at 51 of
  357 origins. System log: connected standby entered 07-31 23:45:23,
  exited 08-02 00:00:35 — over 24 hours asleep.
- This run held `keep_awake()` (`ES_CONTINUOUS | ES_SYSTEM_REQUIRED`),
  so the 07-31 fix did not prevent it. Correction to that entry: on
  Modern Standby (S0) an execution-state request is advisory, not
  binding; it does not hold off the idle transition the way it did under
  the old S3 model.
- What actually governs it is the power policy: `standby-timeout` is 0
  (never) on AC but 300 s on battery. The machine was on AC at the time
  of writing this entry, so the run either began on battery or the lid
  was closed.
- Operating rule for long runs: **AC power, lid open** — that, not
  `keep_awake()`, is the real protection. `keep_awake()` stays as
  defence-in-depth; it costs nothing and does help where the request is
  honoured. Setting `standby-timeout-dc` to 0 would remove the battery
  hazard but is a machine-wide policy change, so it is the user's call.
- Third run lost to this cause. All were fully recoverable thanks to
  per-origin checkpointing, at a cost of wall-clock time only.

### 2026-08-02 — INCIDENT: two concurrent writers on validation_preds/lightgbm.csv

- Cause: the validation run launched 07-31 23:42 was assumed dead after
  the 24-hour standby, on the evidence of a log whose tail showed only
  warnings and a CSV at 51/357 origins. It was not dead — it had been
  suspended with the machine and resumed when the machine woke at
  08-02 00:00:35. A second instance was then launched at 00:18, and both
  appended to the same file for ~10 minutes.
- This is exactly the failure the task monitor's `avoid` list names first
  ("launching a second run of the SAME model (would corrupt its CSV)").
  The rule was there; the mistake was inferring process death from
  artifact state instead of checking the process list.
- Damage: 11 origins written twice (48 rows each), 264 duplicate
  (origin, hour) pairs, 1,824 rows across 65 distinct origins.
- Mitigating fact: **all 264 duplicate pairs carry identical `y_pred`**
  — LightGBM is deterministic at seed 42 with a fixed `n_jobs`, so both
  writers computed the same numbers. The repair is therefore an
  unambiguous de-duplication, not a choice between conflicting values.
  The determinism rule paid for itself here.
- Resolution: the duplicate process was stopped; the original was left to
  finish, since its todo list was computed when the file was empty and it
  will therefore write every origin. De-duplication runs after it
  completes — repairing a file that is being appended to would risk worse
  corruption than the duplicates.
- Note on the resume logic: `completed_origins()` counts only origins with
  exactly 24 rows, so a 48-row origin reads as NOT done and would be run
  again on any future resume, appending a third copy. De-duplicating
  before the next resume is therefore mandatory, not cosmetic.
- Process change: before launching any long run, check the process list
  for an existing instance. A stale log tail and a stalled row count are
  evidence about the artifact, not about the process.
- **Repaired 2026-08-02, same session.** The surviving run was stopped
  cleanly (no `run_full_baselines` processes left), the file backed up,
  then de-duplicated on (origin, hour) keeping the first occurrence.
  Guarded by three assertions that had to hold before the write: zero
  duplicate pairs with conflicting `y_pred`/`y_true`, every origin at
  exactly 24 rows afterwards, and no origin lost. 2,184 rows -> 1,920,
  exactly the 264 expected removals, 80 origins preserved. All four
  validation files now read clean (naive/SARIMAX/LEAR-LASSO complete at
  357 origins; LightGBM at 80/357, resuming at origin 81).

### 2026-08-02 — Daily-direct route completed: SARIMAX, LEAR-LASSO, LSTM

- `src/models/daily.py` had only `DailyNaiveModel` and
  `DailyLightGBMModel`, so the direct arm of RQ4 covered two of the five
  models. Added `DailySARIMAXModel`, `DailyLEARLassoModel` and
  `DailyLSTMModel`; all five are now registered in
  `scripts/run_daily_direct.py`, reusing the same `configs/models.yaml`
  entries as the hourly route so the two routes differ only in the target.
- **LEAR is a transposition, not a reimplementation.** epftoolbox's `LEAR`
  class loops `for h in range(24)` and cannot take a scalar target, so the
  daily arm cannot call it. Every numerically significant piece is still
  taken from epftoolbox itself: `scaling(..., 'Invariant')` (asinh-median)
  on the target and on all inputs except the 7 dow dummies,
  `LassoLarsIC(criterion='aic', max_iter=2500)` for lambda, then
  `Lasso(max_iter=2500, alpha=lambda)`. Verified line-by-line against
  `epftoolbox/models/_lear.py`. The only difference is that the 24-fit loop
  collapses to one fit. Rejected alternative: drop LEAR from the direct
  arm — declined because RQ4 then compares different model sets on the two
  routes, which confounds the very thing it measures.
- **SARIMAX daily exog = the daily mean of the same `exog_*_D0` columns.**
  One model on the baseload series rather than 24 (the baseload is a single
  series, so this is by construction, not simplification). Averaging the D0
  forecasts mirrors exactly what the target does to the 24 prices; it stays
  legal since a D0 forecast is known before the origin and averaging cannot
  import later information. Seasonal period stays 7 — on a daily series
  that is the same weekly cycle the hourly models capture.
- **LSTM subclasses the hourly wrapper read-only**, overriding only the
  output head (width 1), the target's shape going in, and the frame schema
  coming out. The audited hourly path that produced the committed results
  is untouched — the reason `daily.py` exists as parallel code at all.
- 24 tests in `tests/test_daily.py`, all passing. Smoke-tested end to end
  on real benchmark data (3 origins, all five models, plausible
  predictions against identical `y_true`).
- **DECIDED (same day, user's call: optimize for accuracy): the daily
  models get their own Optuna pass against the daily target.** Reusing the
  hourly models' tuned hyperparameters would make any direct-vs-aggregated
  difference partly a difference in tuning effort rather than in the
  target — precisely the confound RQ4 exists to measure around.
  `scripts/tune_daily.py` added: same protocol as the hourly searches (50
  trials, TPE seeded 42, one static fit per trial, validation window
  hard-asserted before the test period), identical search-space bounds so
  neither route is allowed to look harder than the other. New
  `daily_lightgbm` / `daily_lstm` entries in `configs/models.yaml` point at
  `configs/tuned/daily_*_params.yaml`; the wrappers fall back to the
  untuned defaults until those files exist.
- **Only LightGBM and LSTM are tuned, and that is not a cost compromise.**
  They are the only two models with an Optuna search anywhere in this
  project. LEAR-LASSO selects its own lambda per fit via `LassoLarsIC` —
  that IS its tuning, and it re-runs against the daily target
  automatically. SARIMAX's (p,d,q)(P,D,Q,s) order is fixed by
  `configs/models.yaml` on both routes, so both arms already share one
  convention. `DailySARIMAX` and `DailyLEAR-LASSO` therefore keep reading
  the hourly `sarimax` / `lear_lasso` entries.
- Sequencing: the daily tuning pass and the daily walk-forward run both
  wait for the LightGBM/LSTM validation run to finish — one heavy job on
  the machine at a time (user's call, 2026-08-02), given that every run
  incident so far traces to concurrency or power state.
- Red-first TDD was not possible here: a PostToolUse hook runs the offline
  suite after every edit and blocks any edit that leaves it failing, which
  is exactly what a failing-test-first edit does. Implementation was
  written before tests as a result. Noted so the coverage guarantee that
  the red phase normally provides is not assumed to hold.

### 2026-08-02 — Ensemble runner built; spike threshold moved into config

- `scripts/run_ensemble.py` added: fits static weights and calm/spike
  weight sets on `data/processed/validation_preds/`, applies them to
  `data/processed/baselines/`, and writes `ensemble_static.csv` /
  `ensemble_regime.csv` in the same long-frame schema as every model, so
  `dm_matrix()` and the canonical results table consume them unchanged.
  `fit_weights` is always given `test_days`, so the leakage contract is
  checked rather than documented.
- The runner refuses to fit on a partial file: every member must cover an
  identical origin set with exactly 24 rows each and no duplicate
  (origin, hour). This is a direct response to the 08-02 concurrent-writer
  incident — weights silently fitted on a truncated or duplicated window
  would be a wrong result that still looks like a result.
- Guard on the regime arm: fewer than 20 validation days in either regime
  aborts with an explanation rather than fitting a weight set on noise.
- The 84.04 EUR/MWh spike threshold now lives in
  `configs/evaluation.yaml` under `regime.spike_threshold_eur_mwh` instead
  of only in a docstring, per the project's config-driven convention. Its
  provenance (train mean + 3*std, week-2 EDA, train-only) is recorded in
  the config comment so the no-test-leakage property travels with the
  number.
- Not yet run: blocked on the LightGBM/LSTM validation-window pass.

---

### 2026-08-04 — Regime threshold recalibrated to mean+1.5*std; 'spike' renamed 'stressed'

**Validation run completed.** The detached 08-02 run finished. All five
models cover 357 origins (2015-01-12 -> 2016-01-03), exactly 24 rows per
origin, no duplicate (origin, hour), no NaN. Nothing needed rerunning.

**The blocker.** At the registered 84.04 EUR/MWh threshold (train mean +
3*std, week-2 EDA, this file 2026-07-05) the validation window holds only
**3** stressed days, against the runner's `MIN_DAYS_PER_REGIME = 20`. The
guard fired correctly and was NOT weakened. Spike days are near-absent from
exactly the years this thesis uses:

| Year | Days over threshold (84.04) |
|------|------|
| 2012 | 26 / 358 |
| 2013 | 38 / 365 |
| 2014 | 1 / 365 |
| 2015 (validation) | 3 / 365 |
| 2016 (test) | 6 / 366 |
| 2017 (test) | 19 / 365 |

**Why the validation window was not widened instead.** Two validation-only
grounds, both admissible: (i) widening does not rescue k=3.0 — a 730-day
validation window still holds only 4 stressed days, so the >= 20 rule still
fails and the threshold would have to move anyway; reaching 26 requires
going back to ~2013-04, into a structurally different market era; (ii)
compute cost — ~643 extra origins x 5 models, a multi-day walk-forward.

(Correction, same day: an earlier revision of this entry argued the point
"more decisively" from a test-window count. That was inconsistent with this
entry's own rule that test data must not justify design choices, and has
been removed. See the ex-post note below.)

**Decision: keep the mean + k*std family, move k from 3.0 to 1.5**
(= 62.65 EUR/MWh; train statistics on data <= 2015-01-11, mean 37.61,
std 16.70 — strictly before the validation window, so still train-only).

k is chosen by a **validation-only rule**: take the largest k in
{3.0, 2.5, 2.0, 1.5} for which both regimes hold >= 20 validation days.

| k | Threshold | Validation stressed | Test stressed |
|---|-----------|--------------------|---------------|
| 3.0 | 84.04 | 3 / 357 (fails) | 25 / 728 |
| 2.5 | 79.35 | 5 / 357 (fails) | 32 / 728 |
| 2.0 | 71.00 | 10 / 357 (fails) | 44 / 728 |
| **1.5** | **62.65** | **37 / 357** | **77 / 728** |

The test column is recorded **ex post, after k was fixed**, for
transparency only. It is NOT the justification and must not become one:
choosing a threshold by its test behaviour is in-sample selection and would
compromise every downstream number. The rule reads the validation window and
pre-validation train statistics only.

Full disclosure, since the audit trail should not overstate the analyst's
ignorance: the test counts WERE known at decision time (they were computed
while diagnosing the blocker). The defence is not ignorance but determinism
— the selection rule is a total function of validation counts and would
return k=1.5 with the test column blacked out. k=2.0, the next candidate up,
fails the >= 20 rule at 10 validation days regardless of anything on test.

**Scope of the supersession.** 84.04 is superseded as a *regime switch*
only. It is not retracted as a descriptive statistic — the week-2 EDA
finding stands; it is simply the wrong switch for this data, because the
2014-2016 German market barely produced 3-sigma days.

**Rename `spike` -> `stressed`.** At ~1.5 sigma the label marks an elevated
day, not a price spike; calling it a spike would misdescribe the mechanism
in the thesis. Renamed in `configs/evaluation.yaml`
(`regime.stress_threshold_eur_mwh`), `src/evaluation/ensemble.py`,
`scripts/run_ensemble.py`, `tests/test_ensemble.py`, and CLAUDE.md.
`combine_regime_aware()` now raises on a legacy `{'calm','spike'}` weights
dict, with a test pinning it: silently falling through to the calm weight
set for every day would emit a static ensemble mislabeled as regime-aware.
A post-split origin-count assertion was added for the same reason — the
rename initially dropped all stressed days from the output because the
iteration loop still said `("calm", "spike")`, and only a test caught it.

**Results (test period 2016-01-04 -> 2017-12-31, 728 origins).**

| model | MAE | RMSE | sMAPE | rMAE |
|-------|-----|------|-------|------|
| Ensemble (regime-aware) | 3.557 | 6.557 | 14.644 | 0.390 |
| Ensemble (static) | 3.574 | 6.610 | 14.671 | 0.392 |
| LSTM | 3.873 | 7.020 | 15.599 | 0.424 |
| LEAR-LASSO | 3.899 | 6.475 | 16.657 | 0.427 |
| LightGBM | 3.968 | 7.502 | 15.722 | 0.435 |
| SARIMAX | 4.351 | 7.117 | 18.035 | 0.477 |
| naive | 7.750 | 13.257 | 28.595 | 0.849 |

Both ensembles beat the best single model (LSTM) by ~0.30 MAE — the solid
result. The regime-aware gain over static is 0.017 MAE (0.48%) in
aggregate; its significance was left open here pending the DM test and is
resolved in the next entry.

Interpretable side-result worth a sentence in chapter 4: under stress the
weights shift toward LEAR-LASSO (0.237 -> 0.375) and away from SARIMAX
(0.140 -> 0.063), with LSTM up (0.235 -> 0.281). Phrase this as a shift in
*fitted weights*, not as evidence that LEAR-LASSO is intrinsically better
under stress — the weights are fitted on the same window the members were
tuned on (see the leakage-review note in the 2026-08-04 DM entry), so they
partly reflect relative overfitting of that window.

---

### 2026-08-04 — DM test resolves the regime-aware ensemble: significant, and localized

Ran the pairwise DM test (multivariate, 24-h vector, L1 norm — the
epftoolbox/Lago protocol) on the test period, plus a focused
regime-aware-vs-static comparison split by regime.

**REVISED the same day after the leakage review.** The first version of this
entry reported the uncorrected epftoolbox DM p-values (0.0009 / 0.0003) and
claimed rejection at the 1% level. That claim is **withdrawn**. epftoolbox's
DM is mean(d)/sqrt(var(d)/N) with no HAC correction, but the loss
differential is strongly autocorrelated — a stressed day is *defined* by its
predecessor breaching the threshold, so stressed days arrive in runs (77
days in only 31 runs, mean run 2.48, max 7; lag-1 autocorrelation +0.36).
Treating clustered days as independent understates the standard error.

**Two independent corrections are reported as a RANGE**, not one number
(revised again after code review — see the block-length note below):
Newey-West HAC DM, and a circular block bootstrap (20 000 resamples, seed
42) swept over block lengths 3-10. The uncorrected DM is shown for
comparability with Lago et al. but is not a reported result.

| Subset | Delta MAE | DM (uncorr.) | HAC | bootstrap sweep b=3..10 |
|--------|-----------|--------------|-----|--------------------------|
| All 728 days | -0.0173 (-0.48%) | 0.0009 | 0.0226 | 0.0129 - 0.0571 |
| 77 stressed days | -0.1698 (-2.99%) | 0.0003 | **0.0063** | **0.0081 - 0.0439** |
| 651 calm days | +0.0007 (+0.02%) | 0.8624 | 0.8465 | 0.8399 - 0.8525 |

**Why a range and not a single p.** No block rule is authoritative here.
The usual n**(1/3) rule would give the stressed subset a SHORTER block (4)
than the full sample (9) despite its STRONGER dependence (lag-1 +0.356 vs
+0.290), purely because it is shorter — so the "significant" headline would
have been partly an artifact of subset size. Conversely, blocks beyond ~7
exceed the longest observed run of stressed days (max 7, mean 2.48), which
over-corrects by treating months-apart runs as dependent. The claim is
therefore made against the WORST case of the sweep, not the best.

**Claim discipline for chapter 4.** The supportable claim is: *regime-aware
weighting significantly improves accuracy on stressed days (-3.0% MAE;
p between 0.006 and 0.044 across HAC and every block length tried, i.e.
significant at the 5% level under every dependence correction applied),
while over the full test period the improvement is NOT robustly significant
(p 0.013-0.057, straddling 0.05).* Both halves must be stated together.
Never quote the uncorrected 0.0003, and never claim 1% significance.

Multiplicity: three subsets are reported, but they are not three independent
hypotheses (all-days is the union of the other two), so a Bonferroni
correction over them would be inappropriately conservative rather than
merely strict. Note the multiplicity in the text and let the worst-case
range carry the honesty, rather than applying a correction that does not fit
the design.

This is coherent rather than contradictory: a mechanism that fires on 10.6%
of days is diluted toward non-detectability when averaged over all days. The
calm subset (p~0.85) confirms the switch does not fire where it should not —
but report that as a sanity check, not as independent corroboration, since
the same threshold defines both the estimator's switch and the evaluation
partition, making the calm-day null partly mechanical.

**Methodological caveat to carry into chapter 3.** The loss differential is
severely right-skewed (skewness +7.9 on all days, +2.1 on stressed), so an
unstudentized bootstrap of a raw mean converges slowly and is mildly
oversized. Reporting HAC alongside partially covers this: the two methods
bracket the stressed result at 0.006-0.044 and agree on its direction and
significance level, which is stronger evidence than either alone. A
studentized block-t bootstrap would tighten this further and is the obvious
extension if a reviewer presses.

**Other DM results worth carrying into chapter 4.**
- Both ensembles beat every single model at p ~ 0.0000 — the strongest
  claim available in the results.
- LSTM beats LightGBM (p=0.0337), but **LSTM vs LEAR-LASSO is a tie
  (p=0.381)**: the best single neural model is not statistically
  distinguishable from the classical linear benchmark. This must be
  reported — it is directly relevant to the Plan A/Plan B framing and
  omitting it would overstate the deep model's standing.

Reproduce with `scripts/run_dm_ensembles.py` (reads only committed
artifacts). The canonical DM table is exported via the export-results skill
before the v1.0-results tag.

**Leakage review, 2026-08-04** (leakage-reviewer agent, ensemble changes).
No test-set contamination found in code: `fit_weights` never sees a test
origin, `combine_regime_aware` carries only the scalar threshold and the
weight vectors across from validation, and the previous-day labelling rule
is genuinely ex-ante. Actioned from it:
- Threshold slice tightened from <= 2015-01-11 to <= 2015-01-04 so it
  precedes the Optuna tuning window (opens 01-05) as well as the
  weight-fitting window (opens 01-12). Value 62.6522 -> 62.6989; regime
  split unchanged (37/357, 77/728) and all reported metrics unchanged, so
  this is provenance only.
- `tests/test_regime_threshold.py` added: asserts the configured threshold
  equals train mean + 1.5*std on data <= 2015-01-04, pins the window
  ordering, and rejects a legacy `spike_threshold_eur_mwh` key. The
  train-only property is now checked rather than asserted in a comment.
- `y_true` agreement across member frames is now enforced in
  `ensemble._aligned_pivots` and `results.dm_matrix`. Both took the truth
  column from one arbitrary member, so a stale file would have silently
  defined reality for every metric — the shape of the 2026-08-02
  concurrent-writer corruption. Synthetic test fixtures were randomising
  y_true per member, which no real frame does; they now share a fixed
  truth seed, which is also more faithful.
- Confirmed NOT actioned: ensemble weights are fitted on the same window
  the members were Optuna-tuned on (2015-01-05 -> 2016-01-03), so the
  validation predictions are not out-of-sample w.r.t. hyperparameter
  selection. Not a leak (no post-origin or test information), and fixing it
  needs a nested tuning/weighting split plus another walk-forward pass.
  **Disclose in chapter 3**, and soften the interpretive claim below
  accordingly — the weight vector is not an unbiased estimate of relative
  model quality.

---

### 2026-08-04 — OOD stress test built; BLOCKED on network, result not yet produced

`scripts/run_ood_stress.py` implements the sanctioned OOD arm (gameplan
2026-07-11): frozen benchmark-era models evaluated on live Energy-Charts
data. Three deliberately separate stages so the network is touched once:

- `--fit` fits all five models on the FINAL 1092-day calibration window of
  the benchmark data (2015-01-05 -> 2017-12-31 — exactly what the
  walk-forward would have used for one more origin) and persists them to
  `models/frozen/` with a `metadata.json` recording the freeze date and the
  benchmark price level. **Done and committed.**
- `--fetch` pulls the live window and CACHES it to `data/raw/`. **Blocked.**
- default replays: frozen models + cached data -> predictions + metrics.
  Implemented and validated; awaiting real data.

**Why frozen, not refitted.** Refitting on live data would measure
adaptation, not out-of-distribution degradation. The models never see an
hour of live data. Ensemble weights are likewise the frozen
validation-fitted ones — refitting them on the live window would leak the
OOD period into its own evaluation.

**~~BLOCKER: no outbound HTTPS from the workstation.~~ RETRACTED the same
day — this diagnosis was WRONG.** The original claim was that every HTTPS
request failed (Energy-Charts, pypi.org, example.com alike) and that the
cause was therefore environment-level rather than an API or code problem.
That conclusion was drawn from a handful of consecutive failures during one
bad window. Pushing the `v1.0-results` tag minutes later succeeded over
HTTPS to GitHub, which contradicted it outright; a retest then reached
every host, Energy-Charts included, with status 200.

What was actually happening: `api.energy-charts.info` intermittently drops
TLS mid-handshake (`SSLEOFError`, `DECRYPTION_FAILED_OR_BAD_RECORD_MAC`),
especially under sustained use, and the failures clustered enough to look
categorical. The correct lesson is the opposite of the one first recorded —
this is a *flaky remote host* to be retried through, not a dead network to
work around. See the retry fix in the follow-up entry.

Generalisable: "all N of my samples failed" is evidence of a failure, not
evidence of its scope. Diagnosing an environment-level cause needs a
successful control from a different code path — here, git's own HTTPS stack
would have falsified it immediately.

**Reproducibility requirement, deliberately built in.** The live API returns
different data every day, so an uncached OOD result could never be
reproduced — unacceptable under the post-tag freeze rule. The fetch stage
writes a CSV to `data/raw/` that must be committed alongside whatever
numbers it produces. Fetching is chunked (30-day windows) because the API
read-times-out on multi-month ranges, and a failed chunk is reported rather
than discarding the chunks already retrieved.

**Guards, because a bad OOD run fails silently.** A window overlapping the
freeze date would score models on their own training data and report
flattering numbers under an OOD heading. `replay()` refuses it, and
`tests/test_ood_stress.py` pins that guard plus the short-window,
missing-cache and partial-freeze cases. Known limitation, recorded rather
than papered over: **the guard compares dates, not content.** Validating the
pipeline with benchmark test data shifted forward nine years passed the
guard and produced an absurd LightGBM MAE of 1.67 (better than benchmark)
precisely because the underlying rows were training data. Dates alone
cannot detect that; only provenance can. Treat any OOD result whose metrics
IMPROVE on the benchmark as suspect input, not as a finding.

**Expectation to test, not to assume.** The frozen regime threshold
(62.6989 EUR/MWh, from 2012-14 prices) will likely classify nearly every
2026 day as stressed, degenerating the regime switch to a single weight set.
The script detects and reports that explicitly. If it happens it is a
genuine OOD finding — a threshold calibrated on a pre-crisis market does not
partition a post-crisis one — and should be reported as such, not patched by
recalibrating the threshold, which would destroy the frozen-model premise.

---

### 2026-08-04 — OOD stress test RUN; frozen models fall below naive on 2026 data

Supersedes the "blocked" status in the previous entry and in the
`v1.0-results` tag message. The tag is already pushed and is NOT being
rewritten: it froze the benchmark-era results, which this does not touch.
The OOD arm evaluates ALREADY-frozen models and cannot change any tagged
number, so it lands as the addendum `v1.1-ood`.

**Setup.** Models frozen on 2015-01-05 -> 2017-12-31, evaluated on live
DE-LU from `EnergyChartsLoader`, 173 complete days 2026-01-08 -> 2026-06-29
(4343 cached hours, no gaps). No refitting of any kind; ensemble weights are
the frozen validation-fitted ones. Live price mean **98.67 EUR/MWh vs 34.69**
at training time — a 2.84x level shift, consistent with the 3x flagged in
the week-1 smoke test.

| model | MAE | RMSE | sMAPE | rMAE | MAE vs benchmark |
|-------|-----|------|-------|------|------------------|
| naive | 29.18 | 48.73 | 45.90 | **0.808** | 3.8x |
| LEAR-LASSO | 39.27 | 65.31 | 49.72 | 1.087 | 10.1x |
| SARIMAX | 41.39 | 56.50 | 50.18 | 1.145 | 9.5x |
| Ensemble (regime-aware) | 42.17 | 53.38 | 59.40 | 1.167 | 11.9x |
| Ensemble (static) | 44.43 | 55.52 | 61.70 | 1.230 | 12.4x |
| LSTM | 54.93 | 67.51 | 79.16 | 1.520 | 14.2x |
| LightGBM | 66.06 | 77.35 | 92.50 | 1.828 | 16.6x |

**Finding 1 — every trained model is worse than naive (rMAE > 1).** This is
the headline, and rMAE is what makes it defensible: it rescales by a naive
forecast fitted to the live data itself, so it cannot be dismissed as "2026
is simply a harder market". naive alone stays below 1.0 because it carries
no frozen parameters — it re-reads the current price level for free, while
every trained model is anchored to a 35 EUR/MWh world.

**Finding 2 — the ranking inverts.** LightGBM and LSTM were the strongest
single models in-era and degrade the WORST (16.6x, 14.2x); SARIMAX and
LEAR-LASSO were mid-table in-era and degrade the LEAST (9.5x, 10.1x). The
flexible learners absorbed the price regime itself; the structured ones
carried less of it. Directly relevant to the Plan A/Plan B framing and to
the LSTM-vs-LEAR tie in the frozen DM table: the neural model's in-era
advantage was not only statistically indistinguishable, it also proves the
more fragile of the two.

**Finding 3 — the regime switch degenerates.** The frozen threshold
(62.6989 EUR/MWh, from 2012-14 prices) labels **171/173 live days stressed
(98.8%)**, collapsing the regime-aware ensemble to a single weight set. This
was predicted before running and is confirmed. It must NOT be fixed by
recalibrating the threshold — doing so would destroy the frozen-model
premise. It is reported as a limitation of threshold-based regime switching
under distribution shift.

**Honest caveat.** The regime-aware ensemble still edges the static one
here (42.17 vs 44.43), but with 98.8% of days in one regime that is
essentially the stressed weight set applied throughout, not regime
switching doing work. Do not present it as the regime mechanism succeeding
out of distribution.

### 2026-08-04 — Debug sweep: four bugs fixed

Systematic one-by-one check of the whole system. Frozen results verified
untouched throughout (`reports/tables/` regenerates to a zero diff).

1. **`EnergyChartsLoader._get` retried only on HTTP 429**, not on
   connection-level failures — so transient TLS drops killed whole chunks of
   a long fetch permanently. This silently lost all of May 2026 from the
   first OOD fetch. Now retries `RequestException` with the same backoff.
   This is the fix that the retracted "no HTTPS" diagnosis should have been.
2. **`fetch_live` overwrote the cache instead of merging**, so retrying to
   fill a hole would have discarded the chunks that had already succeeded.
   Now merges and warns about hours still missing.
3. **sMAPE returned NaN for the entire 173-day series** because one hour
   (2026-05-29 12:00) had actual == predicted == 0.0 — a perfect forecast of
   a zero price, routine in a 2026 solar-glut market, and epftoolbox's sMAPE
   divides by `(|a|+|p|)/2`. Fixed with a zero-safe variant LOCAL to the OOD
   script: the shared `metrics.smape` wrapper is deliberately untouched so
   no frozen number can move. No such hour exists in 2016-17.
4. **A `Path.relative_to` call inside a print string crashed `fetch_live`**
   when the cache sat outside the repo, aborting the function after its work
   was complete. Replaced with a fallback helper.

Sweep clean elsewhere: 122 offline tests pass, 6 network tests pass (was 2
failing), 13/13 scripts parse, 8/8 configs parse, every processed artifact
has correct origin counts with no duplicates or NaN, all `src` modules
import, no TODO/FIXME.

---


## 2026-08-05 — Full debug sweep: 30 defects closed test-first, frozen results proven unaffected

**Trigger.** The suite reported 128/128 green after the v1.0-results freeze,
which reads as "the code is sound". It was not. Three read-only agent sweeps of
`src/`, `scripts/` and `tests/` found ~30 real defects. The reason a green suite
missed them is structural: only 1 of 12 scripts had any test, and several
existing tests were tautological.

**The freeze question, answered with evidence rather than argument.** Several
defects sat in code that produced the tagged numbers, so the first task was to
determine whether any of them ever *fired* — a latent bug and a bug that fired
demand different responses.

- `data/raw/DE.csv`: tz-naive, 2184 days, **exactly 24 hours on every day**, no
  duplicate timestamps → the DST hour-dropping bug never fired.
- All 7 frozen `data/processed/baselines/*.csv`: **728 contiguous 1-day
  origins, zero NaN** in `y_pred`/`y_true` → neither the ensemble
  weight-collapse nor the previous-row-vs-previous-day regime bug could fire.
- `norm=2` has no call site (`run_dm_ensembles.py` uses `norm=1`) → the
  epftoolbox convention mismatch never fired.
- Frozen ensemble weights are non-uniform (LEAR-LASSO 0.237 → 0.375) →
  independently rules out an equal-weight collapse.
- **Strongest evidence:** with every fix applied, `dm_matrix` re-derives the
  frozen `dm_tests.csv` to a max absolute difference of **1.1e-16**, and all
  four exported tables regenerate with **byte-identical bodies and CSVs**.

Conclusion: this was a code-integrity repair, not a results correction. No
model was re-run, no artifact regenerated. `git diff -- reports/ data/ models/`
stayed empty throughout.

**Defects that could have produced wrong numbers** (all now fixed, each with a
reproducing test that failed first): `fit_weights` silently returned its
equal-weight starting point as "MAE-optimal" whenever any member frame held a
NaN (`res.success` was never checked); `regime_labels` used a positional
`shift(1)` instead of a previous-calendar-day lookup, so with any gap in the
origin set a day inherited a regime label from an arbitrarily older day;
`_aligned_pivots` validated the index but not the hour columns, and its
`np.allclose` check is positional, so mislabelled hours passed and then summed
by label into all-NaN columns; `diebold_mariano_hac` floored a degenerate
variance at 1e-12 and returned p == 0.0 from it; `daily_baseload`'s completeness
guard grouped by different keys than its own aggregation, rejecting valid
multi-model frames while accepting a duplicated hour that hid a missing one;
`_resample` repaired the interior NaNs `fetch_prices` had deliberately
preserved one line earlier.

**Two documentation defects that would have misled a reader.**
1. `diebold_mariano()`'s docstring claimed "is model 1 significantly more
   accurate than model 2". Verified empirically to be backwards: a small p
   supports **p_pred_2**. `dm_matrix` places the row model in `p_pred_2` and
   is correct, so no published number was affected — but the wrong sentence sat
   directly above the function the whole DM chapter rests on. Corrected.
2. `reports/tables/ood_stress.tex`'s caption says "Mean price 98.67"; the data
   gives 98.661573 → **98.66**. A hand-typed rounding error in a committed
   thesis artifact. The exporter now interpolates every caption statistic it
   can compute, so the class of error is gone. The frozen `.tex` was left
   untouched per the freeze rule; the discrepancy has no bearing on any claim
   (the point is that 2026 prices are ~2.8x training-era levels).

**The most serious finding was structural, not a bug.** The only test verifying
the regime stress threshold was computed on train-only data depends on
gitignored `data/raw/DE.csv`. Reproduced: with that file absent the test SKIPS
and the suite still reports green — so **on a clean checkout the non-leakage
guard CLAUDE.md calls non-negotiable was silently not running**, exactly where
an examiner would run it. `THESIS_FULL_DATA=1` worked mechanically but was set
by nothing: no conftest, no CI, no addopts. **Decision: invert the honour
system.** A new `tests/conftest.py` makes missing data FAIL by default, with an
explicit `THESIS_ALLOW_MISSING_DATA=1` opt-out that prints a loud
`!!! MISSING DATA !!!` terminal banner; `THESIS_FULL_DATA=1` is kept as an alias
so NEXT_SESSION.md stays correct. `pytest.ini` gains `--strict-markers`, without
which a typo'd `@pytest.mark.netwrok` silently ran in the offline set.

**Tests strengthened, with mutation evidence.** The DM test asserted only
`0 <= p <= 1` on data with an overwhelming accuracy gap — a constant-returning
DM passed it. Interface tests asserted `isinstance` and then `hasattr`, which it
entails. The OOD guard test asserted the *absence* of two magic substrings, so
any reword de-fanged it silently. Nothing asserted that a model loaded from
`models/frozen/` predicts what the fitted one did — the exact path the OOD
chapter depends on; round-trips added for all 7 wrappers that lacked them. Each
strengthened test was verified to FAIL against deliberately broken behaviour
(constant p-value, reversed p-value, ignored bandwidth, RMSE-based norm=2,
shifted walk-forward window, `save` dropping `_models`/`_scalers`).

**Operational defects.** `task_monitor._find_run_pids` failed OPEN: its bare
`except Exception: return []` meant a PowerShell timeout — which happens exactly
when the machine is loaded by a live run — read as "nothing running", letting
the resume path spawn a duplicate writer on the same CSV. That is the mechanism
behind the 2026-08-02 `validation_preds/lightgbm.csv` corruption; the guard
meant to prevent it is what permitted it. Now fails closed. Also:
`run_full_baselines` never removed partial rows from a torn origin (34 rows for
one day, permanently); `export_tables` crashed on the last table *after*
overwriting the first three; `--last-origin` before `--first-origin` exited 0
having done nothing; `fetch_live` ignored `--cache` and would overwrite the
committed live cache the flag was meant to redirect.

**Suite: 122 → 189 (183 offline + 6 network).** Every fix landed test-first:
reproducing test written and observed failing with the specific wrong value,
then fixed, then re-verified.

---

### 2026-08-05 — SHAP interpretability (section 4-6) built on a separate fit

**The decision that shapes everything else: 4-6 does not explain the frozen
models.** `models/frozen/` was fit on the trailing 1092 days ending 2017-12-31,
which contains the entire 2016-01-04..2017-12-31 test period. SHAP values
computed over test days against those models would be in-sample — in a chapter
whose whole subject is out-of-sample behaviour, and against the project's own
leakage rule. So `scripts/run_shap.py --fit` trains the same wrappers
(`LightGBMModel`, `DailyLightGBMModel`, unmodified) on the trailing 1092 days
ending **2016-01-03**, strictly before the boundary. Every one of the 728
explained days is unseen. Artifacts go to `models/interpretation/`, never
`models/frozen/`.

**Why LightGBM for both arms.** Both are trees, so `TreeExplainer` gives exact
TreeSHAP: no sampling, no background dataset, no seed sensitivity. A thesis
figure that changed between runs would be indefensible; this one cannot. The
exactness is itself asserted — `sum(shap) + expected_value == predict()` to
1e-6, checked against the *wrapper's* predict rather than a re-derived booster
call, so a mismatch between the explained and predicting booster cannot hide.

**Findings.** Renewables day-ahead forecast (`exog_2_D0`) is the largest single
driver (mean |SHAP| 5.86 EUR/MWh hourly), then yesterday's prices (3.77) and
the load forecast (`exog_1_D0`, 3.50); weekday dummies are negligible (0.15).
The hour profile is physically coherent: load dominates the 07:00 ramp,
renewables midday and the 18:00 peak, `price_D-1` the overnight hours. The
beeswarm at h18 shows the merit-order effect directly — high renewables push
price down, high previous-day price pushes it up. **Regime result: under stress
the model leans much harder on persistence** — `price_D-1` rises 3.48 -> 6.15
(+77%) from calm to stressed, while the fundamentals barely move. That is an
independent mechanism for why regime-aware weighting helped in 3-8.
**Hourly vs daily (RQ4):** the direct-daily arm relies proportionally less on
short price lags (`price_D-2` 0.90 -> 0.39) and nearly as much on fundamentals
(`exog_2_D0` 5.86 -> 4.55).

**Cross-check that mattered.** The calm/stressed split delegates to
`ensemble.regime_labels` rather than reimplementing the rule, and independently
reproduced **651 calm / 77 stressed** — the same 77 stressed days the DM regime
tests report. Sections 3-8 and 4-6 therefore describe the same day set, which
is asserted in `tests/test_run_shap.py`.

**Corrections made after the leakage review (agent found no leakage, but four
real defects).**
1. The case-study waterfall originally selected the day with the highest
   *realized* baseload and was captioned "the model under-forecasts the
   extreme". That is circular — conditioning on the argmax of the outcome
   guarantees a calibrated forecast sits below it, so the figure could not
   fail to show under-forecasting whatever the model did. Now selected by the
   model's **own highest prediction** (no outcome used), and the
   under-forecasting claim is instead backed by an aggregate that is not
   selection-fragile: **mean signed error -9.57 EUR/MWh over the 73
   top-decile-baseload days**. Renamed `waterfall_stressed_day` ->
   `waterfall_case_study`, since selection was never by regime.
2. **Captions now disclose that the explained fit is STATIC.** The results
   models recalibrate at every origin (`refit_every_n_days: 1`); the explained
   model is one fit held fixed across two years. Strictly less informed, so
   never a leak, but a reader would otherwise take figure 10 for the
   importance profile of the model behind chapter 4's numbers. It is not, and
   drift in feature reliance across 2016-2017 is invisible here.
3. `test_writable_namespace_covers_every_declared_output` was tautological —
   it iterated the very list the guard checks. Replaced with a test that
   records what `make_figures()` actually writes and asserts every recorded
   path is declared. The determinism test only proved TreeSHAP is
   deterministic (a property of the shap library); it now compares across a
   **refit**, which is what actually binds seed 42 and the fixed `n_jobs`.
4. `_facts_or_skip` used a bare `pytest.skip`, reintroducing the
   clean-checkout hole b246d25 closed. Now routed through
   `conftest.require_thesis_data`, so a missing cache fails.

Also added: a contiguity assertion in `interpretation_train_days` (row count
alone did not pin the window), a set-disjointness check between the training
and explained windows (endpoint comparison is only sufficient while both are
contiguous), and `--limit` now writes to `data/processed/shap/_smoke/` so a
stray smoke run cannot replace the artifacts behind figures 10-15.

**Freeze discipline.** `run_shap.py._assert_writable` refuses any path outside
its own namespace, tested against eight real frozen artifacts plus a `..`
traversal. `git diff -- reports/ data/ models/` is empty; every SHAP artifact
is a new file. `export_tables.py --dry-run` still reproduces the frozen
numbers.

**Suite: 189 -> 237 (231 offline + 6 network).** All new guards
mutation-checked: seven deliberate breaks (inclusive boundary, silent short
window, wrong booster explained, removed column guard, silent 'other' bucket,
sum->max grouping, regime reading its own day) were each confirmed to fail the
corresponding assertion.

---

### 2026-08-05 — Page quota set; writing is now the critical path

Going through the schedule at week 5 of 12: every code deliverable through the
original week 8 is done, and two landed 19 days early (the `v1.0-results`
freeze on 08-04, SHAP today). The PriceCast service layer, planned for week 11,
is also written. **None of that slack helps unless it is spent on pages.**
Writing is at 0 of 100, and the two supervisor dates are fixed: week-9 partial
review opens **2026-08-31**, week-10 full-draft review opens **2026-09-07**.
That is 2.31 pp/day and 3.03 pp/day respectively.

The README's standing rule ("45-60 min thesis writing daily before code, week
2+, page quotas tracked") had lapsed, and this footer is why it went unnoticed:
it read `quota 0` every week, so a quota of zero was always met. Fixed by
setting the quota rather than restating the rule — `configs/schedule.yaml`
holds the dates and targets, `thesis/page_ledger.csv` the daily record, and
`scripts/page_quota.py` reports required pace against each milestone.

The tracker is deliberately built not to flatter: a passed deadline raises
instead of returning a negative rate, a deadline of today demands the whole
remainder today instead of dividing by zero, and pages banked come from the
LATEST ledger row rather than the maximum, so a cut section shows as a drop
instead of leaving a high-water mark standing in for progress. It also warns
when the ledger goes two days stale — the failure mode that produced this
entry.

Week-9 target is 60pp (chapters 3 and 4): both are fully backed by frozen
numbers and `reports/tables/*.tex`, so they can be written now without waiting
on anything.

**Two stale schedule items found while reviewing.** The data-source test table
above still marks row 4 (week-4 indirect re-test) as `Scheduled` although the
2026-07-28 entry records it satisfied; and row 7, the **pre-freeze
reproducibility check** (fresh environment, one model end-to-end from config),
never ran — it was meant to precede the freeze, and the freeze happened on
08-04. It can still be run, but it can no longer be what it was designed to be.
Decide explicitly: run it late, or log it as skipped.

---

Pages banked: 0 / quota 60 by 2026-08-31 | Results table: v1.0-results + v1.1-ood | Backup: [ ]
