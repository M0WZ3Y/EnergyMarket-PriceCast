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

---

Pages banked: 0 / quota 0 | Results table: n/a | Backup: [ ]
