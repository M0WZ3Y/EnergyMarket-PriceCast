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
| 4 | Indirect re-test: walk-forward framework consumes processed benchmark data end-to-end; LEAR sanity check vs published Lago et al. numbers doubles as a silent-data-bug detector | DONE — see 2026-07-28 entry: 728 origins, 17,472 predictions per model, zero NaN, LEAR-LASSO MAE 3.899 in the expected neighborhood |
| 7 | Pre-freeze reproducibility check: fresh environment, one model end-to-end from config — re-verifies benchmark download path from scratch | DONE, BUT RUN LATE — ran 2026-08-06/07, after the 2026-08-04 freeze, so it is a sanity check and not a gate; naive exact match, LEAR-LASSO matches within 1e-12 (see 2026-08-07 entry) |
| 8 or 11 | Live pipeline under real load: OOD stress test pulls a large 2026 window through EnergyChartsLoader (much bigger than week-1 smoke test) | DONE — see 2026-08-04 entry: 173 complete days, 4,343 cached hours, no gaps; tagged `v1.1-ood` |
| 11 | Full live path inside PriceCast: date picker → API fetch → forecast → chart, plus CSV-upload fallback path | DONE — see 2026-08-05 entry: closed, now an automated `@pytest.mark.network` test (commit 4243571) |

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

> **Resolved 2026-08-07.** Both items are closed, and the count was wrong: the
> table had **four** `Scheduled` rows, not two. Rows 4, 8/11 and 11 were all
> already satisfied by entries in this same file (2026-07-28; 2026-08-04,
> `v1.1-ood`; 2026-08-05, commit `4243571`) and now cite that evidence. Row 7
> was run late and matches within 1e-12 — see the 2026-08-07 entry. The lesson
> worth keeping: this note found the two stale rows its author happened to
> notice, and reading it as the full extent of the problem would have left two
> more standing in a table that then looked reviewed.

---

### 2026-08-05 — PriceCast MVP finished; week-11 live-path test CLOSED

`streamlit run app/pricecast.py`. Three data sources (cached demo, live
Energy-Charts, CSV upload), target-day picker, forecast-vs-actual chart,
metrics, attribution. Driven end-to-end in a real browser, not just started:
the demo path serves 173 forecastable days from the committed 2026 window —
the same 173 days the OOD result covers.

**The week-11 data-source test is closed.** The scheduled row ("full live path
inside PriceCast: date picker → API fetch → forecast → chart, plus CSV-upload
fallback") is now an automated `@pytest.mark.network` test that fetches a real
window from Energy-Charts, builds features and produces a 24-hour forecast.
Network suite 6 → 7, all passing.

**The UI is obliged to tell the truth about accuracy, and does.** The served
model is frozen on 2017-12-31; v1.1-ood put every trained model above rMAE 1.0
on live 2026 data. On the demo day the tool forecasts a 48.42 EUR/MWh baseload
against 143.21 realized — MAE 94.79. Presenting that without comment would
misrepresent the research, so a warning renders before any chart, and its
figures are READ from `data/processed/ood/ood_summary.csv` rather than typed
in — the hand-typed-caption error class this project already hit once.

**Design note worth keeping.** `build_features()` drops any day whose own 24
prices contain NaN, correctly, which means it cannot build features for the day
you actually want to forecast — tomorrow's prices do not exist. The service
substitutes a placeholder into the target day's price column. That is only
legitimate because no feature column ever reads the target day's own price, so
the test proves it rather than trusting it: the same day is forecast with
placeholders of -500 and +5000 and the outputs must be identical. If a future
feature ever reads that price, the test fails instead of the app quietly
forecasting from a constant.

All logic lives in `app/forecast_service.py`, which imports no Streamlit;
`app/pricecast.py` only arranges widgets. That is why 29 of the tests need no
browser. Four defects were caught by writing the tests first: a DatetimeIndex
comparison that already returns an ndarray, a non-numeric-CSV test whose token
pandas silently parsed as NaN (so it exercised nothing), a deprecated
`use_container_width` past its removal date, and truncated metric values in the
rendered UI — the last one only visible because the app was actually opened.

**Suite: 237 → 282 (275 offline + 7 network).**

Still open for the MVP: the thesis 5-3 screenshot is not committed. The app was
captured during verification, but a figure that omits the accuracy warning
would be the wrong figure for that section — it should be taken at a window
tall enough to show the banner and the chart together.

---

### 2026-08-07 — Reproducibility check RUN LATE; LEAR-LASSO reproduces to machine precision

Closes row 7 of the 2026-07-13 data-source test table, which had stood at
`Scheduled` since week 7.

**It ran late, and that limits what it proves.** The check was designed to run
*before* the freeze, so that a failure could still change the frozen numbers.
The freeze happened on 2026-08-04; this ran on 2026-08-06/07. It is therefore a
sanity check on already-frozen results, not a gate on them. Recorded plainly
because the alternative — presenting it as if it had run on schedule — would
misrepresent the protocol. The check was run rather than skipped because a
negative result would still have been worth knowing before the thesis leans on
these numbers.

**Method.** Fresh venv at a short path (Windows MAX_PATH), no reuse of the
project environment; benchmark data re-downloaded from scratch into
`data/processed/_repro_check/raw/DE.csv`, exercising the download path itself;
one model end-to-end from an isolated `data_fresh.yaml`. All output confined to
the gitignored `data/processed/_repro_check/` — nothing in `reports/`,
`models/` or the tracked `data/processed/` subdirectories was touched, so the
freeze is intact.

Environments differ only in pandas — fresh 3.0.5 vs project 3.0.3; both
python 3.11.9, numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1, epftoolbox from
git.

**Results.** Two models, both against the frozen `reports/tables/results_canonical.csv`:

- **naive** — exact match, zero difference to 12 decimals on all four metrics
  (728 origins).
- **LEAR-LASSO** — 728 origins, 17,472 predictions, all 24 hours per origin,
  zero NaN:

  | metric | frozen | repro | rel diff |
  |---|---|---|---|
  | MAE | 3.898804600831 | 3.898804600831 | 0.0 |
  | RMSE | 6.474950936820 | 6.474950936820 | 0.0 |
  | sMAPE | 16.657203067271 | 16.657203067271 | 0.0 |
  | rMAE | 0.427153936259 | 0.427153936259 | 1.300e-16 |

  Verdict: MATCH within 1e-12. MAE, RMSE and sMAPE are bitwise identical; rMAE
  differs by 5.551e-17, one unit in the last place — floating-point summation
  order, not a modelling difference.

**Reading.** The seed-42 discipline and the config-driven pipeline hold across a
fresh environment, a fresh data download and a pandas patch-version difference.
This is the reproducibility claim the thesis can make; it is not a claim that
the check ran when it was scheduled to.

**Caveat.** One model, one target (hourly). It does not cover SARIMAX, LightGBM,
LSTM or the ensemble, nor the daily targets. A single-model check was the
design from the start, so this is a bound on the evidence, not a shortfall
against it.

---

### 2026-08-07 — OOD bias recalibration: the level shift explains the failure for some models, not all

Supplementary post-hoc analysis on top of `v1.1-ood`. Nothing frozen was
touched: `scripts/run_ood_recalibration.py` reads `data/processed/ood/`
read-only and writes only to `data/processed/ood_recalibrated/` and
`reports/tables/ood_recalibration.{csv,tex}`. Verified by hash in
`tests/test_ood_recalibration.py` and by `git diff` over `reports/`,
`models/` and `data/processed/ood/`. **This is not a benchmark result and
does not belong in chapter 4** — it feeds section 5-2 (limitations).

**Hypothesis.** Every trained model exceeded rMAE 1.0 on live 2026 data.
That could be a loss of forecasting SKILL, or only a loss of LEVEL — the
model still ranking hours correctly but sitting systematically low, as the
PriceCast demo suggested (48.42 predicted vs 143.21 realized). If removing a
causally-estimated offset restores rMAE toward 1.0, the learned shape
survived the regime change and only the intercept needs re-anchoring.

**Method.** For each model and day *d*, the correction is the mean signed
error (`y_true - y_pred`) over the `w` days STRICTLY BEFORE *d*, added to all
24 hours of *d*. No information from *d* or later enters its own correction;
the causality test is the first thing in the test file, because a leak there
would manufacture precisely the recovery being looked for. Windows 3, 7, 14,
30 and 60 were swept and **all are reported** — none selected after the fact.

**Cold start: excluded, not filled.** The first `w` days have no prior
window. They are dropped, and — the part that matters — the RAW arm is
re-scored on the same reduced day set. Comparing a 173-day raw rMAE against
a 159-day corrected one would let the exclusion itself move the headline. A
secondary `--cold-start expanding` mode (partial history during the cold
start only, then the same rolling window) reproduces every sign and
near-identical magnitudes, so no conclusion here rests on that choice.

**Result — a split, not a confirmation.** Recalibrated rMAE, `exclude`
(raw on the same day subset in parentheses for the two extreme windows):

| model | w=3 | w=7 | w=14 | w=30 | w=60 |
|---|---|---|---|---|---|
| naive *(reference)* | 0.932 | 0.868 | 0.820 | 0.805 | 0.821 |
| SARIMAX | 1.123 | 1.130 | 1.124 | 1.124 | **1.173** |
| LEAR-LASSO | **1.154** | **1.132** | **1.111** | **1.078** | **1.100** |
| LightGBM | 1.082 | 1.078 | 1.081 | 1.105 | 1.159 |
| LSTM | 1.007 | 1.014 | 1.013 | 1.018 | 1.065 |
| Ensemble (static) | **0.891** | 0.906 | 0.914 | 0.926 | 0.979 |
| Ensemble (regime-aware) | **0.878** | 0.895 | 0.898 | 0.901 | 0.947 |

Bold = worse than that model's own raw rMAE on the same days, among the
trained models; naive degrades at every window and is discussed separately
below. **Only the two ensembles cross below rMAE 1.0**, at every window; the
best is regime-aware at w=3 (1.168 → 0.878). No individual trained model
crosses at any window. LSTM comes closest at w=3 (1.007) and still misses.

**The optimum is a SHORT window, which sharpens the hypothesis.** For the
strongly-biased models the correction works best at w=3–7 and decays as the
window lengthens. If the miss were a *constant* level shift, the opposite
would hold — a longer window averages away more estimation noise and would
estimate a constant better. Short windows winning means the bias is
**time-varying**: the models are not sitting a fixed distance below the
market, they are tracking a drifting level with a lag. Conversely, for the
near-unbiased series (naive, LEAR-LASSO) the damage shrinks monotonically as
the window grows — under `expanding`, naive degrades by +0.144 at w=3 but
only +0.017 at w=60, and LEAR-LASSO by +0.071 down to +0.004 — exactly what
is expected when the correction is mostly noise and more averaging means
less of it.

**w=60 is where it breaks down.** SARIMAX flips from a small improvement to
a clear harm (1.133 → 1.173), and every model's gain shrinks. Read this
column with care under `exclude`: it drops the sample to 113 days, so part
of the movement is a change of sample rather than of method. The
`expanding` mode holds all five windows at 172 days and is therefore the
cleaner basis for comparing *across* windows; it reproduces every sign and
the same short-window optimum. `exclude` remains the stricter basis for any
single window's raw-versus-recalibrated claim, since it never mixes a
partially-estimated correction into the corrected arm.

**The improvement inverts with in-era robustness — again.** LightGBM (−0.75)
and LSTM (−0.52) gain most; SARIMAX barely moves (−0.02, and turns harmful
at w=60); **LEAR-LASSO gets WORSE at every window**. Same axis as Finding 2 of the
2026-08-04 OOD entry, and reads coherently with it: the flexible learners
absorbed the 2016-17 price LEVEL into their fitted structure, so their OOD
error is dominated by a removable constant offset. The structured models
carried less level, are closer to unbiased already, and adding a noisy
rolling intercept to a roughly-unbiased forecast just injects variance.

**naive gets worse too (0.79–0.82 → 0.81–0.93), and that is the sanity check.**
naive carries no frozen level, so it has little systematic bias to remove;
the correction can only add estimation noise. A method that improved
everything indiscriminately would be suspect.

**What may and may not be claimed.** Defensible: *for the models that
degraded worst, most of the OOD failure was a removable level shift rather
than lost relative skill, and correcting it brings both ensembles below a
naive forecast.* NOT defensible: "recalibration fixes the OOD failure". It
does not fix LEAR-LASSO or SARIMAX at all, and no single trained model
crosses the line. The v1.1-ood headline — every trained model worse than
naive out of the box — stands unchanged; this explains part of *why*.

**Caveat on what the corrected system is.** The correction consumes realized
prices from previous days. An operator would have those, so it is not a
leak — but the corrected object is no longer the frozen model. It is a
frozen model plus an adaptive intercept, i.e. a hybrid, and must be
described that way rather than as the frozen model performing better.

**Caveat on scope.** 173 live days of one market, one direction of drift
(≈2.8x upward). Nothing here shows the correction would help in a downward
shift or a flat market.

The recalibrated long frames under `data/processed/ood_recalibrated/` are
gitignored like every other regenerable processed output; they rebuild
deterministically from the committed `data/processed/ood/` frames.

---

### 2026-08-07 — Formal Lago et al. (2021) comparison and DM tests against their own forecasts

Closes the last open technical item. New files only:
`scripts/run_lago_comparison.py`, `reports/tables/lago_comparison.{csv,tex}`,
`reports/tables/dm_vs_lago.{csv,tex}`, `tests/test_lago_comparison.py`.
Nothing behind `v1.0-results` or `v1.1-ood` was touched; no model retrained.
The anchor reference was missing from `thesis/references.bib` despite
defining the whole protocol — added as `lago_forecasting_2021`.

**Protocol-equivalence checklist.**

| item | verdict | evidence |
|---|---|---|
| Dataset (EPEX-DE via epftoolbox) | MATCH | pre-verified |
| Test window 2016-01-04..2017-12-31, 728 origins | MATCH | pre-verified; alignment re-checked here |
| rMAE denominator = naive2 (p_{d-7,h}) | MATCH | every call site uses `rmae(..., m="W")` |
| asinh-median VST | **MATCH** | epftoolbox `LEAR.recalibrate` applies `scaling(..., 'Invariant')` to Ytrain and to Xtrain minus the 7 dummies (`_lear.py:64-68`, commented "Invariant, aka asinh-median transformation"). Our wrapper calls `recalibrate()` directly and reuses its `scalerX`/`scalerY` at predict, so it inherits the VST rather than reimplementing it. |
| Calibration windows / cadence | **DIFFERS (partially)** | Cadence matches — both recalibrate daily; `LEARLassoModel.fit` refits at every origin with no cadence shortcut. Windows do not: theirs spans 56/84/1092/1456 days with the LEAR Ensemble as their arithmetic mean, ours runs the **1092 window only**. So our LEAR-LASSO is comparable to their LEAR 1092 variant, and is **not** a like-for-like counterpart of their LEAR Ensemble. |
| Ensemble construction | DIFFERS (structural) | Their DNN Ensemble averages four runs of ONE model family (different hyperparameter/feature draws); ours averages different model FAMILIES. Recorded on the affected rows of the table, not only here. |

**Finding that changes how the comparison must be read: the paper's printed
LEAR numbers and the toolbox's shipped LEAR forecasts disagree.** Scoring the
shipped forecasts with our own metric code reproduces Tables 2/3 *exactly*
for all four DNNs and the DNN Ensemble — the control that proves our
alignment and metrics are correct — but for no LEAR variant:

| model | paper MAE | shipped MAE |
|---|---|---|
| LEAR 56 | 4.619 | 4.283 |
| LEAR 84 | 4.555 | 4.180 |
| LEAR 1092 | 4.108 | 3.930 |
| LEAR 1456 | 4.118 | 3.988 |
| LEAR Ensemble | 3.955 | 3.609 |

Every shipped LEAR scores BETTER than printed. This is not cosmetic: against
the printed LEAR 1092 (4.108) our LEAR-LASSO (3.899) looks comfortably ahead,
while against the shipped one (3.930) it is barely ahead. **The shipped
forecasts are the defensible basis** — identical data, identical metric code,
no transcription — and are what the week-5 checkpoint (2026-07-31) already
used. Both appear in the table as separate rows with an explicit `source`,
never merged. A test pins the discrepancy so it cannot vanish silently, and
fails if a future toolbox release makes them agree.

**Ranking on the shipped-forecast basis (rMAE).** DNN Ensemble 0.374 <
**ours regime-aware 0.390** < **ours static 0.392** < DNN 4 0.394 < LEAR
Ensemble 0.395 < DNN 3 0.406 < DNN 1 0.407 < DNN 2 0.422 < **ours LSTM
0.424** < **ours LEAR-LASSO 0.427** < LEAR 1092 0.431.

**DM tests (HAC, multivariate, both one-sided directions, 728 origins).**

- Our ensembles vs **their DNN Ensemble**: *theirs* significantly better
  (p = 0.013 regime-aware, 0.008 static). Their best system still wins.
- Our ensembles vs **DNN 4**, their best individual model: **no significant
  difference** (p = 0.322, 0.410).
- Our ensembles vs **their LEAR Ensemble**: **no significant difference**
  (p = 0.127, 0.239) — our MAE is lower (3.557 vs 3.609) but the gap does
  not survive testing.
- **LEAR-LASSO vs LEAR 1092** — the true like-for-like, same family, same
  calibration window, same protocol: **ours significantly better**
  (p = 0.0013). Both ensembles also beat LEAR 1092 (p ≈ 1e-7, 2e-6).
- LSTM loses significantly to DNN Ensemble, DNN 4 and LEAR Ensemble, and
  ties LEAR 1092 (p = 0.299).

**What may be claimed, and what may not.** Defensible: *on an identical
protocol and identical data, this thesis's ensembles are statistically
indistinguishable from the best individual model in the published benchmark
and from its LEAR ensemble, and its LEAR-LASSO significantly outperforms the
published LEAR variant it directly corresponds to.* NOT defensible: any
claim of beating the benchmark. Their DNN Ensemble is significantly better
than everything here, and that must be stated wherever the comparison
appears. This refines but does not overturn the week-5 Plan A/Plan B
decision — Plan A remains out of reach; what has improved is the precision
of the fallback claim, which is now a significance result rather than an
ordering of point estimates.

**MAPE.** Reported in the table for the published rows only. The paper's own
MAPE column runs roughly ten times its sMAPE on this market (e.g. DNN 2:
137.4 vs 15.4), driven by negative and near-zero prices — direct support for
this thesis's exclusion of MAPE, to be cited in section 3-5.

The ledger gate was bypassed twice for this work (dry run, then the real
run); both traces are below, as designed.

---

### 2026-08-07 — LSTM seed ensemble: the gap to Lago's DNN Ensemble is NOT closable, but the verdict changes

User decision (2026-08-07) to break the `v1.0-results` freeze and retrain in
order to beat Lago et al.'s DNN Ensemble (MAE 3.4135). **The stated goal was
not achieved and is now shown to be unachievable by ensembling.** A weaker
but real result was achieved instead. Nothing frozen has been modified: all
work is on branch `retrain-v2-seed-ensemble`, output in the new
`data/processed/seed_ensemble{,_val}/` namespaces, both tags intact.

**Why reweighting was ruled out first.** The L1-optimal ensemble weights
fitted directly on the TEST set — illegitimate, therefore a hard upper bound
on any weighting scheme — score MAE 3.558. The legitimate validation-fitted
regime-aware ensemble already scored 3.5569. Headroom from reweighting:
**−0.001, i.e. none.** Only stronger base learners could move the number.

**Method.** Lago's DNN Ensemble is four runs of ONE model family averaged —
variance reduction, not a better model. The same trick was applied to the
LSTM: seeds 42/43/44/45 over the full 728-origin test window, and again over
the 357-origin validation window so ensemble weights could be fitted on
validation and never on test. The seed-42 rule still holds: 42 remains the
default and every single-model result uses it; the other seeds exist only as
members of a labelled seed ensemble.

**The technique worked, and worked better than theirs.**

| member | MAE |
|---|---|
| LSTM seed 42 / 43 / 44 / 45 | 3.873 / 3.925 / 3.898 / 3.875 |
| LSTM 4-seed ensemble | **3.646** |

A 5.9% gain over the best member, against the 5.0% Lago's own ensembling
achieved (DNN 4 3.592 → DNN Ensemble 3.413).

**It still does not close the gap, and cannot.** With the seed-ensembled
LSTM as a member, the test-fitted ORACLE ensemble — cheating, an upper bound
— scores **3.5019** against their 3.4135. Still 0.088 short with weights
that already flatter us. The cause is not the ensembling but the base
learners: their best single model (DNN 4, 3.592) beats our best single model
(LSTM, 3.873) by 0.28, and averaging cannot manufacture that. Closing it
would mean building a Lago-style DNN — outside the sanctioned model list,
and effectively reimplementing their method to match their number.

**What legitimately changed** (weights fitted on validation, scored on test):

| ensemble | MAE | rMAE | DM p vs their DNN Ensemble |
|---|---|---|---|
| frozen regime-aware | 3.5569 | 0.3897 | 0.0127 — significantly worse |
| **new regime-aware** | **3.4994** | **0.3834** | **0.0803 — not significant** |
| new static | 3.5260 | 0.3863 | 0.0460 — still significantly worse |

**The fragility must be reported with the result, not after it.** The two
ensemble variants straddle 0.05: regime-aware clears it at 0.080, static
fails at 0.046. This is the same reporting rule already in force for the
regime-aware-versus-static comparison (both halves always together), and it
applies here for the same reason. The honest statement is: *our regime-aware
ensemble is not significantly worse than the published DNN Ensemble
(p = 0.080), while our static ensemble still is (p = 0.046)*. It is a
failure to reject, not a demonstration of equivalence, and 0.080 is not a
comfortable margin.

Unchanged: both new ensembles remain statistically indistinguishable from
DNN 4 and from their LEAR Ensemble, as before. No beat-the-benchmark claim
is licensed under any variant.

**Freeze status: NOT broken.** The measured upgrade is real but narrow — one
of two ensemble variants moving from p = 0.013 to p = 0.080. Promoting it
cascades into results_canonical, both DM tables, the Lago comparison, the
OOD ensemble rows, the recalibration experiment built on those, and the four
Farsi drafts that cite the old numbers. That promotion is a separate,
deliberate decision and has not been taken here.

A defect worth recording: `seed_ensemble_frame` globbed `lstm_s*.csv`, which
also matches the `lstm_seed_ensemble.csv` average this module writes into the
same directory — folding the average back in as if it were another seed. It
cancelled exactly here (averaging a set with its own mean returns that mean,
so no number changed), but a stale file from a different seed count would
have skewed the result silently. Fixed to a digit-only match; the first
version of the guarding test reproduced the same bug and had to be fixed too.

Offline suite: 362 passed, 7 deselected.

### 2026-08-20 — Seed-ensemble result promoted to a citable supplementary table

**Decision.** The 2026-08-07 seed-ensemble result is promoted from prose into
`reports/tables/seed_ensemble.{csv,tex}`, feeding thesis section **4-5-2**, and
the `retrain-v2-seed-ensemble` branch is merged into `main`. It is
**supplementary, outside `v1.0-results`** — not a new headline.

**Why now.** The result was in a precarious state. The prediction frames sat in
`data/processed/seed_ensemble{,_val}/`, which `.gitignore` excluded, so they
existed on exactly one disk. No exported table carried the numbers. The result
lived only in a commit message and the 2026-08-07 entry above. Four tests in
`tests/test_seed_ensemble.py` read those ignored CSVs and so could not run on a
clean clone.

**What changed.**

1. `.gitignore` now excepts `data/processed/seed_ensemble/` and
   `seed_ensemble_val/`, matching the existing exceptions for `baselines/`,
   `validation_preds/`, `daily_direct/`, `ood/` and `shap/`. Both directories
   are committed whole (~6.4 MB), not just the averages: the validation-window
   members are what the ensemble weights are fitted from, and the test-window
   members are what the coverage tests assert against.
2. `scripts/export_seed_ensemble.py` — new, modelled on
   `run_lago_comparison.py`. Reads the committed frames, recomputes every
   number, writes one new file pair. Nothing frozen is touched.
3. `tests/test_seed_ensemble_table.py` — pins the headline values, both halves
   of the 0.05 straddle, and that `results_canonical.csv` still carries the
   seed-42 numbers (i.e. the freeze did not silently move).
4. `thesis/outline.md` gains **4-5-1** and **4-5-2** under the existing 3pp
   budget for 4-5. Worth recording: `lago_comparison.tex` and `dm_vs_lago.tex`,
   exported 2026-08-07, had **no home in the outline at all** until now. The
   chapter-4 total stays 29pp.

**Numbers, recomputed and reconciling with 2026-08-07:**

| row | MAE | rMAE | DM p vs their DNN Ensemble |
|---|---|---|---|
| LSTM seed 42 (frozen) | 3.8734 | 0.4244 | — |
| LSTM 4-seed ensemble | 3.6460 | 0.3995 | — |
| Ensemble (static), frozen | 3.5742 | 0.3916 | 0.0082 |
| Ensemble (static), seed-ens. | 3.5260 | 0.3863 | 0.0460 |
| Ensemble (regime-aware), frozen | 3.5569 | 0.3897 | 0.0127 |
| Ensemble (regime-aware), seed-ens. | 3.4994 | 0.3834 | 0.0803 |
| Their DNN Ensemble (reference) | 3.4135 | 0.3740 | — |

The static seed-ensembled MAE (3.5260) is newly recorded here; the 2026-08-07
entry quoted only the regime-aware arm.

**Claim discipline for chapter 4 — both halves, always together.** On the
seed-ensembled LSTM the regime-aware ensemble is no longer significantly worse
than their DNN Ensemble (p=0.0803), **while the static ensemble still is**
(p=0.0460). They straddle 0.05. This is a failure to reject, not equivalence,
and the gap is not closed: the test-fitted oracle including this member scores
3.5019, still short of 3.4135, because their best single model (3.592) beats
ours (3.873) by 0.28.

**What was deliberately NOT done.** No re-freeze on 3.4994. No edit to
`results_canonical`, `dm_tests`, `dm_regime_split`, `ood_stress` or
`shap_importance`. SHAP and the OOD stress test were not re-run against the
seed-ensembled LSTM. Sections 4-2, 4-3, 4-6 and the OOD addendum all remain on
the frozen seed-42 LSTM, and 4-5-2 must say so explicitly so the two sets of
numbers are never read as one series.

**Ledger honesty.** The gate blocked the export, as designed, and was bypassed
once with a recorded reason (trace above). The table build was validated first
by an in-process dry run that wrote nothing, so only one bypass trace exists
rather than two. The ledger is still at 0.0 pages dated 2026-08-05 — that is
the real outstanding problem, and this entry does not pretend otherwise.

### 2026-08-20 — NOVELTY GATE (§5 of the new-model handoff): the primary novelty claim does NOT survive

Run per `docs/HANDOFF_new_model_design.md` §5, which requires the answers be
written here before anything is built. Answers below; the arm's §4 premise is
retracted.

**Q1 — Has calibration-window averaging been applied to NEURAL or TREE-BASED
EPF models, or only to linear/parsimonious ones? ANSWER: yes, to neural, and
it predates Lago et al.**

Marcjasz, G. (2020), "Forecasting Electricity Prices Using Deep Neural
Networks: A Robust Hyper-Parameter Selection Scheme", *Energies* 13(18):4605.
Abstract, verbatim:

> "Forecast averaging across calibration window lengths and hyper-parameter
> sets allows the proposed methodology to outperform a parameter-rich least
> absolute shrinkage and selection operator (LASSO)-estimated model and a deep
> neural network (DNN) with non-optimized hyper-parameters in terms of the
> mean absolute forecast error."

That is calibration-window averaging applied to a DNN, over three-year
out-of-sample periods in two markets, published 2020 — a year BEFORE the Lago
et al. review. Grzegorz Marcjasz is also a co-author of Lago et al. (2021)
(`thesis/references.bib` line 477), so this was not obscure to that group.

**The handoff's premise was a misattribution, and this is the important
correction.** §4 states that Lago et al. "note that combining forecasts across
calibration windows significantly outperforms the best ex-post selected single
window, then state it is unknown whether this extends beyond relatively
parsimonious regression models." The "parsimonious" sentence in their §2.1 is
about a DIFFERENT technique — long-term seasonal components, not calibration
windows:

> "By contrast, the applicability of long-term seasonal components has been
> more limited and it is unknown whether their beneficial effect is limited to
> relatively parsimonious regression models or also holds for parameter-rich
> models."

So Lago et al. never left calibration-window averaging for non-linear models
open. The open question the handoff was built on does not exist as stated.

Confirmed correct in the handoff: their DNN Ensemble does NOT vary calibration
windows — it averages four DNNs from four independent runs of the
hyperparameter/feature-selection procedure. Their LEAR Ensemble does vary
windows (8 weeks, 12 weeks, 3 years, 4 years).

**Q2 — Has regime-gated combination or regime-gated window selection been
published? ANSWER: regime-conditional EPF is a mature literature.** Markov
regime-switching with base/spike regimes is long established, and recent work
applies regime-awareness to operational EPF directly (e.g. "Regime-Aware
Conditional Neural Processes with Multi-Criteria Decision Support for
Operational Electricity Price Forecasting", arXiv:2508.00040, 2025). Novelty
here is weak. Our regime-aware ensemble remains defensible as an applied
result — it is already frozen and reported — but not as a novel mechanism.

**Q3 — Per-hour ensemble weights: NOT RESOLVED.** No direct hit either way in
this pass. Treat as unestablished rather than open; it is a small enough idea
that absence of a search hit is weak evidence.

**Residual gap.** Calibration-window averaging for TREE-BASED models
(LightGBM) turned up nothing directly. That is the only part of §4 still
standing, and it is a much narrower claim than the one the arm was designed
around.

**DECISION.** Per §5's own instruction — "If (1) is already done for
non-linear models, this arm has no novelty and should be abandoned in favour
of the contributions listed in §8" — the calibration-window-averaging novelty
arm is ABANDONED as a novelty claim. Marcjasz (2020) must now be CITED in
chapter 2 as prior art; failing to cite it while making this claim would be
the worst available outcome at the defense.

What survives, and on what basis:

- The cheap post-hoc rungs (§3: per-hour convex weights, unconstrained linear
  stacking) remain legitimate as an ABLATION, not as novelty. Under §6.6 a
  null result is still a deliverable: "combination complexity beyond global
  convex weighting yields no significant gain on this benchmark." They run on
  saved frames in minutes and touch nothing frozen.
- The LEAR multi-window sweep remains worth minutes of compute as a possible
  accuracy gain, framed as replication of Hubicka et al. (2019) / Lago et al.
  (2021), not as a new idea.
- The non-linear multi-window arm (days of compute) is NOT justified. Its
  entire rationale was the open question that does not exist.
- The §8 contributions are unaffected and remain the thesis's real
  contributions.

**Not done in this pass, and still open:** §6 (pre-registering the improvement
gate) and §10.3 (the feature-support diagnostic). §10.3 in particular is
untouched and remains cheap and decisive for the operational arm.

**Separate conflict logged for later.** Handoff §10.4 argues the operational
arm may add weather, fuel prices and ENTSO-E data because it is not compared
to the Lago benchmark. CLAUDE.md's data rule is not about protocol
equivalence — it forbids any source requiring registration, and ENTSO-E
requires a registered API key. That conflict must be resolved by an explicit
decision before the operational arm starts, not silently by whichever document
is read last.

### 2026-08-20 — PRE-REGISTRATION of the improvement gate (§6 of the new-model handoff)

Written BEFORE any rung runs, as §6 requires, and not to be revised afterwards.
The point of fixing this in advance is that a search over combination designs
will always surface something that looks better on test; without a criterion
declared beforehand there is no way to tell a real gain from the best of
several noisy draws.

**Scope.** The combination-design ladder only. The non-linear multi-window arm
was abandoned earlier today (novelty gate entry above), so this gate governs
the surviving post-hoc rungs and, if reached, the LEAR multi-window sweep.

**§6.1 Selection on validation, test used once.** Every rung is selected —
weights fitted, rung adopted or rejected — on the VALIDATION window only. The
test window is touched once per adopted rung, to report. A rung rejected on
validation does not get a test score "just to see". Looking is selection.

**§6.2 Criterion: MAE, reported with rMAE.** Improvement means lower MAE.
rMAE = MAE / (naive2 MAE on the same window); the denominator is constant
across models on a window, so the two rank identically. The threshold is set
in MAE because EUR/MWh is interpretable. Both are reported for every rung.
rMAE is NEVER compared across windows — validation and test denominators
differ. No MAPE (negative and near-zero prices). sMAPE and RMSE may be shown
but are not the criterion.

**§6.2.1 Minimum meaningful effect: 0.02 EUR/MWh validation MAE, over the
previous rung.** User's number, chosen 2026-08-20 and justified as follows:
the measured seed-to-seed spread of our own LSTM is 0.052 EUR/MWh (3.873,
3.875, 3.898, 3.925 at seeds 42/45/44/43). A floor of 0.02 sits below that
spread, so it is not a claim of resolution finer than our own run-to-run
variance, while still rejecting gains that are pure noise. Anything smaller,
bought with substantially more parameters, is a liability under Lago et al.
§2.3's critique of unablated hybrid complexity, not an achievement.

**§6.3 Adoption rule.** A rung is adopted only if BOTH: validation MAE
improves by >= 0.02 over the previous rung, AND the DM test on test data
against the previous rung is p < 0.05 in our favour AFTER the §6.4 correction.
A rung that improves the validation point estimate but fails DM is reported as
"improves point estimate, not significant" — never as an improvement.

**§6.4 Multiple comparisons — the family is declared here as SIX tests.**

The ladder is three rungs beyond the existing global convex baseline:

| rung | design |
|---|---|
| 0 | global convex weights (existing static + regime-aware) — baseline, not tested |
| 1 | per-hour convex weights (24 simplex vectors) |
| 2 | per-hour unconstrained linear stacking (negative weights permitted) |
| 3 | regime-gated per-hour weights |

Each rung is DM-tested against the previous rung. Both member sets are run
side by side (user decision 2026-08-20): (a) the frozen seed-42 members, which
isolate the combiner and match chapter 4's frozen headline, and (b) the
seed-ensembled LSTM members from the supplementary arm. 3 rungs x 2 member
sets = **6 adoption tests**, and Holm-Bonferroni is applied across exactly
those six. Raw and corrected p-values are both reported: hiding the raw ones
looks evasive, reporting only the raw ones is wrong.

Any DM test against Lago et al.'s published forecasts is DESCRIPTIVE CONTEXT,
outside the adoption family, corrected separately within itself, and labelled
as such. This split is declared now precisely so it cannot be chosen later to
suit the result.

**§6.5 Stopping rule.** Per member set, stop climbing after TWO consecutive
rungs fail §6.3. No continuing in hope that a later rung rescues the ladder —
that is the search this gate exists to prevent.

**§6.6 Report everything.** Every rung attempted appears in the final table,
adopted or not, with its validation delta and both p-values. If no rung is
adopted, that IS the result: "combination complexity beyond global convex
weighting yields no significant gain on this benchmark" — a clean answer to a
question the benchmark's own authors raised, and unattackable.

**§6.7 Straddle rule.** If two variants of one design land on opposite sides
of 0.05 — as the seed ensemble's regime-aware (p=0.080) and static (p=0.046)
variants did — both are reported together, always.

**Expected outcome, recorded in advance so it cannot be revised later.** The
test-fitted oracle over the current members scores 3.558 against the
legitimate 3.5569, i.e. roughly zero headroom for any reweighting of these
four members under global convex weights. Per-hour and unconstrained variants
sit outside that specific bound and are therefore unmeasured, but there is no
strong prior that they will clear 0.02. A null ladder is the most likely
outcome and is a publishable result under §6.6.

### 2026-08-20 — AMENDMENT to the improvement gate, made before any rung ran

Amending a pre-registration after seeing results is worthless; amending it
before anything runs is just fixing a bug in the protocol. This is the latter,
and the git history shows the ladder script did not exist when this was
written.

**The flaw.** §6.3 as pre-registered adopts a rung when "validation MAE
improves by >= 0.02". But §6.1 fits the weights ON the validation window, so
that quantity is IN-SAMPLE fit. Rung 1 (per-hour convex) has 24x the free
parameters of rung 0, rung 2 more still. Each will show a lower validation MAE
than its predecessor essentially by construction, whether or not it
generalizes. Applied literally, the gate would adopt every rung and the 0.02
floor would gate nothing.

**The fix: an inner split of the validation window, declared now.** The
364-day validation window is divided temporally, never randomly:

  inner-fit    first 273 days  — weights are fitted here
  inner-select last 91 days    — the §6.2.1 criterion is measured here

So "validation MAE" in §6.3 now means OUT-OF-SAMPLE MAE on inner-select, from
weights fitted on inner-fit only. That restores the floor's meaning: extra
parameters now have to earn their place on days they were not fitted on.

Once a rung is adopted, its weights are refitted on the FULL 364-day
validation window and scored ONCE on test, per §6.1. The test window is still
touched exactly once per adopted rung.

**Unchanged:** the 0.02 EUR/MWh floor, the six-test Holm-Bonferroni family,
the stopping rule, the straddle rule, the report-everything rule, and the
recorded expectation that a null ladder is the likely outcome.

**A consequence worth recording in advance.** Rung 3 (regime-gated per-hour)
fits 2 regimes x 24 hours = 48 weight vectors. The validation window holds
only 37 stressed days in 364, so the stressed vectors are fitted on very few
observations per hour. If rung 3 fails, thin data is a likely cause and that
should be reported as the explanation rather than as evidence that
regime-gating cannot work.

### 2026-08-20 — LADDER RESULT: combination complexity beyond global convex weighting buys nothing

Run under the pre-registered gate and its amendment, both committed before
`scripts/run_combination_ladder.py` existed. **No rung was adopted in either
member set.** Under §6.6 that is the result, not a failure to find one.

| member set | rung | val MAE (held-out) | test MAE | test rMAE | val delta vs prev | clears 0.02 | DM p raw | DM p Holm |
|---|---|---|---|---|---|---|---|---|
| frozen | 0 global convex | 3.7459 | 3.5742 | 0.3916 | — | — | — | — |
| frozen | 1 per-hour convex | 3.7695 | 3.6148 | 0.3960 | -0.0236 | no | 1.000 | 1.000 |
| frozen | 2 per-hour unconstrained | 3.7265 | 3.6095 | 0.3955 | +0.0430 | yes | 0.307 | 1.000 |
| frozen | 3 regime-gated per-hour | 3.8247 | 3.6045 | 0.3949 | -0.0982 | no | 0.339 | 1.000 |
| seedens | 0 global convex | 3.6930 | 3.5260 | 0.3863 | — | — | — | — |
| seedens | 1 per-hour convex | 3.7265 | 3.5692 | 0.3910 | -0.0336 | no | 1.000 | 1.000 |
| seedens | 2 per-hour unconstrained | 3.6793 | 3.5718 | 0.3913 | +0.0473 | yes | 0.596 | 1.000 |
| seedens | 3 regime-gated per-hour | 3.7497 | 3.5560 | 0.3900 | -0.0704 | no | 0.179 | 1.000 |

Exactly six DM tests ran, matching the pre-registered family size. Every
Holm-corrected p-value is 1.000.

**Pipeline validity check.** Rung 0 reproduces the frozen ensembles exactly —
3.5742 against `ensemble_static.csv`, and 3.5260 against the seed-ensembled
static arm recorded earlier today. The ladder is therefore measuring what it
claims to.

**The one apparent pass is not one, and this is the trap the gate was built
for.** Rung 2 clears the 0.02 floor *against rung 1*, per §6.3's
against-the-previous-rung wording. But rung 1 is WORSE than rung 0, so rung 2
is mostly recovering rung 1's own loss. Measured against rung 0, the honest
comparison, rung 2 improves held-out validation MAE by only **+0.0194
(frozen)** and **+0.0137 (seedens)** — both BELOW the pre-registered floor.
It also fails §6.3's second condition outright, at raw p = 0.307 and 0.596.
Reported here as "improves point estimate, not significant", exactly as §6.3
requires, and adopted nowhere.

**On test, every rung is worse than rung 0 in both member sets.** Per-hour and
unconstrained weighting do not merely fail to help; they actively hurt out of
sample. Rung 1 costs 0.041 (frozen) and 0.043 (seedens) test MAE.

**Mechanism.** Splitting weights by hour multiplies free parameters by 24
while dividing the observations each vector sees by 24 — roughly 266 days per
hour on the inner-fit window. The members are highly correlated forecasts of
the same quantity, so the per-hour fits chase validation noise. The
held-out inner split is the only reason this is visible: on in-sample
validation MAE every rung would have looked like an improvement, which is
precisely the failure mode the amendment was written to prevent.

**Rung 3 caveat, recorded in advance and now load-bearing.** The regime-gated
rung fits 48 vectors, and its stressed vectors see only 21 inner-fit days.
Its failure is at least partly thin data and should be reported as such, not
as evidence that regime-gating cannot work. Note the frozen regime-aware
ensemble — global weights, two regimes — remains the best model in the
project at 3.5569, so regime CONDITIONING works; it is regime conditioning
combined with per-hour splitting that collapses.

**Claim for the thesis.** "Beyond global convex weighting, additional
combination complexity — per-hour weights, unconstrained stacking, and
regime-gated per-hour weights — yields no significant improvement on this
benchmark, and degrades test accuracy." This is consistent with, and extends,
the test-fitted oracle bound: the oracle showed no headroom *within* global
convex weighting, and the ladder now shows none in three families outside it.

**Scope of the claim, stated so it is not overread.** Four members, one
market, one test period, an MAE criterion, and no intercept anywhere by
design. It does not show that combination complexity never helps in EPF.

**Status:** the ladder is complete. §11 steps 5 is done, step 7 was abandoned
by the novelty gate, and step 6 (LEAR multi-window sweep) and step 4 (§10.3
feature-support diagnostic) remain open.

### 2026-08-20 — PRE-REGISTRATION of the LEAR multi-window sweep (§11 step 6)

Declared before the sweep runs. It sits OUTSIDE the six-test combination-ladder
family declared earlier today, so it gets its own family here rather than being
quietly folded into that one.

**Design.** Run LEAR-LASSO at Lago et al.'s own four calibration windows —
56, 84, 1092 and 1456 days — and combine them by ARITHMETIC MEAN, which is
exactly their LEAR Ensemble construction. Our frozen LEAR-LASSO is the 1092
arm and is reused, not recomputed.

**Why no validation selection is needed, and why that is not a loophole.** The
arithmetic mean has zero free parameters: nothing is fitted, so there is no
selection to leak. The choice of WINDOWS is the only selection, and it is
pre-specified by the published paper rather than chosen from our data — which
is the point of replicating their set instead of searching for our own. Had we
searched windows on our data, §6.1 would apply in full and the inner split
would be mandatory.

**Family: TWO tests, declared now.**

1. LEAR window-ensemble vs our single-window LEAR-LASSO (1092) — the adoption
   test.
2. LEAR window-ensemble vs their shipped LEAR Ensemble (recomputed 3.609) —
   the like-for-like external comparison.

Holm-Bonferroni across these two. Any further comparison (e.g. against their
printed table values, or against our other models) is descriptive context,
reported with raw p-values and labelled as outside this family.

**Adoption rule.** Same as §6.3: the ensemble is adopted as an improvement
only if it lowers test MAE by >= 0.02 EUR/MWh over our 1092 arm AND the DM
test is p < 0.05 in our favour after correction. Anything else is reported as
"improves point estimate, not significant".

**Recorded expectation.** Their window averaging gained 8.2% on LEAR (3.930 ->
3.609 on shipped forecasts). Our 1092 arm is 3.899. A comparable relative gain
would land near 3.58, which would beat their LEAR Ensemble outright. That is
the upside case and it is NOT the prediction — it is what would have to happen
for this to be a win, written down now so it cannot be adjusted afterwards.

**Constraints observed.** The walk-forward window and the model's internal
calibration window must be set together: `walk_forward_splits` sizes
`train_days` from the evaluation config while `LEARLassoModel` reads its own
`calibration_window_days`, so a mismatch would hand LEAR fewer rows than it
believes it has. Data span 2012-01-09 to 2017-12-31 gives exactly 1456 days of
history before the first test origin (2016-01-04), so the longest window fits
with zero slack and no origin may be dropped to accommodate it. Seed 42
unchanged. Output goes to a new gitignored-then-excepted namespace; nothing
frozen is touched and the 1092 arm is read, never rewritten.

### 2026-08-20 — AMENDMENT: Lago's short calibration windows are INFEASIBLE in a current scikit-learn

Made before any sweep result existed — the first run failed during `fit`, so
nothing had been scored when this was written.

**What happened.** Running LEAR-LASSO at Lago et al.'s 56-day window raises,
from inside epftoolbox's `LEAR.recalibrate` -> sklearn `LassoLarsIC.fit`:

> ValueError: You are using LassoLarsIC in the case where the number of
> samples is smaller than the number of features. In this setting, getting a
> good estimate for the variance of the noise is not possible.

Our feature matrix has **247 columns**; the short windows supply 56 and 84
samples. n < p, so LassoLarsIC in scikit-learn 1.9.0 refuses to fit. Both
short windows are structurally infeasible; 1092 and 1456 are unaffected.

**Why this is a finding and not merely an obstacle.** Lago et al.'s published
LEAR Ensemble averages four windows, two of which are in exactly this n < p
regime, and their own stated mechanism is the mixing of a few SHORT windows
(1–4 months) with a few long ones. That ensemble therefore depends on
LassoLarsIC accepting n < p, which older scikit-learn did silently and current
scikit-learn refuses by design. Their headline LEAR Ensemble number is not
reproducible in a current environment without supplying a noise-variance
estimate the paper never specifies.

This belongs beside the existing contribution that their printed LEAR table
does not reproduce from their own shipped forecasts. Both are the same kind of
finding — the LEAR side of that benchmark is harder to reproduce than the DNN
side, where all five rows reproduce exactly.

**Decision (user, 2026-08-20): run the feasible subset, 1092 + 1456.** The
alternative — patching a noise-variance estimate into LassoLarsIC to force the
short windows — was rejected: it injects a hyperparameter absent from the
paper, so the result would no longer be their method and the like-for-like
comparison would be weakened rather than strengthened.

**Consequence, recorded before running so it cannot be presented as a
surprise.** With only 1092 and 1456 the ensemble mixes two LONG, highly
similar windows (3 and 4 years). Lago et al.'s mechanism is short/long
diversity, which is precisely what is missing. **A near-zero gain is the
expected outcome**, and it must NOT be reported as evidence that window
averaging fails for our LEAR — only that it cannot be tested properly here.

**Family amended from two tests to two, unchanged in count but restated:**
(1) 2-window ensemble vs our LEAR-1092; (2) 2-window ensemble vs their shipped
LEAR Ensemble — the latter now an unequal comparison (2 long windows vs their
4 mixed), and it must be labelled as such wherever it appears. The 0.02
EUR/MWh floor and the DM p < 0.05 condition are unchanged.

### 2026-08-20 — CORRECTION: window 1456 cannot cover the frozen 728-origin test window

Correcting a claim I made in the LEAR sweep pre-registration earlier today.
That entry stated the data span gives "exactly 1456 days of history before the
first test origin (2016-01-04), so the longest window fits with zero slack".
**That is wrong.**

The 1456 figure was measured from the RAW data start (2012-01-09).
`build_features` consumes the first 7 days building lag features, so the
usable feature matrix starts **2012-01-16** and only **1449** days precede the
first test origin. With a 1456-day window the walk-forward's first origin
shifts to **2016-01-11** and it yields **721 splits, not 728**.

Verified directly: window 1092 -> 728 splits, first origin 2016-01-04, 1092
train days. Window 1456 -> 721 splits, first origin 2016-01-11, 1456 train
days.

**Consequence.** Every one of Lago et al.'s four calibration windows is now
either infeasible or non-comparable in this setup:

| window | status |
|---|---|
| 56 | LassoLarsIC refuses, n=56 < p=247 |
| 84 | LassoLarsIC refuses, n=84 < p=247 |
| 1092 | runs; this is the frozen arm |
| 1456 | runs, but covers 721 of 728 origins |

So the sweep as specified by their window set **cannot be reproduced here at
all** — not for lack of compute, but for two independent structural reasons.

**How this is reported.** The 1456 arm is still worth running, on the 721
origins it can cover, with the 1092 arm restricted to those same 721 origins
so the comparison is like-for-like. Those numbers are NOT comparable to the
frozen 728-origin table and must never be placed in the same column as it.

**The finding stands and is strengthened.** The reproducibility obstacle is
now two-sided: their short windows need a LassoLarsIC behaviour current
scikit-learn refuses, and their longest window needs more history than a
lag-based feature pipeline leaves available on this dataset. Neither is a
defect in their work; both are real limits on reproducing it, and both belong
next to the existing finding that their printed LEAR table does not reproduce
from their own shipped forecasts.

**Process note.** The coverage guard in `run_lear_windows.py::_check_coverage`
would have refused to average frames over mismatched origin sets, so the wrong
number could not have propagated into a result. But the claim was written into
a pre-registration before being checked against the actual feature index, and
a guard catching a mistake is not the same as not making it.

### 2026-08-20 — LEAR SWEEP RESULT: averaging two long windows buys nothing; the gain lives in the short/long mix

Run under the pre-registration and its two amendments, all committed before
`run_lear_windows.py` produced a number. **The ensemble was NOT adopted.**

All figures below are on the **721 common origins** (2016-01-11 to
2017-12-31), NOT the frozen 728. They must never be placed in the same column
as the results table.

| model | MAE | RMSE | sMAPE | rMAE |
|---|---|---|---|---|
| our LEAR window 1092 (frozen arm) | 3.9039 | 6.4917 | 16.667 | 0.4260 |
| our LEAR window 1456 | 3.9598 | 6.4807 | 16.976 | 0.4321 |
| our LEAR window-ensemble (2 windows) | 3.9074 | 6.4534 | 16.752 | 0.4264 |
| Lago LEAR 1092 (recomputed) | 3.9298 | 6.5259 | 16.795 | 0.4306 |
| Lago LEAR Ensemble (recomputed) | **3.6091** | 6.5083 | 14.744 | 0.3954 |

**Adoption: FAILED on both conditions.** MAE moves 3.9039 -> 3.9074, a delta
of **-0.0035** — the ensemble is slightly WORSE than the single 1092 window,
against a floor of +0.02. DM vs our own 1092 arm is p=0.380, i.e. the
difference is not significant in either direction. Reported as no
improvement, adopted nowhere.

**Pre-registered test 2: we lose clearly.** Their LEAR Ensemble at 3.6091
beats our 2-window ensemble at p < 0.001 (Holm-corrected, same 721 origins,
same metric code). Unequal comparison by construction — their four mixed
windows against our two long ones — and it must be labelled as such wherever
it appears.

**The interesting result is the one that was not the objective.** Their
window averaging gains roughly 8.2% (LEAR 1092 3.9298 -> LEAR Ensemble
3.6091 on these origins). Ours gains **nothing** (-0.09%). The only structural
difference is that their set mixes short windows (56, 84) with long ones,
while ours holds two long, highly correlated windows (3 and 4 years).

That is indirect but real evidence for **their own stated mechanism**: the
benefit comes from short/long DIVERSITY — long windows fitting better, short
windows adapting faster under structural breaks — and not from averaging as
such. Averaging two windows that see nearly the same data produces nearly the
same forecast, and the ensemble inherits it. This is the same mechanism the
OOD recalibration arm measured independently, where short correction windows
(3-7 days) beat long ones.

**A secondary confirmation.** Our LEAR-1092 (3.9039) again beats their LEAR
1092 (3.9298) on identical origins with identical metric code, consistent
with the 3.899 vs 3.930 figure recorded on 728 origins. The gap between our
LEAR and their LEAR ENSEMBLE is therefore entirely the window-averaging
effect, not a weaker base model on our side.

**What must NOT be claimed.** This is not evidence that window averaging fails
for our LEAR. It was never testable here: the short windows are structurally
unreachable in this environment, so the mechanism was absent from the
experiment by construction. The honest claim is narrow — "averaging two long,
similar calibration windows yields no gain" — plus the reproducibility
finding that their full four-window ensemble cannot be rebuilt in a current
scikit-learn.

**§11 status.** Step 6 is now closed. Every step of the design arm is
complete or abandoned except step 4, the §10.3 feature-support diagnostic,
which belongs to the operational arm and is untouched. The arm has produced
three findings and zero accuracy gains, which is a defensible outcome and was
the recorded expectation.


### 2026-08-26 — WRITING DEFERRED: knowledge-graph map rebuilt, no pages banked

Recorded under the mandatory post-task reconciliation rule (NEXT_SESSION.md).
A technical task finished today and the ledger did not move, so this entry is
the required explicit deferral rather than a silent skip.

**What ran.** The interrupted 2026-08-25 `/graphify` build was resumed and
completed. That earlier run died between extraction and graph assembly,
leaving 14 chunk files and outputs still dated 2026-07-30. A coverage audit
against `.graphify_uncached.txt` found the chunks covered only 41 of 48
content files — the seven figures 10-16 had never been dispatched. Those were
extracted and the pipeline finished: **1,649 nodes, 3,323 edges, 108
hand-labelled communities**, from 120 files / ~259k words. Outputs in
`graphify-out/` (`graph.html`, `GRAPH_REPORT.md`, `graph.json`). Extraction
quality 94% EXTRACTED / 5% INFERRED / 0% AMBIGUOUS; the health check flagged
333 dangling-endpoint edges (dropped at build) and 84 isolated nodes.

Read-only with respect to the thesis: nothing outside `graphify-out/` was
modified, no frozen artifact was touched, and `git status` showed only the
untracked `.graphifyignore`.

**Why it was worth the slot.** It is analysis of the existing repo, not new
modelling, and it produced two things the writing needs. First, a traced
account of `build_features()` as the single leakage-enforcement point in the
project: all five models, every tuning script, the walk-forward runner, SHAP,
the OOD arm and the PriceCast app import it, and no path constructs a feature
matrix independently (verified — the only other `.shift()` calls in `src/`,
`scripts/` and `app/` are the documented causal ones in `ensemble.py` and
`run_ood_recalibration.py`). That is the argument section 3-4 has to make.
Second, it establishes that `src/features/pipeline.py` should be treated as
frozen alongside `v1.0-results`, which the freeze hook does NOT currently
cover — the hook intercepts Edit/Write under `reports/`, `models/` and
`data/processed` only.

**What is being deferred, and it is the same thing as always.** Zero pages.
The ledger still holds one row, `2026-08-05, 0.0`, now 21 days old. Four
drafts totalling ~3,300 words (3-5, 3-6, 3-7-1, 3-7-2) sit in
`thesis/drafts/` unconverted. The week-9 partial review opens 2026-08-31, in
five days, against a 60-page quota.

**Why deferred rather than done.** Conversion is a manual step outside this
repo: the official Amirkabir `.docx` template is not here, so no script can
paste into it or read back a real page count. The session had no window in
which that human step could happen. This is an explanation, not a
justification — it is the twelfth consecutive time technical work has been
picked up ahead of the conversion that banks pages, and the ledger gate has
now been bypassed 12 times with zero pages banked afterward.

**Correction carried out of this session.** `CLAUDE.md` records the regime
stress threshold as 62.65 EUR/MWh. The value in `configs/evaluation.yaml` is
**62.6989**, which is what figure 13's "62.70" correctly rounds to, and what
CHECKLIST.md already carries for section 3-8. The CLAUDE.md figure is the
stale one; use 62.6989 when writing. No result is affected — the calm/stressed
split is 651/77 either way.

**Standing instruction added.** `CLAUDE.md`'s session-startup section now
requires querying the graphify map before grepping cold, and re-establishing
state from CHECKLIST.md / decisions.md / the conversion queue / the ledger
rather than from NEXT_SESSION.md, whose week number had drifted three weeks
out of date.


---

Pages banked: 0 / quota 60 by 2026-08-31 | Results table: v1.0-results + v1.1-ood | Backup: [ ]


### 2026-08-07 — LEDGER GATE BYPASSED (run_lago_comparison.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_lago_comparison.py`. Reason given: closing the last open technical item: formal Lago et al. benchmark comparison + DM tests, which back the central chapter-4 claim. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_lago_comparison.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_lago_comparison.py`. Reason given: closing the last open technical item: formal Lago et al. benchmark comparison + DM tests, which back the central chapter-4 claim. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: retrain to close the gap to Lago's DNN Ensemble. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: retrain to close the gap to Lago's DNN Ensemble. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: validation-window seeds so ensemble weights can be fitted legitimately, not on test. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: seed-ensemble evaluation. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: seed-ensemble evaluation. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-07 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: user decision 2026-08-07: seed-ensemble evaluation, rerun after fixing member-glob bug. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-20 — LEDGER GATE BYPASSED (export_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `export_seed_ensemble.py`. Reason given: 2026-08-20: promoting the 2026-08-07 seed-ensemble result to a citable supplementary table so it stops living only as prose in a commit message; no new modelling, reads committed frames only. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-20 — LEDGER GATE BYPASSED (export_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `export_seed_ensemble.py`. Reason given: 2026-08-20: re-export of the same supplementary table to right-align its numeric columns; no number changes, formatting only. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.



### 2026-08-20 — LEDGER GATE BYPASSED (run_lear_windows.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_lear_windows.py`. Reason given: 2026-08-20: combine step of the pre-registered LEAR multi-window sweep; no new modelling, reads completed frames only. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-20 — LEDGER GATE BYPASSED (run_lear_windows.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_lear_windows.py`. Reason given: 2026-08-20: combine step of the pre-registered LEAR multi-window sweep; reads completed frames only. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-26 — LEDGER GATE BYPASSED (run_seed_ensemble.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_seed_ensemble.py`. Reason given: T0 follow-up: commit provenance for the 3.5019 oracle bound, which existed only as prose and a pinned test literal. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-26 — The oracle bound now has provenance: 3.5019 reproduces exactly

**What was wrong.** The T0 provenance audit found that the oracle upper bound
(3.5019) — quoted in the `seed_ensemble` caption, in the 2026-08-07 entry
above, in NEXT_SESSION.md and in the new-model handoff — existed nowhere as a
committed computation. It was prose, plus a literal pinned in
`tests/test_seed_ensemble_table.py`. That test asserted the caption matched the
literal, so it proved the caption and the constant agreed with each other and
nothing more. Every other headline number in the Lago comparison resolved to a
committed results file; this one did not.

**`run_seed_ensemble.py` could not produce it.** Its `evaluate()` fits weights
on VALIDATION by construction and documents why ("scoring a test-window seed
ensemble under test-fitted weights would be in-sample selection"). The oracle
is test-fitted, i.e. precisely what that function exists to avoid. Rerunning
the script in any existing mode would have regenerated the numbers already in
`seed_ensemble.csv` and produced no oracle; rerunning `--seeds` would have
retrained four LSTMs for hours to the same end. No retraining was needed —
the oracle is a derived statistic over forecasts already committed.

**What was added.** A `--oracle` mode on `run_seed_ensemble.py` writing
`reports/tables/oracle_bound.csv`: global convex weights (one scalar per model
across all 24 hours, no intercept — the same family as `run_combination_ladder`
rung 0) fitted directly on TEST, for both member sets.

| member set | oracle MAE | SARIMAX | LEAR-LASSO | LightGBM | LSTM |
|---|---|---|---|---|---|
| frozen seed-42 LSTM | 3.558024 | 0.1160 | 0.3038 | 0.2369 | 0.3433 |
| seed-ensembled LSTM | 3.501884 | 0.0858 | 0.2534 | 0.1265 | 0.5343 |

**Both remembered figures reproduce exactly** — 3.558 at the 2026-08-07 entry
and 3.5019 in the caption. The prose was accurate all along; it was only
unevidenced. Nothing in the substantive argument changes, and no result moved:
`seed_ensemble.csv` is byte-identical, and nothing under `v1.0-results` or
`v1.1-ood` was touched.

**The test now checks a computed value.** `ORACLE_WITH_SEED_MEMBER` (a literal)
is replaced by a read of the artifact, and the caption's figure is asserted
against it. Verified to bite: perturbing `oracle_bound.csv` to 3.4019 fails
both `test_the_gap_to_their_dnn_ensemble_is_not_closed` and
`test_the_caption_reports_both_halves_of_the_straddle`; restoring passes.
pytest 379 passed before and 379 after.

**Standing-constraint note.** The technical-remediation pack says "do not
re-run any experiment". This was a user decision (2026-08-26) taken with that
constraint in view, and it is compatible with it: no model was refit, no result
changed, and the seed-ensemble arm is explicitly outside the `v1.0-results`
freeze. The ledger gate was bypassed once, trace above.

**Ledger reconciliation: no pages banked.** This was technical work and it
banked zero template pages. The ledger still reads 0.0 dated 2026-08-05 — the
real outstanding problem, unchanged by this entry. Deferred again to the next
writing session.


### 2026-08-27 — LEDGER GATE BYPASSED (run_dm_ensembles.py)

`THESIS_SKIP_LEDGER_GATE` was set, so the ledger-progress gate did not run before `run_dm_ensembles.py`. Reason given: T0 follow-up: commit provenance for the block-bootstrap p ranges, which exist only as pinned literals. Ledger state at bypass: last entry 2026-08-05, pages_banked 0.

Recorded automatically by `src/ledger_gate.py`. The bypass exists so an urgent technical task is never hard-blocked by writing admin — but it leaves this trace, so choosing it is visible rather than free.


### 2026-08-26 — The block-bootstrap ranges now have provenance: both reproduce exactly

**Same defect as the oracle bound, same fix.** `BOOTSTRAP_P_RANGE_STRESSED`
and `BOOTSTRAP_P_RANGE_ALL` in `export_tables.py` were literals, and
`test_block_bootstrap_caption_constants_are_pinned` asserted them against
themselves. That pinned them against drift but could never catch them being
wrong. The sweep that produced them lived in `run_dm_ensembles.py`, which
printed to stdout and wrote nothing — so the ranges quoted in the
`dm_regime_split` caption, and the "not robust, therefore not claimed"
finding built on them, rested on numbers no committed artifact contained.

**What was added.** `run_dm_ensembles.py --export` writes
`reports/tables/dm_bootstrap_sensitivity.csv`: one row per (subset, block
length) with the circular block-bootstrap p and the HAC p, for all three
subsets. `reported_range()` defines the caption's range rule next to the
sweep that produces it.

| subset | days | bootstrap p by block (3/4/5/7/9/10) | HAC p | reported range |
|---|---|---|---|---|
| all | 728 | .0129 .0204 .0269 .0396 .0503 .0571 | .0226 | **0.013 - 0.057** |
| stressed | 77 | .0082 .0126 .0193 .0311 .0377 .0439 | .0063 | **0.006 - 0.044** |
| calm | 651 | .8525 .8500 .8509 .8506 .8462 .8399 | .8465 | 0.840 - 0.853 |

**Both pinned ranges reproduce exactly** — 0.006-0.044 and 0.013-0.057. As
with the oracle, the prose was accurate and only unevidenced. No caption
changed: `dm_regime_split.tex` is byte-identical, and no table was
regenerated.

**Why the range floor includes HAC.** On the stressed subset the HAC p
(0.0063) is smaller than every bootstrap value, so a floor taken from the
sweep alone would have reported 0.008 and silently narrowed the claim. The
rule is min(min bootstrap, HAC) to max bootstrap, and it now lives in one
function rather than in whoever last computed it by hand.

**Duplication, deliberately.** `export_tables.py` re-implements the range
rule instead of importing it, because scripts in this repo never import each
other (the same reason `_rel` is duplicated). A new test asserts the two
implementations agree on the same artifact, so the duplication cannot drift
silently. Verified to bite: perturbing the sweep's b=10 row fails
`test_block_bootstrap_caption_ranges_come_from_the_computed_sweep`.

pytest 379 passed before, 380 after (one new test). Ledger gate bypassed
once, trace above.

**Ledger reconciliation: no pages banked.** Technical work again, zero
template pages. Ledger still 0.0 dated 2026-08-05. Deferred to the next
writing session — this is now the second consecutive deferral, and both
class-(b) numbers from the T0 audit are closed, which removes the last
technical excuse for not writing.
