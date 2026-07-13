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
| 3 | New Energy-Charts endpoints (load + renewables): JSON parsing, 15-min→hourly resampling, schema match vs BenchmarkLoader, unit test on a sample month. Plus leakage assertion test on the feature pipeline | IN PROGRESS — current task |
| 4 | Indirect re-test: walk-forward framework consumes processed benchmark data end-to-end; LEAR sanity check vs published Lago et al. numbers doubles as a silent-data-bug detector | Scheduled |
| 7 | Pre-freeze reproducibility check: fresh environment, one model end-to-end from config — re-verifies benchmark download path from scratch | Scheduled |
| 8 or 11 | Live pipeline under real load: OOD stress test pulls a large 2026 window through EnergyChartsLoader (much bigger than week-1 smoke test) | Scheduled |
| 11 | Full live path inside PriceCast: date picker → API fetch → forecast → chart, plus CSV-upload fallback path | Scheduled |

Mitigation note: on the first successful large 2026 pull (week 8 or 11),
cache the window to data/processed/live_2026_cache.csv so the OOD test and
defense demo can run from the cached copy if the API hiccups on defense day.

---

Pages banked: 0 / quota 0 | Results table: n/a | Backup: [ ]
