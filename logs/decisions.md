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

Pages banked: 0 / quota 0 | Results table: n/a | Backup: [ ]
