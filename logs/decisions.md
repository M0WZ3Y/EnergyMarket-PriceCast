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
- [ ] `python scripts/verify_dataset.py` outcome: ______
- [ ] `python scripts/smoke_test_energycharts.py` outcome: ______

### 2026-07-06 — Admin
- [ ] Supervisor scope sign-off sent: ______
- [ ] Week-9 partial review slot booked: ______
- [ ] Week-10 full-draft review slot booked: ______
- [ ] Faculty formalities checked (page limit, deadlines, plagiarism scan): ______
- [ ] Defense format confirmed (duration, demo allowed?): ______

---

Pages banked: 0 / quota 0 | Results table: n/a | Backup: [ ]
