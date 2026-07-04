# Project context for Claude Code

MSc thesis, 12-week schedule: day-ahead electricity price forecasting
(hourly + daily baseload), German market (EPEX-DE benchmark / DE-LU live).

## Non-negotiable constraints
- Data: `BenchmarkLoader` (epftoolbox, thesis results) and
  `EnergyChartsLoader` (live, tool only) — both keyless, one shared schema
  (`price`, `exog_*`, hourly DatetimeIndex). Never introduce a data source
  that needs registration.
- Models: exactly naive, SARIMAX, LEAR-LASSO, LightGBM, LSTM, + weighted
  ensemble. Do not add models.
- Tuning: 50 Optuna trials per model, validation window strictly before
  test window. Walk-forward (rolling-origin) validation only — never
  random splits.
- Metrics: MAE, RMSE, sMAPE, rMAE, Diebold–Mariano. No plain MAPE
  (negative prices exist).
- Leakage rule: no feature may use information after the forecast origin.
  There is an assertion test for this — keep it passing.
- Seed 42 everywhere; every non-trivial decision gets a dated entry in
  logs/decisions.md.
- After the `v1.0-results` tag exists: never rerun or modify model
  results — writing depends on frozen numbers.

## Conventions
- Config-driven: market/zone/splits come from configs/*.yaml, not code.
- Model wrappers implement fit/predict/save/load on a common interface.
- Figures export once, in final captioned form, to reports/figures/.
- Canonical results table (model x target x metric) auto-exports to LaTeX.
