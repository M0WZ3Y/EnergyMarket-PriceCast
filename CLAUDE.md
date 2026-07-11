# Project context for Claude Code

MSc thesis, 12-week personal execution schedule (formal university registration
is 12 months — that's paperwork, not the working timeline). Day-ahead
electricity price forecasting (hourly + daily baseload), German market
(EPEX-DE benchmark / DE-LU live). Thesis body is written in Farsi; the
separate journal article deliverable is English.

## Non-negotiable constraints
- Data: `BenchmarkLoader` (epftoolbox, thesis results) and
  `EnergyChartsLoader` (live, tool only) — both keyless, one shared schema
  (`price`, `exog_*`, hourly DatetimeIndex). Never introduce a data source
  that needs registration.
- Models: exactly naive, SARIMAX, LEAR-LASSO, LightGBM, LSTM, + weighted
  ensemble. Do not add models (RF/XGBoost/SVR/GRU were deliberately cut).
- Targets: hourly (24-price D+1 vector) and daily (baseload average, both
  direct and hourly-aggregated — the comparison itself answers RQ4).
- Tuning: 50 Optuna trials per model, validation window strictly before
  test window. Walk-forward (rolling-origin) validation only — never
  random splits.
- Metrics: MAE, RMSE, sMAPE, rMAE, Diebold-Mariano. No plain MAPE
  (negative prices exist in the data).
- Leakage rule: no feature may use information after the forecast origin.
  There is an assertion test for this — keep it passing.
- Seed 42 everywhere; every non-trivial decision gets a dated entry in
  logs/decisions.md.
- After the `v1.0-results` tag exists (end of week 7): never rerun or
  modify model results — writing depends on frozen numbers.
- Week-5 checkpoint: compare LightGBM walk-forward results against Lago
  et al.'s published numbers. Outcome decides which gameplan leads —
  Plan A (match/beat the published benchmark) or Plan B (innovation-led
  defense) — see logs/decisions.md 2026-07-11 for the full gameplan.
- Week-7 priority order: static ensemble → regime-aware ensemble →
  France (only if slack; France is now third priority, not a default).

## Gameplan (2026-07-11)
- Plan A = match/beat Lago et al.'s published LEAR/DNN numbers on
  EPEX-DE (the only fair "beat" claim).
- Plan B (built regardless, weeks 5-8) = innovation-led defense, all
  sanctioned scope (not scope creep):
  - Regime-aware ensemble weighting — calm/spike weight sets switched on
    the 84.04 EUR/MWh spike threshold from the week-2 EDA.
  - Calm-vs-spike + hourly-vs-daily SHAP comparison.
  - OOD stress test: frozen benchmark-era models evaluated on live 2026
    Energy-Charts data (tool-only loader, per the data rule above).
- Stretch goal (week 7, third priority, only if ahead of schedule):
  rerun final models on France (`dataset='FR'`, config change only).
  Nord Pool was considered and rejected (system-price vs. zonal-price
  mismatch with the live API).

## Six formal assumptions (from the approved university proposal)
Must appear in thesis section 3-2. Keep them in mind when writing any
methodology code/comments: (1) stationarity, (2) data availability,
(3) data quality, (4) model generalization, (5) stable market conditions,
(6) model interpretability.

## Scope vs. the approved proposal
The proposal is generic (any of RF/tree/NN, daily-only focus, MAE/MSE/RMSE,
interpretability as an assumption not a deliverable). This project adds,
confirmed by supervisor as approved: named benchmark tied to published
literature (Lago et al. protocol), hourly actually operationalized, fixed
5-model list, live data feed, significance testing, SHAP as a real
deliverable, the PriceCast tool, and a separate journal article. Also
sanctioned (2026-07-11 gameplan decision, not scope creep): regime-aware
ensemble weighting (calm/spike weight sets) and the OOD stress test of
frozen models on live Energy-Charts data — see Gameplan section below.
Don't scope-creep beyond this list without a logged decision.

## Thesis structure (see thesis/outline.md for full detail)
100-page Farsi body, 5 chapters mapped to Amirkabir's official template:
1. Introduction (7pp) 2. Literature review (17pp) 3. Methodology (37pp)
4. Results & analysis (29pp) 5. Conclusion (10pp). Results/figures produced
by this repo map directly into chapter 3 (methodology sections 3-3
through 3-8) and chapter 4 (results, DM tests, SHAP) — see outline.md for
exact section numbers when generating tables/figures meant for the thesis.

## Conventions
- Config-driven: market/zone/splits come from configs/*.yaml, not code.
- Model wrappers implement fit/predict/save/load on a common interface.
- Figures export once, in final captioned form, to reports/figures/.
- Canonical results table (model x target x metric) auto-exports to LaTeX.
