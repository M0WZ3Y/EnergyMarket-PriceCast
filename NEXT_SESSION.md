# Next session — paste this in

Written 2026-08-02, end of session. Copy everything inside the block.

```
Resume the MSc thesis EPF project. Read logs/decisions.md from the
2026-07-31 "WEEK-5 CHECKPOINT DECIDED" entry onward first — that's the
state of play.

Short version: week-5 checkpoint is decided, Plan A is dead, Plan B
(regime-aware ensemble) leads. All 5 models have complete committed
hourly results; LSTM is best at 3.873 MAE.

First task: resume the LightGBM + LSTM validation-window run. It's at
80/357 origins and is the last thing blocking the ensemble.

  ./.venv/Scripts/python.exe scripts/run_full_baselines.py LightGBM LSTM \
    --first-origin 2015-01-05 --last-origin 2016-01-03 \
    --out-dir data/processed/validation_preds

Before launching, CHECK THE PROCESS LIST for an existing instance — a
duplicate run corrupted this exact file on 2026-08-02.

Then: fit the static and regime-aware ensembles (src/evaluation/ensemble.py,
never run on real data yet; always pass test_days to fit_weights).

Gotchas: use `pytest -m "not epftoolbox and not network"` for the offline
suite. keep_awake() does NOT prevent Modern Standby — keep the laptop on
AC with the lid open for long runs.

Full roadmap is in the approved plan at
~/.claude/plans/hmm-serialized-knuth.md
```

---

## Reference (no need to paste)

**Checkpoint result** — pooled MAE, full test period, identical data
(`max |our y_true - published Real price| = 0.000000`):

| | MAE |
|---|---|
| DNN Ensemble (Lago) | 3.413 |
| LEAR Ensemble (Lago) | 3.609 |
| **LSTM (ours)** | **3.873** |
| LEAR-LASSO (ours) | 3.899 |
| LightGBM (ours) | 3.968 |
| SARIMAX (ours) | 4.351 |

No "beat the benchmark" claim exists in either year. The defensible claim
is reproduction quality: our LEAR-LASSO 3.452 vs their LEAR 1092's 3.474
in 2016, same protocol, identical data.

**Why Plan B is well-founded** — calm→volatile degradation: LightGBM
+1.523, LSTM +1.323, LEAR-LASSO +0.891. Because the three ML models
spread out rather than cluster, this reads as a general property, not a
LightGBM artefact. That is the empirical case for regime-aware weighting.

**Decided this session**
- Ensemble members: SARIMAX, LEAR-LASSO, LightGBM, LSTM. naive is a
  reference model only (rMAE denominator), not a weighted member.
- `data/processed/validation_preds/` is versioned, like `baselines/`.
- `v1.0-results` slips past week 7: daily-direct (RQ4) and the OOD stress
  test are unbuilt, and both produce results that must precede the tag.
  One freeze, not a staged v1.0/v1.1.

**Still to build**
- SARIMAX, LEAR-LASSO and LSTM daily-direct variants (`src/models/daily.py`
  has the pattern: naive + LightGBM done, tested)
- OOD stress test on live Energy-Charts data (`EnergyChartsLoader`)
- Canonical results table + DM table → `reports/tables/` (use the
  `export-results` skill; must be exported BEFORE tagging, the PreToolUse
  hook blocks exports afterwards)
- SHAP (section 4-6, 8pp — nothing imports `shap` yet). Explains fitted
  models, so it correctly comes AFTER the freeze.

**Traps that already bit**
- Three runs were lost to connected standby. AC + lid open is the real
  protection; `keep_awake()` is advisory only on Modern Standby (S0).
- A duplicate run corrupted `validation_preds/lightgbm.csv`. Repaired.
  Check the process list, not the log tail — a stale log says something
  about the artifact, not the process.
- `-m "not epftoolbox"` was never offline; it doesn't deselect the
  `network` marker, so "flaky" Energy-Charts failures were live API calls.
