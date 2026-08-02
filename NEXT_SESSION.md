# Next session — paste this in

Written 2026-08-02, second session of the day. Copy everything inside the block.

```
Resume the MSc thesis EPF project. Read logs/decisions.md from the
2026-07-31 "WEEK-5 CHECKPOINT DECIDED" entry onward — that's the state of
play. Plan A is dead, Plan B (regime-aware ensemble) leads. All 5 models
have complete committed hourly results; LSTM is best at 3.873 MAE.

FIRST, before anything else: check whether the detached validation run is
still going or has finished.

  Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
    Where-Object { $_.CommandLine -like '*run_full_baselines*' } |
    Select-Object ProcessId, CommandLine

  # and what the artifact says:
  ./.venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('data/processed/validation_preds/lightgbm.csv'); print(d['origin'].nunique(),'/357')"

A run was left going in the background (LightGBM 227/357 at handoff, then
LSTM's 357). NEVER launch a second instance while one is live — that
corrupted this exact file on 2026-08-02. If it died, just relaunch; it
resumes from the last complete origin:

  ./.venv/Scripts/python.exe scripts/run_full_baselines.py LightGBM LSTM \
    --first-origin 2015-01-05 --last-origin 2016-01-03 \
    --out-dir data/processed/validation_preds

THEN, in this order:

1. Verify both validation files are complete: 357 origins each, exactly 24
   rows per origin, no duplicate (origin,hour), no NaN.
2. Fit the ensembles — this is the week-7 contribution and everything else
   is behind it:
     ./.venv/Scripts/python.exe scripts/run_ensemble.py --dry-run   # inspect
     ./.venv/Scripts/python.exe scripts/run_ensemble.py             # writes
   It refuses to run on partial/duplicated files and aborts if either
   regime has under 20 validation days, so a clean run IS the check.
3. Commit validation_preds/*.csv plus the two ensemble frames.
4. Daily tuning, then the daily-direct walk-forward (both long):
     ./.venv/Scripts/python.exe scripts/tune_daily.py
     ./.venv/Scripts/python.exe scripts/run_daily_direct.py
5. Then: OOD stress test (unbuilt), results + DM tables via the
   export-results skill, then the v1.0-results tag.

Gotchas:
- Offline suite is `pytest -m "not epftoolbox and not network"`. Anything
  quoting only `-m "not epftoolbox"` predates the fix and hits the live API.
- Long runs need AC power AND the lid open. keep_awake() is advisory only
  on Modern Standby. standby-timeout-dc is still 300s, so a power cut
  suspends any running job within 5 minutes — setting it to 0 is a
  machine-wide policy change and remains the user's call.
- After stopping a long run for ANY reason, verify the output file at the
  interruption boundary (partial origin / duplicates / NaN) before
  resuming. Resume logic ignores a partial origin but does not delete it,
  so re-running that origin would duplicate it.
- A PostToolUse hook runs the offline suite after every edit and blocks
  edits that leave it red, so failing-test-first is not possible. Write
  the implementation first.

Full roadmap is in the approved plan at
~/.claude/plans/hmm-serialized-knuth.md
```

---

## Reference (no need to paste)

### Built this session (committed 720ff1e, pushed)

- **Daily-direct route completed.** `DailySARIMAXModel`,
  `DailyLEARLassoModel`, `DailyLSTMModel` added to `src/models/daily.py`;
  all five models registered in `scripts/run_daily_direct.py`. 24 tests in
  `tests/test_daily.py`, suite green at 108 passed / 5 deselected.
  Smoke-tested end to end on real data.
- **`scripts/run_ensemble.py`** — fits static + calm/spike weights on
  `validation_preds/`, applies to `baselines/`, writes
  `ensemble_static.csv` / `ensemble_regime.csv` in the standard long-frame
  schema. Never run on real data yet: it is blocked on the validation run.
- **`scripts/tune_daily.py`** — 50 Optuna trials against the daily target
  for DailyLightGBM and DailyLSTM. Not yet run; no study DBs exist yet, so
  it starts clean.
- Spike threshold 84.04 EUR/MWh moved from a docstring into
  `configs/evaluation.yaml` (`regime.spike_threshold_eur_mwh`).

### Decisions made this session (full text in logs/decisions.md 2026-08-02)

- **LEAR daily is a transposition, not a reimplementation.** epftoolbox's
  `LEAR` loops `for h in range(24)` and cannot take a scalar target, but
  every numerically significant piece is still epftoolbox's own —
  `scaling(...,'Invariant')`, `LassoLarsIC(aic)`, `Lasso` — verified
  against `_lear.py`. Only the 24-fit loop collapses to one.
- **Daily models get their own Optuna pass** rather than inheriting the
  hourly tuned params, so a direct-vs-aggregated gap is not partly a gap in
  tuning effort. Only LightGBM and LSTM are tuned because only they have an
  Optuna search at all: LEAR self-selects lambda via `LassoLarsIC`, and
  SARIMAX's order is fixed by config on both routes.
- **Daily SARIMAX exog** = daily mean of the same `exog_*_D0` columns,
  mirroring what the target does to the 24 prices.

### Still to build (all must precede the v1.0-results tag)

- Ensemble results (blocked on the run — step 2 above)
- Daily-direct walk-forward, all 5 models, 728 origins
- OOD stress test on live Energy-Charts data (`EnergyChartsLoader`)
- Canonical results table + DM table -> `reports/tables/` (use the
  `export-results` skill; the PreToolUse hook blocks exports once the tag
  exists, so export BEFORE tagging)

SHAP (section 4-6, 8pp, nothing imports `shap` yet) explains fitted models,
so it correctly comes AFTER the freeze.

### Traps that already bit

- Three runs lost to connected standby. AC + lid open is the real
  protection; `keep_awake()` is advisory only on Modern Standby (S0).
- A duplicate run corrupted `validation_preds/lightgbm.csv`. Check the
  process list, not the log tail — a stale log says something about the
  artifact, not the process.
- When checking that process list, note that `.venv/Scripts/python.exe` is
  a launcher stub over the Microsoft Store Python, so ONE run legitimately
  shows TWO PIDs (parent/child, the stub at ~0 CPU seconds). Distinguish a
  real second writer by CPU time and parentage, not by process count.
- `-m "not epftoolbox"` was never offline; it doesn't deselect the
  `network` marker.
