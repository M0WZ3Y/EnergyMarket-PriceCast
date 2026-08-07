# Next session — paste this in

Written 2026-08-04, updated 2026-08-05 after the debug sweep, the SHAP run,
the PriceCast MVP and the page-quota setup. Copy everything inside the block.

```
Resume the MSc thesis EPF project. THE CODE IS ESSENTIALLY DONE. WRITING IS
THE ONLY THING ON THE CRITICAL PATH. Every number the thesis needs exists and
is frozen behind two pushed tags:

  v1.0-results  benchmark-era results (hourly, daily, ensembles, DM)
  v1.1-ood      OOD addendum (frozen models on live 2026 data)

Working tree clean. Do NOT rerun or modify model results. A PreToolUse hook
blocks Edit/Write under reports/figures, reports/tables, models and
data/processed while those tags exist. Suite: 342 tests (335 offline, 7
network).

MANDATORY RECONCILIATION AFTER EVERY TECHNICAL TASK -- read this first.

Whenever a technical or analysis task finishes, and BEFORE starting the next
one, reconcile the writing state:

  1. Compare thesis/drafts/ against thesis/CONVERSION_QUEUE.md. Is there a
     finished draft that has not been pasted into the Amirkabir template?
  2. Compare thesis/page_ledger.csv against what has actually been converted.
     Is there converted work whose page count was never logged?
  3. Then do ONE of two things, never neither:
       - convert and/or log it
         (./.venv/Scripts/python.exe scripts/page_quota.py --add N --note "3-6")
       - or write a dated line in logs/decisions.md saying explicitly what is
         being deferred and why.

This exists because the two drifted badly: on 2026-08-07 the ledger still
read 0.0 pages, dated 2026-08-05, while thesis/drafts/ held four finished
Farsi sections. Technical work kept getting picked up ahead of the
conversion that actually banks pages, and nothing made that visible.

A hard gate now enforces the same thing from the other side:
src/ledger_gate.py exits non-zero before any output-producing script under
scripts/ when the ledger is stale (>48h) or has not moved forward. To run
anyway, set THESIS_SKIP_LEDGER_GATE=<reason>; it runs and appends a trace to
logs/decisions.md, so bypassing is visible rather than free. The gate is
mechanical and only checks the ledger -- this rule is the judgement half,
and covers converted-but-unlogged and drafted-but-unconverted work the gate
cannot see.

Read logs/decisions.md from the 2026-08-04 entries onward for the state of
play, including corrections I had to make to my own earlier claims.

WHAT'S LEFT, in priority order:

1. WRITING. The binding constraint, and now the only one. 0 of 100 Farsi
   pages banked. The quota is finally set and tracked:
     ./.venv/Scripts/python.exe scripts/page_quota.py            # status
     ./.venv/Scripts/python.exe scripts/page_quota.py --add 12 --note "3-3"
   Required pace: 2.31pp/day to the week-9 partial review (opens 2026-08-31),
   3.03pp/day to the week-10 full-draft review (opens 2026-09-07). Both dates
   are booked with the supervisor and do not move. See thesis/page-quota.md.
   Record the page count in the official Amirkabir docx, not words written.

   Suggested order (follows the frozen numbers, waits on nothing):
   chapter 3 (37pp, sections 3-5 and 3-6 already drafted in thesis/drafts/),
   then chapter 4 (29pp, every table and figure it needs now exists),
   then chapters 1, 2, 5 (34pp).

2. Thesis 5-3 screenshot of PriceCast. Not committed. Take it at a window
   tall enough to show BOTH the accuracy warning and the chart -- a 5-3
   figure that omits the warning is the wrong figure. Next free figure
   number is 16.

3. Decide on the week-7 pre-freeze reproducibility check (logs/decisions.md
   data-source table, row 7). It never ran and the freeze already happened,
   so it can no longer be what it was designed to be. Run it late, or log it
   as deliberately skipped -- but decide, don't leave it marked Scheduled.

4. France stretch (dataset='FR'). Untouched, config change only, and it
   competes directly with 3 pages/day. Skip unless genuinely ahead.

5. The English journal article and defense/ assets have no scheduled slot
   anywhere in the log. Worth assigning one.

CLAIM DISCIPLINE — do not restate these loosely in the thesis:
- Regime-aware vs static: significant on the 77 stressed days (p 0.006-0.044
  across HAC and every bootstrap block length), NOT robustly significant over
  all 728 days (p 0.013-0.057). Both halves must appear together.
- Never quote the uncorrected epftoolbox DM p-values; they ignore serial
  dependence. Never claim 1% significance anywhere.
- LSTM vs LEAR-LASSO is a TIE (p=0.404). Do not bury this — the OOD result
  shows the neural model is also the more fragile of the two.
- The calm-day null is a sanity check, not independent corroboration.
- Ensemble weights were fitted on the window the members were tuned on;
  disclose in chapter 3 and phrase the LEAR-LASSO stress shift as a shift in
  fitted weights, not proof of intrinsic superiority.
- OOD: the regime-aware ensemble still edges static (42.17 vs 44.43) but
  98.8% of live days are one regime, so that is the stressed weight set
  applied throughout, NOT regime switching working out of distribution.
- SHAP (4-6): the explained model is a STATIC fit held fixed across the test
  period, while the models behind chapter 4's accuracy numbers recalibrate at
  every origin. Figure 10 is NOT the importance profile of the results model.
  Say so; drift in feature reliance across 2016-2017 is invisible here.
- SHAP case study (figure 15): the day is chosen by the MODEL'S OWN highest
  prediction, deliberately not by the highest realized baseload. The
  under-forecasting claim rests on the aggregate (-9.57 EUR/MWh mean signed
  error over the 73 top-decile days), not on that one day.

Gotchas:
- ALWAYS run `./.venv/Scripts/python.exe`, never bare `python`. The `python`
  on PATH is a Windows Store 3.11 without epftoolbox or lightgbm; it fails
  collection on 5 test modules with ModuleNotFoundError that looks exactly
  like broken code and is not.
- Offline suite: `pytest -m "not network"` — expect 275 passed. Network
  suite: `pytest -m "network"`, 7 tests. api.energy-charts.info intermittently
  drops TLS; the loader retries, so a failure there is the API, not the code.
  DO NOT conclude the network is down — that mistake already cost a
  deliverable once (decisions.md 2026-08-04, retracted entry).
- Missing data now FAILS by default (tests/conftest.py). Opt out only with
  THESIS_ALLOW_MISSING_DATA=1, which prints a loud banner.
- The freeze hook only intercepts Edit/Write TOOL calls, not scripts writing
  the same paths. Check `git diff` after running anything that writes.
- Attribution required wherever live data appears: "Data: Energy-Charts
  (Fraunhofer ISE) / Bundesnetzagentur SMARD.de, CC BY 4.0".
```

---

## Reference (no need to paste)

### Schedule

Week 1 began 2026-07-06. Today is **week 5 of 12** (2026-08-03..08-09); the
schedule ends 2026-09-27. Code is running ~3 weeks ahead of plan: the
`v1.0-results` freeze and SHAP (planned week 8) both landed 19 days early, and
the PriceCast MVP (planned week 11) is done. None of that slack helps unless
it is spent on pages.

### Frozen headline numbers

Benchmark test period 2016-01-04..2017-12-31, 728 origins, hourly MAE:

| model | MAE |
|---|---|
| Ensemble (regime-aware) | 3.557 |
| Ensemble (static) | 3.574 |
| LSTM | 3.873 |
| LEAR-LASSO | 3.899 |
| LightGBM | 3.968 |
| SARIMAX | 4.351 |
| naive | 7.750 |

Daily best: aggregated regime-aware ensemble 2.648; direct LEAR-LASSO 2.899.
**RQ4 answer:** aggregation beats direct modelling for LEAR-LASSO, LightGBM
and LSTM; SARIMAX is the exception and does better direct.

OOD (173 live days, 2026): every trained model exceeds rMAE 1.0 — worse than
naive (0.808). Ranking inverts: LightGBM worst (1.828), LSTM (1.520), while
LEAR-LASSO (1.087) and SARIMAX (1.145) hold up best.

### SHAP findings (section 4-6)

Renewables day-ahead forecast (`exog_2_D0`) is the largest driver at 5.86
EUR/MWh mean |SHAP|, then `price_D-1` (3.77) and the load forecast
`exog_1_D0` (3.50); weekday dummies are negligible (0.15). The hour profile is
physically coherent: load dominates the 07:00 ramp, renewables midday and the
18:00 peak, yesterday's prices the overnight hours.

**Under stress the model leans much harder on persistence** — `price_D-1`
rises 3.48 → 6.15 (+77%) from calm to stressed while fundamentals barely move.
That is an independent mechanism for why regime-aware weighting helped in 3-8.
The split reproduced 651 calm / 77 stressed by delegating to
`ensemble.regime_labels` — the same 77 days the DM regime tests use.

### Exported artifacts

| file | thesis section |
|---|---|
| reports/tables/results_canonical | 4-2, 4-3, 4-4 |
| reports/tables/dm_tests, dm_regime_split | 4-5 |
| reports/tables/ood_stress | 4 / limitations |
| reports/tables/shap_importance | 4-6 |
| reports/figures/01–09 | 3-3-3 (EDA) |
| reports/figures/10–15 | 4-6 (SHAP) |

The first four regenerate byte-identically via `scripts/export_tables.py
--dry-run`. If that ever shows a diff on the first three, a frozen number
moved: investigate before doing anything else.

### Scripts worth knowing

- `page_quota.py` — writing pace against the booked review dates
- `run_shap.py --fit | --figures-only | (default compute+export)`
- `run_ood_stress.py --fit | --fetch | (default replay)`
- `export_tables.py --dry-run` — the four frozen tables
- `run_ensemble.py`, `run_dm_ensembles.py`
- `run_full_baselines.py`, `run_daily_direct.py`, `tune_daily.py` — long runs,
  need AC power and lid open (keep_awake() is advisory on Modern Standby)
- `streamlit run app/pricecast.py` — the MVP

### Traps that already bit (still true)

- Never launch a second long run while one is live — that corrupted
  validation_preds/lightgbm.csv on 2026-08-02.
- `.venv/Scripts/python.exe` is a launcher stub, so ONE run legitimately
  shows TWO PIDs. Distinguish a real second writer by CPU time and parentage.
- An aggregate exit code of 0 means "nothing failed", not "everything ran".
  Verify by artifact, never by exit status.
- A green suite is not a sound codebase: 128/128 passed on 2026-08-04 while
  ~30 real defects sat in the untested surface. Both later sessions found
  further defects only by writing tests first and by actually opening the app.
- Bash tool ≠ PowerShell tool: PowerShell here-strings in Bash corrupt commit
  messages; PowerShell 5.1 has no `<` stdin redirection and no `&&`.
