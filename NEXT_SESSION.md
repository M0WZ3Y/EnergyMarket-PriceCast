# Next session — paste this in

Written 2026-08-04, end of the day the results froze. Copy everything inside
the block.

```
Resume the MSc thesis EPF project. THE CODE PHASE IS DONE. Every number the
thesis needs exists and is frozen behind two pushed tags:

  v1.0-results  benchmark-era results (hourly, daily, ensembles, DM)
  v1.1-ood      OOD addendum (frozen models on live 2026 data)

Working tree is clean and in sync with origin/main. Do NOT rerun or modify
model results. A PreToolUse hook blocks Edit/Write under reports/figures,
reports/tables, models and data/processed while those tags exist.

Read logs/decisions.md from the 2026-08-04 entries onward — that is the
state of play, including two corrections I had to make to my own earlier
claims.

WHAT'S LEFT, in priority order:

1. WRITING. This is now the binding constraint, not code. The standing
   instruction to park drafting (2026-07-30) was conditional on results
   being done — that condition is met. Pages banked: 0 of a 100-page Farsi
   body. Chapters 3 (37pp) and 4 (29pp) are fully backed by frozen numbers
   and reports/tables/*.tex. Week-9 partial review slot is booked for
   2026-08-31..09-06.
2. SHAP (thesis 4-6, 8pp). Correctly post-freeze: it explains already-fitted
   models. Nothing imports `shap` yet. Frozen models are in models/frozen/
   (gitignored, regenerate with `run_ood_stress.py --fit`, deterministic).
   Planned split: calm-vs-stressed and hourly-vs-daily comparison.
3. France stretch goal — third priority, still untouched, config change only
   (dataset='FR'). Skip it unless genuinely ahead; it competes with writing.

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

Gotchas:
- Offline suite: `pytest -m "not network"` (the hook's own invocation) —
  expect 122 passed. Set THESIS_FULL_DATA=1 to turn data-missing skips into
  failures. `-m "not epftoolbox"` alone was never offline.
- Network suite: `pytest -m "network"`, 6 tests. api.energy-charts.info
  intermittently drops TLS; the loader now retries connection errors, so a
  failure here is the API, not the code. DO NOT conclude the network is down
  — I made exactly that mistake and it cost a deliverable (decisions.md
  2026-08-04, retracted entry).
- The freeze hook only intercepts Edit/Write TOOL calls, not scripts writing
  the same paths. That is deliberate (it lets run_ood_stress.py work) but it
  means a script can still overwrite frozen artifacts. Check `git diff` after
  running anything that writes.
- Attribution required wherever live data appears: "Data: Energy-Charts
  (Fraunhofer ISE) / Bundesnetzagentur SMARD.de, CC BY 4.0".
```

---

## Reference (no need to paste)

### Schedule

Week 1 began 2026-07-06. Today is **week 5 of 12** (2026-08-03..08-09).
The v1.0-results freeze was scheduled for end of week 7 (2026-08-23) and
landed ~2.5 weeks early. The 12-week schedule ends 2026-09-27.

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

### Exported tables (reports/tables/, .tex + .csv)

| file | thesis section |
|---|---|
| results_canonical | 4-2, 4-3, 4-4 |
| dm_tests | 4-5 |
| dm_regime_split | 4-5 |
| ood_stress | 4 / limitations + discussion |

All regenerate byte-identically via `scripts/export_tables.py` — verified at
tagging. If that command ever produces a diff on the first three, a frozen
number moved: investigate before doing anything else.

### Two corrections I made to my own earlier claims (both in decisions.md)

- **"No outbound HTTPS from this workstation, environment-level."** Wrong.
  It was a flaky remote host clustering failures. Git's HTTPS worked the
  whole time. Repeated failures through one client are one observation
  repeated, not evidence of scope.
- **"Regime-aware gain is significant at the 1% level."** Withdrawn. The
  uncorrected DM ignores serial dependence; corrected p-values are an order
  of magnitude larger. A later code review then caught that my first
  correction was itself unsound (the block-length rule handed the small
  subset a shorter block despite stronger dependence).

### Scripts worth knowing

- `run_ensemble.py` — static + regime weights, refuses partial/duplicated input
- `run_dm_ensembles.py` — DM with HAC + circular block bootstrap sweep
- `export_tables.py` — all four tables
- `run_ood_stress.py --fit | --fetch | (default replay)`
- `run_full_baselines.py`, `run_daily_direct.py`, `tune_daily.py` — long runs,
  need AC power and lid open (keep_awake() is advisory on Modern Standby)

### Traps that already bit (still true)

- Never launch a second long run while one is live — that corrupted
  validation_preds/lightgbm.csv on 2026-08-02.
- `.venv/Scripts/python.exe` is a launcher stub, so ONE run legitimately
  shows TWO PIDs. Distinguish a real second writer by CPU time and parentage.
- An aggregate exit code of 0 means "nothing failed", not "everything ran".
  A chained job reported success here while its second stage never started.
  Verify by artifact, never by exit status.
- Bash tool ≠ PowerShell tool: PowerShell here-strings in Bash corrupt commit
  messages; PowerShell 5.1 has no `<` stdin redirection and no `&&`.
