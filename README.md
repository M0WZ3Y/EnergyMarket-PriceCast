# Daily & Hourly Electricity Price Forecasting (MSc thesis)

Forecasting day-ahead electricity prices (hourly + daily baseload) for the
German market using ML/DL, benchmarked per Lago et al. (2021), with the
EnergyMarket-PriceCast Streamlit tool as the applied deliverable.

## Data sources (both keyless — no registration anywhere)

| Purpose | Source | Access |
|---|---|---|
| Thesis results | Lago et al. open benchmark (EPEX-DE, 6y hourly) via `epftoolbox` | auto-download on first `read_data` call |
| Live tool feed | Energy-Charts API (Fraunhofer ISE), zone DE-LU | plain REST, CC BY 4.0 |

Attribution (required, CC BY 4.0): *Data: Energy-Charts (Fraunhofer ISE) /
Bundesnetzagentur SMARD.de.*

## Week-1 runbook

```bash
conda env create -f environment.yml
conda activate epf-thesis

# 1. Verify benchmark download + data quality
python scripts/verify_dataset.py

# 2. Smoke-test the live API
python scripts/smoke_test_energycharts.py

# 3. Run offline tests
pytest -m "not network"

# 4. Record outcomes in logs/decisions.md
```

Fallback if the epftoolbox server is down: the five benchmark CSVs are
mirrored in several public research repos; pin the DE csv into `data/raw/`
and `read_data` will use the local copy.

## Repo layout

```
configs/      data.yaml (market, splits, seed) — change market here only
src/data/     BenchmarkLoader + EnergyChartsLoader (one shared schema)
src/features/ feature pipeline (week 3)
src/models/   model wrappers on a common interface (weeks 4–7)
src/evaluation/ walk-forward validation, metrics, DM tests (week 4)
scripts/      verification & utilities
notebooks/    EDA (week 2)
app/          PriceCast Streamlit MVP (week 11)
reports/figures/  final-form figures only (frozen at export)
logs/decisions.md research log — one dated entry per decision
thesis/ defense/  writing and presentation assets
```

## Standing rules

1. Results freeze end of week 7 (`v1.0-results` tag) — no reruns after.
2. 45–60 min thesis writing daily before code (week 2+), page quotas tracked.
3. One canonical results table, auto-exported to LaTeX.
4. Fixed tuning budgets: 50 Optuna trials per model, then stop.
5. Seed 42, pinned environment, every decision logged.
