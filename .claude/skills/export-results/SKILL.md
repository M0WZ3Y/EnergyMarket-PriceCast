---
name: export-results
description: Export a thesis figure or results table following the project's captioning, naming, and LaTeX conventions. Use whenever generating any figure or table destined for the thesis or reports/.
---

# Export thesis figures and tables

Conventions for anything exported to `reports/`. These come from CLAUDE.md and
`thesis/outline.md` — follow them exactly; a figure/table is exported **once,
in final form**, not iterated in place.

## Before exporting

1. Open `thesis/outline.md` and identify the exact section number the artifact
   feeds (e.g. EDA figures → 3-3-3, canonical results tables → 4-2/4-3, DM
   tests → 4-5, SHAP → 4-6). Never guess section numbers.
2. Confirm the data respects the leakage rule (nothing after the forecast
   origin) and was produced with `random_seed: 42`.
3. If the `v1.0-results` tag exists, results artifacts are frozen — do not
   re-export; the PreToolUse hook will block it anyway.

## Figures → `reports/figures/`

- Naming: sequential two-digit prefix + snake_case slug, `NN_short_slug.png`
  (existing: `01_price_distribution.png` … `09_daily_baseload.png`). Check the
  directory and take the next free number; use a letter suffix (`07b_...`)
  only for a companion panel of an existing figure.
- Export via `matplotlib` `savefig` with `dpi=300, bbox_inches="tight"`.
- Self-contained: axis labels with units (EUR/MWh, hour), legend, readable at
  single-column thesis width. Figure text stays English (the Farsi caption is
  written in the thesis document, not baked into the image).
- After exporting, state the figure's target thesis section and a one-line
  caption suggestion, and note it in the session summary so it can be logged.

## Tables → `reports/tables/`

- The canonical results table is model × target × metric with exactly:
  MAE, RMSE, sMAPE, rMAE. Never plain MAPE (negative prices). Diebold–Mariano
  results are a separate table (feeds 4-5).
- Export both `.tex` (via `DataFrame.to_latex`, `booktabs` style, 2-decimal
  floats) and a `.csv` alongside for inspection.
- Models appear in the fixed order: naive, SARIMAX, LEAR-LASSO, LightGBM,
  LSTM, ensemble. Targets: hourly, daily-direct, daily-aggregated.
- Best value per metric column is bolded in the LaTeX output.
