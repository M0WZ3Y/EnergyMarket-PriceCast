"""Diebold-Mariano tests on the ensembles — scripts/run_dm_ensembles.py

Answers the week-7 question the aggregate MAE cannot: is the regime-aware
ensemble's edge over the static one real, and where does it come from?

Reports three things (decision 2026-08-04):
  1. the full pairwise DM matrix over every model + both ensembles;
  2. regime-aware vs static over all test days;
  3. the same split by regime.

(3) is the load-bearing one. The two ensembles differ only slightly on calm
days (their calm weights are close to the static weights), so 651 near-
identical days dilute the aggregate differential toward non-significance.
Splitting by regime shows whether the gain is localized to the days the
mechanism is designed to act on -- a diffuse noise advantage would not be.

Reads only committed artifacts, so it is reproducible after the freeze:
  data/processed/baselines/*.csv  (test period, 2016-01-04 -> 2017-12-31)

The regime threshold comes from configs/evaluation.yaml, never hardcoded —
the labeling rule is previous-day-only (see regime_labels()).

Usage:
    python scripts/run_dm_ensembles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ensemble import regime_labels
from src.evaluation.metrics import diebold_mariano, mae
from src.evaluation.results import dm_matrix, load_long_frame
from src.evaluation.walk_forward import load_evaluation_config

TEST_DIR = REPO_ROOT / "data" / "processed" / "baselines"

FILES = {
    "Ensemble (regime-aware)": "ensemble_regime.csv",
    "Ensemble (static)": "ensemble_static.csv",
    "LSTM": "lstm.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "SARIMAX": "sarimax.csv",
    "naive": "naive.csv",
}

REGIME = "Ensemble (regime-aware)"
STATIC = "Ensemble (static)"


def _report(piv, truth, days, name: str) -> float:
    """MAE of both ensembles on `days`, plus the one-sided DM p-value for
    'regime-aware is more accurate than static'."""
    idx = pd.DatetimeIndex(days)
    real = truth.loc[idx]
    pr, ps = piv[REGIME].loc[idx], piv[STATIC].loc[idx]

    mae_r = mae(real.values.ravel(), pr.values.ravel())
    mae_s = mae(real.values.ravel(), ps.values.ravel())
    # epftoolbox DM: small p supports "p_pred_2 more accurate than p_pred_1",
    # so the regime-aware forecasts go in as p_pred_2.
    p = diebold_mariano(p_real=real.values, p_pred_1=ps.values, p_pred_2=pr.values)

    diff = np.abs(pr.values - ps.values)
    print(f"\n{name}  (n={len(idx)} days)")
    print(
        f"  MAE regime-aware {mae_r:.4f} | static {mae_s:.4f} | "
        f"delta {mae_r - mae_s:+.4f} ({100 * (mae_r - mae_s) / mae_s:+.2f}%)"
    )
    print(f"  mean |pred difference| {diff.mean():.4f} EUR/MWh, max {diff.max():.4f}")
    print(f"  DM p (regime-aware more accurate than static): {p:.4f}")
    return p


def main() -> None:
    frames = {name: load_long_frame(TEST_DIR / f) for name, f in FILES.items()}

    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    labels = regime_labels(frames["LSTM"], threshold=threshold)
    stressed = sorted(o for o, lab in labels.items() if lab == "stressed")
    calm = sorted(o for o, lab in labels.items() if lab == "calm")
    print(
        f"test split: calm {len(calm)}, stressed {len(stressed)} "
        f"(threshold {threshold} EUR/MWh, previous-day rule)"
    )

    print("\n" + "=" * 78)
    print("Pairwise one-sided DM p-values, multivariate (24-h vector, L1 norm)")
    print("cell[row, col] = P(row MORE accurate than col); small => row wins")
    print("=" * 78)
    print(dm_matrix(frames).round(4).to_string(na_rep="  --  "))

    piv = {
        m: f.pivot(index="origin", columns="hour", values="y_pred").sort_index()
        for m, f in frames.items()
    }
    truth = frames["LSTM"].pivot(index="origin", columns="hour", values="y_true").sort_index()

    print("\n" + "=" * 78)
    print("FOCUSED: regime-aware vs static")
    print("=" * 78)
    _report(piv, truth, sorted(labels), "ALL test days")
    _report(piv, truth, stressed, "STRESSED days only (where the mechanism acts)")
    _report(piv, truth, calm, "CALM days only (near-null check)")


if __name__ == "__main__":
    main()
