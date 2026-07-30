"""Week-5 checkpoint: our walk-forward results vs Lago et al. (2021) —
scripts/week5_checkpoint.py

Compares our models' full walk-forward predictions
(data/processed/baselines/*.csv) against the paper's own published
forecasts for DE (downloaded from the epftoolbox repo to
data/raw/Forecasts_DE_DNN_LEAR_ensembles.csv), on the identical test
period (2016-01-04 -> 2017-12-31, 17,472 hours), using the identical
metric implementations (src/evaluation/metrics.py -> epftoolbox.evaluation).
This is the strongest form of the comparison: same data, same period,
same metric code — no transcription from the paper's tables.

The checkpoint decides Plan A vs Plan B (logs/decisions.md 2026-07-11).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import mae, rmae, rmse, smape

PUBLISHED_FILE = REPO_ROOT / "data" / "raw" / "Forecasts_DE_DNN_LEAR_ensembles.csv"
OURS_DIR = REPO_ROOT / "data" / "processed" / "baselines"

OUR_MODELS = {
    "naive (ours)": "naive.csv",
    "SARIMAX (ours)": "sarimax.csv",
    "LEAR-LASSO (ours)": "lear_lasso.csv",
    "LightGBM (ours)": "lightgbm.csv",
}


def _ours_to_hourly(path: Path) -> pd.Series:
    """Long [origin, hour, y_true, y_pred] -> hourly Series on a
    DatetimeIndex matching the published file's convention."""
    df = pd.read_csv(path, parse_dates=["origin"])
    ts = df["origin"] + pd.to_timedelta(df["hour"], unit="h")
    return pd.Series(df["y_pred"].values, index=pd.DatetimeIndex(ts)).sort_index()


def main() -> None:
    pub = pd.read_csv(PUBLISHED_FILE, index_col=0, parse_dates=True)
    real = pub["Real price"]

    rows = []

    def add(name: str, pred: pd.Series, source: str) -> None:
        pred = pred.reindex(real.index)
        if pred.isna().any():
            raise ValueError(f"{name}: {pred.isna().sum()} missing hours after alignment")
        p_real = real.to_frame("price")
        p_pred = pred.to_frame("price")
        rows.append(
            dict(
                model=name,
                source=source,
                MAE=mae(p_real.values, p_pred.values),
                RMSE=rmse(p_real.values, p_pred.values),
                sMAPE=smape(p_real.values, p_pred.values) * 100,
                rMAE=rmae(p_real, p_pred, m="W"),
            )
        )

    for col in pub.columns:
        if col != "Real price":
            add(col, pub[col], "Lago et al. (published)")

    for name, fname in OUR_MODELS.items():
        path = OURS_DIR / fname
        if path.exists():
            add(name, _ours_to_hourly(path), "this thesis")
        else:
            print(f"[skip] {name}: {path.name} not found yet")

    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print("\nDE test period 2016-01-04 -> 2017-12-31, identical metric code\n")
    print(table.to_string(index=False))

    # Cross-check: our y_true must equal the published Real price column,
    # else the two pipelines are not looking at the same data.
    ours = pd.read_csv(OURS_DIR / "naive.csv", parse_dates=["origin"])
    ts = ours["origin"] + pd.to_timedelta(ours["hour"], unit="h")
    y_true = pd.Series(ours["y_true"].values, index=pd.DatetimeIndex(ts)).sort_index()
    max_diff = (y_true - real.reindex(y_true.index)).abs().max()
    print(f"\nsanity: max |our y_true - published Real price| = {max_diff:.6f}")


if __name__ == "__main__":
    main()
