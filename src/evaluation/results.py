"""Long-frame analysis layer — src/evaluation/results.py

Consumers of the walk-forward long frames ([origin, hour, y_true, y_pred,
model], one CSV per model under data/processed/baselines/):

  * daily_baseload  — hourly -> daily baseload aggregation (the
    "aggregated" arm of RQ4's direct-vs-aggregated comparison)
  * dm_matrix       — pairwise one-sided Diebold-Mariano p-values via
    epftoolbox's own DM implementation (multivariate 24-h version)
  * load_long_frame — typed CSV loader

Kept separate from metrics.py (thin epftoolbox wrappers) on purpose:
this module owns frame handling, not metric math.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import diebold_mariano

LONG_COLUMNS = ["origin", "hour", "y_true", "y_pred", "model"]


def load_long_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def daily_baseload(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a long hourly frame to daily baseload (mean of the 24
    hourly prices per origin day). Refuses incomplete days: a partial
    day's mean is not a baseload and would silently skew daily metrics.
    """
    counts = frame.groupby("origin").size()
    bad = counts[counts != 24]
    if len(bad):
        raise ValueError(
            f"daily_baseload: {len(bad)} origin day(s) do not have exactly "
            f"24 hourly rows (first: {bad.index[0]} with {bad.iloc[0]})"
        )
    daily = (
        frame.groupby(["origin", "model"], as_index=False)[["y_true", "y_pred"]]
        .mean()
        .loc[:, ["origin", "y_true", "y_pred", "model"]]
    )
    return daily.sort_values("origin").reset_index(drop=True)


def _pivot_24(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index()


def dm_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise one-sided DM p-values on the multivariate (24-h, L1 norm)
    loss differential. Cell [row, col] = p-value for the alternative
    hypothesis that ROW's forecasts are more accurate than COL's; the
    diagonal is NaN. All frames must cover identical origin sets.
    """
    names = list(frames)
    pivots_pred = {m: _pivot_24(frames[m], "y_pred") for m in names}
    pivots_true = {m: _pivot_24(frames[m], "y_true") for m in names}

    base_index = pivots_pred[names[0]].index
    for m in names[1:]:
        if not pivots_pred[m].index.equals(base_index):
            raise ValueError(
                f"dm_matrix: origins of '{m}' do not align with "
                f"'{names[0]}' -- frames must cover identical origin sets"
            )

    p_real = pivots_true[names[0]]
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for m1 in names:
        for m2 in names:
            if m1 == m2:
                continue
            # epftoolbox DM: small p-value supports "p_pred_2 more
            # accurate than p_pred_1" -> row model goes in as p_pred_2.
            out.loc[m1, m2] = diebold_mariano(
                p_real=p_real.values,
                p_pred_1=pivots_pred[m2].values,
                p_pred_2=pivots_pred[m1].values,
            )
    return out
