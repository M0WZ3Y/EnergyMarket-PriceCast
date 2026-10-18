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

from src.evaluation.metrics import diebold_mariano, diebold_mariano_hac

LONG_COLUMNS = ["origin", "hour", "y_true", "y_pred", "model"]


def load_long_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def daily_baseload(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a long hourly frame to daily baseload (mean of the 24
    hourly prices per origin day). Refuses incomplete days: a partial
    day's mean is not a baseload and would silently skew daily metrics.
    """
    # The guard must group by the SAME keys the aggregation below uses. A
    # frame holding k models has 24*k rows per origin, so counting by
    # 'origin' alone rejected exactly the multi-model input this function
    # exists to aggregate -- and made the 'model' half of its own groupby
    # unreachable.
    group_keys = ["origin", "model"]
    counts = frame.groupby(group_keys).size()
    bad = counts[counts != 24]
    if len(bad):
        raise ValueError(
            f"daily_baseload: {len(bad)} (origin, model) group(s) do not have "
            f"exactly 24 hourly rows (first: {bad.index[0]} with {bad.iloc[0]})"
        )

    # A row COUNT is not completeness. Hours [0..22, 22] is 24 rows with one
    # hour duplicated and hour 23 missing, so the count guard passes while the
    # mean silently double-weights hour 22 and drops the evening peak -- a
    # wrong baseload that looks entirely normal. Require 24 DISTINCT hours.
    distinct = frame.groupby(group_keys)["hour"].nunique()
    dup = distinct[distinct != 24]
    if len(dup):
        raise ValueError(
            f"daily_baseload: {len(dup)} (origin, model) group(s) have 24 rows "
            f"but not 24 distinct hours (first: {dup.index[0]} with "
            f"{dup.iloc[0]} distinct hour(s)) -- a duplicated hour is hiding a "
            "missing one"
        )
    daily = (
        frame.groupby(["origin", "model"], as_index=False)[["y_true", "y_pred"]]
        .mean()
        .loc[:, ["origin", "y_true", "y_pred", "model"]]
    )
    return daily.sort_values("origin").reset_index(drop=True)


def _pivot_24(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index()


def dm_matrix(frames: dict[str, pd.DataFrame], method: str = "hac") -> pd.DataFrame:
    """Pairwise one-sided DM p-values on the multivariate (24-h, L1 norm)
    loss differential. Cell [row, col] = p-value for the alternative
    hypothesis that ROW's forecasts are more accurate than COL's; the
    diagonal is NaN. All frames must cover identical origin sets.

    method='hac' (default, REPORTED) applies a Newey-West correction;
    method='uncorrected' is epftoolbox's own DM, kept for comparability
    with Lago et al. Loss differentials between day-ahead price forecasts
    are serially dependent, so the uncorrected statistic is
    anti-conservative — publishing a corrected p for one comparison and an
    uncorrected p for the rest of the table would not be defensible
    (decision 2026-08-04).
    """
    if method not in ("hac", "uncorrected"):
        raise ValueError(f"dm_matrix: method must be 'hac' or 'uncorrected', got {method!r}")
    test = diebold_mariano_hac if method == "hac" else diebold_mariano
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

    # Same single-source-of-truth hazard as ensemble._aligned_pivots: p_real
    # comes from one arbitrary frame, so a stale y_true there would define
    # reality for every DM p-value in the matrix. Check, don't assume.
    for m in names[1:]:
        # Explicit atol: prices reach exactly 0.0, where allclose's relative
        # term vanishes (see the twin guard in ensemble._aligned_pivots).
        if not np.allclose(
            pivots_true[names[0]].values, pivots_true[m].values, atol=1e-6, equal_nan=True
        ):
            raise ValueError(
                f"dm_matrix: y_true of '{m}' disagrees with '{names[0]}' -- "
                "frames must share identical realized prices"
            )

    p_real = pivots_true[names[0]]
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for m1 in names:
        for m2 in names:
            if m1 == m2:
                continue
            # Both variants share epftoolbox's convention: a small p-value
            # supports "p_pred_2 more accurate than p_pred_1", so the row
            # model goes in as p_pred_2.
            out.loc[m1, m2] = test(
                p_real=p_real.values,
                p_pred_1=pivots_pred[m2].values,
                p_pred_2=pivots_pred[m1].values,
            )
    return out
