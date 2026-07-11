"""Shared feature pipeline for the EPF thesis models.

Builds the price/exogenous lag feature matrix used by every model in the
benchmark (naive, SARIMAX, LEAR-LASSO, LightGBM, LSTM) so all five are
trained on one consistently-derived feature set. The lag convention matches
Lago et al. (2021) / epftoolbox's LEAR implementation exactly — cross-checked
against epftoolbox.models._lear.LEAR._build_and_split_XYs — so results stay
comparable with the published benchmark. See configs/features.yaml for the
lag config and logs/decisions.md (week 3) for the rationale.

Forecast setting: at origin day O (~noon), the model predicts all 24 hourly
prices of target day O+1. Every feature/column name below is expressed in
terms of target_day (= O+1) for direct comparability with epftoolbox:
  - price_D-1_h.. .. price_D-7_h..  : past days' realized 24h price vectors
  - exog*_D-1_h.., exog*_D-7_h..    : past days' exogenous vectors
  - exog*_D0_h..                    : target day's own exogenous vector
                                       (legal — exog_1/exog_2 are day-ahead
                                       load/generation forecasts, known
                                       before the forecast origin)
  - dow_0 .. dow_6                  : target day's weekday, one-hot
  - y_h00 .. y_h23                  : target day's actual 24h price (label)

No column ever reads the target day's own price — that is exactly the
label being predicted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "features.yaml"


def load_feature_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _pivot_to_daily_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape an hourly (timestamp -> columns) frame into one row per
    calendar day, with columns '{col}_h00' .. '{col}_h23'. Days with any
    missing hour get NaN in the affected cells (handled by dropna downstream
    — never filled, so no future/past value ever leaks across days).

    The result is reindexed to a contiguous daily calendar (no gaps) before
    lag columns are built. build_features() later uses positional .shift()
    to build lag columns, which shifts by row position, not by calendar
    date — if a whole day were missing from the index, a positional shift
    would silently pull in a day from further back and mislabel it as a
    closer lag. Reindexing turns a missing day into an explicit all-NaN row
    instead, so it is dropped rather than mislabeled.
    """
    day = df.index.normalize()
    hour = df.index.hour
    wide_parts = []
    for col in df.columns:
        wide = df[col].groupby([day, hour]).first().unstack(level=-1)
        wide.columns = [f"{col}_h{h:02d}" for h in wide.columns]
        wide_parts.append(wide)
    wide_df = pd.concat(wide_parts, axis=1).sort_index()
    full_calendar = pd.date_range(wide_df.index.min(), wide_df.index.max(), freq="D")
    wide_df = wide_df.reindex(full_calendar)
    wide_df.index.name = "day"
    return wide_df


def build_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X, Y) from a standardized hourly DataFrame.

    df must have an hourly DatetimeIndex and columns ['price', 'exog_1',
    'exog_2', ...] (the BenchmarkLoader/EnergyChartsLoader schema).

    Returns
    -------
    X : DataFrame indexed by target_day (midnight timestamps), lag feature
        columns as described in the module docstring.
    Y : DataFrame indexed by target_day, columns y_h00..y_h23 — the actual
        24 hourly prices of that day (the forecasting target).
    """
    if cfg is None:
        cfg = load_feature_config()

    exog_cols = [c for c in df.columns if c.startswith("exog_")]
    price_lag_days = cfg["price_lag_days"]
    exog_lag_days = cfg["exog_lag_days"]

    wide = _pivot_to_daily_wide(df)
    hours = [f"h{h:02d}" for h in range(24)]

    X_parts = []
    for past_day in price_lag_days:
        block = wide[[f"price_{h}" for h in hours]].shift(past_day)
        block.columns = [f"price_D-{past_day}_{h}" for h in hours]
        X_parts.append(block)

    for exog in exog_cols:
        for past_day in exog_lag_days:
            block = wide[[f"{exog}_{h}" for h in hours]].shift(past_day)
            block.columns = [f"{exog}_D-{past_day}_{h}" for h in hours]
            X_parts.append(block)
        if cfg["exog_current_day"]:
            block = wide[[f"{exog}_{h}" for h in hours]]
            block.columns = [f"{exog}_D0_{h}" for h in hours]
            X_parts.append(block)

    if cfg["weekday_dummies"]:
        dow = pd.Categorical(wide.index.dayofweek, categories=range(7))
        dummies = pd.get_dummies(dow, prefix="dow").astype(float)
        dummies.index = wide.index
        X_parts.append(dummies)

    X = pd.concat(X_parts, axis=1)

    Y = wide[[f"price_{h}" for h in hours]].copy()
    Y.columns = [f"y_{h}" for h in hours]

    # Drop days without full lag history (start of series) or with any
    # missing hour anywhere in the required window (both sides at once —
    # a row only survives if every feature AND its label are fully known).
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X.loc[valid], Y.loc[valid]

    X.index.name = "target_day"
    Y.index.name = "target_day"
    return X, Y
