"""Feature pipeline contract + leakage tests.

Uses a fully deterministic synthetic series (price/exog encode their own
day-number and hour) so every lag column's expected value is known exactly,
not just approximately.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pipeline import build_features, load_feature_config

N_DAYS = 15
CFG = dict(
    price_lag_days=[1, 2, 3, 7],
    exog_lag_days=[1, 7],
    exog_current_day=True,
    weekday_dummies=True,
)


def _synthetic_df(n_days: int = N_DAYS) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=24 * n_days, freq="1h")
    day_num = (idx.normalize() - idx.normalize()[0]).days
    hour = idx.hour
    return pd.DataFrame(
        {
            "price": day_num * 100 + hour,
            "exog_1": day_num * 10 + hour,
            "exog_2": day_num + hour * 0.1,
        },
        index=idx,
    )


def test_config_loads():
    cfg = load_feature_config()
    assert cfg["price_lag_days"] == [1, 2, 3, 7]
    assert cfg["exog_lag_days"] == [1, 7]


def test_shape_and_first_target_day():
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    # 4 price-lag blocks + 2 exog * (2 lag + 1 current) blocks + 7 dow dummies
    assert X.shape[1] == 4 * 24 + 2 * (2 * 24 + 24) + 7
    assert Y.shape[1] == 24
    assert len(X) == len(Y)
    # First 7 days lack full D-7 history and must be dropped.
    assert X.index[0] == pd.Timestamp("2020-01-08")


def test_price_lags_exact_values():
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    d = X.index[0]
    day_num = (d - df.index.normalize()[0]).days
    for lag in [1, 2, 3, 7]:
        for h in [0, 5, 23]:
            expected = (day_num - lag) * 100 + h
            assert X.loc[d, f"price_D-{lag}_h{h:02d}"] == expected


def test_exog_lags_and_current_day():
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    d = X.index[0]
    day_num = (d - df.index.normalize()[0]).days
    for h in [0, 12, 23]:
        assert X.loc[d, f"exog_1_D0_h{h:02d}"] == day_num * 10 + h
        assert X.loc[d, f"exog_1_D-1_h{h:02d}"] == (day_num - 1) * 10 + h
        assert X.loc[d, f"exog_1_D-7_h{h:02d}"] == (day_num - 7) * 10 + h


def test_weekday_dummies_one_hot():
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    dow_cols = [c for c in X.columns if c.startswith("dow_")]
    assert len(dow_cols) == 7
    row_sums = X[dow_cols].sum(axis=1)
    assert (row_sums == 1.0).all()
    for d in X.index:
        assert X.loc[d, f"dow_{d.dayofweek}"] == 1.0


def test_target_equals_actual_target_day_price():
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    for d in X.index:
        day_num = (d - df.index.normalize()[0]).days
        for h in [0, 5, 23]:
            assert Y.loc[d, f"y_h{h:02d}"] == day_num * 100 + h


def test_no_feature_reads_target_day_price():
    """Hard leakage guard: no X column may source the target day's own
    price (the exact quantity being predicted)."""
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    d = X.index[-1]
    day_num = (d - df.index.normalize()[0]).days
    target_day_price_values = {day_num * 100 + h for h in range(24)}
    price_cols = [c for c in X.columns if c.startswith("price_")]
    leaked = [c for c in price_cols if X.loc[d, c] in target_day_price_values]
    assert leaked == [], f"columns leaking target-day price: {leaked}"


def test_no_feature_reads_beyond_current_day_exog():
    """Exog D0 (current/target day) is legal (known-in-advance forecast);
    nothing may reference exog from *after* the target day."""
    df = _synthetic_df()
    X, Y = build_features(df, CFG)
    exog_cols = [c for c in X.columns if c.startswith("exog_")]
    allowed_days = {"D0", "D-1", "D-7"}
    for c in exog_cols:
        tag = c.split("_")[2]
        assert tag in allowed_days, f"unexpected exog source tag in {c}"


def test_incomplete_boundary_days_dropped_not_filled():
    """A gap in the source data must shrink the usable window, never be
    silently filled (which could leak an interpolated future/past value)."""
    df = _synthetic_df()
    df = df.drop(df.index[24 * 10 + 3])  # remove one hour mid-series
    X, Y = build_features(df, CFG)
    assert X.isna().sum().sum() == 0
    assert Y.isna().sum().sum() == 0
    # The day containing the missing hour, and any day whose lag window
    # touches it, must be absent.
    affected_day = pd.Timestamp("2020-01-11")
    assert affected_day not in X.index


def test_full_day_gap_does_not_misalign_lags():
    """A whole missing calendar day must become an explicit gap (dropping
    every target day whose lag window touches it), never a positional
    shift that silently relabels a farther day as a closer lag."""
    df = _synthetic_df(n_days=20)
    gap_day = pd.Timestamp("2020-01-11")  # day_num 10
    df = df.drop(df.loc[gap_day : gap_day + pd.Timedelta(hours=23)].index)

    X, Y = build_features(df, CFG)

    # A target day touches the gap when target_day - lag == gap_day for one
    # of the configured lags, i.e. target_day == gap_day + lag. Those rows
    # must be dropped, not filled with a mislabeled neighbor.
    for lag in CFG["price_lag_days"]:
        touching_day = gap_day + pd.Timedelta(days=lag)
        assert touching_day not in X.index, (
            f"{touching_day} should be dropped (D-{lag} lag touches the gap)"
        )

    # A target day far enough past the gap that no configured lag reaches
    # back into it must still have exact, correctly-dated lag values —
    # proving shift() did not silently pull in a farther day.
    far_day = pd.Timestamp("2020-01-20")  # day_num 19; D-7 -> day_num 12
    assert far_day in X.index
    day_num = (far_day - df.index.normalize().min()).days
    for lag in [1, 2, 3, 7]:
        for h in [0, 12, 23]:
            expected = (day_num - lag) * 100 + h
            assert X.loc[far_day, f"price_D-{lag}_h{h:02d}"] == expected
