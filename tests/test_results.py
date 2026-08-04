"""Tests for src/evaluation/results.py — long-frame analysis layer:
daily-baseload aggregation (RQ4 groundwork) and pairwise DM tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.results import daily_baseload, dm_matrix, load_long_frame


def _long_frame(model: str, n_days: int = 30, bias: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Synthetic walk-forward long frame for one model.

    y_true is drawn from a FIXED seed independent of `seed`: every model
    records the same market outcome, as real walk-forward frames do. Only
    y_pred varies with `seed`. dm_matrix() enforces this agreement.
    """
    truth_rng = np.random.default_rng(20210101)
    pred_rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for o in origins:
        y_true = truth_rng.normal(50, 10, size=24)
        y_pred = y_true + pred_rng.normal(bias, 1, size=24)
        for h in range(24):
            rows.append(dict(origin=o, hour=h, y_true=y_true[h], y_pred=y_pred[h], model=model))
    return pd.DataFrame(rows)


def test_daily_baseload_is_mean_of_24_hours():
    frame = _long_frame("m")
    daily = daily_baseload(frame)
    assert daily.columns.tolist() == ["origin", "y_true", "y_pred", "model"]
    assert len(daily) == 30
    one_day = frame[frame.origin == frame.origin.iloc[0]]
    assert np.isclose(daily.iloc[0]["y_true"], one_day["y_true"].mean())
    assert np.isclose(daily.iloc[0]["y_pred"], one_day["y_pred"].mean())


def test_daily_baseload_requires_complete_days():
    frame = _long_frame("m").iloc[:-1]  # drop one hour of the last day
    with pytest.raises(ValueError, match="24"):
        daily_baseload(frame)


def test_dm_matrix_shape_and_diagonal():
    frames = {"a": _long_frame("a", bias=0.0), "b": _long_frame("b", bias=5.0, seed=7)}
    dm = dm_matrix(frames)
    assert dm.shape == (2, 2)
    assert np.isnan(dm.loc["a", "a"])
    # a (unbiased) should be significantly better than b (biased):
    # p-value for "a more accurate than b" small, reverse large
    assert dm.loc["a", "b"] < 0.05
    assert dm.loc["b", "a"] > 0.5


def test_dm_matrix_rejects_frames_whose_y_true_disagrees():
    """p_real is read from one arbitrary frame, so a stale y_true there
    would define reality for every p-value in the matrix. Twin of the
    ensemble guard; without this test the error path is untested and a
    refactor could delete it silently."""
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    frames["b"] = frames["b"].copy()
    frames["b"].loc[0, "y_true"] += 5.0
    with pytest.raises(ValueError, match="y_true"):
        dm_matrix(frames)


def test_dm_matrix_requires_aligned_origins():
    frames = {"a": _long_frame("a"), "b": _long_frame("b").iloc[24:]}  # b missing day 1
    with pytest.raises(ValueError, match="align"):
        dm_matrix(frames)


def test_load_long_frame_roundtrip(tmp_path):
    frame = _long_frame("m")
    p = tmp_path / "m.csv"
    frame.to_csv(p, index=False)
    loaded = load_long_frame(p)
    assert loaded["origin"].dtype.kind == "M"  # parsed as datetime
    assert len(loaded) == len(frame)
