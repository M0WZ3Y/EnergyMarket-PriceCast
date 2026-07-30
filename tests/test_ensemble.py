"""Tests for src/evaluation/ensemble.py — static weighted ensemble over
walk-forward long frames (week-7 scope: static first, regime-aware later)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ensemble import combine_forecasts, fit_weights


def _long_frame(model, n_days=20, bias=0.0, noise=1.0, seed=42):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for o in origins:
        y_true = rng.normal(50, 10, size=24)
        y_pred = y_true + rng.normal(bias, noise, size=24)
        for h in range(24):
            rows.append(dict(origin=o, hour=h, y_true=y_true[h], y_pred=y_pred[h], model=model))
    return pd.DataFrame(rows)


def test_combine_is_weighted_mean_of_predictions():
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    out = combine_forecasts(frames, weights={"a": 0.75, "b": 0.25})
    expected = 0.75 * frames["a"]["y_pred"].values + 0.25 * frames["b"]["y_pred"].values
    assert np.allclose(out["y_pred"].values, expected)
    assert (out["model"] == "ensemble").all()
    # y_true carried through unchanged
    assert np.allclose(out["y_true"].values, frames["a"]["y_true"].values)


def test_combine_normalizes_weights():
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    out1 = combine_forecasts(frames, weights={"a": 3.0, "b": 1.0})
    out2 = combine_forecasts(frames, weights={"a": 0.75, "b": 0.25})
    assert np.allclose(out1["y_pred"].values, out2["y_pred"].values)


def test_combine_rejects_misaligned_frames():
    frames = {"a": _long_frame("a"), "b": _long_frame("b").iloc[24:]}
    with pytest.raises(ValueError, match="align"):
        combine_forecasts(frames, weights={"a": 0.5, "b": 0.5})


def test_combine_rejects_negative_or_zero_weights():
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    with pytest.raises(ValueError, match="weight"):
        combine_forecasts(frames, weights={"a": 1.5, "b": -0.5})
    with pytest.raises(ValueError, match="weight"):
        combine_forecasts(frames, weights={"a": 0.0, "b": 0.0})


def test_fit_weights_favors_the_accurate_model():
    frames = {
        "good": _long_frame("good", bias=0.0, noise=0.5),
        "bad": _long_frame("bad", bias=8.0, noise=0.5, seed=7),
    }
    w = fit_weights(frames)
    assert set(w) == {"good", "bad"}
    assert np.isclose(sum(w.values()), 1.0)
    assert w["good"] > 0.9


def test_fit_weights_beats_or_matches_best_single_model_in_sample():
    frames = {
        "a": _long_frame("a", bias=2.0, noise=2.0),
        "b": _long_frame("b", bias=-2.0, noise=2.0, seed=7),
    }
    w = fit_weights(frames)
    ens = combine_forecasts(frames, weights=w)
    mae_ens = (ens.y_true - ens.y_pred).abs().mean()
    best_single = min((f.y_true - f.y_pred).abs().mean() for f in frames.values())
    assert mae_ens <= best_single + 1e-9
