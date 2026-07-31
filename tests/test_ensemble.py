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


def test_fit_weights_rejects_frames_overlapping_the_test_window():
    """Leakage rule: ensemble weights may never be fitted on predictions
    from the test period. _long_frame spans 2021-01-01..2021-01-20, so a
    test window starting inside it must be refused."""
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    test_days = pd.date_range("2021-01-10", periods=5, freq="D")
    with pytest.raises(AssertionError):
        fit_weights(frames, test_days=test_days)


def test_fit_weights_accepts_frames_strictly_before_the_test_window():
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    test_days = pd.date_range("2021-02-01", periods=5, freq="D")
    w = fit_weights(frames, test_days=test_days)
    assert np.isclose(sum(w.values()), 1.0)


# --------------------------------------------------------------------------
# Regime-aware ensemble (calm/spike weight sets, sanctioned 2026-07-11)
# --------------------------------------------------------------------------

from src.evaluation.ensemble import combine_regime_aware, regime_labels


def _frame_with_spike_days(model, spike_days, n_days=20, bias=0.0, seed=42):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for i, o in enumerate(origins):
        level = 200.0 if i in spike_days else 50.0
        y_true = rng.normal(level, 5, size=24)
        y_pred = y_true + rng.normal(bias, 1, size=24)
        for h in range(24):
            rows.append(dict(origin=o, hour=h, y_true=y_true[h], y_pred=y_pred[h], model=model))
    return pd.DataFrame(rows)


def test_regime_labels_use_previous_day_only():
    """Leakage rule: the regime of origin day D must be decided from the
    PREVIOUS day's realized prices (known before the forecast origin),
    never from day D's own outcome."""
    frame = _frame_with_spike_days("m", spike_days={5})
    labels = regime_labels(frame, threshold=84.04)
    origins = sorted(frame["origin"].unique())
    # day 5 is the spike day; only day 6 (which OBSERVES day 5) is
    # labeled spike -- day 5 itself was preceded by a calm day
    assert labels[origins[6]] == "spike"
    assert labels[origins[5]] == "calm"
    # first origin has no previous day inside the frame -> calm default
    assert labels[origins[0]] == "calm"


def test_combine_regime_aware_switches_weight_sets():
    frames = {
        "a": _frame_with_spike_days("a", spike_days={5}),
        "b": _frame_with_spike_days("b", spike_days={5}, seed=7),
    }
    weights = dict(
        calm={"a": 1.0, "b": 0.0},  # calm days: pure a
        spike={"a": 0.0, "b": 1.0},  # spike-regime days: pure b
    )
    out = combine_regime_aware(frames, weights, threshold=84.04)
    origins = sorted(frames["a"]["origin"].unique())

    a_piv = frames["a"].pivot(index="origin", columns="hour", values="y_pred")
    b_piv = frames["b"].pivot(index="origin", columns="hour", values="y_pred")
    out_piv = out.pivot(index="origin", columns="hour", values="y_pred")

    # day 6 follows the spike -> spike weights (pure b); day 3 calm -> pure a
    assert np.allclose(out_piv.loc[origins[6]], b_piv.loc[origins[6]])
    assert np.allclose(out_piv.loc[origins[3]], a_piv.loc[origins[3]])
    assert (out["model"] == "regime-ensemble").all()
