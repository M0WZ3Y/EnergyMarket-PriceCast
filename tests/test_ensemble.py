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
    """Synthetic walk-forward long frame for one member.

    y_true is drawn from a FIXED seed independent of `seed`, so every member
    sees identical realized prices — as real walk-forward frames do, since
    they all record the same market outcome. Only y_pred varies with `seed`.
    (Before 2026-08-04 each member also randomized y_true, which no real
    frame does and which masked the y_true-agreement guard.)
    """
    truth_rng = np.random.default_rng(20210101)
    pred_rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for o in origins:
        y_true = truth_rng.normal(50, 10, size=24)
        y_pred = y_true + pred_rng.normal(bias, noise, size=24)
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


def test_combine_rejects_frames_whose_y_true_disagrees():
    """A member carrying different realized prices must be refused.

    The truth column is read from one arbitrary member, so a stale or
    shifted y_true there would silently define reality for every metric
    downstream — the shape of the 2026-08-02 concurrent-writer corruption.
    """
    frames = {"a": _long_frame("a"), "b": _long_frame("b", seed=7)}
    frames["b"] = frames["b"].copy()
    frames["b"].loc[0, "y_true"] += 5.0  # one corrupted cell is enough
    with pytest.raises(ValueError, match="y_true"):
        combine_forecasts(frames, weights={"a": 0.5, "b": 0.5})


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
# Regime-aware ensemble (calm/stressed weight sets, sanctioned 2026-07-11;
# 'spike' renamed to 'stressed' on 2026-08-04 with the threshold move to
# mean+1.5*std -- at ~1.5 sigma the label marks an elevated day, not a spike)
# --------------------------------------------------------------------------

import pytest

from src.evaluation.ensemble import combine_regime_aware, regime_labels

# Synthetic separator only. Deliberately NOT the configured production
# threshold: these tests pin the labeling/switching contract, and must not
# start failing whenever the calibrated threshold is retuned.
SYNTHETIC_THRESHOLD = 100.0


def _frame_with_stressed_days(model, stressed_days, n_days=20, bias=0.0, seed=42):
    """As _long_frame, but with chosen days pushed to a high price level.

    y_true again comes from a fixed seed so all members share realized
    prices; only y_pred varies with `seed`.
    """
    truth_rng = np.random.default_rng(20210101)
    pred_rng = np.random.default_rng(seed)
    origins = pd.date_range("2021-01-01", periods=n_days, freq="D")
    rows = []
    for i, o in enumerate(origins):
        level = 200.0 if i in stressed_days else 50.0
        y_true = truth_rng.normal(level, 5, size=24)
        y_pred = y_true + pred_rng.normal(bias, 1, size=24)
        for h in range(24):
            rows.append(dict(origin=o, hour=h, y_true=y_true[h], y_pred=y_pred[h], model=model))
    return pd.DataFrame(rows)


def test_regime_labels_use_previous_day_only():
    """Leakage rule: the regime of origin day D must be decided from the
    PREVIOUS day's realized prices (known before the forecast origin),
    never from day D's own outcome."""
    frame = _frame_with_stressed_days("m", stressed_days={5})
    labels = regime_labels(frame, threshold=SYNTHETIC_THRESHOLD)
    origins = sorted(frame["origin"].unique())
    # day 5 is the high-price day; only day 6 (which OBSERVES day 5) is
    # labeled stressed -- day 5 itself was preceded by a calm day
    assert labels[origins[6]] == "stressed"
    assert labels[origins[5]] == "calm"
    # first origin has no previous day inside the frame -> calm default
    assert labels[origins[0]] == "calm"


def test_combine_regime_aware_switches_weight_sets():
    frames = {
        "a": _frame_with_stressed_days("a", stressed_days={5}),
        "b": _frame_with_stressed_days("b", stressed_days={5}, seed=7),
    }
    weights = dict(
        calm={"a": 1.0, "b": 0.0},  # calm days: pure a
        stressed={"a": 0.0, "b": 1.0},  # stressed-regime days: pure b
    )
    out = combine_regime_aware(frames, weights, threshold=SYNTHETIC_THRESHOLD)
    origins = sorted(frames["a"]["origin"].unique())

    a_piv = frames["a"].pivot(index="origin", columns="hour", values="y_pred")
    b_piv = frames["b"].pivot(index="origin", columns="hour", values="y_pred")
    out_piv = out.pivot(index="origin", columns="hour", values="y_pred")

    # day 6 follows the stressed day -> stressed weights (pure b); day 3 -> pure a
    assert np.allclose(out_piv.loc[origins[6]], b_piv.loc[origins[6]])
    assert np.allclose(out_piv.loc[origins[3]], a_piv.loc[origins[3]])
    assert (out["model"] == "regime-ensemble").all()


def test_combine_regime_aware_rejects_legacy_spike_key():
    """A caller still using the pre-2026-08-04 'spike' key must fail loudly.

    Without this, an unmigrated caller would fall through to the calm
    weight set for every day and silently produce a static ensemble
    mislabeled as regime-aware -- a wrong number that still looks right.
    """
    frames = {
        "a": _frame_with_stressed_days("a", stressed_days={5}),
        "b": _frame_with_stressed_days("b", stressed_days={5}, seed=7),
    }
    legacy = dict(calm={"a": 1.0, "b": 0.0}, spike={"a": 0.0, "b": 1.0})
    with pytest.raises(ValueError, match="calm.*stressed"):
        combine_regime_aware(frames, legacy, threshold=SYNTHETIC_THRESHOLD)
