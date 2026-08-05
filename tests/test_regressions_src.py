"""Regression reproductions for defects found in the 2026-08-04 debug sweep.

Every test in this file FAILED before its fix and passes after. They are
kept together, rather than folded into the topic-named suites, so the set
of known-and-closed defects stays readable as a set.

Freeze note: none of these defects contaminated the v1.0-results numbers.
That was verified empirically before any fix landed (DE.csv is tz-naive
with exactly 24 hours on all 2184 days; the 7 frozen baseline CSVs have
728 contiguous origins and zero NaN; norm=2 has no call site). These tests
guard the code, they do not restate the frozen results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.ensemble import (  # noqa: E402
    _aligned_pivots,
    combine_forecasts,
    fit_weights,
    regime_labels,
)
from src.evaluation.metrics import diebold_mariano_hac, loss_differential  # noqa: E402
from src.evaluation.results import daily_baseload  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def long_frame(model: str, n_days: int = 4, offset: float = 0.0,
               start: str = "2020-01-01") -> pd.DataFrame:
    """Minimal walk-forward long frame: [origin, hour, y_true, y_pred, model]."""
    rows = []
    for day in pd.date_range(start, periods=n_days, freq="D"):
        for hour in range(24):
            truth = 50.0 + hour
            rows.append(
                dict(origin=day, hour=hour, y_true=truth,
                     y_pred=truth + offset, model=model)
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# src/evaluation/ensemble.py
# --------------------------------------------------------------------------
def test_fit_weights_rejects_nan_instead_of_collapsing_to_equal_weights():
    """A single NaN in a member's y_pred made SLSQP return its x0 unchanged.

    fit_weights never checked res.success, so the equal-weight STARTING
    POINT was returned as though it were the MAE-optimal solution. One
    non-converged SARIMAX hour anywhere in the validation frames would
    have silently turned the tuned ensemble into an unweighted average
    while every downstream table still called it 'MAE-optimal weights'.
    """
    good = long_frame("good", offset=0.1)
    bad = long_frame("bad", offset=5.0)

    # Sanity: on clean frames the optimiser genuinely discriminates.
    clean = fit_weights({"good": good, "bad": bad})
    assert clean["good"] > 0.9, f"optimiser did not discriminate: {clean}"

    bad_with_nan = bad.copy()
    bad_with_nan.loc[0, "y_pred"] = np.nan

    with pytest.raises(ValueError, match="(?i)nan"):
        fit_weights({"good": good, "bad": bad_with_nan})


def test_fit_weights_reports_non_convergence_rather_than_returning_x0():
    """Even without NaN, a returned solution must be a converged one.

    Guards the general case: whatever the optimiser does, fit_weights may
    not hand back its equal-weight start dressed up as an optimum.
    """
    good = long_frame("good", offset=0.1)
    bad = long_frame("bad", offset=5.0)
    w = fit_weights({"good": good, "bad": bad})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert not np.isnan(list(w.values())).any()
    # The equal-weight start must not survive as the answer here.
    assert not np.allclose(list(w.values()), 0.5), "returned the x0 start point"


def test_regime_labels_use_the_previous_calendar_day_not_the_previous_row():
    """day_max.shift(1) is POSITIONAL, so with any gap in the origin set the
    label came from an arbitrarily old day.

    Reproduction: origins 01-01, 01-02 (spike), then 01-10. The old code
    labelled 01-10 'stressed' on the basis of a spike EIGHT DAYS earlier.
    A partially re-run walk-forward, a filtered evaluation window, or
    step_days > 1 all produce exactly this shape.
    """
    frame = pd.concat([long_frame("m", n_days=2), long_frame("m", n_days=1, start="2020-01-10")])
    frame.loc[frame["origin"] == pd.Timestamp("2020-01-02"), "y_true"] = 200.0

    labels = regime_labels(frame, threshold=100.0)

    assert labels[pd.Timestamp("2020-01-02")] == "calm"
    # 01-10's previous calendar day (01-09) is absent from the frame, so its
    # regime is unknowable -- it must fall back to the documented default,
    # never inherit the 01-02 spike.
    assert labels[pd.Timestamp("2020-01-10")] == "calm", (
        "label leaked across an 8-day gap: previous ROW was used, not previous DAY"
    )


def test_aligned_pivots_rejects_members_with_different_hour_columns():
    """_aligned_pivots validated the index but never the hour columns.

    np.allclose compares POSITIONALLY, so a member whose hours are labelled
    1..24 passes the y_true agreement check, and combine_forecasts then sums
    by column LABEL -- yielding all-NaN columns instead of an error.
    """
    good = long_frame("good")
    shifted = long_frame("shifted")
    shifted["hour"] = shifted["hour"] + 1  # hours 1..24 instead of 0..23

    with pytest.raises(ValueError, match="(?i)hour"):
        _aligned_pivots({"good": good, "shifted": shifted})


def test_combine_forecasts_never_emits_nan_predictions():
    """End-to-end consequence of the hour-column hole above."""
    good = long_frame("good")
    shifted = long_frame("shifted")
    shifted["hour"] = shifted["hour"] + 1

    with pytest.raises(ValueError):
        combine_forecasts({"good": good, "shifted": shifted},
                          {"good": 0.5, "shifted": 0.5})


# --------------------------------------------------------------------------
# src/evaluation/metrics.py
# --------------------------------------------------------------------------
def test_loss_differential_norm2_matches_epftoolbox_mse_convention():
    """The docstring promises epftoolbox's convention; norm=2 did not match.

    epftoolbox _dm.py multivariate norm=2:
        d = mean(e1**2, axis=1) - mean(e2**2, axis=1)      (MSE per day)
    this module used:
        d = sqrt(mean(e1**2, axis=1)) - sqrt(mean(e2**2))  (RMSE per day)

    RMSE and MSE are not a monotone-equivalent basis for a DIFFERENCE of
    two series, so dbar, the HAC variance and the p-value all diverge.
    dm_matrix advertises 'hac' and 'uncorrected' as the same test with a
    different variance estimator -- untrue the moment norm=2 is used.
    """
    rng = np.random.default_rng(42)
    real = rng.normal(50, 20, size=(30, 24))
    # Heteroscedastic: model 1's error is concentrated on a few days, which
    # is precisely where RMSE and MSE differentials disagree.
    p1 = real + rng.normal(0, 1, size=(30, 24))
    p1[:3] += 40.0
    p2 = real + rng.normal(0, 3, size=(30, 24))

    got = loss_differential(real, p1, p2, norm=2)
    expected = np.mean((real - p1) ** 2, axis=1) - np.mean((real - p2) ** 2, axis=1)

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_diebold_mariano_hac_refuses_a_degenerate_zero_variance():
    """A constant, non-zero loss differential drove var to 0, and the
    var = max(var, 1e-12) floor then manufactured z ~ 3.2e6 and p == 0.0.

    p == 0.0 from a degenerate variance is not a statistical result. With
    every loss differential identical there is no sampling variation to
    test against, so this must raise rather than report certainty.
    """
    real = np.zeros((10, 24))
    p1 = np.ones((10, 24))    # uniformly wrong by exactly 1.0
    p2 = np.zeros((10, 24))   # exactly right

    with pytest.raises(ValueError, match="(?i)(degenerate|variance|constant)"):
        diebold_mariano_hac(real, p1, p2)


def test_diebold_mariano_hac_still_returns_p_half_for_identical_forecasts():
    """The zero-variance guard must not swallow the legitimate degenerate
    case: two IDENTICAL forecasts have dbar == 0 and p == 0.5 is correct.
    """
    real = np.zeros((10, 24))
    same = np.ones((10, 24))
    assert diebold_mariano_hac(real, same, same) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# src/evaluation/results.py
# --------------------------------------------------------------------------
def test_daily_baseload_accepts_the_multi_model_frame_it_aggregates():
    """The completeness guard counted groupby('origin') while the
    aggregation below it groups by ['origin','model'].

    A frame with k models has 24*k rows per origin, so the guard fired on
    exactly the input the function was written for, and the 'model' half of
    its own groupby was unreachable.
    """
    frame = pd.concat([long_frame("a", offset=1.0), long_frame("b", offset=2.0)])

    daily = daily_baseload(frame)

    assert set(daily["model"]) == {"a", "b"}
    assert len(daily) == 8  # 4 origins x 2 models
    a = daily[daily["model"] == "a"].iloc[0]
    assert a["y_true"] == pytest.approx(61.5)   # mean of 50..73
    assert a["y_pred"] == pytest.approx(62.5)


def test_daily_baseload_rejects_a_duplicated_hour_that_hides_a_missing_one():
    """The guard checked the row COUNT only, so an origin with hours
    [0..22, 22] -- 24 rows, one duplicated, one missing -- passed and
    produced a silently wrong baseload.
    """
    frame = long_frame("a", n_days=1)
    frame = frame[frame["hour"] != 23].copy()
    dup = frame[frame["hour"] == 22].copy()
    frame = pd.concat([frame, dup], ignore_index=True)
    assert len(frame) == 24  # the count guard is satisfied

    with pytest.raises(ValueError, match="(?i)(duplicate|hour)"):
        daily_baseload(frame)


# --------------------------------------------------------------------------
# src/evaluation/run_baselines.py
# --------------------------------------------------------------------------
def test_run_model_returns_the_documented_schema_when_no_origins_run():
    """pd.DataFrame.from_records([]) yields a (0, 0) frame with NO columns,
    not the documented [origin, hour, y_true, y_pred, model] schema.

    Downstream, load_long_frame/_pivot_24/daily_baseload then fail with an
    opaque KeyError: 'origin' far from the cause. Reachable whenever Y is
    shorter than calibration_window_days, or first_origin is past the data.
    """
    from src.evaluation.run_baselines import run_model
    from src.models import NaiveModel

    index = pd.date_range("2020-01-01", periods=40, freq="D")
    X = pd.DataFrame({"f": np.arange(40.0)}, index=index)
    Y = pd.DataFrame(
        np.tile(np.arange(24.0), (40, 1)),
        index=index,
        columns=[f"price_h{h:02d}" for h in range(24)],
    )

    out = run_model(
        "naive", NaiveModel(), X, Y,
        eval_cfg={"walk_forward": {"calibration_window_days": 7, "step_days": 1}},
        first_origin=index.max() + pd.Timedelta(days=1),
    )

    assert list(out.columns) == ["origin", "hour", "y_true", "y_pred", "model"]
    assert len(out) == 0
