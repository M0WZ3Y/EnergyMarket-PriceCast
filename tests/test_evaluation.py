"""Evaluation framework tests — src/evaluation/

Leakage-focused: every walk-forward split must keep the entire train
window strictly before its forecast origin, splits must be produced in
chronological (non-random) order, and the validation window must sit
strictly before the test window.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import diebold_mariano, mae, rmae, rmse, smape
from src.evaluation.walk_forward import (
    assert_validation_before_test,
    carve_validation_from_train,
    load_evaluation_config,
    walk_forward_splits,
)

CFG = dict(
    walk_forward=dict(calibration_window_days=10, step_days=1),
    validation=dict(validation_days=5),
)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_config_loads():
    cfg = load_evaluation_config()
    assert cfg["walk_forward"]["calibration_window_days"] > 0
    assert cfg["walk_forward"]["step_days"] >= 1
    assert cfg["validation"]["validation_days"] > 0
    assert cfg["optuna"]["n_trials"] == 50
    assert cfg["random_seed"] == 42


def test_walk_forward_splits_are_chronological_not_random():
    days = _days(20)
    splits = list(walk_forward_splits(days, cfg=CFG))
    origins = [s.origin for s in splits]
    assert origins == sorted(origins)
    # first origin is exactly the day after a full 10-day calibration window
    assert origins[0] == days[10]
    # advances one day at a time (step_days=1)
    assert list(np.diff(origins)) == [pd.Timedelta(days=1)] * (len(origins) - 1)


def test_walk_forward_train_never_reaches_origin_or_beyond():
    days = _days(20)
    for split in walk_forward_splits(days, cfg=CFG):
        assert split.train_days.max() < split.origin
        assert len(split.train_days) == CFG["walk_forward"]["calibration_window_days"]
        assert list(split.test_days) == [split.origin]
        # no overlap between train and test days
        assert not set(split.train_days).intersection(split.test_days)


def test_walk_forward_respects_first_origin_restriction():
    days = _days(30)
    first_origin = days[20]
    splits = list(walk_forward_splits(days, cfg=CFG, first_origin=first_origin))
    assert splits[0].origin == first_origin
    # training history for the restricted origin still comes from everything
    # before it, not just data after the restriction point
    assert splits[0].train_days.max() < first_origin
    assert splits[0].train_days.min() >= days[0]


def test_walk_forward_first_origin_before_calibration_floor_is_overridden():
    """Requesting an origin earlier than a full calibration window of
    history is safe but silently overridden -- the actual first origin is
    days[window], not the requested day. Documented in the docstring;
    this locks that behavior down with a test."""
    days = _days(30)
    first_origin = days[3]  # inside the window=10 floor
    splits = list(walk_forward_splits(days, cfg=CFG, first_origin=first_origin))
    assert splits[0].origin == days[CFG["walk_forward"]["calibration_window_days"]]
    assert splits[0].origin != first_origin


def test_walk_forward_skips_origins_without_full_calibration_history():
    days = _days(20)
    splits = list(walk_forward_splits(days, cfg=CFG))
    # no split should ever have been produced for the first `window` days
    assert all(s.origin >= days[10] for s in splits)


def test_carve_validation_from_train_is_trailing_slice():
    train_days = _days(30)
    fit_days, validation_days = carve_validation_from_train(train_days, cfg=CFG)
    assert len(validation_days) == CFG["validation"]["validation_days"]
    assert len(fit_days) == len(train_days) - len(validation_days)
    assert list(fit_days) + list(validation_days) == list(train_days)
    assert fit_days.max() < validation_days.min()


def test_assert_validation_before_test_passes_when_ordered():
    validation_days = _days(5)
    test_days = pd.date_range(validation_days[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    assert_validation_before_test(validation_days, test_days)  # must not raise


def test_assert_validation_before_test_raises_when_overlapping():
    validation_days = _days(10)
    test_days = _days(10)[-3:]  # overlaps the tail of validation
    with pytest.raises(AssertionError):
        assert_validation_before_test(validation_days, test_days)


def test_assert_validation_before_test_raises_on_empty_validation():
    with pytest.raises(ValueError):
        assert_validation_before_test(pd.DatetimeIndex([]), _days(5))


def test_assert_validation_before_test_raises_on_empty_test():
    with pytest.raises(ValueError):
        assert_validation_before_test(_days(5), pd.DatetimeIndex([]))


def test_carve_validation_rejects_zero_validation_days():
    """train_days[:-0] is empty and train_days[-0:] is everything -- the
    negative-zero slicing trap must raise, not silently invert fit/val."""
    cfg = dict(walk_forward=CFG["walk_forward"], validation=dict(validation_days=0))
    with pytest.raises(ValueError):
        carve_validation_from_train(_days(30), cfg=cfg)


def test_assert_validation_before_test_raises_when_test_precedes_validation():
    test_days = _days(5)
    validation_days = pd.date_range(test_days[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    with pytest.raises(AssertionError):
        assert_validation_before_test(validation_days, test_days)


def test_carve_validation_rejects_window_larger_than_train():
    train_days = _days(3)
    with pytest.raises(ValueError):
        carve_validation_from_train(train_days, cfg=CFG)


def test_evaluation_pipeline_end_to_end_with_real_config():
    """Integration test tying carve_validation_from_train + assert_validation_before_test
    + walk_forward_splits together using the REAL configs/evaluation.yaml values
    (calibration_window_days=1092, validation_days=364), not the shrunk toy CFG
    used elsewhere in this file. This is the seam the thesis's "validation
    window strictly before test window" leakage claim rests on."""
    real_cfg = load_evaluation_config()
    window = real_cfg["walk_forward"]["calibration_window_days"]
    val_days_n = real_cfg["validation"]["validation_days"]

    # enough history for: calibration window + validation window + a test period
    test_period_days = 30
    all_days = _days(window + val_days_n + test_period_days)

    train_days = all_days[: window + val_days_n]
    test_days = all_days[window + val_days_n :]

    fit_days, validation_days = carve_validation_from_train(train_days, cfg=real_cfg)
    assert_validation_before_test(validation_days, test_days)  # must not raise

    splits = list(
        walk_forward_splits(all_days, cfg=real_cfg, first_origin=test_days[0])
    )
    assert splits, "expected at least one walk-forward split over the test period"
    test_days_set = set(test_days)
    for split in splits:
        assert split.origin in test_days_set
        # the core walk-forward guarantee: training never reaches this
        # origin or any day at/after it, regardless of how that day was
        # nominally labeled (test-period days that are already in the past
        # relative to a *later* origin are legitimate history, not leakage)
        assert split.train_days.max() < split.origin
        assert len(split.train_days) == window
    # the strongest, unambiguous check: the FIRST test-period origin's
    # training window must not contain any test-period day at all, since
    # nothing in the test period has happened yet at that point in time
    assert not set(splits[0].train_days).intersection(test_days_set)


# --- metrics: sanity checks that the epftoolbox wrappers behave as expected ---


def _daily_price_frame(n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        rng.normal(40, 10, size=(n_days, 24)),
        index=pd.date_range("2020-01-01", periods=n_days, freq="D"),
    )


def test_metrics_zero_error_when_forecast_is_perfect():
    real = _daily_price_frame(30, np.random.default_rng(42))
    assert mae(real, real) == pytest.approx(0.0)
    assert rmse(real, real) == pytest.approx(0.0)
    assert smape(real, real) == pytest.approx(0.0)


def test_rmae_worse_forecast_scores_higher_than_naive_persistence():
    rng = np.random.default_rng(42)
    real = _daily_price_frame(60, rng)
    good_forecast = real + rng.normal(0, 1, size=real.shape)
    bad_forecast = real + rng.normal(0, 50, size=real.shape)
    assert rmae(real, good_forecast, m="W") < rmae(real, bad_forecast, m="W")


def test_diebold_mariano_returns_a_p_value_in_range():
    rng = np.random.default_rng(42)
    real = _daily_price_frame(60, rng)
    pred_1 = real + rng.normal(0, 1, size=real.shape)
    pred_2 = real + rng.normal(0, 5, size=real.shape)
    p = diebold_mariano(real, pred_1, pred_2)
    assert 0.0 <= p <= 1.0
