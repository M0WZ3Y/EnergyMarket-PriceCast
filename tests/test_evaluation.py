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

from src.evaluation.metrics import (
    diebold_mariano,
    diebold_mariano_hac,
    loss_differential,
    mae,
    rmae,
    rmse,
    smape,
)
from scipy import stats
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
    """Leakage oracle reconstructed INDEPENDENTLY of the production assert.

    walk_forward_splits() asserts `train_days.max() < origin` internally, so
    a test that only re-states that assertion can never be the thing that
    fails (the generator raises first) and disappears entirely under
    `python -O`. What is checked here instead is the emitted split data
    against the source calendar: every training day must be one of the
    `window` calendar days immediately preceding the origin -- an equality
    that is strictly stronger than the inequality the generator asserts and
    that survives assertion-stripping of src/.
    """
    days = _days(20)
    window = CFG["walk_forward"]["calibration_window_days"]
    calendar = list(days)
    splits = list(walk_forward_splits(days, cfg=CFG))
    assert splits, "expected at least one split over a 20-day calendar"

    for split in splits:
        pos = calendar.index(split.origin)
        # the exact expected train window, derived from the calendar alone
        assert list(split.train_days) == calendar[pos - window : pos]
        # every individual train timestamp is strictly earlier than the origin
        assert all(day < split.origin for day in split.train_days)
        assert split.train_days.max() < split.origin
        assert len(split.train_days) == window
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
    calendar = list(all_days)
    for split in splits:
        assert split.origin in test_days_set
        # the core walk-forward guarantee: training never reaches this
        # origin or any day at/after it, regardless of how that day was
        # nominally labeled (test-period days that are already in the past
        # relative to a *later* origin are legitimate history, not leakage).
        # Reconstructed from the calendar rather than restating the
        # production assert inside walk_forward_splits(), so this check is
        # still doing work when src/ runs with assertions stripped.
        pos = calendar.index(split.origin)
        assert list(split.train_days) == calendar[pos - window : pos]
        assert all(day < split.origin for day in split.train_days)
        assert split.train_days.max() < split.origin
        assert len(split.train_days) == window
        assert list(split.test_days) == [split.origin]
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
    """Range alone is a vacuous oracle: a function returning a constant 0.5,
    or one that reports the REVERSED one-sided p-value, satisfies it. With
    pred_1 (sigma=1) overwhelmingly more accurate than pred_2 (sigma=5) the
    one-sided test must actually detect the difference, and the reversed
    argument order must land on the other side of 0.5 -- the same oracle
    pattern dm_matrix is held to in tests/test_results.py.

    Direction follows epftoolbox's convention, the one dm_matrix documents
    and relies on: a SMALL p-value supports "p_pred_2 is more accurate than
    p_pred_1". So the accurate forecast passed as p_pred_2 must reject, and
    swapping the two must not.
    """
    rng = np.random.default_rng(42)
    real = _daily_price_frame(60, rng)
    pred_1 = real + rng.normal(0, 1, size=real.shape)
    pred_2 = real + rng.normal(0, 5, size=real.shape)

    # accurate forecast in the p_pred_2 slot -> the alternative is true
    p = diebold_mariano(real, pred_2, pred_1)
    assert 0.0 <= p <= 1.0
    assert p < 0.05, "pred_1 is 5x more accurate; the test must reject"

    p_reversed = diebold_mariano(real, pred_1, pred_2)
    assert 0.0 <= p_reversed <= 1.0
    assert p_reversed > 0.5, "the one-sided direction must not be inverted"


# --- loss_differential / diebold_mariano_hac: direct tests ----------------


def _dm_inputs(n_days=90, sigma_1=1.0, sigma_2=5.0, seed=42):
    rng = np.random.default_rng(seed)
    real = _daily_price_frame(n_days, rng)
    pred_1 = real + rng.normal(0, sigma_1, size=real.shape)
    pred_2 = real + rng.normal(0, sigma_2, size=real.shape)
    return real, pred_1, pred_2


def test_loss_differential_norm1_is_the_daily_mean_absolute_error_gap():
    real, pred_1, pred_2 = _dm_inputs()
    d = loss_differential(real, pred_1, pred_2, norm=1)
    expected = (
        np.abs(real.to_numpy() - pred_1.to_numpy()).mean(axis=1)
        - np.abs(real.to_numpy() - pred_2.to_numpy()).mean(axis=1)
    )
    assert d.shape == (len(real),)  # one number per day, not per hour
    np.testing.assert_allclose(d, expected)
    # pred_1 is the better forecast, so its loss is lower and d is negative
    assert d.mean() < 0


def test_loss_differential_norm2_is_mean_squared_not_rmse():
    """epftoolbox's _dm.py uses MEAN SQUARED error for norm=2. Taking the
    root first would silently change dbar, the HAC variance and the
    p-value, so the exact basis is pinned here, not just its sign."""
    real, pred_1, pred_2 = _dm_inputs()
    d = loss_differential(real, pred_1, pred_2, norm=2)
    e1 = np.abs(real.to_numpy() - pred_1.to_numpy())
    e2 = np.abs(real.to_numpy() - pred_2.to_numpy())
    np.testing.assert_allclose(d, (e1**2).mean(axis=1) - (e2**2).mean(axis=1))
    # and it is NOT the RMSE-based differential
    rmse_based = np.sqrt((e1**2).mean(axis=1)) - np.sqrt((e2**2).mean(axis=1))
    assert not np.allclose(d, rmse_based)


def test_loss_differential_is_exactly_zero_for_identical_forecasts():
    real, pred_1, _ = _dm_inputs()
    np.testing.assert_allclose(loss_differential(real, pred_1, pred_1), 0.0)


def test_loss_differential_is_antisymmetric_in_its_two_forecasts():
    real, pred_1, pred_2 = _dm_inputs()
    np.testing.assert_allclose(
        loss_differential(real, pred_1, pred_2),
        -loss_differential(real, pred_2, pred_1),
    )


@pytest.mark.parametrize("bad_norm", [0, 3, -1, 1.5])
def test_loss_differential_rejects_norms_other_than_1_or_2(bad_norm):
    real, pred_1, pred_2 = _dm_inputs(n_days=10)
    with pytest.raises(ValueError, match="norm must be 1 or 2"):
        loss_differential(real, pred_1, pred_2, norm=bad_norm)


def test_diebold_mariano_hac_detects_a_real_accuracy_difference():
    """Same convention as the uncorrected wrapper (small p supports
    'p_pred_2 more accurate'), asserted DIRECTLY rather than only through
    dm_matrix."""
    real, pred_1, pred_2 = _dm_inputs()  # pred_1 accurate, pred_2 noisy
    p = diebold_mariano_hac(real, pred_2, pred_1)
    assert 0.0 <= p <= 1.0
    assert p < 0.05
    p_reversed = diebold_mariano_hac(real, pred_1, pred_2)
    assert 0.0 <= p_reversed <= 1.0
    assert p_reversed > 0.5


def test_diebold_mariano_hac_returns_one_half_for_identical_forecasts():
    """Zero differential everywhere means 'neither model is better' -- the
    one-sided answer is 0.5, never a manufactured 0.0 from a variance floor."""
    real, pred_1, _ = _dm_inputs()
    assert diebold_mariano_hac(real, pred_1, pred_1) == 0.5


def test_diebold_mariano_hac_rejects_a_degenerate_constant_differential():
    real, _, _ = _dm_inputs(n_days=20)
    pred_1 = real.copy()
    pred_2 = real + 1.0  # constant loss gap, zero variance across days
    with pytest.raises(ValueError, match="degenerate loss differential"):
        diebold_mariano_hac(real, pred_1, pred_2)


@pytest.mark.parametrize("n_days", [1, 2])
def test_diebold_mariano_hac_refuses_fewer_than_three_days(n_days):
    """Two days cannot support a Newey-West variance estimate; the guard
    must refuse rather than return a number nobody can interpret."""
    real, pred_1, pred_2 = _dm_inputs(n_days=n_days)
    with pytest.raises(ValueError, match="at least 3 days"):
        diebold_mariano_hac(real, pred_1, pred_2, norm=1)


def test_diebold_mariano_hac_accepts_exactly_three_days():
    """The boundary the n < 3 guard defines: 3 days must go through."""
    real, pred_1, pred_2 = _dm_inputs(n_days=3)
    p = diebold_mariano_hac(real, pred_1, pred_2)
    assert 0.0 <= p <= 1.0


def test_diebold_mariano_hac_bandwidth_zero_reproduces_the_iid_statistic():
    """bandwidth=0 drops every autocovariance term, so the HAC variance
    collapses to the plain iid variance of the loss differential. That gives
    an exact closed-form oracle for the returned p-value -- if the kernel
    weighting, the /n, or the one-sided tail were wrong, this fails."""
    real, pred_1, pred_2 = _dm_inputs(n_days=60)
    d = loss_differential(real, pred_1, pred_2, norm=1)
    dbar = d.mean()
    iid_var = ((d - dbar) ** 2).mean()
    expected = float(1.0 - stats.norm.cdf(dbar / np.sqrt(iid_var / len(d))))

    assert diebold_mariano_hac(real, pred_1, pred_2, bandwidth=0) == pytest.approx(expected)


def test_diebold_mariano_hac_larger_bandwidth_is_more_conservative():
    """With positively autocorrelated loss differentials, adding Newey-West
    lags must inflate the variance estimate and therefore the p-value. A
    bandwidth argument that was ignored would make these equal."""
    rng = np.random.default_rng(42)
    n_days = 120
    real = _daily_price_frame(n_days, rng)

    # A persistent (AR-like) accuracy advantage for pred_2, the forecast the
    # one-sided alternative is about. Kept modest so the p-values stay well
    # inside float resolution -- an overwhelming signal underflows both to
    # exactly 0.0 and the comparison would test nothing.
    advantage = np.zeros(n_days)
    for i in range(1, n_days):
        advantage[i] = 0.9 * advantage[i - 1] + rng.normal(0, 1)
    advantage = 0.3 * advantage + 0.15
    noise = rng.normal(0, 5, size=real.shape)
    pred_2 = real + noise
    pred_1 = real + noise + advantage[:, None]

    p_iid = diebold_mariano_hac(real, pred_1, pred_2, bandwidth=0)
    p_hac = diebold_mariano_hac(real, pred_1, pred_2, bandwidth=12)
    assert 0.0 < p_iid < p_hac < 1.0
    # the default bandwidth rule must also land between the two extremes,
    # i.e. it applies a real correction rather than silently using L=0
    p_default = diebold_mariano_hac(real, pred_1, pred_2)
    assert p_default > p_iid
