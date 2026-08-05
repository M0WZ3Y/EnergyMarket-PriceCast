"""SHAP interpretability layer — src/interpretability/shap_analysis.py (thesis 4-6).

Three things must hold for a SHAP figure to be defensible in a thesis, and
each has its own test here:

1. **The explained model never saw the explained days.** The models in
   `models/frozen/` were fit on a trailing window ending 2017-12-31, which
   swallows the whole 2016-01-04..2017-12-31 test period — explaining those
   over test days would be in-sample. Section 4-6 therefore explains a
   separate interpretation-only fit whose training window ends strictly
   before the test boundary. `test_interpretation_train_days_*` is the
   load-bearing guard on that.

2. **The attributions are exact, not approximate.** TreeSHAP is exact, so
   `sum(shap) + expected_value == prediction` holds to float tolerance. That
   identity is a real assertion: it fails if the explainer is wired to the
   wrong booster, or if feature order drifts between fit and explain.

3. **"Stressed" means the same thing here as in chapter 3-8.** The calm/
   stressed split reuses `evaluation.ensemble.regime_labels` rather than
   redefining the rule; a second definition would quietly contradict the
   ensemble chapter. Asserted directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import require_thesis_data, thesis_data_path

from src.evaluation.ensemble import regime_labels
from src.features.pipeline import build_features, daily_target
from src.interpretability.shap_analysis import (
    FEATURE_GROUPS,
    ShapResult,
    feature_group,
    fit_interpretation_models,
    group_importance,
    interpretation_train_days,
    regime_split,
    shap_values_daily,
    shap_values_hourly,
    split_boundary,
)
from src.models.daily import DailyLightGBMModel

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DE = thesis_data_path("raw", "DE.csv")

# Deliberately tiny boosters: these tests check wiring and invariants, not
# forecast quality, and the suite runs on every file edit via a PostToolUse
# hook. Same shape as the real config, 1/10th the trees.
LIGHT_CFG = {
    "lightgbm": {"refit_every_n_days": 1, "params": {"n_estimators": 15, "num_leaves": 7}},
    "daily_lightgbm": {"refit_every_n_days": 1, "params": {"n_estimators": 15, "num_leaves": 7}},
}


def _synthetic_hourly(n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Same generator shape as tests/test_daily.py, so a failure here reads
    against a series the rest of the suite already characterises."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days * 24, freq="h")
    hour = idx.hour.to_numpy()
    base = 40 + 12 * np.sin(2 * np.pi * hour / 24)
    price = base + rng.normal(0, 4, size=len(idx))
    return pd.DataFrame(
        {
            "price": price,
            "exog_1": price + rng.normal(0, 2, size=len(idx)),
            "exog_2": rng.normal(100, 10, size=len(idx)),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def features():
    return build_features(_synthetic_hourly())


@pytest.fixture(scope="module")
def fitted(features):
    """One interpretation fit reused across the SHAP tests (24 + 1 boosters)."""
    X, Y = features
    train_days = X.index[:90]
    hourly, daily = fit_interpretation_models(X, Y, train_days, models_cfg=LIGHT_CFG)
    explain = X.loc[X.index[90:100]]
    return hourly, daily, explain


# ---------------------------------------------------------------------------
# 1. The leakage guard: the interpretation window ends before the test period
# ---------------------------------------------------------------------------


def test_interpretation_train_days_end_strictly_before_the_boundary():
    index = pd.date_range("2012-01-16", "2017-12-31", freq="D")
    boundary = pd.Timestamp("2016-01-04")

    days = interpretation_train_days(index, boundary, calibration_window_days=1092)

    assert days.max() < boundary
    assert days.max() == pd.Timestamp("2016-01-03")
    assert len(days) == 1092
    # Contiguous trailing window, not an arbitrary subset.
    assert days.min() == pd.Timestamp("2016-01-03") - pd.Timedelta(days=1091)


def test_interpretation_train_days_refuses_a_short_window():
    """A silently-shortened training window would produce a differently-fitted
    model than the walk-forward used, with no visible symptom in any figure."""
    index = pd.date_range("2015-01-01", "2016-01-03", freq="D")
    with pytest.raises(ValueError, match="1092"):
        interpretation_train_days(index, pd.Timestamp("2016-01-04"), calibration_window_days=1092)


def test_interpretation_train_days_excludes_the_boundary_day_itself():
    """Off-by-one check: the boundary day is the FIRST test day and must not
    be trained on. An inclusive comparison here is exactly the kind of
    one-day leak the project rule forbids."""
    index = pd.date_range("2015-01-01", "2016-01-31", freq="D")
    days = interpretation_train_days(index, pd.Timestamp("2016-01-04"), calibration_window_days=10)
    assert pd.Timestamp("2016-01-04") not in days
    assert days.max() == pd.Timestamp("2016-01-03")


@pytest.mark.epftoolbox
def test_split_boundary_is_the_first_real_test_day():
    """Derived from the loader's own split, never hardcoded — but pinned to
    the known value so a config change that moves the split is visible."""
    require_thesis_data(RAW_DE, "benchmark DE.csv")
    from src.data.loader import BenchmarkLoader, load_config

    _, test = BenchmarkLoader(load_config()).load()
    boundary = split_boundary(test)

    assert boundary == test.index.min().normalize()
    assert boundary == pd.Timestamp("2016-01-04")


# ---------------------------------------------------------------------------
# 2. TreeSHAP exactness
# ---------------------------------------------------------------------------


def test_hourly_shap_reconstructs_every_hourly_prediction(fitted):
    """The additivity identity, checked against the wrapper's own predict()
    — not against a re-derived booster call — so a mismatch between the
    explained booster and the predicting booster cannot hide."""
    hourly, _, explain = fitted
    results = shap_values_hourly(hourly, explain)

    assert set(results) == set(range(24))
    for day in explain.index:
        predicted = hourly.predict(explain.loc[[day]]).iloc[0]
        for h in range(24):
            res = results[h]
            row = res.values[res.index.get_loc(day)]
            reconstructed = row.sum() + res.expected_value
            assert reconstructed == pytest.approx(predicted[f"y_h{h:02d}"], abs=1e-6)


def test_daily_shap_reconstructs_the_daily_prediction(fitted):
    _, daily, explain = fitted
    res = shap_values_daily(daily, explain)

    for day in explain.index:
        predicted = daily.predict(explain.loc[[day]]).iloc[0]["y_daily"]
        row = res.values[res.index.get_loc(day)]
        assert row.sum() + res.expected_value == pytest.approx(predicted, abs=1e-6)


def test_shap_results_carry_the_fitted_feature_names(fitted):
    """The wrappers fit on X.values, so the boosters carry no feature names.
    If the result did not carry them explicitly, every figure would be
    labelled by position — silently wrong the moment column order changes."""
    hourly, _, explain = fitted
    res = shap_values_hourly(hourly, explain)[0]

    assert isinstance(res, ShapResult)
    assert list(res.columns) == list(explain.columns)
    assert res.values.shape == (len(explain), explain.shape[1])
    assert list(res.index) == list(explain.index)


def test_shap_values_are_deterministic_across_a_refit(features, fitted):
    """Calling the explainer twice on ONE fitted object only asserts that
    TreeSHAP is deterministic, which is a property of the shap library, not of
    this repo. The failure mode that matters is seed or thread-count drift
    making a REFIT produce different attributions — so the model is refitted
    from scratch and the two fits compared.
    """
    X, Y = features
    hourly, _, explain = fitted

    refitted, _ = fit_interpretation_models(X, Y, X.index[:90], models_cfg=LIGHT_CFG)

    for hour in (0, 12, 23):
        first = shap_values_hourly(hourly, explain)[hour].values
        second = shap_values_hourly(refitted, explain)[hour].values
        assert np.array_equal(first, second), f"refit changed attributions at hour {hour}"


def test_shap_refuses_a_column_order_that_does_not_match_the_fit(fitted):
    """Reordered columns still produce numbers — wrong ones. Must raise."""
    hourly, _, explain = fitted
    shuffled = explain[list(explain.columns[::-1])]
    with pytest.raises(ValueError, match="column"):
        shap_values_hourly(hourly, shuffled)


# ---------------------------------------------------------------------------
# 3. Feature grouping: 247 raw columns -> 11 readable families
# ---------------------------------------------------------------------------


def test_every_feature_column_maps_to_exactly_one_group(features):
    X, _ = features
    groups = [feature_group(c) for c in X.columns]

    assert len(groups) == X.shape[1] == 247
    assert set(groups) == set(FEATURE_GROUPS)
    # 24 hours per lag block, 7 dummies in the dow group.
    counts = pd.Series(groups).value_counts()
    assert counts["dow"] == 7
    assert all(counts[g] == 24 for g in FEATURE_GROUPS if g != "dow")


def test_feature_group_rejects_an_unrecognised_column():
    """Silently bucketing an unknown column into 'other' would understate
    whichever family it really belongs to, in a published figure."""
    with pytest.raises(ValueError, match="unrecognised|unknown"):
        feature_group("temperature_D0_h05")


def test_group_importance_conserves_total_attribution(features):
    X, _ = features
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, X.shape[1]))

    imp = group_importance(values, X.columns)

    assert list(imp.index) == [g for g in FEATURE_GROUPS if g in set(imp.index)]
    # Grouping redistributes attribution; it must never create or destroy it.
    assert imp.sum() == pytest.approx(np.abs(values).mean(axis=0).sum())
    assert (imp >= 0).all()


def test_group_importance_rejects_a_shape_mismatch(features):
    X, _ = features
    with pytest.raises(ValueError, match="column"):
        group_importance(np.zeros((5, 3)), X.columns)


# ---------------------------------------------------------------------------
# 4. The calm/stressed split is the ensemble chapter's split
# ---------------------------------------------------------------------------


def test_regime_split_agrees_with_the_ensemble_regime_labels(features):
    """Pins 4-6 to 3-8. If these ever diverge, one chapter's 'stressed days'
    are not the other's and the comparison is meaningless."""
    _, Y = features
    threshold = 45.0

    ours = regime_split(Y, threshold)

    long = (
        Y.rename(columns=lambda c: int(c[3:]))
        .stack()
        .rename("y_true")
        .reset_index()
        .rename(columns={"target_day": "origin", "level_1": "hour"})
    )
    theirs = regime_labels(long, threshold)

    assert dict(ours) == theirs
    assert set(ours.unique()) <= {"calm", "stressed"}


def test_regime_split_ignores_the_target_days_own_prices(features):
    """The label must depend only on the PREVIOUS day. Spiking a day's own
    prices far above the threshold must not relabel that day."""
    _, Y = features
    threshold = 1e6  # nothing is stressed under this
    baseline = regime_split(Y, threshold)
    assert (baseline == "calm").all()

    spiked = Y.copy()
    victim = spiked.index[5]
    spiked.loc[victim] = 1e9

    relabelled = regime_split(spiked, threshold)
    assert relabelled.loc[victim] == "calm", "a day's own prices leaked into its label"
    # ...but the day AFTER it must flip, which is the rule working.
    assert relabelled.loc[spiked.index[6]] == "stressed"


def test_regime_split_covers_every_target_day(features):
    _, Y = features
    labels = regime_split(Y, 45.0)
    assert list(labels.index) == list(Y.index)
    assert labels.notna().all()


# ---------------------------------------------------------------------------
# 5. The fitted interpretation models are the real wrappers, wired to Y
# ---------------------------------------------------------------------------


def test_daily_arm_is_fitted_on_the_baseload_target(features):
    """The daily arm must be fitted on daily_target(Y) — the same reduction
    RQ4 uses — not on some other collapse of Y. Otherwise the hourly-vs-daily
    SHAP comparison in 4-6 contrasts two models trained on targets that were
    never commensurable, and nothing would fail to say so.

    Checked by fitting the reference model directly and demanding identical
    predictions, then confirming a DIFFERENT target would have been visible.
    """
    X, Y = features
    train_days = X.index[:90]
    _, daily = fit_interpretation_models(X, Y, train_days, models_cfg=LIGHT_CFG)

    reference = DailyLightGBMModel(LIGHT_CFG["daily_lightgbm"])
    reference.fit(X.loc[train_days], daily_target(Y.loc[train_days]))

    probe = X.loc[[X.index[95]]]
    assert daily.predict(probe).iloc[0]["y_daily"] == pytest.approx(
        reference.predict(probe).iloc[0]["y_daily"]
    )

    # Mutation control: a plausible-but-wrong target (hour 0 instead of the
    # 24-hour mean) must NOT reproduce the same prediction, or the assertion
    # above would pass regardless of which target was used.
    wrong = DailyLightGBMModel(LIGHT_CFG["daily_lightgbm"])
    wrong.fit(X.loc[train_days], Y.loc[train_days, "y_h00"].rename("y_daily"))
    assert daily.predict(probe).iloc[0]["y_daily"] != pytest.approx(
        wrong.predict(probe).iloc[0]["y_daily"]
    )


def test_interpretation_models_expose_the_fitted_feature_columns(features):
    X, Y = features
    hourly, daily = fit_interpretation_models(X, Y, X.index[:90], models_cfg=LIGHT_CFG)

    assert hourly.is_fitted and daily.is_fitted
    assert list(hourly._feature_columns) == list(X.columns)
    assert list(daily._feature_columns) == list(X.columns)


def test_interpretation_fit_refuses_train_days_outside_the_feature_index(features):
    X, Y = features
    stray = X.index[:5].append(pd.DatetimeIndex([pd.Timestamp("1999-01-01")]))
    with pytest.raises((KeyError, ValueError)):
        fit_interpretation_models(X, Y, stray, models_cfg=LIGHT_CFG)
