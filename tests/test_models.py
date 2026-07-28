"""Model wrapper contract tests — src/models/

Naive and SARIMAX are exercised for real (no epftoolbox dependency).
LEAR-LASSO tests that touch the real epftoolbox.models.LEAR are marked
`epftoolbox` so the suite can still run with `-m "not epftoolbox"` if the
package ever becomes unavailable again (see logs/decisions.md).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.run_baselines import run_model
from src.features.pipeline import build_features
from src.models import BaseModel, LEARLassoModel, NaiveModel, SARIMAXModel, load_models_config
from src.models.base import HOURS, Y_COLUMNS

FEATURE_CFG = dict(
    price_lag_days=[1, 2, 3, 7],
    exog_lag_days=[1, 7],
    exog_current_day=True,
    weekday_dummies=True,
)


def _synthetic_df(n_days: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=24 * n_days, freq="1h")  # 2020-01-01 is a Wednesday
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


def _sarimax_XY(days: pd.DatetimeIndex, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    Y = pd.DataFrame(
        rng.normal(size=(len(days), 24)), index=days, columns=[f"y_{h}" for h in HOURS]
    )
    X = pd.DataFrame(
        rng.normal(size=(len(days), 48)),
        index=days,
        columns=[f"exog_1_D0_{h}" for h in HOURS] + [f"exog_2_D0_{h}" for h in HOURS],
    )
    return X, Y


def test_models_config_loads():
    cfg = load_models_config()
    assert cfg["naive"]["rule"] == "lago_dow"
    assert cfg["sarimax"]["refit_every_n_days"] == 7
    assert cfg["lear_lasso"]["calibration_window_days"] is None
    assert cfg["artifact_dir"] == "models"


def test_all_three_wrappers_conform_to_base_model_interface():
    models_cfg = load_models_config()
    wrappers = [
        NaiveModel(models_cfg["naive"]),
        SARIMAXModel(models_cfg["sarimax"]),
        LEARLassoModel(models_cfg["lear_lasso"]),
    ]
    for model in wrappers:
        assert isinstance(model, BaseModel)
        for method in ("fit", "predict", "save", "load"):
            assert hasattr(model, method)


# --------------------------------------------------------------------------
# NaiveModel
# --------------------------------------------------------------------------

def _naive_XY():
    X, Y = build_features(_synthetic_df(40), FEATURE_CFG)
    model = NaiveModel().fit(X, Y)
    return X, Y, model


def test_naive_monday_uses_d_minus_3():
    X, Y, model = _naive_XY()
    preds = model.predict(X)
    mondays = X.index[X.index.dayofweek == 0]
    assert len(mondays) > 0
    expected = X.loc[mondays, [f"price_D-3_{h}" for h in HOURS]]
    actual = preds.loc[mondays]
    assert np.allclose(actual.values, expected.values)


def test_naive_saturday_uses_d_minus_7():
    X, Y, model = _naive_XY()
    preds = model.predict(X)
    saturdays = X.index[X.index.dayofweek == 5]
    assert len(saturdays) > 0
    expected = X.loc[saturdays, [f"price_D-7_{h}" for h in HOURS]]
    actual = preds.loc[saturdays]
    assert np.allclose(actual.values, expected.values)


def test_naive_sunday_uses_d_minus_7():
    X, Y, model = _naive_XY()
    preds = model.predict(X)
    sundays = X.index[X.index.dayofweek == 6]
    assert len(sundays) > 0
    expected = X.loc[sundays, [f"price_D-7_{h}" for h in HOURS]]
    actual = preds.loc[sundays]
    assert np.allclose(actual.values, expected.values)


def test_naive_midweek_uses_d_minus_1():
    X, Y, model = _naive_XY()
    preds = model.predict(X)
    midweek = X.index[X.index.dayofweek.isin([1, 2, 3, 4])]
    assert len(midweek) > 0
    expected = X.loc[midweek, [f"price_D-1_{h}" for h in HOURS]]
    actual = preds.loc[midweek]
    assert np.allclose(actual.values, expected.values)


def test_naive_fit_is_noop_and_idempotent():
    X, Y = build_features(_synthetic_df(40), FEATURE_CFG)
    model = NaiveModel()
    before = model.predict(X)
    model.fit(X, Y)
    model.fit(Y, X.iloc[:, :24])  # nonsense args -- must not change behavior
    after = model.predict(X)
    assert np.allclose(before.values.astype(float), after.values.astype(float))


def test_naive_output_columns_match_Y_schema():
    X, Y, model = _naive_XY()
    preds = model.predict(X)
    assert preds.columns.tolist() == Y_COLUMNS
    assert Y.columns.tolist() == Y_COLUMNS


# --------------------------------------------------------------------------
# SARIMAXModel
# --------------------------------------------------------------------------

def test_sarimax_predict_raises_on_multi_row_X():
    days = pd.date_range("2021-01-01", periods=5, freq="D")
    X, _ = _sarimax_XY(days)
    model = SARIMAXModel()
    with pytest.raises(ValueError):
        model.predict(X)  # 5 rows, not exactly 1


def test_sarimax_exog_columns_are_D0_not_D_minus_1():
    model = SARIMAXModel()
    cols = model._exog_columns("h05")
    assert cols == ["exog_1_D0_h05", "exog_2_D0_h05"]
    assert not any("D-1" in c or "D-7" in c for c in cols)


def test_sarimax_refit_cadence_uses_append_between_full_refits():
    """Regression test for a bug caught by the leakage-reviewer: cadence
    tracking must be measured from the last *full refit*, not from the
    last fit() call -- walk_forward_splits advances train_end by
    step_days (1) on every call, so if _last_refit_end were updated
    unconditionally, (train_end - _last_refit_end).days would always be 1
    and needs_full_refit could never re-trigger after the first cycle. A
    3-call test can't distinguish the two behaviors (both produce exactly
    one extra refit); this test runs 7 sequential daily calls, matching
    the harness's actual step_days=1 walk-forward cadence, so the
    intended periodic (every 3rd call) pattern is distinguishable from
    the buggy (only ever once) one.
    """
    cfg = dict(refit_every_n_days=3)
    model = SARIMAXModel(cfg)
    days = pd.date_range("2021-01-01", periods=20, freq="D")
    X, Y = _sarimax_XY(days)

    fit_calls = []
    append_calls = []

    def _make_fake_results(row_labels):
        res = MagicMock()
        res.model.data.row_labels = row_labels

        def _append(endog, exog, refit):
            append_calls.append(endog.index.min())
            return _make_fake_results(row_labels.union(endog.index))

        res.append.side_effect = _append
        res.forecast.return_value = pd.Series([0.0])
        return res

    def _sarimax_factory(endog, exog=None, **kwargs):
        inst = MagicMock()

        def _fit(disp=False):
            fit_calls.append(endog.index.max())
            return _make_fake_results(endog.index)

        inst.fit.side_effect = _fit
        return inst

    n_calls = 7  # train_end advances by 1 day per call, matching step_days=1
    with patch("src.models.sarimax.SARIMAX", side_effect=_sarimax_factory):
        for i in range(n_calls):
            end = 5 + i
            model.fit(X.loc[days[0:end]], Y.loc[days[0:end]])

    # Full refits happen on call 1 (start), call 4 (+3 days), call 7 (+3
    # more days) = 3 refits; appends happen on the other 4 calls.
    assert len(fit_calls) == 24 * 3
    assert len(append_calls) == 24 * 4


def test_sarimax_end_to_end_forecast_shape_on_synthetic_data():
    days = pd.date_range("2021-01-01", periods=30, freq="D")
    X, Y = _sarimax_XY(days)
    model = SARIMAXModel(dict(order=[1, 0, 0], seasonal_order=[0, 0, 0, 0], refit_every_n_days=30))
    model.fit(X.iloc[:-1], Y.iloc[:-1])
    preds = model.predict(X.iloc[[-1]])
    assert preds.shape == (1, 24)
    assert preds.columns.tolist() == Y_COLUMNS
    assert not preds.isna().any().any()


# --------------------------------------------------------------------------
# LEARLassoModel (real epftoolbox)
# --------------------------------------------------------------------------

@pytest.mark.epftoolbox
def test_lear_lasso_predict_raises_on_multi_row_X():
    X, _ = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LEARLassoModel()
    with pytest.raises(ValueError):
        model.predict(X)  # multiple rows, not exactly 1 -- must not silently
                           # predict only from row 0 (leakage-reviewer finding)


@pytest.mark.epftoolbox
def test_lear_lasso_column_order_assertion_fires_on_misordered_X():
    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    shuffled_cols = list(X.columns[-7:]) + list(X.columns[:-7])  # dow columns first, not last
    X_bad = X[shuffled_cols]
    model = LEARLassoModel()
    with pytest.raises(ValueError):
        model.fit(X_bad, Y)


@pytest.mark.epftoolbox
def test_lear_lasso_recalibrate_predict_expected_shapes():
    # LassoLarsIC needs more training samples than features (247 features
    # with the default 2-exog feature config), so this needs a much larger
    # synthetic history than the other, non-numerical LEAR-LASSO tests.
    X, Y = build_features(_synthetic_df(280), FEATURE_CFG)
    train_days, test_day = X.index[:-1], X.index[[-1]]
    model = LEARLassoModel(dict(calibration_window_days=len(train_days)))
    model.fit(X.loc[train_days], Y.loc[train_days])
    preds = model.predict(X.loc[test_day])
    assert preds.shape == (1, 24)
    assert preds.columns.tolist() == Y_COLUMNS
    assert not preds.isna().any().any()


# --------------------------------------------------------------------------
# End-to-end: run_baselines.run_model over a tiny synthetic walk-forward
# --------------------------------------------------------------------------

def test_run_baselines_end_to_end_on_synthetic_data():
    X, Y = build_features(_synthetic_df(60), FEATURE_CFG)
    eval_cfg = dict(walk_forward=dict(calibration_window_days=20, step_days=1), random_seed=42)
    first_origin = Y.index[25]

    n_origins = sum(1 for d in Y.index if d >= first_origin)

    naive_result = run_model("naive", NaiveModel(), X, Y, eval_cfg=eval_cfg, first_origin=first_origin)
    assert len(naive_result) == n_origins * 24
    assert not naive_result["y_pred"].isna().any()

    sarimax_model = SARIMAXModel(dict(order=[1, 0, 0], seasonal_order=[0, 0, 0, 0], refit_every_n_days=100))
    X_sarimax = X.rename(
        columns={f"exog_1_D0_{h}": f"exog_1_D0_{h}" for h in HOURS}
    )  # exog_*_D0_* columns already present from build_features
    sarimax_result = run_model(
        "SARIMAX", sarimax_model, X_sarimax, Y, eval_cfg=eval_cfg, first_origin=first_origin
    )
    assert len(sarimax_result) == n_origins * 24
    assert not sarimax_result["y_pred"].isna().any()
