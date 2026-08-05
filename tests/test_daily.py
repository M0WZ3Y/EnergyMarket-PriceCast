"""Direct daily-baseload route — src/models/daily.py + daily_target().

The load-bearing test here is the naive identity: the direct and
aggregated routes must agree exactly for a model where they are
mathematically the same operation. If that ever fails, the two routes are
not reading the same data and the RQ4 comparison is meaningless.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.results import daily_baseload
from src.features.pipeline import build_features, daily_target
from src.models.daily import (
    DailyLEARLassoModel,
    DailyLightGBMModel,
    DailyLSTMModel,
    DailyNaiveModel,
    DailySARIMAXModel,
)
from src.models.naive import NaiveModel


def _synthetic_hourly(n_days=200, seed=42):
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
def features_large():
    """LEAR's LassoLarsIC step needs more training samples than features
    (247 with the default 2-exog feature config), so its numerical tests
    need a longer synthetic history than everything else here — the same
    reason test_models.py sizes its LEAR fixture at 280 days."""
    return build_features(_synthetic_hourly(320))


def test_daily_target_is_the_unweighted_mean_of_the_24_hours(features):
    X, Y = features
    d = daily_target(Y)
    assert d.name == "y_daily"
    assert d.index.equals(Y.index)
    assert np.allclose(d.to_numpy(), Y.to_numpy().mean(axis=1))


def test_daily_target_rejects_a_frame_that_is_not_the_24_hour_target(features):
    _, Y = features
    with pytest.raises(ValueError, match="24 target"):
        daily_target(Y.iloc[:, :5])


def test_daily_target_rejects_nan(features):
    _, Y = features
    dirty = Y.copy()
    dirty.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        daily_target(dirty)


def test_direct_and_aggregated_naive_agree_exactly(features):
    """Both routes are the same linear operation on the same lag block,
    so any disagreement means they are not seeing the same data."""
    X, Y = features
    daily = daily_target(Y)
    origins = X.index[-30:]

    direct, aggregated = [], []
    for origin in origins:
        row = X.loc[[origin]]
        direct.append(DailyNaiveModel().fit(X, daily).predict(row)["y_daily"].iloc[0])

        hourly = NaiveModel().fit(X, Y).predict(row)
        long = pd.DataFrame(
            dict(
                origin=[origin] * 24,
                hour=range(24),
                y_true=Y.loc[origin].to_numpy(),
                y_pred=hourly.iloc[0].to_numpy(),
                model="naive",
            )
        )
        aggregated.append(daily_baseload(long)["y_pred"].iloc[0])

    assert np.allclose(direct, aggregated)


def test_daily_lightgbm_predicts_one_value_per_origin(features):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]

    model = DailyLightGBMModel({"refit_every_n_days": 1}).fit(X.loc[train], daily.loc[train])
    out = model.predict(X.loc[[origin]])

    assert list(out.columns) == ["y_daily"]
    assert len(out) == 1
    assert np.isfinite(out["y_daily"].iloc[0])


def test_daily_lightgbm_beats_a_constant_mean_forecast(features):
    """Sanity that the model learns something from X at all."""
    X, Y = features
    daily = daily_target(Y)
    train, test = X.index[:-30], X.index[-30:]

    model = DailyLightGBMModel({"refit_every_n_days": 10**6})
    model.fit(X.loc[train], daily.loc[train])
    preds = [model.predict(X.loc[[o]])["y_daily"].iloc[0] for o in test]

    mae_model = np.mean(np.abs(daily.loc[test].to_numpy() - np.array(preds)))
    mae_const = np.mean(np.abs(daily.loc[test].to_numpy() - daily.loc[train].mean()))
    assert mae_model <= mae_const


def test_daily_lightgbm_rejects_reordered_features(features):
    X, Y = features
    daily = daily_target(Y)
    model = DailyLightGBMModel().fit(X, daily)
    with pytest.raises(ValueError, match="column order changed"):
        model.fit(X[list(X.columns)[::-1]], daily)


def test_daily_lightgbm_predict_rejects_multiple_rows(features):
    X, Y = features
    daily = daily_target(Y)
    model = DailyLightGBMModel().fit(X, daily)
    with pytest.raises(ValueError, match="exactly one"):
        model.predict(X.iloc[-3:])


def test_daily_lightgbm_is_deterministic(features):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]
    a = DailyLightGBMModel().fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])
    b = DailyLightGBMModel().fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])
    assert a["y_daily"].iloc[0] == b["y_daily"].iloc[0]


# --------------------------------------------------------------------------
# SARIMAX / LEAR-LASSO / LSTM daily-direct variants
#
# These complete the model list for the direct route, so that RQ4 compares
# the same five models on both routes rather than a subset. Each is checked
# for the daily output contract (one row, one 'y_daily' column, finite) and
# for the single-row predict guard; the estimator-specific behaviour is
# already covered for the hourly wrappers in test_models.py.
# --------------------------------------------------------------------------


def test_daily_sarimax_predicts_one_value_per_origin(features):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:120], X.index[120]

    model = DailySARIMAXModel({"refit_every_n_days": 1})
    out = model.fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])

    assert list(out.columns) == ["y_daily"]
    assert len(out) == 1
    assert np.isfinite(out["y_daily"].iloc[0])


def test_daily_sarimax_exog_is_the_daily_mean_of_the_hourly_d0_columns(features):
    """The direct and aggregated routes must differ only in the target, so
    the daily exog has to be the same averaging operation the target
    applies to the 24 prices."""
    X, _ = features
    model = DailySARIMAXModel()
    exog = model._daily_exog(X)

    assert list(exog.columns) == ["exog_1_D0", "exog_2_D0"]
    expected = X[[f"exog_1_D0_h{h:02d}" for h in range(24)]].mean(axis=1)
    assert np.allclose(exog["exog_1_D0"].to_numpy(), expected.to_numpy())


def test_daily_sarimax_holds_parameters_between_refits(features):
    """Cadence > 1 must reuse the fitted parameters, not silently refit
    every origin -- the same contract the hourly wrapper documents."""
    X, Y = features
    daily = daily_target(Y)
    train = X.index[:120]

    model = DailySARIMAXModel({"refit_every_n_days": 7})
    model.fit(X.loc[train], daily.loc[train])
    first_refit = model._last_refit_end
    model.fit(X.loc[X.index[:123]], daily.loc[X.index[:123]])

    assert model._last_refit_end == first_refit


def test_daily_sarimax_predict_rejects_multiple_rows(features):
    X, Y = features
    daily = daily_target(Y)
    train = X.index[:120]
    model = DailySARIMAXModel().fit(X.loc[train], daily.loc[train])
    with pytest.raises(ValueError, match="exactly one"):
        model.predict(X.iloc[-3:])


@pytest.mark.epftoolbox
def test_daily_lear_lasso_predicts_one_value_per_origin(features_large):
    X, Y = features_large
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]

    model = DailyLEARLassoModel()
    out = model.fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])

    assert list(out.columns) == ["y_daily"]
    assert len(out) == 1
    assert np.isfinite(out["y_daily"].iloc[0])


@pytest.mark.epftoolbox
def test_daily_lear_lasso_beats_a_constant_mean_forecast(features_large):
    X, Y = features_large
    daily = daily_target(Y)
    train, test = X.index[:-30], X.index[-30:]

    model = DailyLEARLassoModel().fit(X.loc[train], daily.loc[train])
    preds = np.array([model.predict(X.loc[[o]])["y_daily"].iloc[0] for o in test])

    mae_model = np.mean(np.abs(daily.loc[test].to_numpy() - preds))
    mae_const = np.mean(np.abs(daily.loc[test].to_numpy() - daily.loc[train].mean()))
    assert mae_model <= mae_const


@pytest.mark.epftoolbox
def test_daily_lear_lasso_does_not_mutate_the_callers_frame(features_large):
    """LEAR's scalers write back into the array they are handed; the
    wrapper must copy first or the harness's X is silently rescaled."""
    X, Y = features_large
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]

    model = DailyLEARLassoModel().fit(X.loc[train], daily.loc[train])
    before = X.loc[[origin]].to_numpy(dtype=float, copy=True)
    model.predict(X.loc[[origin]])
    assert np.allclose(X.loc[[origin]].to_numpy(dtype=float), before)


@pytest.mark.epftoolbox
def test_daily_lear_lasso_predict_rejects_multiple_rows(features_large):
    X, Y = features_large
    daily = daily_target(Y)
    model = DailyLEARLassoModel().fit(X, daily)
    with pytest.raises(ValueError, match="exactly one"):
        model.predict(X.iloc[-3:])


def test_daily_lstm_predicts_one_value_per_origin(features):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]

    model = DailyLSTMModel({"units": 4, "epochs": 2, "batch_size": 32})
    out = model.fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])

    assert list(out.columns) == ["y_daily"]
    assert len(out) == 1
    assert np.isfinite(out["y_daily"].iloc[0])


def test_daily_lstm_output_layer_has_one_unit(features):
    """The daily head must be width 1 -- a 24-wide head would silently
    make predict() return the first hour rather than the baseload."""
    X, Y = features
    daily = daily_target(Y)
    model = DailyLSTMModel({"units": 4, "epochs": 1}).fit(X.iloc[:60], daily.iloc[:60])
    assert model._net.output_shape[-1] == 1


def test_daily_lstm_is_deterministic(features):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]
    cfg = {"units": 4, "epochs": 2}
    a = DailyLSTMModel(cfg).fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])
    b = DailyLSTMModel(cfg).fit(X.loc[train], daily.loc[train]).predict(X.loc[[origin]])
    assert a["y_daily"].iloc[0] == b["y_daily"].iloc[0]


def test_daily_lstm_predict_rejects_multiple_rows(features):
    X, Y = features
    daily = daily_target(Y)
    model = DailyLSTMModel({"units": 4, "epochs": 1}).fit(X.iloc[:60], daily.iloc[:60])
    with pytest.raises(ValueError, match="exactly one"):
        model.predict(X.iloc[-3:])


def test_daily_lstm_save_load_roundtrip(features, tmp_path):
    X, Y = features
    daily = daily_target(Y)
    origin = X.index[-1]
    model = DailyLSTMModel({"units": 4, "epochs": 2}).fit(X.iloc[:-1], daily.iloc[:-1])
    before = model.predict(X.loc[[origin]])["y_daily"].iloc[0]

    path = tmp_path / "daily_lstm.pkl"
    model.save(path)
    after = DailyLSTMModel().load(path).predict(X.loc[[origin]])["y_daily"].iloc[0]
    assert before == after


def test_daily_sarimax_save_load_roundtrip(features, tmp_path):
    X, Y = features
    daily = daily_target(Y)
    train, origin = X.index[:120], X.index[120]
    model = DailySARIMAXModel().fit(X.loc[train], daily.loc[train])
    before = model.predict(X.loc[[origin]])["y_daily"].iloc[0]

    path = tmp_path / "daily_sarimax.pkl"
    model.save(path)
    after = DailySARIMAXModel().load(path).predict(X.loc[[origin]])["y_daily"].iloc[0]
    assert before == after


def test_daily_naive_save_load_roundtrip(features, tmp_path):
    """The naive wrapper carries no estimator, so a broken save() would go
    unnoticed until a frozen replay produced a model that is 'fitted' but
    predicts nothing. Round-tripped like every other wrapper."""
    X, Y = features
    daily = daily_target(Y)
    origin = X.index[-1]
    model = DailyNaiveModel().fit(X, daily)
    before = model.predict(X.loc[[origin]])

    path = tmp_path / "daily_naive.pkl"
    model.save(path)
    loaded = DailyNaiveModel().load(path)
    assert loaded.is_fitted
    after = loaded.predict(X.loc[[origin]])

    assert list(after.columns) == ["y_daily"]
    np.testing.assert_allclose(after.to_numpy(dtype=float), before.to_numpy(dtype=float))


@pytest.mark.epftoolbox
def test_daily_lear_lasso_save_load_roundtrip(features_large, tmp_path):
    """LEAR's fitted state is a Lasso plus TWO scalers (X and y). Dropping
    either from the pickle still loads and still predicts — just with
    different numbers — which is exactly the failure the frozen-model replay
    could not detect on its own."""
    X, Y = features_large
    daily = daily_target(Y)
    train, origin = X.index[:-1], X.index[-1]
    model = DailyLEARLassoModel().fit(X.loc[train], daily.loc[train])
    before = model.predict(X.loc[[origin]])

    path = tmp_path / "daily_lear.pkl"
    model.save(path)
    loaded = DailyLEARLassoModel().load(path)
    assert loaded.is_fitted
    after = loaded.predict(X.loc[[origin]])

    assert list(after.columns) == ["y_daily"]
    assert np.isfinite(after["y_daily"].iloc[0])
    np.testing.assert_allclose(after.to_numpy(dtype=float), before.to_numpy(dtype=float))


def test_daily_model_save_load_roundtrip(features, tmp_path):
    X, Y = features
    daily = daily_target(Y)
    origin = X.index[-1]
    model = DailyLightGBMModel().fit(X, daily)
    before = model.predict(X.loc[[origin]])["y_daily"].iloc[0]

    path = tmp_path / "daily_lgbm.pkl"
    model.save(path)
    after = DailyLightGBMModel().load(path).predict(X.loc[[origin]])["y_daily"].iloc[0]
    assert before == after
