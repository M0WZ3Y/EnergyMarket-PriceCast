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
from src.models.daily import DailyLightGBMModel, DailyNaiveModel
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
