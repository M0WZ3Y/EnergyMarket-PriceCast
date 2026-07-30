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


# --------------------------------------------------------------------------
# LightGBM
# --------------------------------------------------------------------------

def _lgbm_cfg(**overrides):
    cfg = dict(
        refit_every_n_days=1,
        params=dict(objective="regression_l1", n_estimators=10, num_leaves=7),
        tuned_params_file=None,
    )
    cfg.update(overrides)
    return cfg


def test_lightgbm_conforms_to_base_model_interface():
    from src.models import LightGBMModel

    model = LightGBMModel(_lgbm_cfg())
    assert isinstance(model, BaseModel)
    for method in ("fit", "predict", "save", "load"):
        assert callable(getattr(model, method))


def test_lightgbm_predict_raises_on_multi_row_X():
    from src.models import LightGBMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LightGBMModel(_lgbm_cfg()).fit(X, Y)
    with pytest.raises(ValueError):
        model.predict(X)


def test_lightgbm_predict_raises_on_reordered_columns():
    from src.models import LightGBMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LightGBMModel(_lgbm_cfg()).fit(X, Y)
    reordered = X.iloc[[-1]][list(X.columns[::-1])]
    with pytest.raises(ValueError):
        model.predict(reordered)


def test_lightgbm_seed_and_determinism_forced_over_config():
    """The project's seed-42 rule must survive any params dict a config or
    tuned-params file could supply -- FORCED_PARAMS always wins."""
    from src.models import LightGBMModel

    from src.models.lgbm import FORCED_PARAMS

    cfg = _lgbm_cfg()
    cfg["params"].update(random_state=7, deterministic=False, n_jobs=8)
    model = LightGBMModel(cfg)
    assert model.params["random_state"] == 42
    assert model.params["deterministic"] is True
    assert model.params["n_jobs"] == FORCED_PARAMS["n_jobs"]


def test_lightgbm_fit_predict_is_deterministic():
    from src.models import LightGBMModel

    X, Y = build_features(_synthetic_df(40), FEATURE_CFG)
    test_day = X.index[[-1]]
    p1 = LightGBMModel(_lgbm_cfg()).fit(X, Y).predict(X.loc[test_day])
    p2 = LightGBMModel(_lgbm_cfg()).fit(X, Y).predict(X.loc[test_day])
    pd.testing.assert_frame_equal(p1, p2)


def test_lightgbm_refit_cadence_only_stamps_on_full_refit():
    """Same regression class as SARIMAX's cadence bug: with cadence N and
    the origin advancing 1 day per fit() call, the model must refit on the
    Nth day, not never-again after the first fit."""
    from src.models import LightGBMModel

    X, Y = build_features(_synthetic_df(45), FEATURE_CFG)
    days = Y.index
    model = LightGBMModel(_lgbm_cfg(refit_every_n_days=3))

    model.fit(X.loc[days[:20]], Y.loc[days[:20]])
    first_models = dict(model._models)
    stamp_after_first = model._last_refit_end

    # +1 and +2 days: within cadence, models must be reused
    model.fit(X.loc[days[1:21]], Y.loc[days[1:21]])
    model.fit(X.loc[days[2:22]], Y.loc[days[2:22]])
    assert model._last_refit_end == stamp_after_first
    assert all(model._models[h] is first_models[h] for h in range(24))

    # +3 days: cadence reached, full refit must trigger
    model.fit(X.loc[days[3:23]], Y.loc[days[3:23]])
    assert model._last_refit_end != stamp_after_first
    assert all(model._models[h] is not first_models[h] for h in range(24))


def test_lightgbm_end_to_end_walk_forward_on_synthetic_data():
    from src.models import LightGBMModel

    X, Y = build_features(_synthetic_df(60), FEATURE_CFG)
    eval_cfg = dict(walk_forward=dict(calibration_window_days=20, step_days=1), random_seed=42)
    first_origin = Y.index[25]
    n_origins = sum(1 for d in Y.index if d >= first_origin)

    result = run_model(
        "LightGBM", LightGBMModel(_lgbm_cfg()), X, Y, eval_cfg=eval_cfg, first_origin=first_origin
    )
    assert len(result) == n_origins * 24
    assert not result["y_pred"].isna().any()


def test_lightgbm_tuned_file_cannot_override_forced_params(tmp_path):
    """Seed-42 rule guarded through the FILE channel, not just cfg[params]:
    a tuned-params yaml trying to change seed/determinism/n_jobs must
    still lose to FORCED_PARAMS."""
    import yaml

    from src.models import LightGBMModel

    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(
        yaml.safe_dump(
            dict(params=dict(random_state=7, deterministic=False, n_jobs=8, num_leaves=31))
        )
    )
    from src.models.lgbm import FORCED_PARAMS

    model = LightGBMModel(_lgbm_cfg(tuned_params_file=str(tuned)))
    assert model.params["random_state"] == 42
    assert model.params["deterministic"] is True
    assert model.params["n_jobs"] == FORCED_PARAMS["n_jobs"]
    # non-forced tuned values do merge
    assert model.params["num_leaves"] == 31


def test_lightgbm_malformed_tuned_file_raises(tmp_path):
    """A tuned file without a 'params' key must fail loudly instead of
    silently merging file metadata (validation_mae, window strings) into
    LGBMRegressor kwargs."""
    import yaml

    from src.models import LightGBMModel

    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(yaml.safe_dump(dict(validation_mae=3.9, n_trials=50)))
    with pytest.raises(ValueError, match="params"):
        LightGBMModel(_lgbm_cfg(tuned_params_file=str(tuned)))


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

def _lstm_cfg(**overrides):
    cfg = dict(
        refit_every_n_days=7,
        sequence_lags=[7, 3, 2, 1],  # oldest -> newest timesteps
        units=8,
        epochs=2,
        batch_size=32,
        tuned_params_file=None,
    )
    cfg.update(overrides)
    return cfg


def test_lstm_conforms_to_base_model_interface():
    from src.models import LSTMModel

    model = LSTMModel(_lstm_cfg())
    assert isinstance(model, BaseModel)
    for method in ("fit", "predict", "save", "load"):
        assert callable(getattr(model, method))


def test_lstm_predict_raises_on_multi_row_X():
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LSTMModel(_lstm_cfg()).fit(X, Y)
    with pytest.raises(ValueError):
        model.predict(X)


def test_lstm_predict_raises_on_reordered_columns():
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LSTMModel(_lstm_cfg()).fit(X, Y)
    reordered = X.iloc[[-1]][list(X.columns[::-1])]
    with pytest.raises(ValueError):
        model.predict(reordered)


def test_lstm_sequence_uses_only_price_lag_columns():
    """The LSTM's sequence branch must be built exclusively from the
    audited price_D-<lag> columns of the shared X — never from any column
    that could carry target-day price information (none exists in X, but
    the selection must be by explicit prefix, not positional)."""
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LSTMModel(_lstm_cfg())
    seq_cols = model._sequence_columns(X.columns)
    assert len(seq_cols) == 4  # one entry per configured lag
    for lag, cols in zip([7, 3, 2, 1], seq_cols):
        assert len(cols) == 24
        assert all(c.startswith(f"price_D-{lag}_") for c in cols)


def test_lstm_refit_cadence_only_stamps_on_full_refit():
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(45), FEATURE_CFG)
    days = Y.index
    model = LSTMModel(_lstm_cfg(refit_every_n_days=3, epochs=1))

    model.fit(X.loc[days[:20]], Y.loc[days[:20]])
    first_net = model._net
    stamp = model._last_refit_end

    model.fit(X.loc[days[1:21]], Y.loc[days[1:21]])
    model.fit(X.loc[days[2:22]], Y.loc[days[2:22]])
    assert model._last_refit_end == stamp
    assert model._net is first_net

    model.fit(X.loc[days[3:23]], Y.loc[days[3:23]])
    assert model._last_refit_end != stamp
    assert model._net is not first_net


def test_lstm_end_to_end_walk_forward_on_synthetic_data():
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(45), FEATURE_CFG)
    eval_cfg = dict(walk_forward=dict(calibration_window_days=15, step_days=1), random_seed=42)
    first_origin = Y.index[30]
    n_origins = sum(1 for d in Y.index if d >= first_origin)

    result = run_model(
        "LSTM",
        LSTMModel(_lstm_cfg(epochs=1, refit_every_n_days=30)),
        X,
        Y,
        eval_cfg=eval_cfg,
        first_origin=first_origin,
    )
    assert len(result) == n_origins * 24
    assert result.columns.tolist() == ["origin", "hour", "y_true", "y_pred", "model"]
    assert not result["y_pred"].isna().any()


def test_lstm_tuned_params_file_merges_over_defaults(tmp_path):
    """Leakage-review bug fix: models.yaml declares tuned_params_file for
    lstm but the wrapper ignored it -- tuned params must merge over the
    config defaults (same channel as LightGBM)."""
    import yaml

    from src.models import LSTMModel

    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(yaml.safe_dump(dict(params=dict(units=16, learning_rate=0.005))))
    model = LSTMModel(_lstm_cfg(units=8, tuned_params_file=str(tuned)))
    assert model.units == 16
    assert model.learning_rate == 0.005
    assert model.epochs == 2  # non-tuned default untouched


def test_lstm_malformed_tuned_file_raises(tmp_path):
    import yaml

    from src.models import LSTMModel

    tuned = tmp_path / "tuned.yaml"
    tuned.write_text(yaml.safe_dump(dict(validation_mae=9.9)))
    with pytest.raises(ValueError, match="params"):
        LSTMModel(_lstm_cfg(tuned_params_file=str(tuned)))


def test_lstm_predict_unfitted_raises_clear_error():
    from src.models import LSTMModel

    X, _ = build_features(_synthetic_df(30), FEATURE_CFG)
    with pytest.raises(RuntimeError, match="not fitted"):
        LSTMModel(_lstm_cfg()).predict(X.iloc[[-1]])


def test_lstm_scalers_untouched_between_refits_and_at_predict():
    """Guards the reviewed leak class: between full refits and during
    predict(), the train-slice scalers must not be refit or mutated."""
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(45), FEATURE_CFG)
    days = Y.index
    model = LSTMModel(_lstm_cfg(refit_every_n_days=5, epochs=1))
    model.fit(X.loc[days[:20]], Y.loc[days[:20]])
    scalers = model._scalers
    seq_mean = scalers[0].mean_.copy()

    model.fit(X.loc[days[1:21]], Y.loc[days[1:21]])  # non-refit call
    model.predict(X.loc[days[[21]]])
    assert model._scalers is scalers
    assert np.array_equal(model._scalers[0].mean_, seq_mean)


def test_lstm_fit_raises_on_backward_train_window():
    """A train window ending BEFORE the last refit would reuse a network
    trained on later data -- genuine leakage if any caller ever iterates
    origins out of order. Must refuse loudly."""
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(45), FEATURE_CFG)
    days = Y.index
    model = LSTMModel(_lstm_cfg(epochs=1))
    model.fit(X.loc[days[:20]], Y.loc[days[:20]])
    with pytest.raises(ValueError, match="backward"):
        model.fit(X.loc[days[:10]], Y.loc[days[:10]])


def test_lstm_load_raises_when_keras_file_missing(tmp_path):
    """load() silently leaving _net=None while is_fitted=True would give
    an opaque AttributeError at predict -- exactly the path the week-8/11
    OOD stress test will exercise. Must raise cleanly instead."""
    from src.models import LSTMModel

    X, Y = build_features(_synthetic_df(30), FEATURE_CFG)
    model = LSTMModel(_lstm_cfg(epochs=1)).fit(X, Y)
    path = tmp_path / "lstm_artifact.pkl"
    model.save(path)
    path.with_suffix(".keras").unlink()

    with pytest.raises(RuntimeError, match="keras"):
        LSTMModel(_lstm_cfg()).load(path)
