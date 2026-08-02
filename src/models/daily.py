"""Direct daily-baseload models — src/models/daily.py

The DIRECT route for the daily target: one estimator trained on the
baseload itself (`features.pipeline.daily_target`), predicting a single
number per origin. The AGGREGATED route needs nothing here — it averages
an hourly model's 24 forecasts after the fact via
`evaluation.results.daily_baseload`. Comparing the two answers RQ4, so
both routes deliberately share the same X (build_features()'s
leakage-audited matrix) and the same output schema
([origin, y_true, y_pred, model]); the target is the only difference.

Why a separate module rather than a flag on the hourly wrappers: every
hourly wrapper is written against `Y_COLUMNS` (24 per-hour sub-models),
and those wrappers produce the frozen v1.0-results numbers. Adding a
target-shape branch inside them would put the audited hourly path at risk
for no benefit, so the daily route is parallel code that reuses the same
features and conventions instead.

Determinism follows the hourly wrappers exactly: seed 42 forced, fixed
n_jobs (LightGBM's deterministic=True only guarantees reproducibility at
an identical thread count).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import HOURS, BaseModel
from src.models.lear_lasso import _assert_dow_columns_last
from src.models.lstm import LSTMModel

DAILY_COLUMN = "y_daily"
DEFAULT_SEED = 42
DEFAULT_REFIT_EVERY_N_DAYS = 1

# Mirrors src/models/sarimax.py's defaults. The seasonal period stays 7:
# on a daily series that is the weekly cycle, the same cycle the hourly
# per-hour models capture with seasonal_order=(1,0,1,7).
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 0, 1, 7)
DEFAULT_EXOG_PREFIXES = ("exog_1_D0", "exog_2_D0")
DEFAULT_SARIMAX_REFIT_EVERY_N_DAYS = 7

FORCED_PARAMS = dict(
    random_state=DEFAULT_SEED,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
    n_jobs=4,
)


def resolve_tuned_params(cfg: dict, params: dict) -> dict:
    """Merge a tuned-params YAML over `params`, if the config names one.

    Deliberately duplicated from LightGBMModel.__init__ rather than
    refactored out of it: that wrapper produced committed, load-bearing
    results, so it is left untouched until after the v1.0-results freeze.
    """
    tuned_file = cfg.get("tuned_params_file")
    if not tuned_file:
        return params
    tuned_path = Path(tuned_file)
    if not tuned_path.is_absolute():
        # Repo root, never the cwd -- a cwd-dependent lookup would silently
        # fall back to untuned defaults when run from elsewhere.
        tuned_path = Path(__file__).resolve().parents[2] / tuned_path
    if not tuned_path.exists():
        return params
    with open(tuned_path) as f:
        tuned = yaml.safe_load(f) or {}
    if "params" not in tuned:
        raise ValueError(
            f"tuned params file {tuned_path} has no 'params' key -- refusing "
            "to merge file metadata into booster params"
        )
    merged = dict(params)
    merged.update(tuned["params"])
    return merged


class DailyModel(BaseModel):
    """Daily-target counterpart of BaseModel.

    Same fit/predict/save/load signatures, but `fit` takes the daily
    target as a Series and `predict` returns a one-column DataFrame
    (`y_daily`) rather than the 24-column hourly frame. The daily runner
    is separate from the hourly one precisely so neither harness has to
    branch on which contract it is holding.
    """

    def _check_single_row(self, X: pd.DataFrame) -> None:
        if len(X) != 1:
            raise ValueError(
                f"{type(self).__name__}.predict expects exactly one "
                f"target_day row (the walk-forward origin), got {len(X)}"
            )

    def _as_frame(self, values, index) -> pd.DataFrame:
        return pd.DataFrame({DAILY_COLUMN: values}, index=index)


class DailyNaiveModel(DailyModel):
    """Lago day-of-week naive on the baseload: Monday -> D-3, weekend ->
    D-7, Tuesday-Friday -> D-1.

    Averaging this model's output over a day is identical to averaging
    the hourly naive's 24 outputs, because both are the same linear
    operation on the same lag block. That identity is asserted in the
    tests and is the cleanest available check that the direct and
    aggregated routes are wired to the same data.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailyNaiveModel":
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_single_row(X)
        dow = X.index.dayofweek
        lag = 3 if dow[0] == 0 else (7 if dow[0] >= 5 else 1)
        cols = [f"price_D-{lag}_{h}" for h in HOURS]
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"DailyNaiveModel.predict: X missing {missing[:3]}")
        return self._as_frame(X[cols].mean(axis=1).to_numpy(), X.index)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "DailyNaiveModel":
        return self._pickle_load(path)


class DailyLightGBMModel(DailyModel):
    """One LGBMRegressor on the baseload target, consuming the same full
    build_features() X as the hourly wrapper."""

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.refit_every_n_days = int(
            self.cfg.get("refit_every_n_days", DEFAULT_REFIT_EVERY_N_DAYS)
        )
        self.params = resolve_tuned_params(self.cfg, dict(self.cfg.get("params", {})))
        self.params.update(FORCED_PARAMS)
        self._model: lgb.LGBMRegressor | None = None
        self._feature_columns: list[str] | None = None
        self._last_refit_end: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailyLightGBMModel":
        train_days = y.index.sort_values()
        train_end = train_days.max()
        needs_full_refit = (
            self._model is None
            or self._last_refit_end is None
            or (train_end - self._last_refit_end).days >= self.refit_every_n_days
        )

        # Runs on EVERY call, not only refits -- with cadence > 1 a reordered
        # X on a skip-day would otherwise pass silently until predict().
        if self._feature_columns is None:
            self._feature_columns = list(X.columns)
        elif list(X.columns) != self._feature_columns:
            raise ValueError(
                "DailyLightGBMModel.fit: X column order changed between calls "
                "-- refusing to fit on silently reordered features"
            )

        if needs_full_refit:
            model = lgb.LGBMRegressor(**self.params)
            model.fit(X.loc[train_days].values, y.loc[train_days].values)
            self._model = model
            # Stamp only on an actual refit: the origin advances one day per
            # call, so an unconditional stamp would pin the cadence at 1 day.
            self._last_refit_end = train_end

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_single_row(X)
        if self._model is None:
            raise ValueError("DailyLightGBMModel.predict called before fit")
        if list(X.columns) != self._feature_columns:
            raise ValueError(
                "DailyLightGBMModel.predict: X columns do not match the "
                "columns the model was fitted on"
            )
        return self._as_frame(self._model.predict(X.values), X.index)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "DailyLightGBMModel":
        return self._pickle_load(path)


class DailySARIMAXModel(DailyModel):
    """ONE SARIMAX on the baseload series, where the hourly wrapper fits 24.

    The hourly model's per-hour independence exists because each hour is
    its own price series; the baseload is a single series, so the direct
    daily route is a single model by construction, not by simplification.

    Exogenous regressors are the daily means of the same `exog_*_D0`
    day-ahead forecast columns the hourly wrapper uses per hour. Averaging
    them mirrors exactly what the target does to the 24 prices, which keeps
    the direct and aggregated routes commensurable (the RQ4 requirement
    that only the target differs). They remain legal: a D0 forecast is
    known before the origin, and averaging cannot import later information.

    Refit cadence and the append-only update between refits follow
    src/models/sarimax.py; see that module's docstring for the bounded
    start-of-window staleness this introduces (never a leakage path).
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.order = tuple(self.cfg.get("order", DEFAULT_ORDER))
        self.seasonal_order = tuple(self.cfg.get("seasonal_order", DEFAULT_SEASONAL_ORDER))
        self.exog_prefixes = tuple(self.cfg.get("exog_columns_prefix", DEFAULT_EXOG_PREFIXES))
        self.refit_every_n_days = int(
            self.cfg.get("refit_every_n_days", DEFAULT_SARIMAX_REFIT_EVERY_N_DAYS)
        )
        self._result = None
        self._last_refit_end: pd.Timestamp | None = None

    def _daily_exog(self, X: pd.DataFrame) -> pd.DataFrame:
        data = {}
        for prefix in self.exog_prefixes:
            cols = [f"{prefix}_{h}" for h in HOURS]
            missing = [c for c in cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"DailySARIMAXModel: X missing exog columns for '{prefix}': {missing[:3]}"
                )
            data[prefix] = X[cols].mean(axis=1)
        return pd.DataFrame(data, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailySARIMAXModel":
        train_days = y.index.sort_values()
        train_end = train_days.max()
        needs_full_refit = (
            self._result is None
            or self._last_refit_end is None
            or (train_end - self._last_refit_end).days >= self.refit_every_n_days
        )

        endog = y.loc[train_days]
        exog = self._daily_exog(X.loc[train_days])

        if needs_full_refit:
            model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
            # Stamp only on a real refit -- see src/models/sarimax.py: an
            # unconditional stamp pins the cadence at 1 day forever.
            self._last_refit_end = train_end
        else:
            prev_end = self._result.model.data.row_labels.max()
            new_days = train_days[train_days > prev_end]
            if len(new_days) > 0:
                self._result = self._result.append(
                    endog.loc[new_days], exog=exog.loc[new_days], refit=False
                )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_single_row(X)
        if self._result is None:
            raise ValueError("DailySARIMAXModel.predict called before fit")
        forecast = self._result.forecast(steps=1, exog=self._daily_exog(X))
        return self._as_frame([forecast.iloc[0]], X.index)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "DailySARIMAXModel":
        return self._pickle_load(path)


class DailyLEARLassoModel(DailyModel):
    """LEAR's own estimation recipe applied to the baseload target.

    epftoolbox's `LEAR` class is hardwired to a 24-column Y (it loops
    `for h in range(24)` and fits one Lasso per hour), so the direct daily
    route cannot call it. What it CAN reuse -- and does here -- is every
    numerically significant piece of that recipe, taken from epftoolbox
    itself rather than reimplemented:

      * `epftoolbox.data.scaling(..., 'Invariant')` (asinh-median) on the
        target and on all inputs except the 7 day-of-week dummies,
      * `LassoLarsIC(criterion='aic', max_iter=2500)` to select lambda,
      * `Lasso(max_iter=2500, alpha=lambda)` for the final fit.

    Verified line-by-line against `epftoolbox/models/_lear.py`
    (recalibrate/predict). The only difference is that the 24-iteration
    loop collapses to a single fit, because the target is one number per
    day. Calling this a transposition of LEAR rather than a new model is
    therefore accurate, and it keeps the hourly and daily LEAR arms
    comparable -- which is the whole point of RQ4.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        # Import-time check only, matching LEARLassoModel: keeps the rest of
        # src.models importable without epftoolbox installed.
        from epftoolbox.data import scaling  # noqa: F401

        self._model = None
        self._scaler_x = None
        self._scaler_y = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailyLEARLassoModel":
        from epftoolbox.data import scaling
        from sklearn.linear_model import Lasso, LassoLarsIC

        # The dummies must be the last 7 columns: 'Invariant' scaling is
        # applied to X[:, :-7] positionally, exactly as LEAR does it.
        _assert_dow_columns_last(X)
        train_days = y.index.sort_values()

        Xtrain = X.loc[train_days].to_numpy(dtype=float, copy=True)
        ytrain = y.loc[train_days].to_numpy(dtype=float).reshape(-1, 1)

        [ytrain_s], self._scaler_y = scaling([ytrain], "Invariant")
        [x_no_dummies], self._scaler_x = scaling([Xtrain[:, :-7]], "Invariant")
        Xtrain[:, :-7] = x_no_dummies

        alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xtrain, ytrain_s[:, 0]).alpha_
        self._model = Lasso(max_iter=2500, alpha=alpha).fit(Xtrain, ytrain_s[:, 0])

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_single_row(X)
        if self._model is None:
            raise ValueError("DailyLEARLassoModel.predict called before fit")
        _assert_dow_columns_last(X)

        # .copy() because the scaler writes back into the array; the harness
        # still holds the original DataFrame (same guard as LEARLassoModel).
        Xtest = X.to_numpy(dtype=float, copy=True)
        Xtest[:, :-7] = self._scaler_x.transform(Xtest[:, :-7])
        pred_s = self._model.predict(Xtest)
        pred = self._scaler_y.inverse_transform(pred_s.reshape(1, -1))
        return self._as_frame(pred.ravel(), X.index)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "DailyLEARLassoModel":
        return self._pickle_load(path)


class DailyLSTMModel(DailyModel, LSTMModel):
    """The hourly LSTM with a 1-unit output head instead of 24.

    Subclasses `LSTMModel` to inherit the sequence/static feature split,
    the scaling, the refit cadence, the determinism setup and the
    keras-aware save/load -- all of it read-only. Nothing in the hourly
    wrapper is modified, so the audited path that produced the committed
    hourly results is untouched (the reason daily.py exists at all).

    Only three things change, and all three follow from the target being
    one number rather than 24: the output layer's width, the target's
    shape on the way in, and the predicted frame's schema on the way out.
    """

    def _build_net(self, n_static: int):
        net = super()._build_net(n_static)
        keras = _keras()
        # Rebuild the head at width 1. Cheaper and less brittle than
        # duplicating the parent's architecture: if the hourly topology
        # changes, this follows it automatically.
        out = keras.layers.Dense(1)(net.layers[-1].input)
        daily_net = keras.Model(net.inputs, out)
        daily_net.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mae"
        )
        return daily_net

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DailyLSTMModel":
        target = y.to_frame(DAILY_COLUMN) if isinstance(y, pd.Series) else y
        LSTMModel.fit(self, X, target)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_single_row(X)
        if self._net is None or self._scalers is None:
            raise ValueError("DailyLSTMModel.predict called before fit")
        if list(X.columns) != self._feature_columns:
            raise ValueError(
                "DailyLSTMModel.predict: X columns do not match the columns "
                "the model was fitted on"
            )

        seq_scaler, static_scaler, y_scaler = self._scalers
        seq, static = self._split(X)
        seq_s = seq_scaler.transform(seq.reshape(1, -1)).reshape(seq.shape)
        static_s = static_scaler.transform(static)
        pred_s = self._net.predict([seq_s, static_s], verbose=0)
        pred = y_scaler.inverse_transform(pred_s)
        return self._as_frame(np.asarray(pred).ravel(), X.index)


def _keras():
    from src.models.lstm import _tf

    return _tf().keras
