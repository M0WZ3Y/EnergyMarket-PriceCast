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
import pandas as pd
import yaml

from src.models.base import HOURS, BaseModel

DAILY_COLUMN = "y_daily"
DEFAULT_SEED = 42
DEFAULT_REFIT_EVERY_N_DAYS = 1

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
