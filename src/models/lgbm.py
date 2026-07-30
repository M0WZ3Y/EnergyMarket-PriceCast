"""LightGBM model — src/models/lgbm.py

24 independent per-hour LGBMRegressors (one per hour-of-day), mirroring
LEAR-LASSO's per-hour independence, each consuming the FULL shared
build_features() X (price lags, exog lags/D0, weekday dummies) — the same
single leakage-audited feature source of truth every other wrapper uses.

Module is named lgbm.py (not lightgbm.py) so it can never be confused
with, or shadow, the lightgbm package itself.

Recalibration cadence: full refit every `refit_every_n_days` (default 1 =
strict daily recalibration, matching the LEAR protocol — LightGBM fits in
seconds on a 1092-day window, so unlike SARIMAX no cadence compromise is
needed). Between refits (if cadence > 1 is ever configured) the fitted
model is simply reused as-is; predictions are still produced every origin.

Hyperparameters: `params` from configs/models.yaml, with the tuned-params
file (written by scripts/tune_lightgbm.py, 50 Optuna trials on a
validation window strictly before the test period) merged over the
defaults when it exists. Determinism: seed and `deterministic=True` are
always forced (seed 42 project rule).
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from src.models.base import HOURS, Y_COLUMNS, BaseModel

DEFAULT_REFIT_EVERY_N_DAYS = 1
DEFAULT_SEED = 42

# Non-negotiable settings merged over any config/tuned params: reproducible
# fits regardless of what the tuning search space contains. n_jobs is a
# FIXED constant, not "-1"/auto: LightGBM's deterministic=True only
# guarantees identical results for an identical thread count, so the
# count must never float with the machine. (Raised 1 -> 4 on 2026-07-29
# after single-threaded tuning trials ran up to 37 min each; see
# logs/decisions.md.)
FORCED_PARAMS = dict(
    random_state=DEFAULT_SEED,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
    n_jobs=4,
)


class LightGBMModel(BaseModel):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.refit_every_n_days = int(
            self.cfg.get("refit_every_n_days", DEFAULT_REFIT_EVERY_N_DAYS)
        )
        self.params = dict(self.cfg.get("params", {}))

        tuned_file = self.cfg.get("tuned_params_file")
        if tuned_file:
            # Relative paths resolve against the repo root, not the cwd --
            # a cwd-dependent lookup would silently fall back to untuned
            # defaults when run from elsewhere, a reproducibility hazard
            # for the frozen v1.0-results numbers.
            tuned_path = Path(tuned_file)
            if not tuned_path.is_absolute():
                tuned_path = Path(__file__).resolve().parents[2] / tuned_path
            if tuned_path.exists():
                with open(tuned_path) as f:
                    tuned = yaml.safe_load(f) or {}
                if "params" not in tuned:
                    raise ValueError(
                        f"tuned params file {tuned_path} has no 'params' key -- "
                        "refusing to merge file metadata into booster params"
                    )
                self.params.update(tuned["params"])

        self.params.update(FORCED_PARAMS)
        self._models: dict[int, lgb.LGBMRegressor] = {}
        self._feature_columns: list[str] | None = None
        self._last_refit_end: pd.Timestamp | None = None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LightGBMModel":
        train_days = Y.index.sort_values()
        train_end = train_days.max()
        needs_full_refit = (
            not self._models
            or self._last_refit_end is None
            or (train_end - self._last_refit_end).days >= self.refit_every_n_days
        )

        # Column-order guard runs on EVERY fit() call, not only refit calls
        # -- with cadence > 1 a reordered X on a skip-day would otherwise
        # pass silently until predict().
        if self._feature_columns is None:
            self._feature_columns = list(X.columns)
        elif list(X.columns) != self._feature_columns:
            raise ValueError(
                "LightGBMModel.fit: X column order changed between calls "
                "-- refusing to fit on silently reordered features"
            )

        if needs_full_refit:
            X_train = X.loc[train_days]
            for h_idx, hour in enumerate(HOURS):
                model = lgb.LGBMRegressor(**self.params)
                model.fit(X_train.values, Y.loc[train_days, f"y_{hour}"].values)
                self._models[h_idx] = model
            # Only stamp on an actual refit (same rationale as SARIMAXModel:
            # the origin advances 1 day per call, so an unconditional stamp
            # would keep the cadence check at 1 day forever).
            self._last_refit_end = train_end

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) != 1:
            raise ValueError(
                f"LightGBMModel.predict expects exactly one target_day row "
                f"(the walk-forward origin), got {len(X)}"
            )
        if list(X.columns) != self._feature_columns:
            raise ValueError(
                "LightGBMModel.predict: X columns do not match the columns "
                "the model was fitted on"
            )

        preds = {
            f"y_{hour}": self._models[h_idx].predict(X.values)[0]
            for h_idx, hour in enumerate(HOURS)
        }
        return pd.DataFrame([preds], index=X.index, columns=Y_COLUMNS)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "LightGBMModel":
        return self._pickle_load(path)
