"""SARIMAX baseline — src/models/sarimax.py

24 independent per-hour statsmodels SARIMAX models (one per hour-of-day),
mirroring LEAR-LASSO's own per-hour independence. Exogenous regressors are
the target day's own day-ahead exog_1/exog_2 forecasts (the "D0" columns
build_features() provides) — legal, since those are known before the
forecast origin.

Recalibration cadence: unlike LEAR-LASSO (which fully re-fits every origin,
cheap for a 24-way Lasso), SARIMAX fully refits only every
`refit_every_n_days` (default 7) and uses statsmodels' cheap
`append(..., refit=False)` state-space update in between. The harness
still calls fit()/predict() every origin and still *forecasts* every
origin day (fair daily comparison against the other models) — only the
underlying SARIMAX parameters are held fixed between full refits. This is
a deliberate, logged deviation from strict daily recalibration: fully
refitting 24 seasonal SARIMAX models across ~730 walk-forward origins
(~17.5k fits) is not practical, and there is no upstream literature
precedent (unlike LEAR) constraining SARIMAX's cadence specifically. See
logs/decisions.md for the full rationale.

One side effect of the append-only update: because build_features()'s
walk-forward train_days window is a fixed-size *sliding* window (drops the
oldest day as the newest is added), append-only updates cannot also drop
the oldest observations from the fitted model's own history without a
full rebuild. Between refits, the fitted SARIMAX therefore keeps a few
extra (already-out-of-window) days of history rather than exactly
matching the harness's nominal train_days start boundary. This is a minor,
bounded (<= refit_every_n_days days) staleness at the *start* of the
window only -- it never reads information from after the forecast origin,
so it does not violate the project's leakage rule.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import HOURS, Y_COLUMNS, BaseModel

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 0, 1, 7)
DEFAULT_EXOG_PREFIXES = ("exog_1_D0", "exog_2_D0")
DEFAULT_REFIT_EVERY_N_DAYS = 7


class SARIMAXModel(BaseModel):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.order = tuple(self.cfg.get("order", DEFAULT_ORDER))
        self.seasonal_order = tuple(self.cfg.get("seasonal_order", DEFAULT_SEASONAL_ORDER))
        self.exog_prefixes = tuple(self.cfg.get("exog_columns_prefix", DEFAULT_EXOG_PREFIXES))
        self.refit_every_n_days = int(self.cfg.get("refit_every_n_days", DEFAULT_REFIT_EVERY_N_DAYS))
        self._results: dict[int, object] = {}
        self._last_refit_end: pd.Timestamp | None = None

    def _exog_columns(self, hour: str) -> list[str]:
        return [f"{prefix}_{hour}" for prefix in self.exog_prefixes]

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "SARIMAXModel":
        train_days = Y.index.sort_values()
        train_end = train_days.max()
        needs_full_refit = (
            not self._results
            or self._last_refit_end is None
            or (train_end - self._last_refit_end).days >= self.refit_every_n_days
        )

        for h_idx, hour in enumerate(HOURS):
            endog = Y.loc[train_days, f"y_{hour}"]
            exog = X.loc[train_days, self._exog_columns(hour)]

            if needs_full_refit or h_idx not in self._results:
                model = SARIMAX(
                    endog,
                    exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self._results[h_idx] = model.fit(disp=False)
            else:
                prev_end = self._results[h_idx].model.data.row_labels.max()
                new_days = train_days[train_days > prev_end]
                if len(new_days) > 0:
                    self._results[h_idx] = self._results[h_idx].append(
                        endog.loc[new_days], exog=exog.loc[new_days], refit=False
                    )

        if needs_full_refit:
            # Only stamp _last_refit_end when a full refit actually happened
            # -- walk_forward_splits advances train_end by step_days (1) on
            # every call, so unconditionally updating this here would make
            # (train_end - _last_refit_end).days == 1 forever, and
            # needs_full_refit could never re-trigger after the first cycle.
            self._last_refit_end = train_end
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) != 1:
            raise ValueError(
                f"SARIMAXModel.predict expects exactly one target_day row "
                f"(the walk-forward origin), got {len(X)}"
            )

        preds = {}
        for h_idx, hour in enumerate(HOURS):
            exog_row = X[self._exog_columns(hour)]
            forecast = self._results[h_idx].forecast(steps=1, exog=exog_row)
            preds[f"y_{hour}"] = forecast.iloc[0]

        return pd.DataFrame([preds], index=X.index, columns=Y_COLUMNS)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "SARIMAXModel":
        return self._pickle_load(path)
