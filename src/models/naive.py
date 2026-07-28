"""Naive day-of-week baseline — src/models/naive.py

The standard Lago et al. "similar day" naive: no parameters, no fitting —
it is a pure column-selection rule over the lag columns build_features()
already provides.

  - Monday             -> price_D-3 (last Friday; D-1/D-2 are the weekend,
                          not representative of a weekday)
  - Saturday, Sunday   -> price_D-7 (same weekday last week)
  - Tuesday..Friday    -> price_D-1 (yesterday)

Requires configs/features.yaml's price_lag_days to include 1, 3, and 7
(the shipped default [1, 2, 3, 7] does).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.base import HOURS, Y_COLUMNS, BaseModel


class NaiveModel(BaseModel):
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "NaiveModel":
        # No parameters to learn -- fit is a no-op by contract, present
        # only so NaiveModel conforms to the shared BaseModel interface.
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        dow = X.index.dayofweek  # Monday=0 .. Sunday=6, matches build_features()
        is_monday = dow == 0
        is_weekend = dow.isin([5, 6])
        is_midweek = ~(is_monday | is_weekend)

        out = pd.DataFrame(index=X.index, columns=Y_COLUMNS, dtype=float)
        out.loc[is_monday, :] = X.loc[is_monday, [f"price_D-3_{h}" for h in HOURS]].values
        out.loc[is_weekend, :] = X.loc[is_weekend, [f"price_D-7_{h}" for h in HOURS]].values
        out.loc[is_midweek, :] = X.loc[is_midweek, [f"price_D-1_{h}" for h in HOURS]].values
        return out

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "NaiveModel":
        return self._pickle_load(path)
