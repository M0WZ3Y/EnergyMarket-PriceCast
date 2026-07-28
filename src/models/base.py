"""Common model wrapper interface — src/models/base.py

Every EPF model (naive, SARIMAX, LEAR-LASSO, and later LightGBM, LSTM)
implements fit/predict/save/load on this one interface (CLAUDE.md
Conventions: "Model wrappers implement fit/predict/save/load on a common
interface"), so src/evaluation/run_baselines.py can drive any of them from
walk_forward_splits() without per-model special-casing.
"""

from __future__ import annotations

import abc
import pickle
from pathlib import Path

import pandas as pd

HOURS = [f"h{h:02d}" for h in range(24)]
Y_COLUMNS = [f"y_{h}" for h in HOURS]


class BaseModel(abc.ABC):
    """Common interface every EPF model wrapper implements.

    fit()/predict() always take/return DataFrames in build_features()'s
    exact shape (X indexed by target_day, predictions with y_h00..y_h23
    columns matching Y's schema) so the harness never needs to branch on
    model type.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "BaseModel":
        """Fit/recalibrate on one walk-forward split's train_days worth of
        (X, Y), both indexed by target_day per build_features(). Must be
        idempotent — called fresh on every origin under daily
        recalibration, so must not accumulate state across calls beyond
        what a model's own docstring explicitly documents (e.g. SARIMAX's
        refit cadence). Returns self for chaining.
        """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict the 24 hourly prices for the row(s) in X (indexed by
        target_day). Returns a DataFrame indexed like X, columns
        y_h00..y_h23 — same column names as build_features()'s Y.
        """

    @abc.abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abc.abstractmethod
    def load(self, path: str | Path) -> "BaseModel": ...

    def _pickle_save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def _pickle_load(self, path: str | Path) -> "BaseModel":
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))
        return self
