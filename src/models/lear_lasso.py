"""LEAR-LASSO baseline — src/models/lear_lasso.py

Resolves the week-3/4 open item (logs/decisions.md): this wrapper feeds
the shared src/features/pipeline.py build_features() output directly into
epftoolbox.models.LEAR's low-level API (LEAR.recalibrate(Xtrain, Ytrain) /
LEAR.predict(X)), not the high-level recalibrate_and_forecast_next_day
(which reruns its own internal df -> X/Y builder). Confirmed by reading
epftoolbox/models/_lear.py: recalibrate() accepts pre-built
[n_days, n_features] / [n_days, 24] numpy arrays and does not rebuild
features itself; it expects the last 7 columns of X to be day-of-week
dummies -- exactly build_features()'s existing column order (price/exog
lag blocks first, dow_0..dow_6 appended last). Using the shared pipeline
keeps one leakage-audited feature source of truth and gives LEAR the same
fit(X, Y)/predict(X) call shape as every other model wrapper.

epftoolbox.models.LEAR is imported inside __init__, not at module level,
so the rest of src/models stays importable without epftoolbox installed
(same pattern as src/data/loader.py).

Caution (verified against _lear.py source): LEAR.recalibrate() and
LEAR.predict() both mutate their X argument in place while rescaling the
non-dummy columns. This wrapper always passes a fresh .copy() of the
underlying array so repeated calls never double-scale or corrupt the
DataFrame the harness still holds a reference to.

Environment note (see logs/decisions.md): epftoolbox's own
LEAR.predict() (epftoolbox/models/_lear.py:109) does
`Yp[h] = self.models[h].predict(X)`, relying on numpy's old implicit
array-to-scalar coercion for the length-1 array each Lasso model returns.
numpy>=1.25 deprecated this and numpy 2.x removed it outright, raising
"setting an array element with a sequence" -- epftoolbox declares only
`numpy>=1` and was never updated for this. Downgrading numpy in this repo
isn't viable (the installed TensorFlow, an epftoolbox dependency via its
DNN model, requires numpy 2.x). recalibrate() itself (LassoLarsIC + Lasso
fitting -- the numerically significant part) is unaffected and used
as-is; only predict()'s trivial final assembly loop is reproduced here,
using the exact same fitted `self._lear.models[h]` / `scalerX` /
`scalerY` objects epftoolbox's own recalibrate() produced, so this is a
numpy-2.x compatibility shim, not a reimplementation of LEAR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import HOURS, Y_COLUMNS, BaseModel
from src.evaluation.walk_forward import load_evaluation_config

DOW_COLUMNS = [f"dow_{i}" for i in range(7)]


def _assert_dow_columns_last(X: pd.DataFrame) -> None:
    tail = list(X.columns[-7:])
    if tail != DOW_COLUMNS:
        raise ValueError(
            "LEAR-LASSO requires build_features()'s last 7 columns to be "
            f"the day-of-week dummies {DOW_COLUMNS} (LEAR.recalibrate/predict "
            f"scale everything except the last 7 columns); got {tail}. Check "
            "configs/features.yaml column ordering."
        )


class LEARLassoModel(BaseModel):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        from epftoolbox.models import LEAR

        window = self.cfg.get("calibration_window_days")
        if window is None:
            window = load_evaluation_config()["walk_forward"]["calibration_window_days"]
        self._lear = LEAR(calibration_window=window)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LEARLassoModel":
        _assert_dow_columns_last(X)
        Xtrain = X.to_numpy(dtype=float, copy=True)
        Ytrain = Y.to_numpy(dtype=float, copy=True)
        self._lear.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) != 1:
            raise ValueError(
                f"LEARLassoModel.predict expects exactly one target_day row "
                f"(the walk-forward origin), got {len(X)}"
            )
        _assert_dow_columns_last(X)
        Xtest = X.to_numpy(dtype=float, copy=True)

        # Reproduces epftoolbox.models.LEAR.predict() exactly (see module
        # docstring's "Environment note"), except for the final assignment,
        # which is numpy-2.x safe: `.predict(Xtest)[0]` explicitly extracts
        # the scalar instead of relying on numpy's removed implicit
        # length-1-array-to-scalar coercion.
        Xtest[:, :-7] = self._lear.scalerX.transform(Xtest[:, :-7])
        Yp = np.array([self._lear.models[h].predict(Xtest)[0] for h in range(24)])
        preds = self._lear.scalerY.inverse_transform(Yp.reshape(1, -1))

        return pd.DataFrame(preds, index=X.index, columns=Y_COLUMNS)

    def save(self, path: str | Path) -> None:
        self._pickle_save(path)

    def load(self, path: str | Path) -> "LEARLassoModel":
        return self._pickle_load(path)
