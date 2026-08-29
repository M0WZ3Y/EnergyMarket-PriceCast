"""LEAR with BIC model selection — A DIAGNOSTIC, NOT A THESIS MODEL.

NOT part of the sanctioned model set. CLAUDE.md fixes that set at naive,
SARIMAX, LEAR-LASSO, LightGBM, LSTM and the weighted ensemble, and this file
does not extend it: it is deliberately NOT registered in `src.models.__init__`,
its name is underscore-prefixed, and nothing in the thesis pipeline imports it.
It exists to answer one question about the ABLATION HARNESS and should never
produce a reported thesis result.

THE QUESTION. epftoolbox's LEAR selects the LASSO penalty with
`LassoLarsIC(criterion='aic')`, refit inside every `recalibrate` call, so alpha
IS re-selected per variant, per hour, per origin -- nothing is cached from the
baseline. But AIC's penalty is a constant 2 per parameter regardless of sample
size, and its noise-variance estimate needs n > p and degrades as p approaches
n. Measured on this data at the first ablation origin:

    variant    p     p/n     mean alpha    non-zero coefs
    baseline   247   0.25    0.00067       111
    B1_ramp    274   0.28    0.00111       103
    ALL        800   0.81    0.00051       319

Alpha FALLS as p triples, and the number of retained coefficients nearly
triples with it. That is backwards: a well-behaved selection rule tightens the
penalty as the feature count grows, so useless columns shrink toward zero and a
wide variant decays GRACEFULLY toward the best narrow one. Under AIC at
p/n = 0.81 it does not, which means part of ALL's measured degradation is a
property of the selection criterion rather than of the features.

WHAT THIS CHANGES. Exactly one line: `criterion='bic'`. BIC's penalty is
log(n) per parameter -- about 6.9 here versus AIC's 2 -- so it selects a larger
alpha and a sparser model as p grows. Everything else reproduces
`epftoolbox.models.LEAR.recalibrate` verbatim: the same Invariant
(asinh-median) scaling of Y, the same scaling of all features except the
trailing 7 dummies, the same per-hour loop, the same `Lasso(max_iter=2500)`
refit at the selected alpha.

Reimplementing `recalibrate` is unfortunate but unavoidable: epftoolbox
hardcodes the criterion inside the method, so there is no parameter to pass.
The body below is kept line-for-line comparable to the original so the
divergence stays auditable to that single word.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC

from src.models.lear_lasso import Y_COLUMNS, LEARLassoModel, _assert_dow_columns_last


class LEARLassoBICModel(LEARLassoModel):
    """LEAR-LASSO with BIC instead of AIC. Harness diagnostic only."""

    criterion = "bic"

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LEARLassoBICModel":
        from epftoolbox.data import scaling

        _assert_dow_columns_last(X)
        Xtrain = X.to_numpy(dtype=float, copy=True)
        Ytrain = Y.to_numpy(dtype=float, copy=True)

        # Identical to LEAR.recalibrate from here down, except `criterion`.
        [Ytrain], self._lear.scalerY = scaling([Ytrain], "Invariant")
        [Xtrain_no_dummies], self._lear.scalerX = scaling([Xtrain[:, :-7]], "Invariant")
        Xtrain[:, :-7] = Xtrain_no_dummies

        self._lear.models = {}
        for h in range(24):
            param_model = LassoLarsIC(criterion=self.criterion, max_iter=2500)
            param = param_model.fit(Xtrain, Ytrain[:, h]).alpha_
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(Xtrain, Ytrain[:, h])
            self._lear.models[h] = model

        self.is_fitted = True
        return self

    def selected_alphas(self, X: pd.DataFrame, Y: pd.DataFrame) -> np.ndarray:
        """The 24 per-hour alphas this criterion picks. Reporting aid."""
        from epftoolbox.data import scaling

        Xtrain = X.to_numpy(dtype=float, copy=True)
        Ytrain = Y.to_numpy(dtype=float, copy=True)
        [Ytrain], _ = scaling([Ytrain], "Invariant")
        [Xnd], _ = scaling([Xtrain[:, :-7]], "Invariant")
        Xtrain[:, :-7] = Xnd
        return np.array(
            [
                LassoLarsIC(criterion=self.criterion, max_iter=2500)
                .fit(Xtrain, Ytrain[:, h])
                .alpha_
                for h in range(24)
            ]
        )
