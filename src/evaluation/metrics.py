"""Forecast accuracy metrics — src/evaluation/metrics.py

Thin wrappers around epftoolbox.evaluation, reused verbatim from the
Lago et al. reference implementation rather than reimplemented, so
results stay numerically comparable to the published benchmark — the
same convention already used for the feature pipeline (see
logs/decisions.md, week 3).

No plain MAPE: electricity prices go negative, which makes percentage
error undefined/explosive (CLAUDE.md). Use sMAPE and rMAE instead.
"""

from __future__ import annotations

from epftoolbox.evaluation import DM
from epftoolbox.evaluation import MAE as mae
from epftoolbox.evaluation import RMSE as rmse
from epftoolbox.evaluation import rMAE as rmae
from epftoolbox.evaluation import sMAPE as smape

__all__ = ["mae", "rmse", "smape", "rmae", "diebold_mariano"]


def diebold_mariano(p_real, p_pred_1, p_pred_2, norm: int = 1, version: str = "multivariate") -> float:
    """Diebold-Mariano test p-value: is model 1 significantly more accurate
    than model 2 (one-sided)? Thin wrapper around epftoolbox.evaluation.DM.
    """
    return DM(p_real=p_real, p_pred_1=p_pred_1, p_pred_2=p_pred_2, norm=norm, version=version)
