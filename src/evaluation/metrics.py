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

import numpy as np
from epftoolbox.evaluation import DM
from epftoolbox.evaluation import MAE as mae
from epftoolbox.evaluation import RMSE as rmse
from epftoolbox.evaluation import rMAE as rmae
from epftoolbox.evaluation import sMAPE as smape
from scipy import stats

__all__ = [
    "mae",
    "rmse",
    "smape",
    "rmae",
    "diebold_mariano",
    "diebold_mariano_hac",
    "loss_differential",
]


def diebold_mariano(p_real, p_pred_1, p_pred_2, norm: int = 1, version: str = "multivariate") -> float:
    """Diebold-Mariano test p-value: is model 1 significantly more accurate
    than model 2 (one-sided)? Thin wrapper around epftoolbox.evaluation.DM.

    UNCORRECTED: assumes independent loss differentials. Electricity price
    forecast errors are serially dependent, so this understates standard
    errors and overstates significance. Kept for numerical comparability
    with Lago et al.; prefer diebold_mariano_hac() for reported results
    (decision 2026-08-04).
    """
    return DM(p_real=p_real, p_pred_1=p_pred_1, p_pred_2=p_pred_2, norm=norm, version=version)


def loss_differential(p_real, p_pred_1, p_pred_2, norm: int = 1) -> np.ndarray:
    """Per-day multivariate loss differential, positive where p_pred_2 wins.

    Matches epftoolbox's 'multivariate' convention: each day contributes one
    number, the norm of its 24-h error vector. norm=1 -> mean absolute error
    across the day; norm=2 -> RMSE across the day.
    """
    real = np.asarray(p_real, dtype=float)
    e1 = np.abs(real - np.asarray(p_pred_1, dtype=float))
    e2 = np.abs(real - np.asarray(p_pred_2, dtype=float))
    if norm == 1:
        l1, l2 = e1.mean(axis=1), e2.mean(axis=1)
    elif norm == 2:
        l1, l2 = np.sqrt((e1**2).mean(axis=1)), np.sqrt((e2**2).mean(axis=1))
    else:
        raise ValueError(f"loss_differential: norm must be 1 or 2, got {norm}")
    return l1 - l2


def diebold_mariano_hac(p_real, p_pred_1, p_pred_2, norm: int = 1, bandwidth: int | None = None) -> float:
    """HAC (Newey-West) Diebold-Mariano p-value — the REPORTED variant.

    Same one-sided question and same argument convention as
    diebold_mariano() (small p supports 'p_pred_2 more accurate'), but the
    variance of the mean loss differential is estimated with a Bartlett
    kernel instead of assuming independence.

    Day-ahead price forecast errors are autocorrelated — regimes persist,
    and every model shares the same market shocks — so the uncorrected
    statistic is anti-conservative. Bandwidth defaults to the standard
    4*(n/100)**(2/9) rule (Newey-West 1994).
    """
    d = loss_differential(p_real, p_pred_1, p_pred_2, norm=norm)
    n = len(d)
    if n < 3:
        raise ValueError(f"diebold_mariano_hac: need at least 3 days, got {n}")

    dbar = float(d.mean())
    e = d - dbar
    L = bandwidth if bandwidth is not None else max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    L = min(L, n - 1)

    var = float(e @ e) / n
    for k in range(1, L + 1):
        var += 2.0 * (1.0 - k / (L + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    if var <= 0:  # negative HAC estimates are possible; fall back to iid
        var = float(e @ e) / n
    var = max(var, 1e-12)

    return float(1.0 - stats.norm.cdf(dbar / np.sqrt(var / n)))
