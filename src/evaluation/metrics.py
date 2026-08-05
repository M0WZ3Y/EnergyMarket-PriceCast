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
    """Diebold-Mariano test p-value (one-sided). Thin wrapper around
    epftoolbox.evaluation.DM.

    DIRECTION: a SMALL p-value supports "p_pred_2 is more accurate than
    p_pred_1" -- the same convention as diebold_mariano_hac() and the one
    dm_matrix() relies on when it places the row model in p_pred_2. This
    follows from epftoolbox's d = loss(pred_1) - loss(pred_2): pred_1 being
    worse makes d positive, the statistic positive, and p small. Until
    2026-08-05 this docstring asserted the opposite ("is model 1 more
    accurate than model 2"), which was simply wrong; the code, dm_matrix
    and every exported table were always on the convention stated here, so
    no published number was affected -- only this description of it.

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
    number summarizing its 24-h error vector. norm=1 -> mean absolute error
    across the day; norm=2 -> MEAN SQUARED error across the day.

    norm=2 is deliberately MSE, not RMSE, because that is what epftoolbox's
    _dm.py computes (d = mean(e1**2, axis=1) - mean(e2**2, axis=1)). The two
    are not interchangeable here: a monotone transform of each series does
    not survive a DIFFERENCE of two series, so taking the square root first
    changes dbar, the HAC variance and the p-value. dm_matrix advertises
    'hac' and 'uncorrected' as the same test with a different variance
    estimator, which is only true while both use this basis.
    """
    real = np.asarray(p_real, dtype=float)
    e1 = np.abs(real - np.asarray(p_pred_1, dtype=float))
    e2 = np.abs(real - np.asarray(p_pred_2, dtype=float))
    if norm == 1:
        l1, l2 = e1.mean(axis=1), e2.mean(axis=1)
    elif norm == 2:
        l1, l2 = (e1**2).mean(axis=1), (e2**2).mean(axis=1)
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

    iid_var = float(e @ e) / n

    # A loss differential with no sampling variation carries no evidence: the
    # DM statistic dbar/sqrt(var/n) has nothing to standardize against. There
    # are two such cases and they are NOT the same.
    #
    #   dbar == 0 too (identical forecasts): the differential is exactly zero
    #   everywhere. "Neither model is better" is the correct, meaningful
    #   answer, and the one-sided p-value for it is 0.5.
    #
    #   dbar != 0 (e.g. one model uniformly wrong by exactly 1.0): the old
    #   code floored var at 1e-12, which turned a degenerate input into
    #   z ~ 3.2e6 and p == 0.0 -- certainty manufactured out of an absent
    #   variance estimate, not a statistical result. Refuse it.
    #
    # The tolerance is scaled by dbar so the check means "the differential is
    # constant relative to its own level", not an absolute EUR^2 threshold.
    scale = max(abs(dbar), 1.0)
    if iid_var <= (1e-12 * scale) ** 2:
        if abs(dbar) <= 1e-12 * scale:
            return 0.5
        raise ValueError(
            "diebold_mariano_hac: degenerate loss differential -- the "
            f"differential is constant at {dbar!r} with zero variance across "
            f"all {n} days, so there is no sampling variation to test against. "
            "A p-value here would be an artifact of the variance floor, not "
            "evidence; check whether the two forecast frames are as intended."
        )

    var = iid_var
    for k in range(1, L + 1):
        var += 2.0 * (1.0 - k / (L + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    if var <= 0:  # negative HAC estimates are possible; fall back to iid
        var = iid_var

    return float(1.0 - stats.norm.cdf(dbar / np.sqrt(var / n)))
