"""Static weighted ensemble — src/evaluation/ensemble.py

Week-7 scope, first arm (static ensemble; the regime-aware calm/spike
variant builds on top of this). Operates on walk-forward long frames
([origin, hour, y_true, y_pred, model]) rather than on model objects:
the member models already produce their predictions independently under
the harness, so the ensemble is a convex combination of those forecasts.

Leakage rule for weight fitting: `fit_weights` minimizes MAE on
whatever frames it is GIVEN — the caller must pass validation-period
frames (predictions on days strictly before the test window), never
test-period frames. The week-7 runner will produce those validation
frames with the same walk-forward harness before any test-set weight is
applied; fitting weights on test predictions and then reporting metrics
on the same period would be in-sample selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_pivots(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Pivot each frame to [origin x hour] and verify identical coverage."""
    names = list(frames)
    preds = {m: frames[m].pivot(index="origin", columns="hour", values="y_pred") for m in names}
    base = preds[names[0]].index
    for m in names[1:]:
        if not preds[m].index.equals(base):
            raise ValueError(
                f"ensemble: origins of '{m}' do not align with '{names[0]}' "
                "-- frames must cover identical origin sets"
            )
    truth = frames[names[0]].pivot(index="origin", columns="hour", values="y_true")
    return truth, preds


def combine_forecasts(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
    name: str = "ensemble",
) -> pd.DataFrame:
    """Convex combination of member forecasts.

    Weights must be non-negative with a positive sum; they are normalized
    to sum to 1, so only relative sizes matter.
    """
    if set(weights) != set(frames):
        raise ValueError("ensemble: weights and frames must cover the same models")
    w = np.array([weights[m] for m in frames], dtype=float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("ensemble: weights must be non-negative with a positive sum")
    w = w / w.sum()

    truth, preds = _aligned_pivots(frames)
    combined = sum(wi * preds[m] for wi, m in zip(w, frames))

    long = combined.stack().rename("y_pred").reset_index()
    long = long.merge(
        truth.stack().rename("y_true").reset_index(), on=["origin", "hour"]
    )
    long["model"] = name
    return long[["origin", "hour", "y_true", "y_pred", "model"]]


def fit_weights(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
    """MAE-minimizing convex weights over the given frames.

    LEAKAGE CONTRACT: pass validation-period frames only (see module
    docstring). Solved on the probability simplex with SLSQP from an
    equal-weight start; deterministic (no random component).
    """
    from scipy.optimize import minimize

    names = list(frames)
    truth, preds = _aligned_pivots(frames)
    P = np.stack([preds[m].values.ravel() for m in names])  # [n_models, n_obs]
    y = truth.values.ravel()

    def mae(w: np.ndarray) -> float:
        return float(np.mean(np.abs(y - w @ P)))

    n = len(names)
    res = minimize(
        mae,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return {m: float(wi) for m, wi in zip(names, w)}
