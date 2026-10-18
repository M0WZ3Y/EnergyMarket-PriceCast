"""Static weighted ensemble — src/evaluation/ensemble.py

Week-7 scope, first arm (static ensemble; the regime-aware calm/stressed
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
    # Hour LABELS must be checked before any value comparison. np.allclose
    # below compares positionally, so a member whose hours are labelled 1..24
    # instead of 0..23 lines up cell-for-cell and sails through the y_true
    # agreement check -- and combine_forecasts then adds the pivots by column
    # LABEL, where 0..23 and 1..24 share only 23 labels and the union columns
    # come out all-NaN. An all-NaN forecast column is far worse than an error,
    # so the label mismatch is caught here, first.
    base_hours = preds[names[0]].columns
    for m in names[1:]:
        if not preds[m].columns.equals(base_hours):
            raise ValueError(
                f"ensemble: hour columns of '{m}' do not match '{names[0]}' "
                f"({list(preds[m].columns)} vs {list(base_hours)}) -- member "
                "frames must use identical hour labels"
            )

    truth = frames[names[0]].pivot(index="origin", columns="hour", values="y_true")

    # The truth column is taken from ONE arbitrary member, so a stale or
    # shifted y_true in that file would silently define reality for every
    # metric downstream. The 2026-08-02 concurrent-writer incident produced
    # exactly this shape of corruption, so agreement is checked, not assumed.
    for m in names[1:]:
        other = frames[m].pivot(index="origin", columns="hour", values="y_true")
        # atol is explicit: the test data contains y_true cells at exactly
        # 0.0 EUR/MWh, where allclose's relative term vanishes and only atol
        # remains. The default 1e-8 would flag a member merely written with
        # fewer decimals; 1e-6 still catches any real corruption.
        if not np.allclose(truth.values, other.values, atol=1e-6, equal_nan=True):
            bad = int((~np.isclose(truth.values, other.values, atol=1e-6, equal_nan=True)).sum())
            raise ValueError(
                f"ensemble: y_true of '{m}' disagrees with '{names[0]}' in "
                f"{bad} cell(s) -- member frames must share identical realized "
                "prices; one of these files is stale or corrupted"
            )
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


def regime_labels(
    frame: pd.DataFrame, threshold: float, default: str = "calm"
) -> dict[pd.Timestamp, str]:
    """Label each origin day 'calm' or 'stressed' using ONLY information
    known before that origin: day D is 'stressed' iff the PREVIOUS day's
    realized prices contain at least one hour above `threshold`. The
    threshold is train-only (train mean + k*std, k fixed by a
    validation-only rule) and comes from configs/evaluation.yaml under
    regime.stress_threshold_eur_mwh -- never hardcode it here. The first
    origin, having no previous day inside the frame, gets `default`.
    Never reads day D's own outcome -- that would leak the label.
    """
    day_max = frame.groupby("origin")["y_true"].max().sort_index()

    # The lookup is by CALENDAR DAY, not by position. A positional
    # day_max.shift(1) reads "the previous ROW", which equals the previous
    # day only when the origin set is gapless -- and it is not gapless after
    # a partially re-run walk-forward, a filtered evaluation window, or any
    # step_days > 1. In those cases the shift silently imports the regime of
    # an arbitrarily old day (an 8-day-old spike, say) into today's label.
    # Origins are normalized first so a timestamp carrying a time-of-day
    # still resolves to its day.
    by_day = day_max.copy()
    by_day.index = pd.DatetimeIndex(by_day.index).normalize()
    by_day = by_day.groupby(level=0).max()

    labels = {}
    for origin in day_max.index:
        prev_day = pd.Timestamp(origin).normalize() - pd.Timedelta(days=1)
        if prev_day not in by_day.index:
            # The previous calendar day is not in the frame, so yesterday's
            # regime is unknowable here -- fall back to the documented
            # default rather than reaching further back for a stale day.
            labels[origin] = default
            continue
        prev = by_day.loc[prev_day]
        labels[origin] = "stressed" if prev > threshold else "calm"
    return labels


def combine_regime_aware(
    frames: dict[str, pd.DataFrame],
    weights: dict[str, dict[str, float]],
    threshold: float,
    name: str = "regime-ensemble",
) -> pd.DataFrame:
    """Regime-aware convex combination: each origin day uses the weight
    set of its regime ('calm'/'stressed' keys in `weights`), with the
    regime decided by regime_labels() (previous-day information only).
    """
    if set(weights) != {"calm", "stressed"}:
        raise ValueError(
            "regime weights must have exactly the keys 'calm' and 'stressed' "
            f"(got {sorted(weights)}). The 'spike' label was renamed to "
            "'stressed' on 2026-08-04 when the threshold moved to mean+1.5*std."
        )

    first = frames[next(iter(frames))]
    labels = regime_labels(first, threshold)

    parts = []
    for regime in ("calm", "stressed"):
        days = [o for o, lab in labels.items() if lab == regime]
        if not days:
            continue
        sub = {m: f[f["origin"].isin(days)] for m, f in frames.items()}
        parts.append(combine_forecasts(sub, weights[regime], name=name))
    out = pd.concat(parts, ignore_index=True)

    # Every input origin must survive the regime split. A label the loop
    # does not iterate over would silently drop those days from the output
    # instead of failing -- the exact defect the rename introduced.
    if len(out) != sum(len(f) for f in frames.values()) // len(frames):
        raise ValueError(
            f"regime split lost origins: {len(labels)} labeled, "
            f"{out['origin'].nunique()} in output -- unhandled regime label?"
        )
    return out.sort_values(["origin", "hour"]).reset_index(drop=True)


def fit_weights(
    frames: dict[str, pd.DataFrame], test_days: pd.DatetimeIndex | None = None
) -> dict[str, float]:
    """MAE-minimizing convex weights over the given frames.

    LEAKAGE CONTRACT: pass validation-period frames only (see module
    docstring). Passing `test_days` turns that contract from a comment
    into a check -- every origin in `frames` must fall strictly before
    the test window, or this raises. Production callers should always
    pass it; it stays optional only so unit tests can fit weights on
    synthetic frames that have no test window.

    Solved on the probability simplex with SLSQP from an equal-weight
    start; deterministic (no random component).
    """
    from scipy.optimize import minimize

    from src.evaluation.walk_forward import assert_validation_before_test

    names = list(frames)

    # NaN screening comes first, and it is not cosmetic. SLSQP evaluates a
    # NaN objective, cannot compare it to anything, and returns its STARTING
    # POINT -- equal weights -- with res.success False. Without this check a
    # single non-converged SARIMAX hour anywhere in the validation frames
    # would turn the tuned ensemble into a plain unweighted average while
    # every downstream table still labelled it "MAE-optimal weights".
    for m in names:
        f = frames[m]
        for col in ("y_pred", "y_true"):
            n_nan = int(f[col].isna().sum())
            if n_nan:
                raise ValueError(
                    f"fit_weights: member '{m}' has {n_nan} NaN value(s) in "
                    f"'{col}' -- weights cannot be fitted on NaN (the optimiser "
                    "would silently return its equal-weight starting point)"
                )

    truth, preds = _aligned_pivots(frames)
    if test_days is not None:
        assert_validation_before_test(truth.index, test_days)
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
    # A non-converged SLSQP run returns x0 -- the equal-weight start -- which
    # is a perfectly plausible-looking weight vector and therefore the most
    # dangerous possible failure mode: the caller gets an unweighted average
    # labelled "MAE-optimal". Never return a solution the optimiser did not
    # claim to have found.
    if not res.success:
        raise ValueError(
            f"fit_weights: SLSQP did not converge ({res.message!r}); refusing "
            "to return the equal-weight starting point as an optimum"
        )

    w = np.clip(res.x, 0.0, None)
    # Clipping can in principle zero out every component (an all-negative x),
    # and w/0 would hand back silent NaN weights. Fail loudly instead.
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"fit_weights: optimiser returned weights that do not sum to a "
            f"positive number (sum={total}); cannot normalize to the simplex"
        )
    w = w / total
    return {m: float(wi) for m, wi in zip(names, w)}
