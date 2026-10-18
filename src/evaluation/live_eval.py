"""Evaluation report metrics for v2 forecasts — src/evaluation/live_eval.py

Every function takes long frames [origin, hour, y_true, y_pred]. Anything that
is FITTED (the uncertainty band) is fitted on the validation frame and applied
to test, never the reverse. These are descriptive diagnostics; none of them
selects a model (selection is validation MAE in run_live_ladder.score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index()


def bias(frame: pd.DataFrame) -> float:
    """Mean forecast minus actual. Negative = forecasts too low."""
    return float((frame["y_pred"] - frame["y_true"]).mean())


def by_hour(frame: pd.DataFrame) -> pd.Series:
    return (frame["y_pred"] - frame["y_true"]).abs().groupby(frame["hour"]).mean()


def by_weekday_type(frame: pd.DataFrame) -> dict:
    wk = pd.to_datetime(frame["origin"]).dt.dayofweek >= 5
    err = (frame["y_pred"] - frame["y_true"]).abs()
    return {"weekday": float(err[~wk].mean()), "weekend": float(err[wk].mean())}


def by_regime(frame: pd.DataFrame, threshold: float) -> dict:
    """MAE on calm vs stressed days, labelled from the PREVIOUS day only (regime_labels)."""
    from src.evaluation.ensemble import regime_labels

    labels = regime_labels(frame, threshold=threshold)
    lab = frame["origin"].map(labels)
    err = (frame["y_pred"] - frame["y_true"]).abs()
    return {k: {"mae": float(err[lab == k].mean()), "days": int(frame.loc[lab == k, "origin"].nunique())}
            for k in ("calm", "stressed")}


def negative_hours(frame: pd.DataFrame) -> dict:
    """How the model handles hours whose ACTUAL price was below zero."""
    neg = frame["y_true"] < 0
    pred_neg = frame["y_pred"] < 0
    n = int(neg.sum())
    return {
        "n_negative_hours": n,
        "mae_on_negative_hours": float((frame.loc[neg, "y_pred"] - frame.loc[neg, "y_true"]).abs().mean()) if n else float("nan"),
        "detected_share": float((pred_neg & neg).sum() / n) if n else float("nan"),
        "false_alarm_share": float((pred_neg & ~neg).sum() / max(1, int(pred_neg.sum()))),
    }


def rolling_mae(frame: pd.DataFrame, days: int = 30) -> pd.Series:
    daily = (frame["y_pred"] - frame["y_true"]).abs().groupby(frame["origin"]).mean().sort_index()
    return daily.rolling(days, min_periods=days).mean()


def baseload_error(frame: pd.DataFrame) -> dict:
    t = frame.groupby("origin")[["y_true", "y_pred"]].mean()
    e = t["y_pred"] - t["y_true"]
    return {"mae": float(e.abs().mean()), "bias": float(e.mean())}


def band_coverage(val: pd.DataFrame, test: pd.DataFrame, lo_q: float = 0.10, hi_q: float = 0.90) -> dict:
    """Per-hour residual quantiles fitted on VALIDATION, applied to test forecasts."""
    res = (val["y_true"] - val["y_pred"]).groupby(val["hour"])
    lo, hi = res.quantile(lo_q), res.quantile(hi_q)
    t_lo = test["y_pred"] + test["hour"].map(lo)
    t_hi = test["y_pred"] + test["hour"].map(hi)
    inside = (test["y_true"] >= t_lo) & (test["y_true"] <= t_hi)
    return {"nominal": hi_q - lo_q, "coverage": float(inside.mean()), "mean_width": float((t_hi - t_lo).mean())}


def _best_window(v: np.ndarray, k: int, cheapest: bool) -> int:
    means = np.convolve(v, np.ones(k) / k, mode="valid")
    return int(means.argmin() if cheapest else means.argmax())


def window_value(frame: pd.DataFrame, k: int = 2) -> dict:
    """Did the forecast find tomorrow's cheapest/dearest k-hour windows, and what would acting on it earn?

    hit = the forecast window overlaps the true one. Value capture models one
    battery cycle per day: buy in the forecast-cheapest window, sell in the
    forecast-dearest window, both priced at ACTUAL prices, divided by the
    perfect-hindsight spread. 100% = as good as knowing the prices.
    """
    yt, yp = _wide(frame, "y_true"), _wide(frame, "y_pred")
    hits_c = hits_d = 0
    earned = ideal = 0.0
    for day in yt.index:
        a, f = yt.loc[day].to_numpy(), yp.loc[day].to_numpy()
        fc, fd = _best_window(f, k, True), _best_window(f, k, False)
        ac, ad = _best_window(a, k, True), _best_window(a, k, False)
        hits_c += abs(fc - ac) < k
        hits_d += abs(fd - ad) < k
        earned += a[fd:fd + k].mean() - a[fc:fc + k].mean()
        ideal += a[ad:ad + k].mean() - a[ac:ac + k].mean()
    n = len(yt)
    return {"k": k, "days": n, "cheapest_hit_rate": hits_c / n, "dearest_hit_rate": hits_d / n,
            "value_capture": earned / ideal if ideal else float("nan"),
            "earned_per_day": earned / n, "ideal_per_day": ideal / n}


BAND_CANDIDATES = [(w, pooled) for w in (28, 56, 91) for pooled in (False, True)]


def rolling_band(frame: pd.DataFrame, window: int, pooled: bool = False,
                 lo_q: float = 0.10, hi_q: float = 0.90) -> pd.DataFrame:
    """Per-origin band offsets from the errors of the `window` origins strictly BEFORE it.

    Day D is forecast on D-1, when every earlier day's prices (hence errors) are
    already published, so only origins < D are used. pooled=False: per-hour
    quantiles; pooled=True: one quantile pair over all 24 hours of the window.
    Returns [origin, hour, lo, hi] (offsets to add to y_pred); origins with
    fewer than `window` earlier days are omitted.
    """
    err = frame.assign(e=frame["y_true"] - frame["y_pred"]).pivot(index="origin", columns="hour", values="e").sort_index()
    out = []
    for i in range(window, len(err)):
        past = err.iloc[i - window:i]
        if pooled:
            v = past.to_numpy().ravel()
            lo, hi = np.full(24, np.nanquantile(v, lo_q)), np.full(24, np.nanquantile(v, hi_q))
        else:
            lo, hi = past.quantile(lo_q).to_numpy(), past.quantile(hi_q).to_numpy()
        out.append(pd.DataFrame({"origin": err.index[i], "hour": err.columns, "lo": lo, "hi": hi}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["origin", "hour", "lo", "hi"])


def band_score(frame: pd.DataFrame, band: pd.DataFrame) -> dict:
    """Coverage and mean width of a [origin, hour, lo, hi] band on the frame's rows that it covers."""
    m = frame.merge(band, on=["origin", "hour"])
    inside = (m["y_true"] >= m["y_pred"] + m["lo"]) & (m["y_true"] <= m["y_pred"] + m["hi"])
    return {"coverage": float(inside.mean()) if len(m) else float("nan"),
            "mean_width": float((m["hi"] - m["lo"]).mean()) if len(m) else float("nan"),
            "days": int(m["origin"].nunique())}


def select_band(val: pd.DataFrame, nominal: float = 0.80, candidates=BAND_CANDIDATES) -> tuple[int, bool]:
    """Pick (window, pooled) on VALIDATION only: coverage closest to nominal, then narrowest.

    Every candidate is scored on the same validation days (those the longest
    window can cover), so the comparison is like for like.
    """
    from src.evaluation.live_ladder import _assert_validation_only

    _assert_validation_only(val, "select_band")  # leakage review 2026-10-16, F7
    bands = {c: rolling_band(val, *c) for c in candidates}
    common = set.intersection(*(set(b["origin"]) for b in bands.values()))
    scores = {c: band_score(val, b[b["origin"].isin(common)]) for c, b in bands.items()}
    return min(scores, key=lambda c: (round(abs(scores[c]["coverage"] - nominal), 3), scores[c]["mean_width"]))
