"""Regime-segmented error evaluation — src/evaluation/regimes.py

Stage 5 of the power-engineering audit. Splits forecast error by the
PHYSICAL condition of the hour, so a block added to capture (say) steep
ramps can be checked against the hours it was meant to help rather than
against the pooled average, where a real regime effect is diluted to noise.

This module is ADDITIVE. It does not touch `src/evaluation/metrics.py`,
`results.py` or the regime-aware ensemble's `regime_labels()` in
`ensemble.py`. Note the two are different things and deliberately so:

  * `ensemble.regime_labels()` labels a DAY calm/stressed from the PREVIOUS
    day's realized prices, because an ensemble weight has to be chosen
    before the day starts.
  * this module labels an HOUR for POST-HOC error analysis, so it may use
    the realized outcome of that hour. Nothing here feeds a model or a
    weight -- using it as a feature would be leakage, which is why the
    labellers live in the evaluation package and not in features/.

Segments
--------
negative_price   realized price < 0
spike            realized price above a high quantile of the eval window
steep_ramp       |hour-to-hour change in residual load| in its top decile
high_res         renewable share of forecast load in its top decile
low_residual     residual load in its bottom decile
coupling_stress  NOT IMPLEMENTED -- needs cross-border data (see below)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse, smape

#: Quantile defining the spike segment (realized price tail).
SPIKE_Q = 0.95
#: Quantile defining the steep-ramp / high-RES / low-residual segments.
TAIL_Q = 0.90

#: Segments this module cannot produce, with the reason. Kept as data so a
#: caller can report the gap instead of silently returning fewer segments
#: than it asked for.
UNAVAILABLE_SEGMENTS = {
    "coupling_stress": (
        "needs neighbour day-ahead prices and NTC / scheduled cross-border "
        "flows (ENTSO-E Transparency Platform, security token required); "
        "not in the project's keyless information set"
    )
}


def _hourly_frame(wide: pd.DataFrame, series: str) -> pd.Series:
    """Flatten a daily-wide 24h block into an (origin, hour)-indexed Series."""
    cols = [f"{series}_h{h:02d}" for h in range(24)]
    out = wide[cols].copy()
    out.columns = range(24)
    stacked = out.stack()
    stacked.index.names = ["origin", "hour"]
    return stacked


def physical_context(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-(origin, hour) physical context used to define segments.

    Built from the same daily-wide frame the feature pipeline uses, so a
    segment means exactly what the corresponding feature means.

    Returns columns [residual_load, res_share, ramp].
    """
    load = wide[[f"exog_1_h{h:02d}" for h in range(24)]].copy()
    res = wide[[f"exog_2_h{h:02d}" for h in range(24)]].copy()
    load.columns = range(24)
    res.columns = range(24)

    resload = load - res
    grad = resload.diff(axis=1)
    grad[0] = resload[0] - resload[23].shift(1)
    share = res.div(load.where(load > 0))

    ctx = pd.DataFrame(
        {
            "residual_load": resload.stack(),
            "res_share": share.stack(),
            "ramp": grad.stack(),
        }
    )
    ctx.index.names = ["origin", "hour"]
    return ctx


def segment_masks(
    frame: pd.DataFrame, context: pd.DataFrame | None = None
) -> dict[str, pd.Series]:
    """Boolean mask per segment, aligned to `frame`'s rows.

    `frame` is a long results frame with columns [origin, hour, y_true,
    y_pred, ...] -- the shape `run_baselines.run_model` emits.

    `context` is the output of `physical_context`; when omitted, only the
    price-based segments (negative_price, spike) are produced. Asking for a
    physical segment without context would otherwise return an all-False
    mask, which reads as "this regime never occurred" rather than "this
    regime was not measured".

    Thresholds are computed on THIS evaluation window. That makes the
    segments a description of the window being scored, and it means two runs
    over the same window get identical masks -- which is the property the
    ablation comparison needs.
    """
    y = frame["y_true"].to_numpy(dtype=float)
    masks: dict[str, pd.Series] = {}

    masks["all"] = pd.Series(True, index=frame.index)
    masks["negative_price"] = pd.Series(y < 0, index=frame.index)
    spike_cut = np.nanquantile(y, SPIKE_Q)
    masks["spike"] = pd.Series(y >= spike_cut, index=frame.index)

    if context is None:
        return masks

    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame["origin"]), frame["hour"].astype(int)],
        names=["origin", "hour"],
    )
    ctx = context.reindex(keys)

    ramp = ctx["ramp"].abs().to_numpy(dtype=float)
    masks["steep_ramp"] = pd.Series(
        ramp >= np.nanquantile(ramp, TAIL_Q), index=frame.index
    )

    share = ctx["res_share"].to_numpy(dtype=float)
    masks["high_res"] = pd.Series(
        share >= np.nanquantile(share, TAIL_Q), index=frame.index
    )

    resload = ctx["residual_load"].to_numpy(dtype=float)
    masks["low_residual"] = pd.Series(
        resload <= np.nanquantile(resload, 1 - TAIL_Q), index=frame.index
    )

    return masks


def segmented_metrics(
    frame: pd.DataFrame, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-segment MAE / RMSE / sMAPE for one model's long results frame.

    sMAPE is reported as NaN on the negative_price segment: it is a
    percentage error and electricity prices go negative there, which is
    exactly why CLAUDE.md bans plain MAPE. Emitting a number would give a
    meaningless figure a metric's authority.
    """
    rows = []
    for name, mask in segment_masks(frame, context).items():
        sub = frame.loc[mask.fillna(False)]
        if sub.empty:
            rows.append(dict(segment=name, n=0, mae=np.nan, rmse=np.nan, smape=np.nan))
            continue
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        rows.append(
            dict(
                segment=name,
                n=len(sub),
                mae=float(mae(yt, yp)),
                rmse=float(rmse(yt, yp)),
                smape=(
                    np.nan
                    if name == "negative_price" or (yt <= 0).any()
                    else float(smape(yt, yp))
                ),
            )
        )
    return pd.DataFrame(rows).set_index("segment")


def compare_segmented(
    baseline: pd.DataFrame,
    variant: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-segment metric deltas, variant minus baseline.

    Negative `mae_delta` means the variant is better on that segment. The
    baseline frame is never modified and never overwritten -- it is the
    frozen record the comparison is against.
    """
    b = segmented_metrics(baseline, context)
    v = segmented_metrics(variant, context)
    out = b.join(v, lsuffix="_base", rsuffix="_new")
    out["mae_delta"] = out["mae_new"] - out["mae_base"]
    out["rmse_delta"] = out["rmse_new"] - out["rmse_base"]
    out["mae_pct"] = 100.0 * out["mae_delta"] / out["mae_base"]
    return out[
        ["n_base", "mae_base", "mae_new", "mae_delta", "mae_pct", "rmse_delta"]
    ]
