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
#:
#: coupling_stress moved OUT of this table on 2026-08-28: neighbour
#: day-ahead prices turned out to be keyless on Energy-Charts and are now
#: pinned in the snapshot, so the segment is built from the realized DE-vs-
#: neighbour price spread. NTC would sharpen it (a spread is the SYMPTOM of a
#: binding interconnector, capacity is the cause) but is not required to
#: define the regime.
UNAVAILABLE_SEGMENTS: dict[str, str] = {
    "outage_scarcity": (
        "a true scarcity regime needs generation outages / available capacity "
        "(ENTSO-E Transparency Platform, security token required). The 'spike' "
        "segment is a price-tail stand-in for it, not the same thing: it "
        "selects hours where price WAS high, not hours where capacity WAS "
        "tight, so a model can score well on it without having anticipated "
        "scarcity at all."
    ),
    "reservoir_hydro": (
        "hydro reservoir FILLING RATES (the seasonal opportunity-cost signal) "
        "need an ENTSO-E token. 'high_hydro' below uses observed pumped-"
        "storage activity instead, which is a dispatch response rather than "
        "the state variable that drives it."
    ),
}

#: Quantile above which a mechanism-specific segment is defined.
MECHANISM_Q = 0.90


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


def mechanism_context(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-(origin, hour) context for the MECHANISM-specific segments.

    Built from the pinned snapshot, so it exists only where that snapshot
    does. Returns an empty frame when the relevant series is not pinned --
    callers then omit the segment rather than reporting an all-False mask,
    because "this regime never occurred" and "this regime was not measured"
    must not look identical.

    These segments describe the PHYSICAL STATE of the hour, which is what a
    physics check needs. They deliberately use realized data: nothing here
    feeds a model, and a post-hoc segment may use the outcome of the hour it
    labels. Putting these in a feature would be leakage, which is why they
    live in evaluation/ and not features/.

    Columns (whichever are available):
      gas_share       gas / (gas + hard coal + lignite) generation
      coupling_spread |DE price - mean neighbour price|
      pumped_activity |pumped-storage generation| + |pumping|
    """
    from src.data.sources import snapshot

    out = pd.DataFrame(index=index)

    if snapshot.has("ec_public_power"):
        pp = snapshot.load_series("ec_public_power")
        gas = pp.get("Fossil gas")
        coal = pp.get("Fossil hard coal")
        lig = pp.get("Fossil brown coal / lignite")
        if gas is not None and coal is not None:
            thermal = gas + coal + (lig if lig is not None else 0)
            share = gas / thermal.where(thermal > 0)
            out["gas_share"] = share.reindex(index)
        gen = pp.get("Hydro pumped storage")
        con = pp.get("Hydro pumped storage consumption")
        if gen is not None or con is not None:
            act = (gen.abs() if gen is not None else 0) + (
                con.abs() if con is not None else 0
            )
            out["pumped_activity"] = act.reindex(index)

    if snapshot.has("ec_price_neighbours") and snapshot.has("ec_price_de"):
        nb = snapshot.load_series("ec_price_neighbours")
        de = snapshot.load_series("ec_price_de")
        if "price_de" in de.columns and len(nb.columns):
            spread = (nb.mean(axis=1) - de["price_de"]).abs()
            out["coupling_spread"] = spread.reindex(index)

    return out


def _stack_hourly(ctx: pd.DataFrame) -> pd.DataFrame:
    """Reshape an hourly-indexed context frame to an (origin, hour) index."""
    if ctx.empty:
        return ctx
    idx = pd.MultiIndex.from_arrays(
        [ctx.index.normalize(), ctx.index.hour], names=["origin", "hour"]
    )
    out = ctx.copy()
    out.index = idx
    return out[~out.index.duplicated(keep="first")]


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

    # Mechanism-specific segments, present only where the snapshot supports
    # them. Each is the regime a particular block physically targets, which
    # is what makes a physics check possible rather than a guess from
    # aggregate MAE.
    for col, name, high in (
        ("gas_share", "gas_marginal", True),
        ("coupling_spread", "coupling_stress", True),
        ("pumped_activity", "high_hydro", True),
    ):
        if col not in ctx.columns:
            continue
        vals = ctx[col].to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            continue
        cut = np.nanquantile(vals, MECHANISM_Q if high else 1 - MECHANISM_Q)
        masks[name] = pd.Series(vals >= cut if high else vals <= cut, index=frame.index)

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
