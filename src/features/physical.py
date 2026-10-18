"""Physical (power-engineering) feature blocks — src/features/physical.py

Additive companions to `src/features/pipeline.build_features`. Nothing here
runs unless a caller explicitly enables it (see `configs/features.yaml`
`physical_blocks:`, every flag defaulting to false), so the frozen
v1.0-results / v1.1-ood numbers are reproduced bit-identically by the
default path.

Each public `*_block` function takes the SAME daily-wide frame that
`build_features` works on -- one row per target day, columns
`{series}_h00..{series}_h23` -- and returns a DataFrame of new columns on
that same index. They never mutate their input and never touch the existing
lag columns.

INFORMATION-SET DISCIPLINE (the rule that governs this whole module):
a target-day (D0) column is legal only if its inputs are day-ahead
FORECASTS. `exog_1` (Amprion load forecast) and `exog_2` (PV+wind forecast)
are forecasts published before gate closure -- that is exactly why
`configs/features.yaml` already licenses `exog_*_D0_h*`. `price` is a
REALIZATION, so no block here reads target-day price, and any block needing
a distributional reference derives it from STRICTLY PAST days only
(`.shift(1)` before any rolling window -- see `_trailing`).

Blocks whose physics needs data this project cannot access keylessly
(carbon, fuel curves, NTC, outages, hydro) are implemented as explicit
stubs that raise `FeatureDataUnavailable`. They are declared, tested and
auditable -- never silently fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS = [f"h{h:02d}" for h in range(24)]


class FeatureDataUnavailable(RuntimeError):
    """Raised by a block whose required series is not in the sanctioned
    keyless information set.

    This is a deliberate, load-bearing failure. The alternative -- returning
    zeros, a constant, or an interpolated stand-in -- would put a fabricated
    physical driver into the feature matrix and make every downstream metric
    a statement about invented data. The stub fails instead.
    """

    def __init__(self, mechanism: str, requires: tuple[str, ...], blocked_by: str):
        self.mechanism = mechanism
        self.requires = requires
        self.blocked_by = blocked_by
        super().__init__(
            f"{mechanism}: cannot be built. Requires {list(requires)}, which is "
            f"not in the project's keyless information set (price, exog_1, "
            f"exog_2). Blocked by: {blocked_by}. Fabricating a stand-in series "
            f"is not an acceptable substitute."
        )


def _hour_cols(wide: pd.DataFrame, series: str) -> pd.DataFrame:
    """The 24 hourly columns of one series, renamed to bare h00..h23."""
    missing = [f"{series}_{h}" for h in HOURS if f"{series}_{h}" not in wide.columns]
    if missing:
        raise KeyError(
            f"physical features need the full 24h block of '{series}'; "
            f"missing {len(missing)} column(s), e.g. {missing[:3]}"
        )
    out = wide[[f"{series}_{h}" for h in HOURS]].copy()
    out.columns = HOURS
    return out


def residual_load_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Target-day residual load, 24 columns h00..h23.

    residual_load = forecast_load - forecast_wind_solar = exog_1 - exog_2.

    Both operands are day-ahead forecasts, so this is a D0 (target-day)
    quantity that is legal at the forecast origin. Shared helper: several
    blocks below build on it, and they must all mean the same thing.
    """
    load = _hour_cols(wide, "exog_1")
    res = _hour_cols(wide, "exog_2")
    return load - res


def _trailing(daily: pd.Series, window: int, func: str) -> pd.Series:
    """A rolling statistic over STRICTLY PAST days.

    `.shift(1)` runs BEFORE the rolling window, so the value at day D is
    computed from days D-window .. D-1 and never includes D itself. Rolling
    first and shifting after would be the same arithmetic in the wrong order
    and would fold the target day into its own reference distribution -- the
    exact in-sample contamination this project's walk-forward rules exist to
    prevent.
    """
    past = daily.shift(1)
    roll = past.rolling(window=window, min_periods=window)
    return getattr(roll, func)()


# --------------------------------------------------------------------------
# Buildable blocks
# --------------------------------------------------------------------------


def residual_load_block(wide: pd.DataFrame, lag_days: tuple[int, ...] = (1,)) -> pd.DataFrame:
    """Residual load for the target day, plus its lagged 24h vectors.

    Columns
    -------
    resload_D0_h00..h23   target-day residual load (forecast-based, legal)
    resload_D-{k}_h00..h23 residual load k days back (realized-forecast, past)
    """
    rl = residual_load_frame(wide)

    parts = []
    d0 = rl.copy()
    d0.columns = [f"resload_D0_{h}" for h in HOURS]
    parts.append(d0)

    for k in lag_days:
        block = rl.shift(k)
        block.columns = [f"resload_D-{k}_{h}" for h in HOURS]
        parts.append(block)

    return pd.concat(parts, axis=1)


def residual_load_gradient_block(wide: pd.DataFrame) -> pd.DataFrame:
    """Hour-to-hour residual-load ramp on the target day, plus its extremes.

    The gradient at hour h is residual_load[h] - residual_load[h-1]. Hour 00
    has no in-day predecessor; it is set to the ramp from the LAST hour of
    the previous day (h23 of D-1), which is a real overnight ramp and is
    still forecast-based. It is NOT filled with 0, which would fabricate a
    flat hour, nor with h01's value, which would duplicate information.

    Columns
    -------
    resload_grad_D0_h00..h23  hourly ramp (MW/h)
    resload_ramp_max_D0       steepest up-ramp of the day
    resload_ramp_min_D0       steepest down-ramp of the day
    resload_ramp_absmax_D0    largest ramp magnitude either direction
    """
    rl = residual_load_frame(wide)
    grad = rl.diff(axis=1)
    # h00's predecessor is the previous day's h23.
    grad[HOURS[0]] = rl[HOURS[0]] - rl[HOURS[-1]].shift(1)

    out = grad.copy()
    out.columns = [f"resload_grad_D0_{h}" for h in HOURS]
    out["resload_ramp_max_D0"] = grad.max(axis=1)
    out["resload_ramp_min_D0"] = grad.min(axis=1)
    out["resload_ramp_absmax_D0"] = grad.abs().max(axis=1)
    return out


def merit_order_block(wide: pd.DataFrame, window: int = 365) -> pd.DataFrame:
    """Merit-order position proxies for the target day.

    Two complementary views, neither of which needs an external supply curve:

    merit_pos_D0_h00..h23
        Where each hour's residual load sits inside the distribution of
        residual load over the trailing `window` days, expressed as a
        z-score against a PAST-ONLY mean and std (see `_trailing`). A high
        value means the dispatchable fleet is being pushed into the steep,
        expensive region of the stack.

    res_share_D0_h00..h23
        Renewable share of forecast load, exog_2 / exog_1. This is the
        merit-order (price-suppression) effect stated directly, and it is
        scale-free, so it stays comparable as the fleet grows.

    Both are target-day quantities built from day-ahead forecasts only; the
    distributional reference comes from strictly earlier days.
    """
    rl = residual_load_frame(wide)
    daily_mean = rl.mean(axis=1)
    ref_mean = _trailing(daily_mean, window, "mean")
    ref_std = _trailing(daily_mean, window, "std")
    # A degenerate (zero-variance) trailing window would make the z-score
    # infinite. That is a data pathology, not a signal -- mark it missing so
    # build_features drops the row rather than feeding an inf to a model.
    ref_std = ref_std.where(ref_std > 0)

    pos = rl.sub(ref_mean, axis=0).div(ref_std, axis=0)
    pos.columns = [f"merit_pos_D0_{h}" for h in HOURS]

    load = _hour_cols(wide, "exog_1")
    res = _hour_cols(wide, "exog_2")
    share = res.div(load.where(load > 0))
    share.columns = [f"res_share_D0_{h}" for h in HOURS]

    return pd.concat([pos, share], axis=1)


def scarcity_block(wide: pd.DataFrame, window: int = 365) -> pd.DataFrame:
    """Supply-tightness PROXY for the target day.

    The true scarcity signal is forecast_load / available_capacity, where
    available capacity is installed dispatchable capacity net of planned
    outages. That needs the ENTSO-E outage feed, which requires a security
    token this project may not use, so no true reserve margin is produced
    here -- see `price_formation.scarcity_reserve_margin.blocked_by`.

    What IS buildable keylessly: residual load relative to the highest
    residual load seen in the trailing `window` days. Capacity is roughly
    flat year to year, so the recent maximum is a stable stand-in for the
    dispatchable ceiling, and the ratio says how close the target day comes
    to the tightest recent condition. Values near 1 mean the fleet is being
    asked for as much as it has recently had to deliver.

    Every column carries `_proxy` in its name so it can never be read as a
    measured reserve margin.

    Columns
    -------
    tightness_proxy_D0_h00..h23  residual load / trailing-max residual load
    tightness_proxy_peak_D0      the day's maximum of the above
    """
    rl = residual_load_frame(wide)
    daily_max = rl.max(axis=1)
    ref_max = _trailing(daily_max, window, "max")
    ref_max = ref_max.where(ref_max > 0)

    tight = rl.div(ref_max, axis=0)
    out = tight.copy()
    out.columns = [f"tightness_proxy_D0_{h}" for h in HOURS]
    out["tightness_proxy_peak_D0"] = tight.max(axis=1)
    return out


# --------------------------------------------------------------------------
# Declared-unavailable blocks (stubs that fail loudly, never fabricate)
# --------------------------------------------------------------------------


def marginal_fuel_carbon_block(wide: pd.DataFrame) -> pd.DataFrame:
    """STUB. Needs gas/coal/EUA settlement prices (registered feed)."""
    raise FeatureDataUnavailable(
        "marginal_fuel_carbon",
        ("gas_price", "coal_price", "eua_price"),
        "EEX/ICE gas, coal and EUA futures settlement prices (registered feed)",
    )


def clean_spreads_block(wide: pd.DataFrame) -> pd.DataFrame:
    """STUB. Clean spark/dark spreads need the same fuel+carbon series.

    For the record, so a future implementation does not have to re-derive
    them:
        clean_spark = price - gas/eta_gas   - eua * ef_gas/eta_gas
        clean_dark  = price - coal/eta_coal - eua * ef_coal/eta_coal
    Note both use `price`, a REALIZATION -- a same-day spread would leak the
    target. Only lagged spreads (D-1 and back) would ever be legal here.
    """
    raise FeatureDataUnavailable(
        "clean_spreads",
        ("gas_price", "coal_price", "eua_price"),
        "EEX/ICE gas, coal and EUA futures settlement prices (registered feed)",
    )


def market_coupling_block(wide: pd.DataFrame) -> pd.DataFrame:
    """STUB. Needs neighbour day-ahead prices and NTC / scheduled flows.

    Additional constraint recorded for any future implementation: neighbour
    day-ahead prices clear in the SAME Euphemia session as DE-LU, so the
    target day's neighbour price is not knowable at the forecast origin.
    Only lagged neighbour prices and forecast NTC would be legal.
    """
    raise FeatureDataUnavailable(
        "market_coupling",
        ("neighbor_price", "ntc", "scheduled_flow"),
        "ENTSO-E Transparency Platform (security token required)",
    )


def storage_hydro_block(wide: pd.DataFrame) -> pd.DataFrame:
    """STUB. Needs reservoir filling rate / hydro generation."""
    raise FeatureDataUnavailable(
        "storage_hydro",
        ("reservoir_level", "hydro_generation"),
        "ENTSO-E aggregated filling rate / hydro generation (token required)",
    )


#: Every block by name, so `price_formation.Mechanism.builder` resolves
#: programmatically instead of by a hand-maintained if/else.
BLOCKS = {
    "residual_load_block": residual_load_block,
    "residual_load_gradient_block": residual_load_gradient_block,
    "merit_order_block": merit_order_block,
    "scarcity_block": scarcity_block,
    "marginal_fuel_carbon_block": marginal_fuel_carbon_block,
    "clean_spreads_block": clean_spreads_block,
    "market_coupling_block": market_coupling_block,
    "storage_hydro_block": storage_hydro_block,
}


def build_physical_blocks(wide: pd.DataFrame, enabled: dict | None = None) -> pd.DataFrame:
    """Run the enabled physical blocks and concatenate their columns.

    `enabled` maps block name -> bool (or a dict of kwargs). Unknown names
    raise rather than being ignored, so a typo in configs/features.yaml
    fails at build time instead of silently producing a smaller feature set
    that would still train, still score, and quietly not be the experiment
    anyone thought they ran.

    Returns an empty frame on `wide`'s index when nothing is enabled -- the
    default, and the path that reproduces the frozen results.
    """
    if not enabled:
        return pd.DataFrame(index=wide.index)

    parts = []
    for name, setting in enabled.items():
        if name not in BLOCKS:
            raise KeyError(
                f"unknown physical block '{name}'; known blocks: "
                f"{sorted(BLOCKS)}"
            )
        if not setting:
            continue
        kwargs = setting if isinstance(setting, dict) else {}
        parts.append(BLOCKS[name](wide, **kwargs))

    if not parts:
        return pd.DataFrame(index=wide.index)
    return pd.concat(parts, axis=1)
