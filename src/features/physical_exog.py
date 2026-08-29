"""Snapshot-derived physical feature blocks — src/features/physical_exog.py

Blocks 2-6 of the power-engineering feature set: the ones needing external
data. Kept separate from `physical.py` (blocks 1 and the keyless proxies,
already committed and tested) so that module stays untouched.

Every block here reads the PINNED snapshot through
`src.data.sources.snapshot` and never the network. Each returns an empty
frame when its input series is not pinned, so a missing Tier-B source makes
the block skip rather than fail the pipeline -- and never fills a substitute
value.

COVERAGE, which is a real cost and is stated up front: the Energy-Charts
endpoints serve nothing before 2015-01-01 (verified by probing; see
`clients.EC_DATA_START`). The benchmark window starts 2012-01-09, so every
block built on them is NaN for the first three years, and those rows are
dropped by `build_features`. Enabling them roughly halves the available
training history. That is a trade to make deliberately, not to discover as
an unexplained row count.

INFORMATION SET. Every column produced here is registered with
`src.features.leakage_guard` at the lag it actually reads, and the guard
fails the build if any of them would read past gate closure. The two traps:
neighbour day-ahead prices clear in the SAME coupled auction as DE-LU
(published ~12:42 CET, after the 12:00 origin) so they are lagged; realized
generation and flows for the delivery day are not known at the origin, so
they are lagged too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sources import snapshot
from src.features import leakage_guard as lg

HOURS = [f"h{h:02d}" for h in range(24)]

# --------------------------------------------------------------------------
# Emission factors and plant efficiencies (IPCC/EEA defaults, thesis-citable)
# --------------------------------------------------------------------------
#: tCO2 per MWh of THERMAL input.
EF_GAS_T_PER_MWH_TH = 0.202
EF_HARDCOAL_T_PER_MWH_TH = 0.340
#: Reference net electrical efficiencies for the marginal European fleet.
ETA_CCGT = 0.50
ETA_HARDCOAL = 0.38

#: tCO2 per MWh of ELECTRICITY -- the form the carbon cost enters SRMC in.
EF_GAS_T_PER_MWH_EL = EF_GAS_T_PER_MWH_TH / ETA_CCGT        # 0.404
EF_COAL_T_PER_MWH_EL = EF_HARDCOAL_T_PER_MWH_TH / ETA_HARDCOAL  # 0.895

#: Production types in Energy-Charts' /public_power response.
GAS_TYPES = ("Fossil gas",)
COAL_TYPES = ("Fossil hard coal",)
LIGNITE_TYPES = ("Fossil brown coal / lignite",)
NUCLEAR_TYPES = ("Nuclear",)
RES_TYPES = ("Solar", "Wind onshore", "Wind offshore", "Hydro Run-of-River", "Biomass")
PUMPED_TYPES = ("Hydro pumped storage",)
PUMPED_CONSUMPTION_TYPES = ("Hydro pumped storage consumption",)

#: Dispatchable technologies in /installed_power, for the capacity margin.
DISPATCHABLE_CAPACITY_TYPES = (
    "Nuclear",
    "Fossil brown coal / lignite",
    "Fossil hard coal",
    "Fossil gas",
    "Fossil oil",
    "Other, non-renewable",
    "Hydro",
    "Hydro pumped storage",
    "Biomass",
)


def _empty(index) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def _to_daily_wide(series_or_frame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align an hourly external frame to the benchmark calendar, then pivot
    to one row per day with h00..h23 columns.

    Uses the same day/hour convention as `pipeline._pivot_to_daily_wide` so
    an external column means the same thing as a benchmark one.
    """
    df = series_or_frame.to_frame() if isinstance(series_or_frame, pd.Series) else series_or_frame
    aligned = snapshot.align_to_index(df, index)
    day = aligned.index.normalize()
    hour = aligned.index.hour
    parts = []
    for col in aligned.columns:
        w = aligned[col].groupby([day, hour]).first().unstack(level=-1)
        w.columns = [f"{col}_h{h:02d}" for h in w.columns]
        parts.append(w)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    out.index.name = "day"
    return out


def _lagged_block(wide_ext: pd.DataFrame, cols: list[str], lag: int, name: str) -> pd.DataFrame:
    block = wide_ext[cols].shift(lag)
    block.columns = [f"{name}_D-{lag}_{c.split('_')[-1]}" for c in cols]
    return block


# ==========================================================================
# Block 2 — explicit merit order / marginal technology
# ==========================================================================


def merit_order_explicit_block(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, lags: tuple[int, ...] = (1, 7)
) -> pd.DataFrame:
    """Observed dispatch shares by technology, lagged.

    Makes explicit what `exog_2` only implies. The share of generation coming
    from gas versus coal versus lignite says which technology was setting the
    price recently, which is the merit-order position the price actually
    responds to.

    REALIZED generation, so lagged only -- the delivery day's dispatch is
    unknown at the origin.
    """
    if not snapshot.has("ec_public_power"):
        return _empty(wide.index)

    pp = snapshot.load_series("ec_public_power")
    present = [c for c in pp.columns]

    def _sum(types) -> pd.Series | None:
        cols = [t for t in types if t in present]
        return pp[cols].sum(axis=1) if cols else None

    gas, coal, lig = _sum(GAS_TYPES), _sum(COAL_TYPES), _sum(LIGNITE_TYPES)
    nuc, res = _sum(NUCLEAR_TYPES), _sum(RES_TYPES)
    if gas is None or coal is None:
        return _empty(wide.index)

    total = pp[[c for c in present if "consumption" not in c.lower()]].sum(axis=1)
    total = total.where(total > 0)

    shares = pd.DataFrame(
        {
            "gasshare": gas / total,
            "coalshare": coal / total,
            "ligshare": (lig / total) if lig is not None else np.nan,
            "nucshare": (nuc / total) if nuc is not None else np.nan,
            "resgenshare": (res / total) if res is not None else np.nan,
            "thermalshare": (gas + coal + (lig if lig is not None else 0)) / total,
        }
    ).dropna(axis=1, how="all")

    wide_ext = _to_daily_wide(shares, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for lag in lags:
        for base in shares.columns:
            cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
            if cols:
                parts.append(_lagged_block(wide_ext, cols, lag, base))
    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_public_power", lag_days=min(lags),
        note="realized dispatch shares; delivery-day mix unknown at origin",
    )
    return out


def capacity_structure_block(wide: pd.DataFrame) -> pd.DataFrame:
    """Installed dispatchable capacity for the target year.

    Structural and published in advance, so unlike generation this needs no
    lag -- the one capacity input legal at lag 0.
    """
    if not snapshot.has("ec_installed_power"):
        return _empty(wide.index)

    ip = snapshot.load_series("ec_installed_power", parse_dates=False)
    cols = [c for c in DISPATCHABLE_CAPACITY_TYPES if c in ip.columns]
    if not cols:
        return _empty(wide.index)

    disp = ip[cols].sum(axis=1)
    disp.index = [int(str(y)[:4]) for y in disp.index]

    years = pd.Index(wide.index.year)
    out = pd.DataFrame(index=wide.index)
    out["disp_capacity_D0"] = years.map(disp.to_dict()).to_numpy(dtype=float)
    lg.declare_block(
        out.columns, source="ec_installed_power", lag_days=0,
        note="structural capacity, published in advance",
    )
    return out


# ==========================================================================
# Block 3 — carbon cost and fuel-switching (REDUCED: see module note)
# ==========================================================================


def carbon_block(wide: pd.DataFrame, emit_all_variants: bool = False) -> pd.DataFrame:
    """EUA carbon cost, converted to EUR per MWh of electricity.

    WHAT THIS IS NOT: a clean spark or dark spread. Those need gas (TTF) and
    coal (API2) prices, which are Montel-licensed; Ember publishes only
    series derived from them, making Ember a citation rather than a source.
    So the fuel leg is absent and only the CARBON leg of SRMC is built here.

    What it is: carbon cost per MWh_el for a reference CCGT and a reference
    hard-coal unit, and the difference between them -- the carbon-driven part
    of the coal-to-gas switching signal, which is real and citable on its own
    because coal's higher emission factor means a rising EUA price
    monotonically favours gas regardless of where fuel prices sit.

    LAG: the last auction STRICTLY BEFORE the target day. Auctions clear
    ~11:00 CET, an hour before the 12:00 gate closure, so a same-day price is
    arguably legal -- but auctions do not run daily, a fill is needed
    regardless, and a thesis leakage claim should not rest on a one-hour
    margin. See leakage_guard.SOURCE_TIMING.
    """
    if not snapshot.has("eex_eua_auctions"):
        return _empty(wide.index)

    eua = snapshot.load_series("eex_eua_auctions")
    price_col = "eua_price_eur_t"
    if price_col not in eua.columns:
        return _empty(wide.index)

    # The EU-wide auction is the reference; fall back to the day's mean where
    # only national auctions ran.
    if "auction_name" in eua.columns:
        eu = eua[eua["auction_name"].astype(str).str.strip().str.upper() == "EU"]
        daily = eu.groupby(eu.index.normalize())[price_col].mean()
        allday = eua.groupby(eua.index.normalize())[price_col].mean()
        daily = daily.reindex(allday.index).fillna(allday)
    else:
        daily = eua.groupby(eua.index.normalize())[price_col].mean()

    # Reindex to a contiguous calendar, then take the last auction strictly
    # before each day: shift(1) FIRST, then forward-fill. Filling before
    # shifting would let the target day's own auction reach itself.
    cal = pd.date_range(daily.index.min(), max(daily.index.max(), wide.index.max()), freq="D")
    daily = daily.reindex(cal).shift(1).ffill()

    eua_lagged = daily.reindex(wide.index)
    out = pd.DataFrame(index=wide.index)

    # ONE column, by the redundancy rule. The obvious four -- the EUA price,
    # the gas carbon cost, the coal carbon cost and the gas-vs-coal switching
    # advantage -- are each a CONSTANT MULTIPLE of the same series, because
    # the emission factors and efficiencies are fixed scalars. Emitting all
    # four measured r = 1.0000 between every pair and drove the matrix
    # singular; they carry one driver's worth of information between them and
    # would split that driver's SHAP attribution four ways at random.
    #
    # carbon_cost_gas is kept because it is denominated in EUR/MWh_el, the
    # same units as the target, so its coefficient is directly readable as a
    # pass-through rate. The other three are recoverable from it by the
    # constants above whenever they are wanted for interpretation:
    #     eua           = carbon_cost_gas / EF_GAS_T_PER_MWH_EL
    #     carbon_coal   = carbon_cost_gas * (EF_COAL / EF_GAS)
    #     switch_adv    = carbon_cost_gas * (EF_COAL / EF_GAS - 1)
    out["carbon_cost_gas_D-1"] = eua_lagged * EF_GAS_T_PER_MWH_EL

    if emit_all_variants:
        out["eua_eur_t_D-1"] = eua_lagged
        out["carbon_cost_coal_D-1"] = eua_lagged * EF_COAL_T_PER_MWH_EL
        out["carbon_switch_advantage_D-1"] = (
            eua_lagged * (EF_COAL_T_PER_MWH_EL - EF_GAS_T_PER_MWH_EL)
        )

    lg.declare_block(
        out.columns, source="eex_eua_auctions", lag_days=1,
        note="last auction strictly before the target day",
    )
    return out


def fuel_switch_proxy_block(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, lag: int = 1
) -> pd.DataFrame:
    """Observed gas-vs-coal dispatch split — a PROXY for fuel switching.

    Not a clean spread and named so it cannot be mistaken for one. Where the
    fleet actually landed between gas and coal yesterday is the market's own
    revealed answer to the switching question, which is available even though
    the fuel prices that drive it are not.

    Realized generation, so lagged.
    """
    if not snapshot.has("ec_public_power"):
        return _empty(wide.index)

    pp = snapshot.load_series("ec_public_power")
    gas_cols = [c for c in GAS_TYPES if c in pp.columns]
    coal_cols = [c for c in COAL_TYPES if c in pp.columns]
    if not gas_cols or not coal_cols:
        return _empty(wide.index)

    gas = pp[gas_cols].sum(axis=1)
    coal = pp[coal_cols].sum(axis=1)
    denom = (gas + coal).where((gas + coal) > 0)
    ratio = (gas / denom).to_frame("gascoalsplit")

    wide_ext = _to_daily_wide(ratio, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)
    cols = [c for c in wide_ext.columns if c.startswith("gascoalsplit_h")]
    out = _lagged_block(wide_ext, cols, lag, "switch_proxy").reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_public_power", lag_days=lag,
        note="PROXY for fuel switching, not a clean spread",
    )
    return out


# ==========================================================================
# Block 4 — market coupling
# ==========================================================================


def coupling_block(
    wide: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    zones: tuple[str, ...] = ("FR", "NL", "BE", "AT", "CH"),
    lags: tuple[int, ...] = (1, 7),
) -> pd.DataFrame:
    """Lagged neighbour day-ahead prices and their spread to Germany.

    THE TRAP THIS ENCODES: neighbour day-ahead prices clear in the SAME
    coupled EUPHEMIA auction as DE-LU and are published together, ~12:42 CET
    -- after the 12:00 gate closure. A same-day neighbour price is therefore
    a leak, notwithstanding that it is an "exogenous" input. Only lags are
    built, and the leakage guard fails the build if that ever changes.

    Lag 1 carries yesterday's coupling state; lag 7 carries the same weekday,
    which matters because interconnector congestion has a strong weekly
    profile.
    """
    if not snapshot.has("ec_price_neighbours"):
        return _empty(wide.index)

    nb = snapshot.load_series("ec_price_neighbours")
    keep = [f"price_{z}" for z in zones if f"price_{z}" in nb.columns]
    if not keep:
        return _empty(wide.index)

    frame = nb[keep].copy()
    frame.columns = [c.replace("price_", "nbprice") for c in keep]

    # Spread to the German price, which is what actually says whether the
    # border was binding -- a neighbour price alone does not.
    if snapshot.has("ec_price_de"):
        de = snapshot.load_series("ec_price_de")
        if "price_de" in de.columns:
            de_s = de["price_de"].reindex(frame.index)
            for z in zones:
                col = f"nbprice{z}"
                if col in frame.columns:
                    frame[f"nbspread{z}"] = frame[col] - de_s

    wide_ext = _to_daily_wide(frame, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for lag in lags:
        for base in frame.columns:
            cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
            if cols:
                parts.append(_lagged_block(wide_ext, cols, lag, base))
    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_price_neighbours", lag_days=min(lags),
        note="COUPLED AUCTION — same-day neighbour price leaks; lagged only",
    )
    return out


def cross_border_flow_block(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, lag: int = 1
) -> pd.DataFrame:
    """Lagged net physical cross-border flow, plus its magnitude.

    REALIZED flows, not scheduled: the delivery day's flow is not known at
    the origin, so only lags are legal. Scheduled/forecast NTC would be legal
    at lag 0 but needs an ENTSO-E token (see `entsoe_ntc` in the guard).
    """
    if not snapshot.has("ec_cross_border"):
        return _empty(wide.index)

    cb = snapshot.load_series("ec_cross_border")
    if cb.empty:
        return _empty(wide.index)

    net = cb.sum(axis=1).to_frame("netimport")
    net["absflow"] = cb.abs().sum(axis=1)

    wide_ext = _to_daily_wide(net, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for base in ("netimport", "absflow"):
        cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
        if cols:
            parts.append(_lagged_block(wide_ext, cols, lag, base))
    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_cross_border", lag_days=lag,
        note="REALIZED flows; scheduled NTC would need an ENTSO-E token",
    )
    return out


# ==========================================================================
# Block 5 — scarcity / reserve margin (keyless part)
# ==========================================================================


def reserve_margin_block(wide: pd.DataFrame) -> pd.DataFrame:
    """Forecast load against installed dispatchable capacity.

    Closer to a true reserve margin than `physical.scarcity_block`'s
    trailing-maximum proxy, because the denominator is real published
    capacity rather than a recent high-water mark. It is still not the true
    margin: capacity net of PLANNED AND FORCED OUTAGES needs the ENTSO-E
    outage feed. So the columns keep an explicit `_nooutage` marker -- the
    denominator is installed, not available, capacity, and on a day with a
    large outage the margin computed here is optimistic.

    Numerator is a day-ahead forecast and denominator is structural, so this
    is legal at lag 0.
    """
    if not snapshot.has("ec_installed_power"):
        return _empty(wide.index)

    cap = capacity_structure_block(wide)
    if cap.empty or "disp_capacity_D0" not in cap.columns:
        return _empty(wide.index)

    load_cols = [f"exog_1_{h}" for h in HOURS]
    if not all(c in wide.columns for c in load_cols):
        return _empty(wide.index)

    denom = cap["disp_capacity_D0"].where(cap["disp_capacity_D0"] > 0)
    load = wide[load_cols]

    margin = load.div(denom, axis=0)
    out = margin.copy()
    out.columns = [f"resmargin_nooutage_D0_{h}" for h in HOURS]
    out["resmargin_nooutage_peak_D0"] = margin.max(axis=1)
    lg.declare_block(
        out.columns, source="exog_1", lag_days=0,
        note="forecast load / installed capacity; NOT net of outages",
    )
    return out


# ==========================================================================
# Block 6 — storage / hydro (keyless part)
# ==========================================================================


def storage_block(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, lag: int = 1
) -> pd.DataFrame:
    """Lagged pumped-storage generation and pumping.

    Pumped storage arbitrages the intraday price shape, so its observed
    behaviour carries the market's own read on the spread between peak and
    off-peak. Reservoir FILLING RATES -- the seasonal opportunity-cost signal
    -- need an ENTSO-E token and are absent here.

    Realized generation, so lagged.
    """
    if not snapshot.has("ec_public_power"):
        return _empty(wide.index)

    pp = snapshot.load_series("ec_public_power")
    gen_cols = [c for c in PUMPED_TYPES if c in pp.columns]
    con_cols = [c for c in PUMPED_CONSUMPTION_TYPES if c in pp.columns]
    if not gen_cols and not con_cols:
        return _empty(wide.index)

    frame = pd.DataFrame(index=pp.index)
    if gen_cols:
        frame["pumpgen"] = pp[gen_cols].sum(axis=1)
    if con_cols:
        frame["pumpcon"] = pp[con_cols].sum(axis=1)
    if gen_cols and con_cols:
        frame["pumpnet"] = frame["pumpgen"] + frame["pumpcon"]

    wide_ext = _to_daily_wide(frame, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for base in frame.columns:
        cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
        if cols:
            parts.append(_lagged_block(wide_ext, cols, lag, base))
    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_public_power", lag_days=lag,
        note="realized pumped storage; reservoir levels need ENTSO-E",
    )
    return out


#: Technologies counted as dispatchable generation in /public_power. Must
#: correspond to DISPATCHABLE_CAPACITY_TYPES so numerator and denominator
#: describe the same fleet -- a utilisation ratio built from mismatched sets
#: is not a utilisation of anything.
DISPATCHABLE_GEN_TYPES = (
    "Nuclear",
    "Fossil brown coal / lignite",
    "Fossil hard coal",
    "Fossil gas",
    "Fossil oil",
    "Hydro water reservoir",
    "Hydro pumped storage",
    "Biomass",
    "Waste",
    "Others",
)


def dispatchable_headroom_block(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, lag: int = 1
) -> pd.DataFrame:
    """A NON-DEGENERATE keyless scarcity proxy: unused dispatchable capacity.

    Deliberately separate from `reserve_margin_block`, not folded into it,
    because the two answer different questions and one of them is provably
    empty.

    WHY THE OTHER ONE IS EMPTY. `reserve_margin_block` computes forecast load
    divided by installed capacity. Installed capacity is published yearly and
    is constant within a year, so that ratio is an EXACT linear function of
    load inside every year -- the within-year residual is zero to machine
    precision (scripts/check_scarcity_proxy.py). It cannot carry scarcity
    information because it carries nothing that `exog_1` does not already.

    WHY THIS ONE IS NOT. The failure above is not caused by the denominator
    being capacity; it is caused by dividing a series the model ALREADY HAS by
    a within-year constant. The fix is a numerator the baseline does not
    contain. Realized dispatchable GENERATION varies day to day and is absent
    from the benchmark information set, so:

        headroom     = installed dispatchable capacity - dispatchable generation
        utilisation  = dispatchable generation / installed capacity

    both vary daily and both carry genuinely new information. Headroom is the
    physically meaningful one for scarcity: it is how much dispatchable plant
    was still idle, which is what a tight system runs out of.

    STILL A PROXY, and the naming says so. Capacity here is INSTALLED, not
    available: a unit on outage counts as headroom that does not exist, so
    this reads optimistic exactly on the days scarcity matters most. The true
    margin needs the ENTSO-E outage feed. What this block buys is a fair test
    -- if a DAILY-VARYING keyless proxy also fails, the case for the outage
    feed is far stronger than if the only proxy tested were the one guaranteed
    to be degenerate.

    LAG. Generation is realized, so lagged; capacity is structural and legal at
    lag 0. The combination is declared at the generation lag, which is the
    binding constraint.
    """
    if not (snapshot.has("ec_public_power") and snapshot.has("ec_installed_power")):
        return _empty(wide.index)

    cap = capacity_structure_block(wide)
    if cap.empty or "disp_capacity_D0" not in cap.columns:
        return _empty(wide.index)

    pp = snapshot.load_series("ec_public_power")
    gen_cols = [t for t in DISPATCHABLE_GEN_TYPES if t in pp.columns]
    if not gen_cols:
        return _empty(wide.index)

    disp_gen = pp[gen_cols].sum(axis=1).to_frame("dispgen")
    wide_ext = _to_daily_wide(disp_gen, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    cols = [c for c in wide_ext.columns if c.startswith("dispgen_h")]
    gen_lagged = _lagged_block(wide_ext, cols, lag, "dispgen").reindex(wide.index)

    denom = cap["disp_capacity_D0"].where(cap["disp_capacity_D0"] > 0)

    out = pd.DataFrame(index=wide.index)
    for h in HOURS:
        g = gen_lagged[f"dispgen_D-{lag}_{h}"]
        out[f"headroom_proxy_D-{lag}_{h}"] = denom - g
        out[f"disputil_proxy_D-{lag}_{h}"] = g.div(denom)
    out[f"headroom_proxy_min_D-{lag}"] = out[
        [f"headroom_proxy_D-{lag}_{h}" for h in HOURS]
    ].min(axis=1)

    lg.declare_block(
        out.columns, source="ec_public_power", lag_days=lag,
        note="dispatchable headroom/utilisation; capacity is INSTALLED not available",
    )
    return out


#: |spread| above which a border is treated as binding. Not zero: prices are
#: published to 2 decimals, so exact equality is the right test in principle
#: but floating-point round-trips through CSV make a small tolerance safer.
BINDING_EUR = 0.01


def coupling_state_block(
    wide: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    zones: tuple[str, ...] = ("FR", "NL", "BE", "CH"),
    lag: int = 1,
    freq_window: int = 7,
    include_freq: bool = False,
) -> pd.DataFrame:
    """Market coupling decomposed into BINDING STATE and SIGNED MAGNITUDE.

    Replaces the raw continuous spread of `coupling_block`, which was measured
    to make its own target regime WORSE (coupling_stress MAE +10.8% at 80
    origins) despite carrying information the baseline lacks (median max|r|
    0.599, nothing above 0.90). Independent-but-harmful points at the
    encoding, and the encoding is the thing changed here.

    WHY THE RAW SPREAD FAILS. A coupled border clears DE and its neighbour at
    exactly the same price, so the spread distribution is a point mass at zero
    plus a continuous tail: FR 33.1% exactly zero, NL 38.0%, BE 31.2%. One
    linear coefficient has to serve both the discrete question "is this border
    congested at all" and the continuous question "by how much", and the
    column's scale is dominated by the tail, so the discrete part is buried.
    (CH is the exception at 0.1% zero -- Switzerland sits outside EU market
    coupling, so DE-CH essentially never clears identically. The zero-
    inflation argument applies to the coupled borders, not to CH.)

    WHAT IS EMITTED, per target hour:
      coupnbind_D-{lag}_h..    how many borders were congested (0..len(zones))
      coupsigned_D-{lag}_h..   mean SIGNED spread over congested borders.
                               Sign is kept because DE importing and DE
                               exporting are different price regimes; folding
                               them into |spread| would make one parameter
                               serve two opposite states, which is the same
                               mistake one level down.
      coupbindfreq_D-{lag}_h.. share of the previous `freq_window` days in
                               which that hour had any border congested.
                               OFF BY DEFAULT -- measured and rejected, see
                               below.

    WHY THE FREQUENCY FEATURE IS OFF. It was intended as the persistent,
    slow-moving part of congestion. Measured, a 7-day window destroys the
    signal it was meant to summarise:

        feature        CoV     ac(1)   distinct values
        coupnbind      0.33-0.52   0.20-0.42   5
        coupbindfreq   0.034       0.94-0.96   6

    and it correlates only +0.10 with the same-day binding count it is
    supposed to track. A near-constant column that is uncorrelated with the
    state it summarises cannot inform the forecast and can only spend degrees
    of freedom, which at p/n ~ 0.8 is not free. Enable it with
    include_freq=True and a much shorter window if the idea is revisited.

    LEAKAGE, and this one is a trap. Target-day congestion is NOT knowable at
    the forecast origin: it is an outcome of the very auction being forecast,
    published with the prices. A binding indicator FEELS structural, which is
    exactly why it is easy to build at lag 0 by accident -- installed capacity
    genuinely is legal at lag 0, and this looks similar. It is not. Every
    column here is built from realized prices at D-lag and earlier, declared
    against `ec_price_neighbours` (COUPLED_AUCTION, minimum lag 1), and the
    guard rejects the whole build if that is ever violated.
    """
    if not (snapshot.has("ec_price_neighbours") and snapshot.has("ec_price_de")):
        return _empty(wide.index)

    nb = snapshot.load_series("ec_price_neighbours")
    de = snapshot.load_series("ec_price_de")
    if "price_de" not in de.columns:
        return _empty(wide.index)

    de_s = de["price_de"].reindex(nb.index)
    present = [z for z in zones if f"price_{z}" in nb.columns]
    if not present:
        return _empty(wide.index)

    spreads = pd.DataFrame(
        {z: nb[f"price_{z}"] - de_s for z in present}, index=nb.index
    )
    binding = spreads.abs() > BINDING_EUR

    state = pd.DataFrame(index=nb.index)
    state["coupnbind"] = binding.sum(axis=1).astype(float)
    # Mean over congested borders only. Where nothing is congested the mean is
    # undefined, and 0 is the correct value there rather than a fill: a fully
    # coupled hour genuinely has zero signed divergence.
    state["coupsigned"] = spreads.where(binding).mean(axis=1).fillna(0.0)

    wide_ext = _to_daily_wide(state, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for base in ("coupnbind", "coupsigned"):
        cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
        if cols:
            parts.append(_lagged_block(wide_ext, cols, lag, base))

    # Persistent congestion: rolling share of recent days with any binding
    # border, computed on ALREADY-LAGGED values so the window can never reach
    # the target day.
    nbind_cols = [c for c in wide_ext.columns if c.startswith("coupnbind_h")]
    if include_freq and nbind_cols:
        any_bind = (wide_ext[nbind_cols] > 0).astype(float).shift(lag)
        freq = any_bind.rolling(window=freq_window, min_periods=freq_window).mean()
        freq.columns = [f"coupbindfreq_D-{lag}_{c.split('_')[-1]}" for c in nbind_cols]
        parts.append(freq)

    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_price_neighbours", lag_days=lag,
        note="congestion STATE is an auction outcome, legal only lagged",
    )
    return out


#: Borders that clear inside EU market coupling with DE-LU. CH is NOT one:
#: Switzerland sits outside the coupled day-ahead market, so DE-CH prices
#: essentially never clear identically (0.1% exactly equal, against 31-38% for
#: FR/NL/BE). Pooling it with the coupled borders averages two different
#: mechanisms into one number.
COUPLED_ZONES = ("FR", "NL", "BE")
UNCOUPLED_ZONES = ("CH",)


def coupling_split_block(
    wide: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    lag: int = 1,
) -> pd.DataFrame:
    """Coupling with the UNCOUPLED border separated from the coupled ones.

    The uniform encoding was wrong from the start, and the project's own data
    says so: FR/NL/BE spreads are exactly zero 31-38% of the time because a
    coupled border clears both zones at one price, while DE-CH is exactly zero
    0.1% of the time because Switzerland is outside EU market coupling. Those
    are two different physical mechanisms, and `coupling_block` pooled them.

    Consequences of the pooling, both of which this block avoids:
      - The binding-state question ("is this border congested") is meaningful
        for FR/NL/BE and almost vacuous for CH, which is essentially always
        "congested" in the sense of having a non-zero spread.
      - A mean spread taken across all four borders mixes a zero-inflated
        variable with a continuous one, so the coupled borders' point mass is
        diluted by CH's always-on variation.

    Emitted per target hour:
      coupbind_D-{lag}_h..     borders congested among the COUPLED set only
      coupspr_D-{lag}_h..      mean signed spread over congested coupled borders
      chspr_D-{lag}_h..        DE-CH signed spread, as its own feature
    """
    if not (snapshot.has("ec_price_neighbours") and snapshot.has("ec_price_de")):
        return _empty(wide.index)

    nb = snapshot.load_series("ec_price_neighbours")
    de = snapshot.load_series("ec_price_de")
    if "price_de" not in de.columns:
        return _empty(wide.index)
    de_s = de["price_de"].reindex(nb.index)

    coupled = [z for z in COUPLED_ZONES if f"price_{z}" in nb.columns]
    if not coupled:
        return _empty(wide.index)

    sp = pd.DataFrame({z: nb[f"price_{z}"] - de_s for z in coupled}, index=nb.index)
    binding = sp.abs() > BINDING_EUR

    state = pd.DataFrame(index=nb.index)
    state["coupbind"] = binding.sum(axis=1).astype(float)
    state["coupspr"] = sp.where(binding).mean(axis=1).fillna(0.0)
    for z in UNCOUPLED_ZONES:
        if f"price_{z}" in nb.columns:
            state["chspr"] = nb[f"price_{z}"] - de_s

    wide_ext = _to_daily_wide(state, hourly_index)
    if wide_ext.empty:
        return _empty(wide.index)

    parts = []
    for base in state.columns:
        cols = [c for c in wide_ext.columns if c.startswith(f"{base}_h")]
        if cols:
            parts.append(_lagged_block(wide_ext, cols, lag, base))
    if not parts:
        return _empty(wide.index)

    out = pd.concat(parts, axis=1).reindex(wide.index)
    lg.declare_block(
        out.columns, source="ec_price_neighbours", lag_days=lag,
        note="coupled borders and the uncoupled CH border kept separate",
    )
    return out


# ==========================================================================
# CONTROL BLOCKS — diagnostics for the B5 spike anomaly, not features
# ==========================================================================
#
# B5's spike improvement appeared twice independently (standalone Holm 0.028;
# B1+B5 raw 0.036) while the block's within-year residual against exog_1 is
# 3e-15 -- an exact linear function of load. A column set carrying zero
# independent information should not improve any regime, so either the
# correlation analysis missed something or the gain is an artifact of adding
# 25 columns at that scale to a LASSO at p/n ~ 0.8.
#
# These two blocks make that testable. Each emits EXACTLY 25 columns matched
# to B5's shape and scale while carrying no usable alignment with the target.
# If they reproduce the spike gain, it is a penalty/degrees-of-freedom
# artifact and B5's spike result must be retracted. If they do not, the
# anomaly is real.
#
# Neither is a thesis feature. Both are named `control_` and default off.

#: Fixed offset for the shifted-load control. 199 days: prime, so it never
#: aligns with the weekly cycle, and large enough that the target-day
#: relationship is destroyed. STRICTLY PAST, so the control stays inside the
#: information set -- a control that leaked would answer a different question
#: than the one being asked.
CONTROL_SHIFT_DAYS = 199

#: Seed for the noise control. Fixed so the experiment is reproducible; the
#: whole point is a control that can be re-run and get the same answer.
CONTROL_SEED = 42


def control_noise_block(wide: pd.DataFrame) -> pd.DataFrame:
    """CONTROL: 25 Gaussian columns matched to B5's per-column mean and sd.

    Carries no information whatsoever -- not about the target, not about
    anything. Matched in COUNT and SCALE to reserve_margin_block so that any
    effect it produces can only come from the act of adding that many columns
    at that magnitude, never from their content.
    """
    if not snapshot.has("ec_installed_power"):
        return _empty(wide.index)
    ref = reserve_margin_block(wide)
    if ref.empty:
        return _empty(wide.index)

    rng = np.random.default_rng(CONTROL_SEED)
    out = pd.DataFrame(index=wide.index)
    for c in ref.columns:
        col = ref[c]
        mu = float(col.mean(skipna=True))
        sd = float(col.std(skipna=True))
        vals = rng.normal(mu, sd if sd > 0 else 1.0, size=len(wide))
        name = c.replace("resmargin_nooutage", "ctrlnoise")
        s = pd.Series(vals, index=wide.index)
        # Mirror the reference column's missingness so the two variants drop
        # the same rows; otherwise the control would be scored on a different
        # day set and the comparison would be rigged.
        out[name] = s.where(col.notna())

    lg.declare_block(
        out.columns, source="calendar", lag_days=0,
        note="SYNTHETIC CONTROL: Gaussian noise, no information content",
    )
    return out


def control_shifted_load_block(wide: pd.DataFrame) -> pd.DataFrame:
    """CONTROL: real load columns shifted 199 days into the past.

    A stronger control than pure noise. It preserves load's marginal
    distribution AND its autocorrelation structure exactly -- it IS load --
    while destroying alignment with the target day. So it isolates the effect
    of adding realistically-shaped columns, as opposed to Gaussian ones, which
    a LASSO may treat differently.

    The shift is backwards, so every value is strictly past and the control
    stays inside the information set. A control that leaked would be answering
    a different question than the one asked.
    """
    load_cols = [f"exog_1_{h}" for h in HOURS]
    if not all(c in wide.columns for c in load_cols):
        return _empty(wide.index)
    if not snapshot.has("ec_installed_power"):
        return _empty(wide.index)
    ref = reserve_margin_block(wide)
    if ref.empty:
        return _empty(wide.index)

    shifted = wide[load_cols].shift(CONTROL_SHIFT_DAYS)
    out = pd.DataFrame(index=wide.index)
    for h in HOURS:
        out[f"ctrlshift_D0_{h}"] = shifted[f"exog_1_{h}"]
    out["ctrlshift_peak_D0"] = shifted.max(axis=1)

    # Same missingness as the reference, for the same reason as above.
    mask = ref.notna().all(axis=1)
    out = out.where(mask, other=np.nan)

    lg.declare_block(
        out.columns, source="exog_1", lag_days=CONTROL_SHIFT_DAYS,
        note=f"CONTROL: load shifted {CONTROL_SHIFT_DAYS}d back, alignment destroyed",
    )
    return out


#: Blocks that need the hourly benchmark index as well as the daily-wide
#: frame (they align an external hourly series onto it).
NEEDS_HOURLY_INDEX = {
    "merit_order_explicit_block",
    "fuel_switch_proxy_block",
    "coupling_block",
    "cross_border_flow_block",
    "storage_block",
    "dispatchable_headroom_block",
    "coupling_state_block",
    "coupling_split_block",
}

EXOG_BLOCKS = {
    "merit_order_explicit_block": merit_order_explicit_block,
    "capacity_structure_block": capacity_structure_block,
    "carbon_block": carbon_block,
    "fuel_switch_proxy_block": fuel_switch_proxy_block,
    "coupling_block": coupling_block,
    "cross_border_flow_block": cross_border_flow_block,
    "reserve_margin_block": reserve_margin_block,
    "storage_block": storage_block,
    "dispatchable_headroom_block": dispatchable_headroom_block,
    "coupling_state_block": coupling_state_block,
    "coupling_split_block": coupling_split_block,
    "control_noise_block": control_noise_block,
    "control_shifted_load_block": control_shifted_load_block,
}


def build_exog_blocks(
    wide: pd.DataFrame, hourly_index: pd.DatetimeIndex, enabled: dict | None = None
) -> pd.DataFrame:
    """Run the enabled snapshot-derived blocks and concatenate their columns.

    An unknown block name raises rather than being ignored, for the same
    reason as in `physical.build_physical_blocks`: a silently-dropped block
    still trains and still scores, and quietly is not the experiment anyone
    thought they ran.
    """
    if not enabled:
        return pd.DataFrame(index=wide.index)

    parts = []
    for name, setting in enabled.items():
        if name not in EXOG_BLOCKS:
            raise KeyError(
                f"unknown exog block '{name}'; known blocks: {sorted(EXOG_BLOCKS)}"
            )
        if not setting:
            continue
        kwargs = setting if isinstance(setting, dict) else {}
        fn = EXOG_BLOCKS[name]
        if name in NEEDS_HOURLY_INDEX:
            parts.append(fn(wide, hourly_index, **kwargs))
        else:
            parts.append(fn(wide, **kwargs))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(index=wide.index)
    return pd.concat(parts, axis=1)
