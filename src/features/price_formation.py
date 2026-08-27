"""Price-formation reference map — src/features/price_formation.py

A machine-readable mapping from each physical price-formation mechanism in a
day-ahead electricity market to the pipeline feature(s) that represent it.
This is a data structure the audit code consumes (see `audit.py`), not
documentation: `MECHANISMS` is iterated programmatically to classify the
feature set as present / misspecified / missing and to score the gaps.

The chain modelled here is the standard merit-order account of how a
day-ahead price is set:

    residual load  ->  merit-order position  ->  marginal fuel + carbon
                   ->  clean spreads  ->  market coupling
                   ->  ramping / flexibility  ->  scarcity / reserve margin
                   ->  storage / hydro

DATA REALITY (as of this repo): the benchmark information set is exactly
three series -- `price`, `exog_1` (Amprion day-ahead LOAD forecast) and
`exog_2` (day-ahead PV+WIND forecast); see data/raw/DE.csv. Everything
derivable from those two forecasts is buildable here. Everything else
(carbon, fuel curves, cross-border capacity, outages, hydro reservoir
levels) needs a registered data feed, which CLAUDE.md forbids
("Never introduce a data source that needs registration"). Those mechanisms
are therefore declared with `available=False` and a named blocking source.
They are represented honestly as gaps, never as fabricated columns.

INFORMATION-SET DISCIPLINE: every `builder` referenced here consumes only
quantities knowable before gate closure (day-ahead forecasts and strictly
past realizations). `requires` lists the raw input columns a mechanism
needs; `realized_only=True` marks a mechanism that CANNOT be built from
forecasts at all and must stay unbuilt even if the data appeared.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mechanism:
    """One physical price-formation mechanism and its feature representation.

    Attributes
    ----------
    key : short stable identifier, used as the audit's join key.
    description : the physical effect on the day-ahead price.
    requires : raw loader columns needed to build it.
    features : the feature-column PREFIXES that represent it in X. Empty
        when the mechanism is unrepresented.
    builder : name of the callable in `src.features.physical` that emits
        those columns, or None when nothing can be built.
    available : whether the required data exists in the project's sanctioned
        keyless sources. False => structurally missing, not a coding gap.
    blocked_by : when available is False, the registered feed that would be
        needed. Recorded so the gap is auditable rather than forgotten.
    driver_strength : 1-5, literature-weighted influence on DE day-ahead
        price formation. Used by the Stage-3 gap scorer.
    """

    key: str
    description: str
    requires: tuple[str, ...]
    driver_strength: int
    features: tuple[str, ...] = ()
    builder: str | None = None
    available: bool = True
    blocked_by: str | None = None
    realized_only: bool = False
    notes: str = ""


MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        key="residual_load",
        description=(
            "Demand net of must-run renewable infeed. The quantity the "
            "dispatchable fleet must actually serve, and the single strongest "
            "physical driver of the day-ahead price level."
        ),
        requires=("exog_1", "exog_2"),
        driver_strength=5,
        features=("resload_D0_h", "resload_D-1_h"),
        builder="residual_load_block",
        available=True,
        notes=(
            "exog_1 - exog_2. Both are day-ahead FORECASTS, so the D0 "
            "(target-day) value is knowable at the forecast origin -- the "
            "same legality argument that already licenses exog_*_D0_h in "
            "configs/features.yaml."
        ),
    ),
    Mechanism(
        key="residual_load_gradient",
        description=(
            "Hour-to-hour change in residual load. Steep ramps force fast, "
            "expensive units on or off and drive intraday price shape."
        ),
        requires=("exog_1", "exog_2"),
        driver_strength=4,
        features=("resload_grad_D0_h", "resload_ramp_max_D0", "resload_ramp_min_D0"),
        builder="residual_load_gradient_block",
        available=True,
        notes="First difference of residual load across the 24 target hours.",
    ),
    Mechanism(
        key="merit_order_position",
        description=(
            "Where residual load sits on the supply stack. Non-linear: the "
            "same MW change moves price far more in the steep peaking region "
            "than on the flat mid-merit shoulder."
        ),
        requires=("exog_1", "exog_2"),
        driver_strength=5,
        features=("merit_pos_D0_h", "res_share_D0_h"),
        builder="merit_order_block",
        available=True,
        notes=(
            "Represented WITHOUT an external supply curve: the empirical "
            "quantile of target-day residual load within a TRAILING window of "
            "strictly past days is a monotone proxy for stack position, and "
            "the renewable share captures the merit-order (price-suppression) "
            "effect directly. The quantile reference uses past days only."
        ),
    ),
    Mechanism(
        key="marginal_fuel_carbon",
        description=(
            "Short-run marginal cost of the price-setting thermal unit: fuel "
            "cost plus EUA carbon cost divided by efficiency."
        ),
        requires=("gas_price", "coal_price", "eua_price"),
        driver_strength=5,
        features=(),
        builder=None,
        available=False,
        blocked_by="EEX/ICE gas, coal and EUA futures settlement prices (registered feed)",
        notes=(
            "No fuel or carbon series exists in the benchmark information "
            "set. Declared as a structural gap; stubbed in physical.py so the "
            "absence is explicit and testable, never silently fabricated."
        ),
    ),
    Mechanism(
        key="clean_spreads",
        description=(
            "Clean spark spread (gas) and clean dark spread (coal): power "
            "price minus carbon-inclusive generation cost. Their crossover is "
            "the coal-to-gas switching point that sets the marginal "
            "technology."
        ),
        requires=("gas_price", "coal_price", "eua_price"),
        driver_strength=4,
        features=(),
        builder=None,
        available=False,
        blocked_by="EEX/ICE gas, coal and EUA futures settlement prices (registered feed)",
        notes=(
            "Depends entirely on marginal_fuel_carbon. Cannot be approximated "
            "from load/RES forecasts without inventing prices."
        ),
    ),
    Mechanism(
        key="market_coupling",
        description=(
            "Euphemia couples DE-LU with its neighbours: prices converge "
            "while interconnector capacity is free and separate when it "
            "binds. Neighbour prices and NTC set the import/export bound."
        ),
        requires=("neighbor_price", "ntc", "scheduled_flow"),
        driver_strength=4,
        features=(),
        builder=None,
        available=False,
        blocked_by="ENTSO-E Transparency Platform (security token required)",
        notes=(
            "Neighbour day-ahead prices are published simultaneously with "
            "DE-LU, so same-day neighbour prices are NOT knowable at the "
            "forecast origin -- only lagged neighbour prices and forecast NTC "
            "would be legal. Recorded here so any future implementation "
            "inherits the constraint rather than rediscovering it."
        ),
    ),
    Mechanism(
        key="ramping_flexibility",
        description=(
            "Cost of moving the dispatchable fleet: min-up/min-down times and "
            "ramp limits make steep net-load ramps expensive and can decouple "
            "adjacent hours."
        ),
        requires=("exog_1", "exog_2"),
        driver_strength=3,
        features=("resload_grad_D0_h", "resload_ramp_max_D0"),
        builder="residual_load_gradient_block",
        available=True,
        notes=(
            "Shares its representation with residual_load_gradient -- the "
            "same columns serve both mechanisms. The audit reports it as "
            "covered-by-proxy, not as an independent feature block."
        ),
    ),
    Mechanism(
        key="scarcity_reserve_margin",
        description=(
            "Tightness of supply: forecast load against available "
            "dispatchable capacity net of planned outages. Drives the convex "
            "upper tail of the price distribution."
        ),
        requires=("exog_1", "available_capacity", "planned_outages"),
        driver_strength=4,
        features=("tightness_proxy_D0_h", "tightness_proxy_peak_D0"),
        builder="scarcity_block",
        available=True,
        blocked_by=(
            "ENTSO-E installed capacity + planned outage feed (token) for the "
            "TRUE reserve margin; only the proxy below is buildable keylessly"
        ),
        notes=(
            "PARTIAL. The true reserve margin needs an outage feed. What IS "
            "buildable keylessly is a tightness proxy: target-day residual "
            "load relative to the trailing-window maximum residual load, i.e. "
            "how close the system is to the highest dispatchable requirement "
            "recently observed. Emitted under an explicit `_proxy` naming so "
            "it can never be mistaken for a true reserve margin."
        ),
    ),
    Mechanism(
        key="storage_hydro",
        description=(
            "Reservoir and pumped-storage opportunity cost links today's "
            "price to expected future prices and smooths the intraday shape."
        ),
        requires=("reservoir_level", "hydro_generation"),
        driver_strength=2,
        features=(),
        builder=None,
        available=False,
        blocked_by="ENTSO-E aggregated filling rate / hydro generation (token required)",
        notes=(
            "Lowest-strength mechanism for DE-LU specifically (limited "
            "reservoir hydro); listed for completeness of the chain."
        ),
    ),
)


MECHANISMS_BY_KEY: dict[str, Mechanism] = {m.key: m for m in MECHANISMS}

#: Mechanisms that can be built from the sanctioned keyless information set.
BUILDABLE: tuple[str, ...] = tuple(m.key for m in MECHANISMS if m.builder is not None)

#: Mechanisms blocked purely by data access, not by code.
DATA_BLOCKED: tuple[str, ...] = tuple(m.key for m in MECHANISMS if not m.available)


def chain() -> list[str]:
    """The mechanism keys in causal order, as declared in MECHANISMS."""
    return [m.key for m in MECHANISMS]
