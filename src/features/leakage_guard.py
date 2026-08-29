"""Information-set leakage guard — src/features/leakage_guard.py

The project's non-negotiable rule is that no feature may use information
dated after the forecast origin. `tests/test_features.py` already checks that
by perturbation for the original columns. This module adds the complementary
check the external data makes necessary: a DECLARATIVE one, where every
feature column states which information it draws on and the guard fails the
build if that statement is inconsistent with the pre-gate-closure set.

Why declarative as well as empirical. A perturbation test can only catch a
leak in data it happens to perturb. Once features come from nine external
series with different publication times, the question "is this column legal?"
stops being about arithmetic and becomes about WHEN each source publishes --
which no amount of perturbing a local DataFrame can answer. The publication
time has to be stated, and then checked.

THE TIMELINE, which every rule below is measured against:

    D-1 ~11:00 CET   EUA auction clears
    D-1  12:00 CET   day-ahead GATE CLOSURE for delivery day D  <-- ORIGIN
    D-1 ~12:42 CET   EUPHEMIA clears; DE-LU *and every neighbour* price
                     published simultaneously
    D    00:00-23:00 delivery; generation, flows and prices realize

Two traps this encodes, both named in the task brief:

  1. Neighbour day-ahead prices clear in the SAME coupled auction as DE-LU.
     They are published AFTER the origin, so a same-day neighbour price is a
     leak even though it is "only" an exogenous input. Legal at lag >= 1 day.

  2. Realized flows and realized outages for day D are not known at the
     origin. Only scheduled/forecast values, or lags, are legal.

`SOURCE_TIMING` records each raw series' availability class.
`declare` registers what a feature column reads. `assert_no_leakage` is the
build-failing check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Availability(Enum):
    """When a raw series becomes known, relative to the forecast origin."""

    #: A forecast published before gate closure for the delivery day itself.
    #: Legal at lag 0 (the target day).
    EX_ANTE_FORECAST = "ex_ante_forecast"

    #: Structural information published well in advance (installed capacity,
    #: calendar). Legal at lag 0.
    STRUCTURAL = "structural"

    #: Clears in the same coupled auction as the target, i.e. AFTER the
    #: origin. Legal only at lag >= 1 day.
    COUPLED_AUCTION = "coupled_auction"

    #: Known only after delivery (realized generation, realized flows,
    #: realized price). Legal only at lag >= 1 day.
    REALIZED = "realized"

    #: Auction/settlement that clears BEFORE gate closure on its own day.
    #: Treated as lag >= 1 here by deliberate choice -- see EUA note below.
    PRE_GATE_SETTLEMENT = "pre_gate_settlement"


#: Minimum legal lag in days, by availability class.
MIN_LAG_DAYS = {
    Availability.EX_ANTE_FORECAST: 0,
    Availability.STRUCTURAL: 0,
    Availability.COUPLED_AUCTION: 1,
    Availability.REALIZED: 1,
    Availability.PRE_GATE_SETTLEMENT: 1,
}


SOURCE_TIMING: dict[str, Availability] = {
    # --- benchmark series (already in the frozen pipeline) --------------
    "price": Availability.REALIZED,
    "exog_1": Availability.EX_ANTE_FORECAST,   # Amprion day-ahead load forecast
    "exog_2": Availability.EX_ANTE_FORECAST,   # day-ahead PV+wind forecast
    "calendar": Availability.STRUCTURAL,
    # --- pinned external snapshot ---------------------------------------
    "ec_price_de": Availability.COUPLED_AUCTION,
    "ec_price_neighbours": Availability.COUPLED_AUCTION,
    "ec_public_power": Availability.REALIZED,
    "ec_cross_border": Availability.REALIZED,
    "ec_installed_power": Availability.STRUCTURAL,
    # EUA auctions clear ~11:00 CET, an hour BEFORE the 12:00 gate closure,
    # so a same-day carbon price is arguably legal. It is deliberately
    # classified as lag >= 1 anyway: auctions do not run every day, so a
    # fill is required regardless, and resting a thesis-critical leakage
    # claim on a one-hour margin buys a marginally fresher price at the cost
    # of a defensible one. The conservative reading is the cheap one here.
    "eex_eua_auctions": Availability.PRE_GATE_SETTLEMENT,
    # --- Tier B (present only once a token lands) -----------------------
    "entsoe_outages": Availability.REALIZED,
    "entsoe_ntc": Availability.EX_ANTE_FORECAST,   # forecast NTC is published ahead
    "entsoe_reservoir": Availability.REALIZED,
}


@dataclass(frozen=True)
class ColumnDeclaration:
    """What one feature column reads, and at what lag."""

    column: str
    source: str
    lag_days: int
    note: str = ""

    @property
    def availability(self) -> Availability:
        if self.source not in SOURCE_TIMING:
            raise KeyError(
                f"column '{self.column}' declares unknown source "
                f"'{self.source}'. Add it to SOURCE_TIMING with its "
                f"availability class -- an undeclared source cannot be "
                f"checked, and an unchecked source is how a leak gets in."
            )
        return SOURCE_TIMING[self.source]

    @property
    def min_lag(self) -> int:
        return MIN_LAG_DAYS[self.availability]

    @property
    def is_legal(self) -> bool:
        return self.lag_days >= self.min_lag


class LeakageError(AssertionError):
    """Raised when a declared feature would read past the forecast origin."""


_REGISTRY: dict[str, ColumnDeclaration] = {}


def declare(column: str, source: str, lag_days: int, note: str = "") -> ColumnDeclaration:
    """Register what a feature column reads. Returns the declaration."""
    d = ColumnDeclaration(column=column, source=source, lag_days=lag_days, note=note)
    _REGISTRY[column] = d
    return d


def declare_block(columns, source: str, lag_days: int, note: str = "") -> None:
    for c in columns:
        declare(c, source=source, lag_days=lag_days, note=note)


def registry() -> dict[str, ColumnDeclaration]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    _REGISTRY.clear()


#: Column-name conventions used by the existing pipeline, so declarations can
#: be inferred for columns that follow them rather than hand-listed.
_LAG_PATTERNS = (
    (re.compile(r"_D-(\d+)_h\d{2}$"), lambda m: int(m.group(1))),
    (re.compile(r"_D0_h\d{2}$"), lambda m: 0),
    (re.compile(r"_D0$"), lambda m: 0),
    (re.compile(r"_lag(\d+)d"), lambda m: int(m.group(1))),
)


def infer_lag(column: str) -> int | None:
    """Read the lag out of a column name, or None if it follows no convention.

    Returning None rather than guessing 0 is deliberate: an unrecognised
    column must be reported as undeclared, not silently assumed to be the
    riskiest legal value.
    """
    for pat, fn in _LAG_PATTERNS:
        m = pat.search(column)
        if m:
            return fn(m)
    return None


def assert_no_leakage(columns=None, strict: bool = True) -> list[ColumnDeclaration]:
    """Fail the build if any declared column reads past the forecast origin.

    Parameters
    ----------
    columns : optional iterable of column names actually present in X. When
        given, every one of them must have a declaration (under `strict`);
        this is what catches a NEW column that nobody declared, which is the
        realistic way a leak arrives -- not by someone declaring an illegal
        lag, but by adding a column and never declaring it at all.
    strict : when True, an undeclared column is itself a failure.

    Returns the offending declarations (empty on success) after raising, so
    the exception message carries the full list rather than the first item.
    """
    violations = [d for d in _REGISTRY.values() if not d.is_legal]

    undeclared: list[str] = []
    if columns is not None:
        known = set(_REGISTRY)
        undeclared = [c for c in columns if c not in known]

    problems = []
    for d in violations:
        problems.append(
            f"  {d.column}: reads '{d.source}' "
            f"({d.availability.value}, needs lag >= {d.min_lag}d) "
            f"at lag {d.lag_days}d"
            + (f" -- {d.note}" if d.note else "")
        )
    if strict and undeclared:
        for c in sorted(undeclared)[:20]:
            hint = infer_lag(c)
            problems.append(
                f"  {c}: UNDECLARED"
                + (f" (name implies lag {hint}d)" if hint is not None else "")
            )
        if len(undeclared) > 20:
            problems.append(f"  ... and {len(undeclared) - 20} more undeclared")

    if problems:
        raise LeakageError(
            "information-set violation: feature(s) would read data published "
            "after the forecast origin (day-ahead gate closure, 12:00 CET).\n"
            + "\n".join(problems)
            + "\n\nNeighbour day-ahead prices clear in the SAME coupled auction "
            "as DE-LU and are published ~12:42 CET, after the origin -- they "
            "are legal only at lag >= 1 day. Realized generation, flows and "
            "outages for the delivery day are not known at the origin either."
        )
    return violations


def summary() -> str:
    """Human-readable status line, printed by the ablation harness."""
    if not _REGISTRY:
        return "leakage guard: no columns declared"
    by_class: dict[str, int] = {}
    worst: dict[str, int] = {}
    for d in _REGISTRY.values():
        k = d.availability.value
        by_class[k] = by_class.get(k, 0) + 1
        worst[k] = min(worst.get(k, 99), d.lag_days)
    parts = [
        f"{k}={n}@min_lag{worst[k]}d" for k, n in sorted(by_class.items())
    ]
    return f"leakage guard: {len(_REGISTRY)} columns declared — " + ", ".join(parts)
