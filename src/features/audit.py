"""Feature audit and gap prioritization — src/features/audit.py

Stage 2 and Stage 3 of the power-engineering audit, as executable code.

`audit_features()` walks the Stage-1 reference map
(`src.features.price_formation.MECHANISMS`) against a real feature matrix
and classifies every physical price-formation mechanism as:

    present     -- all of its declared feature columns exist in X
    partial     -- some but not all of them exist, or the mechanism is
                   only covered by a proxy shared with another mechanism
    missing     -- none exist, but the data to build them IS available
    unavailable -- none exist and the required data is outside the
                   project's keyless information set (a structural gap,
                   not a coding gap)

`prioritize_gaps()` scores the actionable gaps by
driver_strength x data_availability and returns them ordered.

The result of `audit_features` is the source of truth for the rest of the
work: "missing" items become additions, "unavailable" items stay declared
and unbuilt, "present" items are frozen and left untouched.

Everything here is a pure function over a DataFrame -- no I/O, no saved
report. Callers print it.
"""

from __future__ import annotations

import pandas as pd

from src.features.price_formation import MECHANISMS, Mechanism

#: How much of a mechanism's physics survives when only a proxy is
#: available. Applied to driver_strength in the Stage-3 score so a
#: proxy-covered mechanism ranks below a directly-measured one.
PROXY_DISCOUNT = 0.5

#: data_availability scores used by prioritize_gaps().
AVAILABILITY_FULL = 1.0  # buildable from price/exog_1/exog_2 today
AVAILABILITY_PROXY = 0.5  # only a proxy is buildable keylessly
AVAILABILITY_NONE = 0.0  # needs a registered feed


def _columns_for(mech: Mechanism, columns: set[str]) -> list[str]:
    """Which of X's columns represent this mechanism.

    `Mechanism.features` holds column PREFIXES (e.g. "resload_D0_h"), since
    a 24-hour block expands to 24 columns. A prefix counts as satisfied when
    at least one real column starts with it.
    """
    hits = []
    for prefix in mech.features:
        hits.extend(sorted(c for c in columns if c.startswith(prefix)))
    return hits


def _availability(mech: Mechanism) -> float:
    if not mech.available:
        return AVAILABILITY_NONE
    # A mechanism that is `available` but still names a blocker is only
    # partially buildable -- the scarcity case, where the true reserve
    # margin needs an outage feed and only the tightness proxy is keyless.
    if mech.blocked_by:
        return AVAILABILITY_PROXY
    return AVAILABILITY_FULL


def classify(mech: Mechanism, columns: set[str]) -> tuple[str, list[str]]:
    """Classify one mechanism against a set of feature-column names."""
    hits = _columns_for(mech, columns)

    if not mech.features:
        # Nothing was ever declared to represent it.
        return ("unavailable" if not mech.available else "missing"), hits

    if not hits:
        return ("unavailable" if not mech.available else "missing"), hits

    expected_prefixes = len(mech.features)
    satisfied = sum(
        1 for p in mech.features if any(c.startswith(p) for c in columns)
    )
    if satisfied < expected_prefixes:
        return "partial", hits
    # Fully represented -- but a proxy-only mechanism is never "present",
    # because the column existing does not mean the physics is measured.
    if mech.blocked_by:
        return "partial", hits
    return "present", hits


def audit_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stage 2. Inventory the feature matrix against the price-formation map.

    Parameters
    ----------
    X : the feature matrix from `build_features` (only its columns are read).

    Returns
    -------
    DataFrame, one row per mechanism, in causal-chain order, with columns
    [mechanism, status, driver_strength, data_availability, n_columns,
     builder, requires, blocked_by, example_columns].
    """
    columns = set(X.columns)
    rows = []
    for mech in MECHANISMS:
        status, hits = classify(mech, columns)
        rows.append(
            dict(
                mechanism=mech.key,
                status=status,
                driver_strength=mech.driver_strength,
                data_availability=_availability(mech),
                n_columns=len(hits),
                builder=mech.builder or "",
                requires=",".join(mech.requires),
                blocked_by=mech.blocked_by or "",
                example_columns=",".join(hits[:2]),
            )
        )
    return pd.DataFrame(rows).set_index("mechanism")


def prioritize_gaps(audit: pd.DataFrame) -> pd.DataFrame:
    """Stage 3. Score and order the gaps by driver strength x availability.

    Only rows whose status is not "present" are gaps. `unavailable` rows
    score 0 by construction (availability 0) and sort to the bottom -- they
    are listed rather than dropped, because a structural gap that vanishes
    from the report is a gap nobody remembers to revisit.

    A `partial` row keeps its driver strength discounted by PROXY_DISCOUNT:
    half the physics is already captured, so the remaining headroom is worth
    less than an untouched mechanism of the same strength.
    """
    gaps = audit[audit["status"] != "present"].copy()

    effective = gaps["driver_strength"].astype(float)
    effective = effective.where(gaps["status"] != "partial", effective * PROXY_DISCOUNT)
    gaps["effective_strength"] = effective
    gaps["score"] = gaps["effective_strength"] * gaps["data_availability"]

    gaps["actionable"] = gaps["score"] > 0
    gaps = gaps.sort_values(
        ["score", "driver_strength"], ascending=[False, False]
    )
    gaps["rank"] = range(1, len(gaps) + 1)
    return gaps[
        [
            "rank",
            "status",
            "driver_strength",
            "effective_strength",
            "data_availability",
            "score",
            "actionable",
            "builder",
            "blocked_by",
        ]
    ]
