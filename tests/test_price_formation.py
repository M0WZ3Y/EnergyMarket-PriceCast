"""Stage 1-3 contract tests: the price-formation map and the feature audit.

These guard the audit's *integrity*, not its verdict. The point of a
machine-readable reference map is that it cannot quietly drift out of sync
with the code that consumes it -- a mechanism naming a builder that does not
exist, or a "buildable" mechanism with no columns declared, would make the
audit report confidently wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import physical
from src.features.audit import audit_features, prioritize_gaps
from src.features.price_formation import (
    BUILDABLE,
    DATA_BLOCKED,
    MECHANISMS,
    MECHANISMS_BY_KEY,
)


def test_mechanism_keys_unique():
    keys = [m.key for m in MECHANISMS]
    assert len(keys) == len(set(keys)), f"duplicate mechanism keys: {keys}"


def test_every_declared_builder_exists():
    """A builder name in the map must resolve in physical.BLOCKS.

    Without this, renaming a block would leave the map pointing at a ghost
    and the audit would report a mechanism as buildable that nothing can
    build.
    """
    from src.features import physical_exog

    known = set(physical.BLOCKS) | set(physical_exog.EXOG_BLOCKS)
    for m in MECHANISMS:
        if m.builder is None:
            continue
        assert m.builder in known, (
            f"mechanism '{m.key}' names builder '{m.builder}', which is in "
            f"neither physical.BLOCKS nor physical_exog.EXOG_BLOCKS "
            f"({sorted(known)})"
        )


def test_unavailable_mechanisms_declare_a_blocker_and_build_nothing():
    """A mechanism we cannot get data for must say so and must not pretend
    to have features. Silence here is how a structural gap turns into a
    forgotten one."""
    for key in DATA_BLOCKED:
        m = MECHANISMS_BY_KEY[key]
        assert m.blocked_by, f"'{key}' is unavailable but names no blocking source"
        assert m.features == (), f"'{key}' is unavailable but declares features"
        assert m.builder is None, f"'{key}' is unavailable but names a builder"


def test_buildable_mechanisms_declare_features():
    for key in BUILDABLE:
        m = MECHANISMS_BY_KEY[key]
        assert m.features, f"'{key}' names a builder but declares no feature columns"


def test_driver_strength_in_range():
    for m in MECHANISMS:
        assert 1 <= m.driver_strength <= 5, f"'{m.key}' strength {m.driver_strength}"


def test_audit_reports_missing_when_columns_absent():
    """A buildable mechanism with no columns is 'missing'; one whose DATA
    cannot be obtained at all is 'unavailable'. The distinction is the whole
    point of the audit -- 'missing' is a to-do, 'unavailable' is a limit.

    market_coupling moved from unavailable to missing on 2026-08-28, when
    lagged neighbour prices became buildable from the pinned snapshot.
    clean_spreads is used here instead because it is the one mechanism still
    structurally blocked (Montel-licensed gas and coal prices)."""
    X = pd.DataFrame(columns=["price_D-1_h00", "dow_0"])
    a = audit_features(X)
    assert a.loc["residual_load", "status"] == "missing"
    assert a.loc["market_coupling", "status"] == "missing"
    assert a.loc["clean_spreads", "status"] == "unavailable"


def test_audit_reports_present_when_columns_exist():
    cols = [f"resload_D0_h{h:02d}" for h in range(24)]
    cols += [f"resload_D-1_h{h:02d}" for h in range(24)]
    a = audit_features(pd.DataFrame(columns=cols))
    assert a.loc["residual_load", "status"] == "present"
    assert a.loc["residual_load", "n_columns"] == 48


def test_proxy_only_mechanism_never_reports_present():
    """scarcity is covered by a proxy, not a measured reserve margin.

    Reporting it as 'present' would let a proxy be mistaken for the physics
    it stands in for -- the single most misleading thing this audit could
    say.
    """
    cols = [f"tightness_proxy_D0_h{h:02d}" for h in range(24)]
    cols += ["tightness_proxy_peak_D0"]
    a = audit_features(pd.DataFrame(columns=cols))
    assert a.loc["scarcity_reserve_margin", "status"] == "partial"


def test_prioritization_ranks_available_gaps_above_blocked_ones():
    a = audit_features(pd.DataFrame(columns=["dow_0"]))
    gaps = prioritize_gaps(a)
    actionable = gaps[gaps["actionable"]]
    blocked = gaps[~gaps["actionable"]]
    assert not actionable.empty and not blocked.empty
    assert actionable["rank"].max() < blocked["rank"].min(), (
        "a data-blocked mechanism outranked a buildable one; a gap nobody "
        "can act on must never head the queue"
    )
    assert (blocked["score"] == 0).all()


def test_prioritization_lists_blocked_gaps_rather_than_dropping_them():
    a = audit_features(pd.DataFrame(columns=["dow_0"]))
    gaps = prioritize_gaps(a)
    for key in DATA_BLOCKED:
        assert key in gaps.index, f"blocked mechanism '{key}' vanished from the report"


@pytest.mark.parametrize("key", DATA_BLOCKED)
def test_blocked_builders_raise_instead_of_fabricating(key):
    """The load-bearing guarantee of this whole module: no stub ever returns
    a made-up series. It raises."""
    name = f"{key}_block"
    assert name in physical.BLOCKS
    with pytest.raises(physical.FeatureDataUnavailable):
        physical.BLOCKS[name](pd.DataFrame())
