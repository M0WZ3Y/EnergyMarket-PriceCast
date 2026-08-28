"""Snapshot-derived feature blocks (blocks 2-6).

These tests run against synthetic pinned data where possible so they do not
depend on the committed snapshot being present, and skip cleanly where the
real snapshot is genuinely required.

The properties under test are the ones that would silently corrupt results
rather than raise: a block reading the target day when it should read a lag,
a missing series producing zeros instead of nothing, and the redundancy rule
being violated by columns that are exact rescalings of one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import leakage_guard as lg
from src.features import physical_exog as px


@pytest.fixture(autouse=True)
def clean_registry():
    lg.clear_registry()
    yield
    lg.clear_registry()


@pytest.fixture
def hourly_index() -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=24 * 60, freq="h")


@pytest.fixture
def wide(hourly_index) -> pd.DataFrame:
    from src.features.pipeline import _pivot_to_daily_wide

    n = len(hourly_index)
    hour = np.arange(n) % 24
    day = np.arange(n) // 24
    return _pivot_to_daily_wide(
        pd.DataFrame(
            {
                "price": 40.0 + hour,
                "exog_1": 20000.0 + 300 * hour + 10 * day,
                "exog_2": 5000.0 + 100 * hour,
            },
            index=hourly_index,
        )
    )


# ----------------------------------------------------------- skip behaviour


def test_blocks_return_empty_when_series_is_not_pinned(wide, hourly_index, monkeypatch):
    """A missing series must yield NO columns, never zeros.

    This is the ENTSO-E case and the normal state until a token lands. A
    block that returned zeros would put a fabricated physical driver into X,
    and every metric computed afterwards would be a statement about invented
    data -- with nothing anywhere reporting a problem.
    """
    monkeypatch.setattr(px.snapshot, "has", lambda key: False)

    for name, fn in px.EXOG_BLOCKS.items():
        out = fn(wide, hourly_index) if name in px.NEEDS_HOURLY_INDEX else fn(wide)
        assert out.shape[1] == 0, f"{name} produced columns with no pinned data"
        assert list(out.index) == list(wide.index), f"{name} changed the index"


def test_unknown_block_name_raises(wide, hourly_index):
    with pytest.raises(KeyError, match="unknown exog block"):
        px.build_exog_blocks(wide, hourly_index, {"coupling_blok": True})


def test_disabled_blocks_produce_nothing(wide, hourly_index):
    out = px.build_exog_blocks(wide, hourly_index, {"coupling_block": False})
    assert out.shape[1] == 0


# ----------------------------------------------------------- redundancy rule


@pytest.mark.skipif(
    not px.snapshot.has("eex_eua_auctions"), reason="EUA snapshot not pinned"
)
def test_carbon_block_emits_one_column_by_default(wide):
    """The EUA price, the gas carbon cost, the coal carbon cost and the
    switching advantage are constant multiples of one another (fixed emission
    factors), so emitting all four measured r = 1.0000 between every pair.
    One driver gets one column."""
    out = px.carbon_block(wide)
    assert list(out.columns) == ["carbon_cost_gas_D-1"]


@pytest.mark.skipif(
    not px.snapshot.has("eex_eua_auctions"), reason="EUA snapshot not pinned"
)
def test_carbon_variants_are_exact_rescalings_when_requested(wide):
    """Verifies the documented recovery constants actually hold, so dropping
    the three columns really is lossless rather than merely claimed to be."""
    out = px.carbon_block(wide, emit_all_variants=True)
    base = out["carbon_cost_gas_D-1"].dropna()
    if base.empty:
        pytest.skip("no overlapping carbon data for this synthetic window")
    ratio = px.EF_COAL_T_PER_MWH_EL / px.EF_GAS_T_PER_MWH_EL
    np.testing.assert_allclose(
        out.loc[base.index, "carbon_cost_coal_D-1"], base * ratio, rtol=1e-9
    )
    np.testing.assert_allclose(
        out.loc[base.index, "eua_eur_t_D-1"], base / px.EF_GAS_T_PER_MWH_EL, rtol=1e-9
    )


# ----------------------------------------------------------- leakage


@pytest.mark.skipif(
    not px.snapshot.has("ec_price_neighbours"), reason="neighbour snapshot not pinned"
)
def test_coupling_block_emits_only_lagged_columns(wide, hourly_index):
    """No D0 column may exist for a coupled-auction source."""
    out = px.coupling_block(wide, hourly_index)
    if out.empty:
        pytest.skip("no neighbour overlap for this synthetic window")
    d0 = [c for c in out.columns if "_D0_" in c]
    assert d0 == [], f"same-day neighbour price columns: {d0[:5]}"
    assert all("_D-" in c for c in out.columns)


@pytest.mark.skipif(
    not px.snapshot.has("ec_public_power"), reason="generation snapshot not pinned"
)
def test_merit_order_block_emits_only_lagged_columns(wide, hourly_index):
    out = px.merit_order_explicit_block(wide, hourly_index)
    if out.empty:
        pytest.skip("no generation overlap for this synthetic window")
    assert [c for c in out.columns if "_D0_" in c] == []


@pytest.mark.skipif(
    not px.snapshot.has("eex_eua_auctions"), reason="EUA snapshot not pinned"
)
def test_carbon_uses_only_auctions_strictly_before_the_target_day(wide):
    """Property test on the shift/ffill ORDER.

    The carbon series is forward-filled because auctions do not run daily.
    Filling BEFORE shifting would let the target day's own auction reach
    itself; shifting first is what makes it the last auction strictly
    before. Recomputed here from the raw snapshot rather than restated.
    """
    from src.data.sources import snapshot

    out = px.carbon_block(wide)
    vals = out["carbon_cost_gas_D-1"].dropna()
    if vals.empty:
        pytest.skip("no overlapping carbon data")

    eua = snapshot.load_series("eex_eua_auctions")
    daily = eua.groupby(eua.index.normalize())["eua_price_eur_t"].mean()

    day = vals.index[len(vals) // 2]
    implied = vals.loc[day] / px.EF_GAS_T_PER_MWH_EL
    strictly_before = daily[daily.index < day]
    assert not strictly_before.empty
    # The value used must be attainable from auctions strictly before `day`;
    # it must NOT equal an auction held ON that day when that differs.
    same_day = daily[daily.index == day]
    if len(same_day) and not np.isclose(same_day.iloc[0], strictly_before.iloc[-1]):
        assert not np.isclose(implied, same_day.iloc[0]), (
            "carbon feature used the target day's own auction"
        )


# ----------------------------------------------------------- naming


@pytest.mark.skipif(
    not px.snapshot.has("ec_installed_power"), reason="capacity snapshot not pinned"
)
def test_reserve_margin_columns_declare_they_exclude_outages(wide):
    """The denominator is INSTALLED capacity, not capacity net of outages --
    the true margin needs the ENTSO-E outage feed. On a day with a large
    outage this margin is optimistic, so the name has to say so."""
    out = px.reserve_margin_block(wide)
    if out.empty:
        pytest.skip("capacity years do not overlap this synthetic window")
    assert all("nooutage" in c for c in out.columns), list(out.columns)[:4]


@pytest.mark.skipif(
    not px.snapshot.has("ec_public_power"), reason="generation snapshot not pinned"
)
def test_fuel_switch_columns_are_named_as_a_proxy(wide, hourly_index):
    """It is a dispatch split, not a clean spread; the fuel prices that would
    make it a true spread are Montel-licensed and absent."""
    out = px.fuel_switch_proxy_block(wide, hourly_index)
    if out.empty:
        pytest.skip("no generation overlap")
    assert all(c.startswith("switch_proxy") for c in out.columns)
