"""The declarative information-set guard.

`tests/test_features.py` checks leakage empirically, by perturbing inputs and
watching what moves. That works while every input is a local column. Once
features come from external series with different publication times, the
question stops being arithmetic and becomes "when does this source publish?"
-- which no amount of perturbing a DataFrame can answer. So the timing is
declared, and these tests check the declaration is enforced.

The two traps under test are the ones named in the task brief, and both are
counter-intuitive enough to be re-introduced by someone acting reasonably:

  1. Neighbour day-ahead prices look like ordinary exogenous inputs, but they
     clear in the SAME coupled EUPHEMIA auction as DE-LU and are published
     ~12:42 CET -- after the 12:00 gate closure. Same-day use is a leak.
  2. Realized flows and outages for the delivery day are not known at the
     origin, however "historical" the series feels.
"""

from __future__ import annotations

import pytest

from src.features import leakage_guard as lg


@pytest.fixture(autouse=True)
def clean_registry():
    lg.clear_registry()
    yield
    lg.clear_registry()


def test_ex_ante_forecast_is_legal_on_the_target_day():
    """exog_1/exog_2 are day-ahead forecasts published before gate closure --
    which is exactly why the frozen pipeline already licenses exog_*_D0_h*."""
    lg.declare("exog_1_D0_h12", source="exog_1", lag_days=0)
    assert lg.assert_no_leakage(strict=False) == []


def test_structural_data_is_legal_on_the_target_day():
    lg.declare("disp_capacity_D0", source="ec_installed_power", lag_days=0)
    assert lg.assert_no_leakage(strict=False) == []


def test_same_day_neighbour_price_is_rejected():
    """Trap 1. The coupled auction publishes DE-LU and its neighbours
    together, after the origin."""
    lg.declare("nbpriceFR_D0_h12", source="ec_price_neighbours", lag_days=0)
    with pytest.raises(lg.LeakageError, match="coupled_auction"):
        lg.assert_no_leakage(strict=False)


def test_lagged_neighbour_price_is_accepted():
    lg.declare("nbpriceFR_D-1_h12", source="ec_price_neighbours", lag_days=1)
    assert lg.assert_no_leakage(strict=False) == []


def test_same_day_realized_generation_is_rejected():
    """Trap 2. The delivery day's dispatch is unknown at the origin."""
    lg.declare("gasshare_D0_h12", source="ec_public_power", lag_days=0)
    with pytest.raises(lg.LeakageError, match="realized"):
        lg.assert_no_leakage(strict=False)


def test_same_day_realized_flow_is_rejected():
    lg.declare("netimport_D0_h12", source="ec_cross_border", lag_days=0)
    with pytest.raises(lg.LeakageError, match="realized"):
        lg.assert_no_leakage(strict=False)


def test_target_day_price_is_rejected():
    """The label itself. Legal only as a lag, which is what price_D-1 is."""
    lg.declare("price_D0_h12", source="price", lag_days=0)
    with pytest.raises(lg.LeakageError):
        lg.assert_no_leakage(strict=False)


def test_carbon_is_treated_conservatively_as_lag_one():
    """EUA auctions clear ~11:00 CET, an hour BEFORE the 12:00 gate closure,
    so same-day carbon is arguably legal. It is deliberately classified as
    lag >= 1 anyway -- auctions do not run daily so a fill is needed
    regardless, and a leakage claim should not rest on a one-hour margin."""
    lg.declare("carbon_cost_gas_D0", source="eex_eua_auctions", lag_days=0)
    with pytest.raises(lg.LeakageError, match="pre_gate_settlement"):
        lg.assert_no_leakage(strict=False)


def test_unknown_source_is_rejected_rather_than_assumed_safe():
    """An undeclared source cannot be checked, and an unchecked source is how
    a leak gets in."""
    lg.declare("mystery_D0_h00", source="some_new_feed", lag_days=0)
    with pytest.raises(KeyError, match="unknown source"):
        lg.assert_no_leakage(strict=False)


def test_undeclared_column_fails_under_strict_mode():
    """The realistic failure mode: not someone declaring an illegal lag, but
    someone adding a column and never declaring it at all."""
    lg.declare("exog_1_D0_h12", source="exog_1", lag_days=0)
    with pytest.raises(lg.LeakageError, match="UNDECLARED"):
        lg.assert_no_leakage(columns=["exog_1_D0_h12", "brand_new_col_D0_h00"])


def test_infer_lag_reads_the_naming_convention():
    assert lg.infer_lag("price_D-7_h03") == 7
    assert lg.infer_lag("resload_D0_h03") == 0
    assert lg.infer_lag("resload_ramp_max_D0") == 0
    assert lg.infer_lag("dow_3") is None, "unrecognised names must not be guessed as 0"


def test_every_source_in_timing_table_has_a_min_lag():
    for src, av in lg.SOURCE_TIMING.items():
        assert av in lg.MIN_LAG_DAYS, f"{src}: {av} has no minimum lag defined"


def test_real_pipeline_with_all_blocks_passes_the_guard():
    """End-to-end: build the full feature set and assert every column it
    declares is inside the pre-gate-closure information set."""
    pd = pytest.importorskip("pandas")
    from src.data.sources import snapshot
    from src.features.pipeline import build_features, load_feature_config

    if not snapshot.has("ec_public_power"):
        pytest.skip("physical snapshot not pinned in this checkout")

    from src.data.loader import BenchmarkLoader, load_config

    tr, te = BenchmarkLoader(load_config()).load()
    df = pd.concat([tr, te])

    cfg = dict(load_feature_config())
    cfg["exog_blocks"] = {
        "merit_order_explicit_block": True,
        "capacity_structure_block": True,
        "carbon_block": True,
        "fuel_switch_proxy_block": True,
        "coupling_block": True,
        "cross_border_flow_block": True,
        "reserve_margin_block": True,
        "storage_block": True,
    }
    lg.clear_registry()
    X, _ = build_features(df, cfg)
    assert len(lg.registry()) > 0, "no columns declared -- the test is vacuous"
    assert lg.assert_no_leakage(strict=False) == []
