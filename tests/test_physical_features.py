"""Physical feature blocks: correctness + the leakage guarantees.

The project's non-negotiable rule is that no feature may use information
after the forecast origin. The physical blocks add TARGET-DAY (D0) columns,
which is the highest-risk kind of column to add, so the leakage properties
are tested as properties -- computed and checked -- rather than asserted in
a comment.

Three distinct leak classes are covered:
  1. reading the target day's own PRICE (the label),
  2. a rolling/distributional reference that includes the target day itself,
  3. the default config silently acquiring a D0 column (which would change
     the frozen v1.0-results feature set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import physical
from src.features.pipeline import build_features, load_feature_config

HOURS = [f"h{h:02d}" for h in range(24)]

ALL_BLOCKS = {
    "residual_load_block": True,
    "residual_load_gradient_block": True,
    "merit_order_block": {"window": 30},
    "scarcity_block": {"window": 30},
}


@pytest.fixture
def hourly() -> pd.DataFrame:
    """A deterministic hourly frame in the loader's schema.

    Values are constructed so every derived quantity is hand-checkable:
    load has a daily ramp, renewables a distinct one, price is unrelated
    (so any dependence of a feature on price is a leak, not a coincidence).
    """
    idx = pd.date_range("2020-01-01", periods=24 * 120, freq="h")
    n = len(idx)
    day = np.arange(n) // 24
    hour = np.arange(n) % 24
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "price": rng.normal(40, 15, n),
            "exog_1": 20000 + 300 * hour + 50 * day,
            "exog_2": 5000 + 100 * hour + 10 * day,
        },
        index=idx,
    )


@pytest.fixture
def wide(hourly) -> pd.DataFrame:
    from src.features.pipeline import _pivot_to_daily_wide

    return _pivot_to_daily_wide(hourly)


# ---------------------------------------------------------------- formulas


def test_residual_load_is_load_minus_renewables(wide):
    out = physical.residual_load_block(wide)
    expected = wide["exog_1_h07"] - wide["exog_2_h07"]
    pd.testing.assert_series_equal(
        out["resload_D0_h07"], expected, check_names=False
    )


def test_residual_load_lag_is_the_previous_day(wide):
    out = physical.residual_load_block(wide, lag_days=(1,))
    assert out["resload_D-1_h07"].iloc[5] == pytest.approx(out["resload_D0_h07"].iloc[4])
    assert pd.isna(out["resload_D-1_h07"].iloc[0]), "first day has no D-1 to read"


def test_gradient_is_the_hour_to_hour_difference(wide):
    out = physical.residual_load_gradient_block(wide)
    rl = physical.residual_load_frame(wide)
    expected = rl["h07"] - rl["h06"]
    pd.testing.assert_series_equal(
        out["resload_grad_D0_h07"], expected, check_names=False
    )


def test_gradient_hour00_uses_previous_day_h23_not_zero(wide):
    """h00 has no in-day predecessor. Filling it with 0 would fabricate a
    flat hour every single day -- a constant lie in a ramp feature."""
    out = physical.residual_load_gradient_block(wide)
    rl = physical.residual_load_frame(wide)
    expected = rl["h00"].iloc[3] - rl["h23"].iloc[2]
    assert out["resload_grad_D0_h00"].iloc[3] == pytest.approx(expected)
    assert pd.isna(out["resload_grad_D0_h00"].iloc[0])
    assert (out["resload_grad_D0_h00"].dropna() != 0).any()


def test_res_share_is_renewables_over_load(wide):
    out = physical.merit_order_block(wide, window=30)
    expected = wide["exog_2_h07"] / wide["exog_1_h07"]
    pd.testing.assert_series_equal(
        out["res_share_D0_h07"], expected, check_names=False
    )


def test_scarcity_columns_are_named_as_a_proxy(wide):
    """No column may read as a measured reserve margin. The naming is the
    only thing standing between a proxy and a misread result."""
    out = physical.scarcity_block(wide, window=30)
    assert all("proxy" in c for c in out.columns), list(out.columns)


# ---------------------------------------------------------------- leakage


def test_trailing_reference_excludes_the_target_day(wide):
    """The distributional reference for merit-order position must be built
    from strictly earlier days.

    Property test: mutate ONLY day k's residual load and confirm day k's own
    reference statistics are unchanged. If the rolling window included day
    k, its z-score denominator would move.
    """
    k, window = 60, 30
    out = physical.merit_order_block(wide, window=window)
    rl = physical.residual_load_frame(wide)

    # Recompute day k's z-score by hand from the days that are ALLOWED to
    # inform it: k-window .. k-1, with day k itself excluded. Rolling before
    # shifting would instead use k-window+1 .. k, so this equality is exactly
    # the property under test, not a restatement of the implementation.
    daily_mean = rl.mean(axis=1)
    past = daily_mean.iloc[k - window : k]
    assert len(past) == window
    expected = (rl["h07"].iloc[k] - past.mean()) / past.std()

    assert np.isclose(out["merit_pos_D0_h07"].iloc[k], expected), (
        "merit-order position at day k does not match a reference built from "
        "days k-window..k-1 -- day k is leaking into its own reference window"
    )

    # Second, independent angle: perturbing day k must not move any EARLIER
    # day, since a past day's reference cannot see a future day at all.
    tampered = wide.copy()
    for h in HOURS:
        tampered.iloc[k, tampered.columns.get_loc(f"exog_1_{h}")] += 500_000
    after = physical.merit_order_block(tampered, window=window)
    pd.testing.assert_series_equal(
        out["merit_pos_D0_h07"].iloc[:k], after["merit_pos_D0_h07"].iloc[:k]
    )


def test_trailing_helper_never_includes_the_current_row():
    s = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
    out = physical._trailing(s, window=2, func="max")
    # Row 3 holds the outlier; its own reference is max(rows 1,2) = 3.
    assert out.iloc[3] == 3.0
    # Row 4's reference is max(rows 2,3) = 100 -- the outlier, now past.
    assert out.iloc[4] == 100.0


def test_no_physical_column_reads_the_target_day_price(hourly):
    """Hard leakage guard, mirroring test_features.py's price check.

    Property test: perturb ONLY the target day's price and confirm no
    physical feature column moves. A feature that shifts is reading the
    label.
    """
    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = ALL_BLOCKS

    X0, _ = build_features(hourly, cfg)

    tampered = hourly.copy()
    target_day = X0.index[40]
    mask = tampered.index.normalize() == target_day
    tampered.loc[mask, "price"] += 1_000.0

    X1, _ = build_features(tampered, cfg)

    physical_cols = [
        c
        for c in X0.columns
        if c.startswith(("resload_", "merit_pos_", "res_share_", "tightness_proxy_"))
    ]
    assert physical_cols, "no physical columns were built -- the test is vacuous"

    moved = [
        c
        for c in physical_cols
        if not np.isclose(X0.loc[target_day, c], X1.loc[target_day, c], equal_nan=True)
    ]
    assert moved == [], f"physical columns reading the target-day price: {moved}"


def test_physical_features_do_not_change_when_a_future_day_changes(hourly):
    """No feature for day D may move when day D+1's inputs change.

    This is the forecast-origin rule stated as a property. It catches the
    classic ordering bug -- rolling first and shifting after -- that a
    same-day check cannot see.
    """
    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = ALL_BLOCKS

    X0, _ = build_features(hourly, cfg)
    target_day = X0.index[40]
    future_day = X0.index[41]

    tampered = hourly.copy()
    mask = tampered.index.normalize() == future_day
    tampered.loc[mask, ["exog_1", "exog_2"]] += 500_000.0

    X1, _ = build_features(tampered, cfg)

    physical_cols = [
        c
        for c in X0.columns
        if c.startswith(("resload_", "merit_pos_", "res_share_", "tightness_proxy_"))
    ]
    moved = [
        c
        for c in physical_cols
        if not np.isclose(X0.loc[target_day, c], X1.loc[target_day, c], equal_nan=True)
    ]
    assert moved == [], f"columns for day D moved when day D+1 changed: {moved}"


# ---------------------------------------------------------------- wiring


def test_default_config_adds_no_physical_columns(hourly):
    """The frozen-results guarantee.

    v1.0-results / v1.1-ood were produced from the default config. If the
    default path ever grows a physical column, every frozen number silently
    stops describing the pipeline that exists.
    """
    X, _ = build_features(hourly)
    leaked = [
        c
        for c in X.columns
        if c.startswith(("resload_", "merit_pos_", "res_share_", "tightness_proxy_"))
    ]
    assert leaked == [], f"default config emitted physical columns: {leaked}"


def test_weekday_dummies_stay_last_when_blocks_are_enabled(hourly):
    """LEARLassoModel._assert_dow_columns_last depends on this ordering.

    The physical blocks are inserted BEFORE the dummies for exactly this
    reason; appending them after would break LEAR with a confusing error far
    from the cause.
    """
    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = ALL_BLOCKS
    X, _ = build_features(hourly, cfg)
    assert list(X.columns)[-7:] == [f"dow_{i}" for i in range(7)]


def test_unknown_block_name_raises_rather_than_being_ignored():
    """A typo in configs/features.yaml must fail loudly. Ignoring it would
    still train, still score, and quietly not be the experiment intended."""
    with pytest.raises(KeyError, match="unknown physical block"):
        physical.build_physical_blocks(pd.DataFrame(), {"resload_blok": True})


def test_disabled_blocks_produce_no_columns(wide):
    out = physical.build_physical_blocks(wide, {"residual_load_block": False})
    assert out.shape[1] == 0
    assert out.index.equals(wide.index)
