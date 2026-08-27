"""Regime-segmented error evaluation (stage 5).

The risk this module carries is not arithmetic, it is misreporting: a
segment that silently comes back empty reads as "this regime never
occurred" when the truth is "this regime was never measured", and a
percentage error computed across negative prices reads as a metric when it
is noise. Both are tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import regimes


@pytest.fixture
def frame() -> pd.DataFrame:
    """A long results frame in run_baselines.run_model's schema."""
    rng = np.random.default_rng(42)
    origins = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for o in origins:
        for h in range(24):
            y = float(rng.normal(40, 30))
            rows.append(
                dict(origin=o, hour=h, y_true=y, y_pred=y + rng.normal(0, 5), model="m")
            )
    return pd.DataFrame(rows)


@pytest.fixture
def wide() -> pd.DataFrame:
    from src.features.pipeline import _pivot_to_daily_wide

    idx = pd.date_range("2020-01-01", periods=24 * 40, freq="h")
    n = len(idx)
    hour = np.arange(n) % 24
    day = np.arange(n) // 24
    return _pivot_to_daily_wide(
        pd.DataFrame(
            {
                "price": np.zeros(n),
                "exog_1": 20000 + 300 * hour + 50 * day,
                "exog_2": 5000 + 400 * hour - 20 * day,
            },
            index=idx,
        )
    )


def test_price_segments_available_without_context(frame):
    masks = regimes.segment_masks(frame)
    assert set(masks) == {"all", "negative_price", "spike"}
    assert masks["all"].all()


def test_physical_segments_require_context_rather_than_returning_empty(frame, wide):
    """Without context, a physical segment must be ABSENT, not all-False.

    An all-False mask would be reported as n=0 and read as "this regime
    never occurred" -- a measurement gap disguised as a finding.
    """
    assert "steep_ramp" not in regimes.segment_masks(frame)
    with_ctx = regimes.segment_masks(frame, regimes.physical_context(wide))
    assert "steep_ramp" in with_ctx
    assert with_ctx["steep_ramp"].any()


def test_negative_price_segment_selects_exactly_the_negative_hours(frame):
    masks = regimes.segment_masks(frame)
    assert (frame.loc[masks["negative_price"], "y_true"] < 0).all()
    assert (frame.loc[~masks["negative_price"], "y_true"] >= 0).all()


def test_spike_segment_is_the_upper_tail(frame):
    masks = regimes.segment_masks(frame)
    cut = np.quantile(frame["y_true"], regimes.SPIKE_Q)
    assert (frame.loc[masks["spike"], "y_true"] >= cut).all()
    assert 0 < masks["spike"].sum() < len(frame)


def test_smape_suppressed_where_prices_go_negative(frame):
    """CLAUDE.md bans plain MAPE because prices go negative. Reporting a
    percentage error on exactly those hours would give a meaningless number
    a metric's authority."""
    m = regimes.segmented_metrics(frame)
    assert np.isnan(m.loc["negative_price", "smape"])
    assert m.loc["negative_price", "n"] > 0
    assert not np.isnan(m.loc["negative_price", "mae"])


def test_segmented_metrics_match_manual_computation_on_a_segment(frame):
    m = regimes.segmented_metrics(frame)
    sub = frame[frame["y_true"] < 0]
    expected = np.mean(np.abs(sub["y_true"] - sub["y_pred"]))
    assert m.loc["negative_price", "mae"] == pytest.approx(expected)
    assert m.loc["negative_price", "n"] == len(sub)


def test_all_segment_equals_pooled_metric(frame):
    m = regimes.segmented_metrics(frame)
    expected = np.mean(np.abs(frame["y_true"] - frame["y_pred"]))
    assert m.loc["all", "mae"] == pytest.approx(expected)
    assert m.loc["all", "n"] == len(frame)


def test_compare_reports_improvement_as_a_negative_delta(frame):
    better = frame.copy()
    better["y_pred"] = better["y_true"]  # a perfect variant
    cmp = regimes.compare_segmented(frame, better)
    assert (cmp["mae_delta"] <= 0).all()
    assert cmp.loc["all", "mae_new"] == pytest.approx(0.0)


def test_compare_does_not_mutate_the_baseline(frame):
    before = frame.copy()
    regimes.compare_segmented(frame, frame.copy())
    pd.testing.assert_frame_equal(frame, before)


def test_coupling_stress_is_declared_unavailable_not_silently_omitted():
    """The one segment the prompt asks for that this data cannot support.
    It must be named, with a reason, rather than quietly missing."""
    assert "coupling_stress" in regimes.UNAVAILABLE_SEGMENTS
    assert "ENTSO-E" in regimes.UNAVAILABLE_SEGMENTS["coupling_stress"]


def test_physical_context_residual_load_matches_the_feature_definition(wide):
    """A segment must mean exactly what the corresponding feature means,
    or the ablation is scored against a different regime than it targets."""
    from src.features import physical

    ctx = regimes.physical_context(wide)
    rl = physical.residual_load_frame(wide)
    day = wide.index[5]
    assert ctx.loc[(day, 7), "residual_load"] == pytest.approx(rl["h07"].iloc[5])
