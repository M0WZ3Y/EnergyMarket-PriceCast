"""Wave B regression reproductions — loader and feature-pipeline defects.

Companion to test_regressions_src.py; see that module's docstring for the
freeze note. None of these defects touched the v1.0-results numbers: the
benchmark data is tz-naive with exactly 24 hours on every one of its 2184
days, and the live OOD cache is UTC, so neither the DST path nor the
quarter-hour resample path was ever exercised by the frozen run.

They matter for what comes next: the France stretch goal (`dataset='FR'`)
and any live-data work both go through exactly this code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import EnergyChartsLoader, load_config  # noqa: E402
from src.features.pipeline import _pivot_to_daily_wide  # noqa: E402


# --------------------------------------------------------------------------
# src/data/loader.py
# --------------------------------------------------------------------------
def test_resample_keeps_an_interior_quarter_hour_gap_visible(monkeypatch):
    """fetch_prices goes to real trouble to preserve interior NaNs, then
    hands the frame to _resample, which destroys them one line later.

    `.resample("1h").mean()` skips NaN by default, so a quarter-hour gap
    inside an hour is absorbed into the mean of the surviving quarters:
    [1, 2, NaN, 4] became 2.333 instead of NaN. The documented invariant
    ("an interior NaN is a real API gap and must stay visible, not be
    silently dropped -- dropping it would misalign downstream lag
    features") was false in the price path. The renewables path got this
    right with min_count; the price path did not.
    """
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)

    def fake_get(endpoint, params):
        # Two hours at 15-minute resolution; hour 0 has an interior gap.
        return {
            "unix_seconds": list(range(0, 8 * 900, 900)),
            "price": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
        }

    monkeypatch.setattr(loader, "_get", fake_get)
    out = loader.fetch_prices("2026-06-01", "2026-06-01")

    assert pd.isna(out["price"].iloc[0]), (
        f"interior gap was silently repaired to {out['price'].iloc[0]!r} "
        "instead of staying NaN"
    )
    # The complete hour must still aggregate normally.
    assert out["price"].iloc[1] == pytest.approx(6.5)


def test_resample_leaves_complete_hours_untouched(monkeypatch):
    """Guard: the min_count fix must not turn healthy data into NaN."""
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)

    def fake_get(endpoint, params):
        return {
            "unix_seconds": list(range(0, 4 * 900, 900)),
            "price": [1.0, 2.0, 3.0, 4.0],
        }

    monkeypatch.setattr(loader, "_get", fake_get)
    out = loader.fetch_prices("2026-06-01", "2026-06-01")
    assert out["price"].iloc[0] == pytest.approx(2.5)


def test_fetch_exog_warns_when_the_inner_join_drops_hours(monkeypatch, caplog):
    """fetch_exog inner-joins three endpoints and silently returns a
    shorter window than requested.

    A caller asking for 7 days can get 6.5 back with no indication, then
    build lag features off the shortened index. Trimming a ragged tail is
    legitimate -- fetch_prices deliberately cuts unpublished nulls -- so
    the defect is the SILENCE, not the trim. The loss must be reported.
    """
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)

    def fake_get(endpoint, params):
        if endpoint == "price":
            return {
                "unix_seconds": list(range(0, 4 * 3600, 3600)),
                "price": [10.0, 11.0, 12.0, 13.0],
            }
        # load/renewables are one hour short of the price series
        return {
            "unix_seconds": list(range(0, 3 * 3600, 3600)),
            "forecast_values": [1.0, 2.0, 3.0],
        }

    monkeypatch.setattr(loader, "_get", fake_get)
    with caplog.at_level(logging.WARNING):
        out = loader.fetch_exog("2026-06-01", "2026-06-01")

    assert len(out) == 3
    assert list(out.columns) == ["price", "exog_1", "exog_2"]
    assert any("drop" in r.message.lower() or "align" in r.message.lower()
               for r in caplog.records), (
        "inner join silently discarded an hour with no warning logged"
    )


# --------------------------------------------------------------------------
# src/features/pipeline.py
# --------------------------------------------------------------------------
def test_pivot_refuses_to_silently_drop_the_repeated_dst_hour():
    """On the October 'fall back' day Germany has 25 hours and local hour
    02 occurs twice. `groupby([day, hour]).first()` kept the first and
    silently DISCARDED the second real hourly price.

    There was no NaN and no error: the day looked like a perfectly normal
    24-hour day, survived the `valid` mask, and flowed into X, Y,
    daily_target and every metric downstream. A dropped price must fail
    loudly, not vanish.
    """
    idx = pd.date_range("2019-10-25", "2019-10-29 23:00", freq="h", tz="Europe/Berlin")
    df = pd.DataFrame({"price": np.arange(float(len(idx)))}, index=idx)
    assert len(idx) == 121, "fixture must span the 25-hour DST day"

    with pytest.raises(ValueError, match="(?i)(duplicate|dst|repeated)"):
        _pivot_to_daily_wide(df)


def test_pivot_still_accepts_ordinary_tz_naive_days():
    """Guard: the DST check must not disturb the benchmark path, which is
    tz-naive with exactly 24 hours per day -- the shape that produced
    every frozen number.
    """
    idx = pd.date_range("2019-01-01", periods=72, freq="h")
    df = pd.DataFrame({"price": np.arange(72.0)}, index=idx)

    wide = _pivot_to_daily_wide(df)

    assert len(wide) == 3
    assert wide.loc[pd.Timestamp("2019-01-02"), "price_h00"] == 24.0
    assert wide.isna().sum().sum() == 0
