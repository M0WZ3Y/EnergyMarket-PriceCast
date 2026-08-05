"""PriceCast forecast service — app/forecast_service.py (thesis 5-3).

Everything here runs against the service layer, which imports no Streamlit:
the UI is a thin shell over these functions precisely so the logic is testable
without a browser or a Streamlit runtime.

The load-bearing test is `test_the_price_placeholder_cannot_reach_the_forecast`.
Forecasting a day whose prices are not yet published requires a placeholder,
because `build_features()` drops any day whose own 24 prices contain NaN. That
placeholder is only legitimate because no feature column ever reads the target
day's own price (src/features/pipeline.py:23) — so the test proves it, by
forecasting the same day twice with wildly different placeholders and demanding
identical output. If a future feature ever did read the target day's price,
this test fails loudly instead of the app quietly forecasting from a constant.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conftest import require_thesis_data

from app.forecast_service import (
    DEMO_CACHE_PATH,
    MIN_HISTORY_DAYS,
    OOD_SUMMARY_PATH,
    ForecastResult,
    InsufficientHistory,
    attribution,
    fetch_live_window,
    forecast_for_day,
    forecastable_days,
    history_window,
    load_cached_demo,
    load_model,
    ood_context,
    validate_uploaded_frame,
)
from src.features.pipeline import build_features
from src.models.lgbm import LightGBMModel

LIGHT_CFG = {"params": {"n_estimators": 15, "num_leaves": 7}}


def _hourly(n_days: int = 40, start: str = "2026-01-01", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    price = 40 + 12 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4, size=len(idx))
    return pd.DataFrame(
        {
            "price": price,
            "exog_1": price + rng.normal(0, 2, size=len(idx)),
            "exog_2": rng.normal(100, 10, size=len(idx)),
        },
        index=idx,
    ).rename_axis("timestamp")


@pytest.fixture(scope="module")
def model():
    df = _hourly(60)
    X, Y = build_features(df)
    m = LightGBMModel(LIGHT_CFG)
    m.fit(X, Y)
    return m


# ---------------------------------------------------------------------------
# History window
# ---------------------------------------------------------------------------


def test_history_window_covers_the_longest_lag():
    """price_lag_days maxes at 7, plus the target day itself: 8 days minimum.
    A shorter request produces an empty feature frame and a confusing error
    somewhere much deeper."""
    start, end = history_window(pd.Timestamp("2026-06-10"))
    assert (pd.Timestamp(end) - pd.Timestamp(start)).days >= MIN_HISTORY_DAYS
    assert MIN_HISTORY_DAYS >= 8


def test_history_window_ends_on_the_target_day():
    """The target day's own exog_*_D0 columns are required features (they are
    day-ahead forecasts, legal before the origin), so the window must include
    the target day, not stop the day before."""
    target = pd.Timestamp("2026-06-10")
    _, end = history_window(target)
    assert pd.Timestamp(end).normalize() >= target


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------


def test_forecast_returns_all_24_hours(model):
    df = _hourly(40)
    target = df.index.normalize().unique()[20]

    result = forecast_for_day(df, target, model)

    assert isinstance(result, ForecastResult)
    assert list(result.forecast.index) == [f"h{h:02d}" for h in range(24)]
    assert result.forecast.notna().all()
    assert result.target_day == target


def test_forecast_reports_actuals_when_the_day_is_published(model):
    df = _hourly(40)
    target = df.index.normalize().unique()[20]

    result = forecast_for_day(df, target, model)

    assert result.actual is not None
    assert len(result.actual) == 24
    expected = df.loc[df.index.normalize() == target, "price"].to_numpy()
    assert np.allclose(result.actual.to_numpy(), expected)


def test_forecast_works_when_the_target_days_prices_are_not_published(model):
    """The real live case: tomorrow's prices do not exist yet, but tomorrow's
    day-ahead load/renewables forecasts do."""
    df = _hourly(40)
    target = df.index.normalize().unique()[20]
    unpublished = df.copy()
    unpublished.loc[unpublished.index.normalize() == target, "price"] = np.nan

    result = forecast_for_day(unpublished, target, model)

    assert result.forecast.notna().all()
    assert result.actual is None, "actuals must be reported absent, not filled in"


def test_the_price_placeholder_cannot_reach_the_forecast(model):
    """Two absurdly different placeholders must give identical forecasts.

    This is what makes forecasting an unpublished day legitimate rather than a
    hack: X provably never reads the target day's own price.
    """
    df = _hourly(40)
    target = df.index.normalize().unique()[20]
    blank = df.copy()
    blank.loc[blank.index.normalize() == target, "price"] = np.nan

    low = forecast_for_day(blank, target, model, price_placeholder=-500.0)
    high = forecast_for_day(blank, target, model, price_placeholder=5000.0)

    pd.testing.assert_series_equal(low.forecast, high.forecast)


def test_forecast_refuses_a_day_without_enough_history(model):
    df = _hourly(40)
    too_early = df.index.normalize().unique()[2]

    with pytest.raises(InsufficientHistory) as exc:
        forecast_for_day(df, too_early, model)

    # The message must say what is missing, not just that something is.
    assert "history" in str(exc.value).lower()
    assert str(MIN_HISTORY_DAYS) in str(exc.value)


def test_forecast_refuses_a_target_day_outside_the_supplied_data(model):
    df = _hourly(40)
    with pytest.raises(InsufficientHistory):
        forecast_for_day(df, pd.Timestamp("2030-01-01", tz="UTC"), model)


def test_forecast_refuses_a_day_with_a_missing_hour(model):
    """A 23-hour day silently becomes NaN features and is dropped. The user
    must be told the data is incomplete, not shown a generic failure."""
    df = _hourly(40)
    target = df.index.normalize().unique()[20]
    holed = df.drop(df.index[(df.index.normalize() == target) & (df.index.hour == 5)])

    with pytest.raises(InsufficientHistory):
        forecast_for_day(holed, target, model)


# ---------------------------------------------------------------------------
# CSV upload validation
# ---------------------------------------------------------------------------


def _csv(frame: pd.DataFrame) -> io.StringIO:
    buf = io.StringIO()
    frame.to_csv(buf)
    buf.seek(0)
    return buf


def test_upload_accepts_the_shared_schema():
    df = _hourly(12)
    out = validate_uploaded_frame(_csv(df))

    assert list(out.columns) == ["price", "exog_1", "exog_2"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == len(df)


@pytest.mark.parametrize("missing", ["price", "exog_1", "exog_2"])
def test_upload_rejects_a_missing_schema_column(missing):
    df = _hourly(12).drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        validate_uploaded_frame(_csv(df))


def test_upload_rejects_duplicate_timestamps():
    """build_features raises on duplicate (day, hour) pairs because pivoting
    would silently discard a real price. Caught at the door with a message a
    user can act on."""
    df = _hourly(12)
    doubled = pd.concat([df, df.iloc[[5]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        validate_uploaded_frame(_csv(doubled))


def test_upload_rejects_a_non_hourly_index():
    df = _hourly(12).iloc[::3]
    with pytest.raises(ValueError, match="hourly"):
        validate_uploaded_frame(_csv(df))


def test_upload_rejects_a_frame_with_no_parseable_timestamp():
    bad = pd.DataFrame({"price": [1.0, 2.0], "exog_1": [1.0, 2.0], "exog_2": [1.0, 2.0]})
    with pytest.raises(ValueError):
        validate_uploaded_frame(_csv(bad))


def test_upload_rejects_non_numeric_prices():
    """Note the value: pandas parses 'n/a', 'NA', '-' and friends as NaN on
    read, so those would arrive as a numeric column with holes rather than as
    a type error. Only a genuinely unparseable token exercises this branch."""
    df = _hourly(12).astype({"price": object})
    df.iloc[3, df.columns.get_loc("price")] = "forty two"
    with pytest.raises(ValueError, match="numeric"):
        validate_uploaded_frame(_csv(df))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_missing_model_names_the_command_that_creates_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_ood_stress"):
        load_model(tmp_path / "absent")


# ---------------------------------------------------------------------------
# Selectable days
# ---------------------------------------------------------------------------


def test_forecastable_days_exclude_the_unbacked_head_of_the_series():
    """The first 7 days can never be forecast — they lack the D-7 lag. Offering
    them in the date picker would guarantee an error the user cannot fix."""
    df = _hourly(40)
    days = forecastable_days(df)

    all_days = df.index.normalize().unique()
    assert list(days) == list(all_days[MIN_HISTORY_DAYS - 1 :])
    assert len(days) == 40 - (MIN_HISTORY_DAYS - 1)


def test_forecastable_days_agree_with_what_forecasting_actually_accepts(model):
    """The picker's offer and the forecaster's behaviour must not disagree:
    every offered day must forecast, and the day just before the first offered
    day must not."""
    df = _hourly(20)
    days = forecastable_days(df)

    for day in list(days)[:3]:
        forecast_for_day(df, day, model)

    just_before = days[0] - pd.Timedelta(days=1)
    with pytest.raises(InsufficientHistory):
        forecast_for_day(df, just_before, model)


def test_forecastable_days_skip_a_day_with_a_hole_and_the_week_after_it():
    """A missing hour poisons that day and every day whose lag window contains
    it — the picker must not offer any of them."""
    df = _hourly(40)
    victim = df.index.normalize().unique()[15]
    holed = df.drop(df.index[(df.index.normalize() == victim) & (df.index.hour == 5)])

    days = forecastable_days(holed)

    assert victim not in days
    for offset in range(1, MIN_HISTORY_DAYS):
        assert victim + pd.Timedelta(days=offset) not in days


def test_forecastable_days_tolerate_an_unpublished_price_on_the_target_day():
    """The live case: tomorrow has exogenous forecasts but no prices yet, and
    must still be offered."""
    df = _hourly(40)
    last = df.index.normalize().unique()[-1]
    df.loc[df.index.normalize() == last, "price"] = np.nan

    assert last in forecastable_days(df)


def test_forecastable_days_require_the_exogenous_columns():
    """Unlike price, a missing exog on the target day is fatal — exog_*_D0 are
    required features."""
    df = _hourly(40)
    last = df.index.normalize().unique()[-1]
    df.loc[df.index.normalize() == last, "exog_1"] = np.nan

    assert last not in forecastable_days(df)


# ---------------------------------------------------------------------------
# Cached demo + attribution
# ---------------------------------------------------------------------------


def test_the_committed_demo_cache_loads_in_the_shared_schema():
    require_thesis_data(DEMO_CACHE_PATH, "live OOD demo cache")
    df = load_cached_demo()

    assert list(df.columns) == ["price", "exog_1", "exog_2"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(forecastable_days(df)) > 0, "the demo cache must offer forecastable days"


def test_ood_context_reads_the_frozen_numbers_rather_than_hardcoding_them():
    """The app's honesty banner quotes rMAE figures. They must come from the
    v1.1-ood artifact — a hand-typed statistic in a committed caption is an
    error this project already made once."""
    require_thesis_data(OOD_SUMMARY_PATH, "OOD summary")
    ctx = ood_context()

    assert ctx is not None
    assert ctx["model_rmae"] > 1.0, "the served model is worse than naive on 2026 data"
    assert ctx["naive_rmae"] < 1.0
    frozen = pd.read_csv(OOD_SUMMARY_PATH, index_col=0)
    assert ctx["model_rmae"] == pytest.approx(frozen.loc["LightGBM", "rMAE"])


def test_ood_context_degrades_instead_of_crashing_when_absent(tmp_path):
    from app.forecast_service import ood_context

    assert ood_context(tmp_path / "nope.csv") is None


def test_attribution_is_the_licence_required_string():
    """CC BY 4.0 requires it wherever the live data appears; the app renders
    it on every view. Sourced from the config, never retyped."""
    text = attribution()
    assert "Energy-Charts" in text
    assert "CC BY 4.0" in text


# ---------------------------------------------------------------------------
# Live path — this test IS the week-11 data-source check
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_the_full_live_path_produces_a_forecast():
    """decisions.md week-11 row: date picker -> API fetch -> forecast -> chart.

    A failure here is the Energy-Charts API (it drops TLS intermittently), not
    this code — the loader already retries connection errors.
    """
    target = pd.Timestamp.now('UTC').normalize().tz_localize(None) - pd.Timedelta(days=3)
    start, end = history_window(target)

    df = fetch_live_window(start, end)
    assert list(df.columns) == ["price", "exog_1", "exog_2"]

    days = forecastable_days(df)
    assert len(days) > 0, f"no forecastable day in {start}..{end}"

    model = load_model()
    result = forecast_for_day(df, days[-1], model)
    assert len(result.forecast) == 24
    assert result.forecast.notna().all()


def test_loaded_model_predicts_what_the_fitted_one_did(tmp_path, model):
    """Round-trip: the app must serve the same forecasts the fitted model
    produced, or the tool contradicts the thesis it demonstrates."""
    path = tmp_path / "lightgbm"
    model.save(path)

    restored = load_model(path)

    df = _hourly(40)
    target = df.index.normalize().unique()[20]
    pd.testing.assert_series_equal(
        forecast_for_day(df, target, model).forecast,
        forecast_for_day(df, target, restored).forecast,
    )
