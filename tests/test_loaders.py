"""Loader contract tests.

Offline tests always run; network-dependent tests are marked and can be
skipped with:  pytest -m "not network"
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import BenchmarkLoader, EnergyChartsLoader, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["benchmark"]["dataset"] in {"DE", "FR", "BE", "NP", "PJM"}
    assert "bzn" in cfg["live"]
    assert cfg["random_seed"] == 42


def test_benchmark_standardize_schema():
    raw = pd.DataFrame(
        {"Price": [50.0, 55.0], "Exogenous 1": [1.0, 2.0], "Exogenous 2": [3.0, 4.0]},
        index=pd.date_range("2016-01-01", periods=2, freq="1h"),
    )
    out = BenchmarkLoader._standardize(raw)
    assert list(out.columns) == ["price", "exog_1", "exog_2"]
    assert out.index.name == "timestamp"


@pytest.mark.network
def test_benchmark_download():
    cfg = load_config()
    train, test = BenchmarkLoader(cfg).load()
    for df in (train, test):
        assert "price" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
    assert len(train) > len(test)


def test_energycharts_load_and_renewables_always_request_day_ahead(monkeypatch):
    """Safety-critical invariant: exog_1/exog_2 must stay forecast-sourced,
    never realized/actuals — a leakage regression if this ever drifts."""
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)
    seen_params = []

    def fake_get(endpoint, params):
        seen_params.append(dict(params))
        n = 4
        return {
            "unix_seconds": list(range(0, n * 3600, 3600)),
            "forecast_values": [1.0] * n,
        }

    monkeypatch.setattr(loader, "_get", fake_get)
    loader.fetch_load("2026-06-01", "2026-06-01")
    loader.fetch_renewables("2026-06-01", "2026-06-01")

    assert seen_params, "no requests captured"
    for params in seen_params:
        assert params["forecast_type"] == "day-ahead"


def test_energycharts_renewables_missing_component_is_nan(monkeypatch):
    """A gap in one renewable component must surface as NaN, not silently sum as 0."""
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)

    def fake_get(endpoint, params):
        if params["production_type"] == "wind_offshore":
            return {"unix_seconds": [0, 3600], "forecast_values": [None, 5.0]}
        return {"unix_seconds": [0, 3600], "forecast_values": [1.0, 2.0]}

    monkeypatch.setattr(loader, "_get", fake_get)
    df = loader.fetch_renewables("2026-06-01", "2026-06-01")
    assert pd.isna(df["renewables"].iloc[0])


@pytest.mark.network
def test_energycharts_fetch_week():
    cfg = load_config()
    df = EnergyChartsLoader(cfg).fetch_prices("2026-05-01", "2026-05-07")
    assert "price" in df.columns
    assert df.index.tz is not None
    # hourly after resampling: ~24 rows/day
    assert 100 <= len(df) <= 200


@pytest.mark.network
def test_energycharts_fetch_load_month():
    """Load endpoint: JSON parsing + 15-min -> hourly resampling on a sample month."""
    cfg = load_config()
    df = EnergyChartsLoader(cfg).fetch_load("2026-06-01", "2026-07-01")
    assert list(df.columns) == ["load"]
    assert df.index.tz is not None
    assert df.index.name == "timestamp"
    # ~24 rows/day over ~30 days
    assert 600 <= len(df) <= 800
    assert df["load"].notna().all()


@pytest.mark.network
def test_energycharts_fetch_renewables_month():
    """Renewables endpoint: solar+wind sum, JSON parsing + resampling on a sample month."""
    cfg = load_config()
    df = EnergyChartsLoader(cfg).fetch_renewables("2026-06-01", "2026-07-01")
    assert list(df.columns) == ["renewables"]
    assert df.index.tz is not None
    assert 600 <= len(df) <= 800
    assert (df["renewables"] >= 0).all()


@pytest.mark.network
def test_energycharts_fetch_exog_schema_matches_benchmark():
    """Combined price+load+renewables must match BenchmarkLoader's column schema."""
    cfg = load_config()
    df = EnergyChartsLoader(cfg).fetch_exog("2026-06-01", "2026-06-08")
    assert list(df.columns) == ["price", "exog_1", "exog_2"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert not df.empty
