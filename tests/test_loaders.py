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


@pytest.mark.network
def test_energycharts_fetch_week():
    cfg = load_config()
    df = EnergyChartsLoader(cfg).fetch_prices("2026-05-01", "2026-05-07")
    assert "price" in df.columns
    assert df.index.tz is not None
    # hourly after resampling: ~24 rows/day
    assert 100 <= len(df) <= 200
