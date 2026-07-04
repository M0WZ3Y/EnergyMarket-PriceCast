"""Data loaders for the EPF thesis pipeline.

Two interchangeable loaders behind one interface:

- BenchmarkLoader  -> epftoolbox open benchmark datasets (all thesis results).
                      Keyless auto-download on first call, cached locally.
- EnergyChartsLoader -> Energy-Charts REST API by Fraunhofer ISE (live data
                      for the PriceCast tool). Keyless, CC BY 4.0 attribution.

Both emit the same standardized schema:
    DatetimeIndex (hourly, tz-aware where available) with columns
    ['price', 'exog_1', 'exog_2', ...]

Usage:
    from src.data.loader import BenchmarkLoader, load_config
    cfg = load_config()
    train, test = BenchmarkLoader(cfg).load()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class BenchmarkLoader:
    """Loads the Lago et al. open benchmark datasets via epftoolbox.

    First call downloads the CSV from the epftoolbox server and caches it
    under cfg['benchmark']['raw_path']; later calls read from cache.
    No API key or registration required.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["benchmark"]
        self.raw_path = REPO_ROOT / self.cfg["raw_path"]
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (df_train, df_test) in the standardized schema."""
        # Imported here so the rest of the package works without epftoolbox
        from epftoolbox.data import read_data

        df_train, df_test = read_data(
            dataset=self.cfg["dataset"],
            path=str(self.raw_path),
            years_test=self.cfg["years_test"],
        )
        return self._standardize(df_train), self._standardize(df_test)

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """epftoolbox names columns 'Price', 'Exogenous 1', ... -> our schema."""
        rename = {}
        for col in df.columns:
            if col.lower() == "price":
                rename[col] = "price"
            elif col.lower().startswith("exogenous"):
                rename[col] = f"exog_{col.split()[-1]}"
        out = df.rename(columns=rename)
        out.index.name = "timestamp"
        return out


class EnergyChartsLoader:
    """Live day-ahead prices from the Energy-Charts API (Fraunhofer ISE).

    Keyless REST API. License CC BY 4.0 — attribute Energy-Charts.info.
    Recent data is 15-minute resolution (European 15-min settlement);
    we resample to hourly means to match the pipeline schema.

    Week-3 TODO: add load / renewable-generation fetching. Endpoint names
    and parameters should be taken from the interactive API docs at
    https://api.energy-charts.info/ (Swagger UI) rather than guessed.
    The `_get` helper below works for any of them.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["live"]
        self.base_url = self.cfg["base_url"].rstrip("/")

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info("GET %s params=%s", url, params)
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_prices(self, start: str, end: str) -> pd.DataFrame:
        """Day-ahead prices for cfg['bzn'] between start/end (YYYY-MM-DD).

        Returns a DataFrame with hourly DatetimeIndex (UTC) and column
        'price' in EUR/MWh.
        """
        data = self._get(
            "price", {"bzn": self.cfg["bzn"], "start": start, "end": end}
        )
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    data["unix_seconds"], unit="s", utc=True
                ),
                "price": data["price"],
            }
        ).set_index("timestamp")

        # Drop trailing nulls (future slots not yet published)
        df = df.dropna(subset=["price"])

        if self.cfg.get("resample_to_hourly", True):
            df = df.resample("1h").mean()

        df.index.name = "timestamp"
        return df

    @property
    def attribution(self) -> str:
        return self.cfg["attribution"]
