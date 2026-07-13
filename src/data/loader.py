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
import time
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
    """Live day-ahead prices, load and generation from the Energy-Charts API
    (Fraunhofer ISE).

    Keyless REST API. License CC BY 4.0 — attribute Energy-Charts.info.
    Recent data is 15-minute resolution (European 15-min settlement);
    we resample to hourly means to match the pipeline schema.

    Endpoint parameter naming is NOT uniform: `/price` takes `bzn` (bidding
    zone, e.g. 'DE-LU') while `/public_power_forecast` takes `country`
    (e.g. 'de') — confirmed against the live openapi.json spec at
    https://api.energy-charts.info/openapi.json, not guessed. Both are kept
    in configs/data.yaml under `live.bzn` / `live.country` and must be kept
    in sync manually if the market changes (e.g. the France stretch goal).
    """

    # production_type values that sum to epftoolbox's exog_2 convention
    # (day-ahead solar + wind generation forecast).
    _RENEWABLE_TYPES = ("solar", "wind_onshore", "wind_offshore")

    def __init__(self, cfg: dict):
        self.cfg = cfg["live"]
        self.base_url = self.cfg["base_url"].rstrip("/")

    # fetch_renewables/fetch_exog fire several sequential requests per call;
    # the API 429s on bursts even well under any documented quota, so retry
    # with backoff instead of treating a burst as a hard failure.
    _MAX_RETRIES = 3

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        for attempt in range(self._MAX_RETRIES + 1):
            logger.info("GET %s params=%s", url, params)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429 and attempt < self._MAX_RETRIES:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                logger.warning("429 from %s, retrying in %.1fs", url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()

    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.get("resample_to_hourly", True):
            df = df.resample("1h").mean()
        df.index.name = "timestamp"
        return df

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

        # Trim only the trailing run of not-yet-published nulls; an interior
        # NaN is a real API gap and must stay visible, not be silently
        # dropped (dropping it would misalign downstream lag features).
        last_valid = df["price"].last_valid_index()
        df = df.loc[:last_valid] if last_valid is not None else df.iloc[0:0]
        return self._resample(df)

    def _fetch_forecast(self, production_type: str, start: str, end: str) -> pd.Series:
        """Day-ahead forecast series for one /public_power_forecast type."""
        data = self._get(
            "public_power_forecast",
            {
                "country": self.cfg["country"],
                "production_type": production_type,
                "forecast_type": "day-ahead",
                "start": start,
                "end": end,
            },
        )
        s = pd.Series(
            data["forecast_values"],
            index=pd.to_datetime(data["unix_seconds"], unit="s", utc=True),
            name=production_type,
        )
        s.index.name = "timestamp"
        return s

    def fetch_load(self, start: str, end: str) -> pd.DataFrame:
        """Day-ahead total load forecast (epftoolbox exog_1 equivalent).

        Returns a DataFrame with hourly DatetimeIndex (UTC) and column
        'load' in MW.
        """
        s = self._fetch_forecast("load", start, end).rename("load")
        return self._resample(s.to_frame())

    def fetch_renewables(self, start: str, end: str) -> pd.DataFrame:
        """Day-ahead solar + wind generation forecast (exog_2 equivalent).

        Sums solar, wind_onshore and wind_offshore day-ahead forecasts —
        the same components Lago et al.'s DE dataset uses for exog_2.
        Returns a DataFrame with hourly DatetimeIndex (UTC) and column
        'renewables' in MW.
        """
        parts = [self._fetch_forecast(t, start, end) for t in self._RENEWABLE_TYPES]
        # min_count keeps an hour NaN if any component is missing, rather
        # than sum()'s default skipna=True silently treating a gap as 0
        # and underestimating exog_2.
        combined = (
            pd.concat(parts, axis=1)
            .sum(axis=1, min_count=len(self._RENEWABLE_TYPES))
            .rename("renewables")
        )
        return self._resample(combined.to_frame())

    def fetch_exog(self, start: str, end: str) -> pd.DataFrame:
        """Price + load + renewables in the BenchmarkLoader schema.

        Returns a DataFrame with columns ['price', 'exog_1', 'exog_2'] —
        exog_1 = day-ahead load forecast, exog_2 = day-ahead renewable
        generation forecast, matching BenchmarkLoader's column naming so
        the live loader is a drop-in replacement for the feature pipeline.
        """
        price = self.fetch_prices(start, end)
        load = self.fetch_load(start, end).rename(columns={"load": "exog_1"})
        renewables = self.fetch_renewables(start, end).rename(
            columns={"renewables": "exog_2"}
        )
        out = price.join(load, how="inner").join(renewables, how="inner")
        out.index.name = "timestamp"
        return out

    @property
    def attribution(self) -> str:
        return self.cfg["attribution"]
