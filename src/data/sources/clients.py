"""Network clients for the physical data snapshot — src/data/sources/clients.py

THE ONLY MODULE IN THE PROJECT THAT MAY PERFORM A NETWORK CALL for feature
data. It is imported by `scripts/fetch_physical_snapshot.py` and by nothing
in `src/features/`. `tests/test_no_network_in_features.py` enforces that
separation, because "features read the pinned snapshot" is a claim that is
worthless unless something checks it.

Three clients, by access tier:

  EnergyChartsClient  Tier A. Keyless, CC BY 4.0. The hourly price,
                      generation and cross-border endpoints start
                      2015-01-01 (see EC_DATA_START), NOT 2011.
  EEXCarbonClient     Tier A. EEX publishes the EU ETS primary-auction
                      archive publicly; carbon is the one fuel-side series
                      obtainable without a licensed feed.
  EntsoeClient        Tier B. Requires ENTSOE_API_TOKEN in the environment.
                      Inert and loud when the token is absent -- it raises,
                      and dependent features skip. It never invents data.

TIMEZONE HANDLING is explicit everywhere and is not incidental. Energy-Charts
returns UTC epoch seconds; the benchmark price index is tz-NAIVE local German
time with exactly 24 hours per day. Converting UTC -> Europe/Berlin -> naive
reproduces that convention, including the two DST days per year where local
time has 23 or 25 hours. Those days are handled explicitly in
`snapshot.align_to_index` rather than being averaged away here, so the
ambiguity stays visible at the point where it matters.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from datetime import date

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LOCAL_TZ = "Europe/Berlin"

ENERGY_CHARTS_BASE = "https://api.energy-charts.info"
ENERGY_CHARTS_LICENCE = "CC BY 4.0"
ENERGY_CHARTS_ATTRIB = (
    "Data: Energy-Charts (Fraunhofer ISE) / Bundesnetzagentur SMARD.de, CC BY 4.0"
)

#: The DE bidding zone changed on 2018-10-01: before that, Germany, Austria
#: and Luxembourg formed one zone (DE-AT-LU); after, DE-LU. The benchmark
#: window (2012-2017) is entirely DE-AT-LU, and `/price?bzn=DE-LU` returns
#: HTTP 404 "no content available" for those dates. Hardcoding DE-LU would
#: therefore yield nothing for the thesis window -- silently, if the caller
#: treated 404 as "no data for this hour".
BZN_SPLIT_DATE = date(2018, 10, 1)

#: Earliest date Energy-Charts actually serves, verified by probing /price,
#: /public_power and /cbpf year by year on 2026-08-28. All three return no
#: data before this date.
#:
#: This contradicts the common claim that Energy-Charts covers "data from
#: 2011" -- that holds for some aggregate series, not for the hourly price,
#: generation and cross-border endpoints used here.
#:
#: It matters because the benchmark window starts 2012-01-09: any block built
#: on these endpoints covers only about half of it, and the pre-2015 rows go
#: NaN and are dropped. That is a real cost in training history, so it is
#: recorded here rather than discovered later as an unexplained row count.
#:
#: The failure is also disguised: requesting a pre-2015 window returns
#: HTTP 400 with the message "end must be >= start", which is untrue of the
#: request and would send anyone debugging it after the date format instead
#: of the coverage floor.
EC_DATA_START = date(2015, 1, 1)

#: Neighbours of DE-LU that clear in the coupled day-ahead auction.
NEIGHBOUR_ZONES = ("FR", "NL", "BE", "AT", "CH", "CZ", "PL", "DK1", "DK2")

EEX_EUA_ARCHIVE_URL = (
    "https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
    "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
    "emission-spot-primary-market-auction-report-2012-2025-data.zip"
)
EEX_LICENCE = "EEX public auction report (free redistribution of results)"

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
ENTSOE_TOKEN_ENV = "ENTSOE_API_TOKEN"


class TokenNotConfigured(RuntimeError):
    """Raised when a Tier-B client is used without its token.

    Deliberately loud. The alternative -- returning an empty frame -- would
    let dependent features quietly train on nothing.
    """


def de_bidding_zone(day: date) -> str:
    """The German bidding zone in force on `day`."""
    return "DE-LU" if day >= BZN_SPLIT_DATE else "DE-AT-LU"


def utc_seconds_to_local_naive(unix_seconds) -> pd.DatetimeIndex:
    """UTC epoch seconds -> tz-naive local German time.

    Matches the benchmark index convention exactly. Kept as one function so
    every series in the snapshot is converted the same way; a per-call
    reimplementation is how two series end up an hour apart.
    """
    idx = pd.to_datetime(pd.Series(unix_seconds), unit="s", utc=True)
    return pd.DatetimeIndex(idx.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None))


class _HttpClient:
    """Shared retry/backoff. The Energy-Charts host 429s on modest bursts and
    intermittently drops TLS mid-handshake, exactly as the existing
    EnergyChartsLoader documents; both are retried rather than treated as
    hard failures, since either one silently costs a chunk of a long fetch.
    """

    user_agent = "EPF-thesis-research/1.0 (academic; contact via repo)"

    def __init__(self, max_retries: int = 5, backoff: float = 4.0):
        self.max_retries = max_retries
        self.backoff = backoff

    def get(self, url: str, params: dict | None = None, timeout: int = 90):
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers={"User-Agent": self.user_agent},
                )
                if r.status_code == 429:
                    wait = self.backoff * (attempt + 1)
                    logger.warning("429 from %s; sleeping %.0fs", url, wait)
                    time.sleep(wait)
                    last = r
                    continue
                return r
            except requests.exceptions.RequestException as exc:
                last = exc
                wait = self.backoff * (attempt + 1)
                logger.warning("%s on %s; retrying in %.0fs", type(exc).__name__, url, wait)
                time.sleep(wait)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} retries: {last!r}")


def _to_hourly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a sub-hourly frame to hourly means.

    /public_power and /cbpf are served at 15-minute resolution (European
    15-min settlement) while the pipeline schema is hourly. Without this,
    aligning onto the hourly index silently keeps only the value stamped
    :00 -- an instantaneous reading standing in for an hourly quantity,
    which is wrong in a way nothing downstream could detect: the column
    would look complete, plausible and correctly dated.

    A frame already at hourly-or-coarser resolution is returned unchanged,
    so this is safe to apply unconditionally.
    """
    if len(df) < 2:
        return df
    step = pd.Series(df.index).diff().dropna().min()
    if step >= pd.Timedelta(hours=1):
        return df
    return df.resample("h").mean()


class EnergyChartsClient(_HttpClient):
    """Tier A. Keyless Energy-Charts REST API (Fraunhofer ISE)."""

    source = "Energy-Charts"
    licence = ENERGY_CHARTS_LICENCE

    def _series(self, endpoint: str, params: dict) -> dict:
        r = self.get(f"{ENERGY_CHARTS_BASE}/{endpoint}", params=params)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()

    def day_ahead_price(self, bzn: str, start: str, end: str) -> pd.Series:
        """Day-ahead price for one bidding zone, tz-naive local index."""
        j = self._series("price", {"bzn": bzn, "start": start, "end": end})
        if not j or "unix_seconds" not in j:
            return pd.Series(dtype=float)
        idx = utc_seconds_to_local_naive(j["unix_seconds"])
        return pd.Series(j["price"], index=idx, dtype=float).sort_index()

    def public_power(self, country: str, start: str, end: str) -> pd.DataFrame:
        """REALIZED generation by production type.

        Realized, therefore NOT knowable for the target day at the forecast
        origin. Legal only as a lag; the leakage guard enforces that, not
        this client.
        """
        j = self._series("public_power", {"country": country, "start": start, "end": end})
        if not j or "unix_seconds" not in j:
            return pd.DataFrame()
        idx = utc_seconds_to_local_naive(j["unix_seconds"])
        data = {
            pt["name"]: pd.Series(pt["data"], index=idx, dtype=float)
            for pt in j.get("production_types", [])
        }
        return _to_hourly_mean(pd.DataFrame(data).sort_index())

    def cross_border_flows(self, country: str, start: str, end: str) -> pd.DataFrame:
        """REALIZED physical cross-border flows (/cbpf). Lagged use only."""
        j = self._series("cbpf", {"country": country, "start": start, "end": end})
        if not j or "unix_seconds" not in j:
            return pd.DataFrame()
        idx = utc_seconds_to_local_naive(j["unix_seconds"])
        data = {
            c["name"]: pd.Series(c["data"], index=idx, dtype=float)
            for c in j.get("countries", [])
        }
        return _to_hourly_mean(pd.DataFrame(data).sort_index())

    def installed_power(self, country: str) -> pd.DataFrame:
        """Installed capacity by technology, yearly.

        Structural and published in advance, so unlike generation this is
        knowable ex ante -- the one capacity input that needs no lag.
        """
        j = self._series(
            "installed_power",
            {"country": country, "time_step": "yearly", "installation_decommission": "false"},
        )
        if not j:
            return pd.DataFrame()
        years = j.get("time") or j.get("years") or []
        data = {
            pt["name"]: pd.Series(pt["data"], dtype=float)
            for pt in j.get("production_types", [])
        }
        df = pd.DataFrame(data)
        if len(years) == len(df):
            df.index = [str(y) for y in years]
        return df


def _read_auction_sheet(raw: bytes, max_scan: int = 12) -> pd.DataFrame | None:
    """Read one EEX auction workbook, locating its header row.

    Scans the first `max_scan` rows for the one holding both 'Date' and an
    'Auction Price' column, then re-reads with that row as the header and
    drops all-empty leading columns. Returns None when no such row exists.
    """
    probe = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None, nrows=max_scan)
    header_row = None
    for i in range(len(probe)):
        vals = [str(v).strip() for v in probe.iloc[i].tolist()]
        if any(v == "Date" for v in vals) and any("Auction Price" in v for v in vals):
            header_row = i
            break
    if header_row is None:
        return None
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=header_row)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


class EEXCarbonClient(_HttpClient):
    """Tier A. EU ETS primary-auction settlement prices, published by EEX.

    This is the one fuel-side price obtainable without a licensed feed. Gas
    (TTF) and coal (API2) are Montel-licensed; Ember publishes only series
    DERIVED from them, which makes Ember a citation rather than a source.
    """

    source = "EEX"
    licence = EEX_LICENCE

    def eua_auction_prices(self) -> pd.DataFrame:
        """Daily EUA auction settlement prices, 2012-2025.

        Returns a frame indexed by auction date with columns
        [eua_price_eur_t, auction_name]. Multiple auctions can occur on one
        date (EU, DE, PL run separate auctions); the EU-wide auction is the
        reference, and where it is absent the day's mean is used.
        """
        r = self.get(EEX_EUA_ARCHIVE_URL, timeout=180)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))

        frames = []
        for name in z.namelist():
            if not name.lower().endswith((".xls", ".xlsx")):
                continue
            raw = z.read(name)
            # The header row MOVES across the archive: 2012-2016 put it on
            # row 2, 2017 onward on row 5 behind a leading blank column.
            # Hardcoding a row silently yielded an all-"Unnamed" frame for
            # every file from 2017 on, which the price-column check then
            # skipped -- losing a third of the series with no error and no
            # log line. So find the header instead of assuming it.
            try:
                df = _read_auction_sheet(raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not parse %s: %s", name, exc)
                continue
            if df is None:
                logger.warning("no auction header found in %s", name)
                continue
            price_col = next(
                (c for c in df.columns if "Auction Price" in c), None
            )
            if price_col is None or "Date" not in df.columns:
                logger.warning("no Date/Auction Price columns in %s", name)
                continue
            sub = df[["Date", "Auction Name", price_col]].copy()
            sub.columns = ["date", "auction_name", "eua_price_eur_t"]
            frames.append(sub)

        if not frames:
            raise RuntimeError("EEX archive parsed but yielded no auction rows")

        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["eua_price_eur_t"] = pd.to_numeric(out["eua_price_eur_t"], errors="coerce")
        out = out.dropna(subset=["date", "eua_price_eur_t"])
        return out.sort_values("date").reset_index(drop=True)


class EntsoeClient(_HttpClient):
    """Tier B. ENTSO-E Transparency Platform. Requires a free academic token.

    Supplies what Energy-Charts does not: generation outages (true scarcity),
    NTC (coupling capacity as opposed to realized flow), and hydro reservoir
    levels. Register at transparency.entsoe.eu and email
    transparency@entsoe.eu with subject "RESTful API access"; granted in
    about three working days.

    The token is read from the environment and never written to disk, logged,
    or committed. With no token configured every method raises
    TokenNotConfigured, and the features that depend on this client skip
    cleanly rather than falling back to a fabricated series.
    """

    source = "ENTSO-E"
    licence = "ENTSO-E Transparency Platform terms (attribution required)"

    #: documentType codes, per the Transparency Platform API guide.
    DOC_UNAVAILABILITY_GENERATION = "A80"
    DOC_INSTALLED_CAPACITY = "A68"
    DOC_NTC_FORECAST = "A61"
    DOC_HYDRO_RESERVOIR = "A72"

    def __init__(self, token: str | None = None, **kw):
        super().__init__(**kw)
        self._token = token if token is not None else os.environ.get(ENTSOE_TOKEN_ENV)

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def _require_token(self) -> str:
        if not self._token:
            raise TokenNotConfigured(
                f"{ENTSOE_TOKEN_ENV} is not set. The ENTSO-E Transparency "
                f"Platform requires a free academic token: register at "
                f"transparency.entsoe.eu, then email transparency@entsoe.eu "
                f"with 'RESTful API access' as the subject (granted in ~3 "
                f"working days). Export the token as {ENTSOE_TOKEN_ENV}. "
                f"Until then the features depending on this client skip -- "
                f"they are never filled with substitute values."
            )
        return self._token

    def query(self, params: dict) -> str:
        """Raw XML for one Transparency Platform query."""
        token = self._require_token()
        r = self.get(ENTSOE_BASE, params={**params, "securityToken": token})
        r.raise_for_status()
        return r.text

    def generation_outages(self, domain: str, start: str, end: str) -> str:
        return self.query(
            {
                "documentType": self.DOC_UNAVAILABILITY_GENERATION,
                "biddingZone_Domain": domain,
                "periodStart": start,
                "periodEnd": end,
            }
        )

    def ntc_forecast(self, in_domain: str, out_domain: str, start: str, end: str) -> str:
        return self.query(
            {
                "documentType": self.DOC_NTC_FORECAST,
                "in_Domain": in_domain,
                "out_Domain": out_domain,
                "periodStart": start,
                "periodEnd": end,
            }
        )

    def hydro_reservoir(self, domain: str, start: str, end: str) -> str:
        return self.query(
            {
                "documentType": self.DOC_HYDRO_RESERVOIR,
                "processType": "A16",
                "in_Domain": domain,
                "periodStart": start,
                "periodEnd": end,
            }
        )
