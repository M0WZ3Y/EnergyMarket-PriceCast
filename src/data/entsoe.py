"""ENTSO-E Transparency Platform loader — src/data/entsoe.py

Produces the shared schema (`price`, `exog_1`, `exog_2`; hourly, tz-naive
Europe/Berlin) from the Transparency Platform, so live serving can feed the
models the SAME definitions they were trained on. That is the point of this
module, and the reason for each choice below (spec:
docs/superpowers/specs/2026-10-15-live-entsoe-recalibration-design.md).

  exog_1  A65 day-ahead total load forecast for the AMPRION control area.
          The benchmark's exog_1 is Amprion zonal load; serving German
          national load instead was defect D2 (x2.553, logs/decisions.md
          2026-09-02).
  exog_2  A69 day-ahead wind + solar forecast, B16 + B18 + B19 summed.
  price   A44 day-ahead price, DE-AT-LU before the 2018-10-01 split, DE-LU after;
          classification sequence 1 only (the SDAC price; 2026-10-17 c).

TIME. The benchmark index is tz-naive local German time with exactly 24 rows
per day. Serving UTC hours to those models was defect D1 (2026-10-15). The
DST convention is copied from what data/raw/DE.csv actually does on
2015-03-29: the nonexistent 02:00 is linearly interpolated for price and
renewables (7.01 = mean(8.93, 5.08)) and repeats the previous hour for load
(17011 twice). The 25-hour autumn day collapses its doubled 02:00 by mean.
Any OTHER gap stays NaN -- build_features drops incomplete days, and an
invented hour in a real gap would be indistinguishable from data.

TOKEN HYGIENE. Library exceptions embed the full request URL, and the token
travels in the query string. Every exception leaving this module is
re-raised as EntsoeFetchError with the text redacted and `from None`, so the
original (URL-bearing) exception is not chained into a printed traceback.
This is not hypothetical: an unredacted SSLError leaked the token into a
session transcript on 2026-10-15.

Network use is confined to EntsoeLoader. Thesis numbers read the pinned
snapshot in data/raw/entsoe/, never this loader directly.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"

DE_LU = "10Y1001A1001A82H"
DE_AT_LU = "10Y1001A1001A63L"
DE = "10Y1001A1001A83F"
AMPRION = "10YDE-RWENET---I"

BZN_SPLIT = date(2018, 10, 1)
WIND_SOLAR_PSR = ("B16", "B18", "B19")
MAX_CHUNK_DAYS = 365  # API maximum is one year; each request costs ~36 s of latency regardless of size

LONG_COLUMNS = ["psr", "res_minutes", "timestamp", "value"]

_RE_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_RE_TOKEN_PARAM = re.compile(r"(securityToken=)[^&\s'\"]+")


class EntsoeFetchError(RuntimeError):
    """A fetch or document error, with any credential already removed."""


def redact(text: object, token: str | None = None) -> str:
    s = str(text)
    if token:
        s = s.replace(token, "<REDACTED>")
    s = _RE_TOKEN_PARAM.sub(r"\1<REDACTED>", s)
    return _RE_UUID.sub("<REDACTED>", s)


# -- parsing -----------------------------------------------------------------
def _resolution_minutes(res: str) -> int:
    m = re.fullmatch(r"PT(\d+)M", res)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"PT(\d+)H", res)
    if m:
        return 60 * int(m.group(1))
    raise EntsoeFetchError(f"unsupported resolution {res!r}")


SDAC_SEQUENCE = 1  # A44 classification sequence of the SDAC day-ahead price (2026-10-17 c)


def _empty_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "psr": pd.Series(dtype=object),
            "res_minutes": pd.Series(dtype=int),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "value": pd.Series(dtype=float),
        }
    )


def parse_timeseries(xml: str, generation_only: bool = False, price_sequence: int | None = None) -> pd.DataFrame:
    """Transparency Platform XML -> long frame [psr, res_minutes, timestamp, value].

    `timestamp` is the UTC start of each interval. An acknowledgement saying
    "No matching data" is a legitimate empty answer; any other
    acknowledgement is an error. Curve type A03 omits positions whose value
    repeats the previous one, so skipped positions are forward-filled up to
    the period end.

    generation_only (A75 actual generation per type): skip TimeSeries that carry
    only an outBiddingZone_Domain, i.e. the CONSUMPTION direction. For wind and
    solar that series is a single A03 point of 0 which the forward-fill expands
    into a full period of zeros; left in, it collides with the real generation
    and wins the dedupe whenever it comes first in the document (it did for
    2021-2025 B19; logs/decisions.md 2026-10-16 d).

    price_sequence (A44): keep only TimeSeries whose classification sequence is
    this number (a series with no sequence element is kept). DE-LU A44 carries
    two series per day: sequence 1 is the SDAC day-ahead price, sequence 2 is
    another price, published earlier and listed FIRST, so without the filter it
    won the dedupe and was stored as "price" for 2019-2026 (logs/decisions.md
    2026-10-17 c).
    """
    root = ET.fromstring(xml)
    tag = root.tag.split("}")[-1]
    ns = {"n": root.tag[1:].split("}")[0]} if root.tag.startswith("{") else {"n": ""}

    def find(el, path):
        return el.find(path, ns) if ns["n"] else el.find(path.replace("n:", ""))

    def findall(el, path):
        return el.findall(path, ns) if ns["n"] else el.findall(path.replace("n:", ""))

    if tag == "Acknowledgement_MarketDocument":
        reason = " ".join((t.text or "") for t in root.iter() if t.tag.endswith("text"))
        if "No matching data" in reason:
            return _empty_long()
        raise EntsoeFetchError(f"ENTSO-E acknowledgement: {redact(reason)[:300]}")

    rows = []
    for ts in findall(root, "n:TimeSeries"):
        if generation_only and find(ts, "n:inBiddingZone_Domain.mRID") is None                 and find(ts, "n:outBiddingZone_Domain.mRID") is not None:
            continue
        if price_sequence is not None:
            seq_el = find(ts, "n:classificationSequence_AttributeInstanceComponent.position")
            if seq_el is not None and int(seq_el.text) != price_sequence:
                continue
        psr_el = find(ts, "n:MktPSRType/n:psrType")
        psr = psr_el.text if psr_el is not None else "-"
        curve_el = find(ts, "n:curveType")
        curve = curve_el.text if curve_el is not None else "A01"
        for period in findall(ts, "n:Period"):
            start = pd.Timestamp(find(period, "n:timeInterval/n:start").text)
            end = pd.Timestamp(find(period, "n:timeInterval/n:end").text)
            res = _resolution_minutes(find(period, "n:resolution").text)
            step = pd.Timedelta(minutes=res)
            n_slots = int((end - start) / step)
            values = np.full(n_slots, np.nan)
            for pt in findall(period, "n:Point"):
                pos = int(find(pt, "n:position").text)
                q = find(pt, "n:quantity")
                if q is None:
                    q = find(pt, "n:price.amount")
                if 1 <= pos <= n_slots:
                    values[pos - 1] = float(q.text)
            if curve == "A03":
                values = pd.Series(values).ffill().to_numpy()
            stamps = start + step * np.arange(n_slots)
            keep = ~np.isnan(values)
            rows.append(
                pd.DataFrame(
                    {
                        "psr": psr,
                        "res_minutes": res,
                        "timestamp": pd.DatetimeIndex(stamps)[keep],
                        "value": values[keep],
                    }
                )
            )
    if not rows:
        return _empty_long()
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out[LONG_COLUMNS]


# -- time ----------------------------------------------------------------------
def _dedupe(long: pd.DataFrame) -> pd.DataFrame:
    """Overlapping TimeSeries for one psr: finest resolution wins, then first.

    "Finest" is decided per HOUR, not over the whole span: a series that moved
    from PT60M to PT15M (SDAC on 2025-10-01) keeps its hourly values before the
    switch. Span-wide, every pre-switch hour was dropped (2026-10-17 c).
    """
    if long.empty:
        return long
    finest = long.groupby([long["psr"], long["timestamp"].dt.floor("h")])["res_minutes"].transform("min")
    long = long[long["res_minutes"] == finest]
    return long.drop_duplicates(subset=["psr", "timestamp"], keep="first")


def _utc_hourly(long: pd.DataFrame) -> pd.Series:
    """Duration-weighted hourly mean in UTC; an hour missing any minutes is NaN."""
    if long.empty:
        return pd.Series(dtype=float)
    hour = long["timestamp"].dt.floor("h")
    w = long["res_minutes"].astype(float)
    num = (long["value"] * w).groupby(hour).sum()
    minutes = w.groupby(hour).sum()
    s = (num / minutes).where(minutes >= 60)
    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    return s.reindex(full)


def _to_local_naive(s_utc: pd.Series, fill: str) -> pd.Series:
    if s_utc.empty:
        return pd.Series(dtype=float)
    local = s_utc.copy()
    local.index = s_utc.index.tz_convert(LOCAL_TZ).tz_localize(None)
    local = local.groupby(level=0).mean()  # autumn: doubled 02:00 -> mean
    full = pd.date_range(local.index.min(), local.index.max(), freq="h")
    local = local.reindex(full)
    # Nonexistent local hours (spring DST). Only these are filled.
    probe = full.tz_localize(LOCAL_TZ, nonexistent="NaT", ambiguous=True)
    gaps = full[probe.isna()]
    if len(gaps):
        if fill == "interpolate":
            filled = local.interpolate(method="linear", limit=1, limit_area="inside")
        elif fill == "ffill":
            filled = local.ffill(limit=1)
        else:
            raise ValueError(f"fill must be 'interpolate' or 'ffill', got {fill!r}")
        local.loc[gaps] = filled.loc[gaps]
    local.index.name = "timestamp"
    return local


def utc_to_local_hourly(long: pd.DataFrame, fill: str) -> pd.Series:
    """Long UTC frame (one psr) -> hourly tz-naive local series, benchmark DST rule."""
    return _to_local_naive(_utc_hourly(_dedupe(long)), fill)


def utc_hourly_frame_to_local(df_utc: pd.DataFrame, fills: dict[str, str]) -> pd.DataFrame:
    """Already-hourly UTC frame (e.g. Energy-Charts as served) -> local naive.

    Used for ladder rung R1a, which isolates the timezone defect D1 by
    changing nothing about the served data except its clock.
    """
    if df_utc.index.tz is None:
        raise ValueError("expected a tz-aware UTC index")
    cols = {c: _to_local_naive(df_utc[c].astype(float), fills.get(c, "interpolate")) for c in df_utc.columns}
    out = pd.DataFrame(cols)
    out.index.name = "timestamp"
    return out


# -- loader ------------------------------------------------------------------------
def _utc_param(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M")


class EntsoeLoader:
    """Live ENTSO-E fetches in the shared schema. Network only; see module doc."""

    source = "ENTSO-E Transparency Platform"
    attribution = "Data: ENTSO-E Transparency Platform (transparency.entsoe.eu)"

    def __init__(self, client=None, chunk_days: int = MAX_CHUNK_DAYS,
                 exog_2_domain: str = DE_LU, exog_2_domain_pre_split: str = DE_AT_LU):
        if client is None:
            from src.data.sources.clients import EntsoeClient

            client = EntsoeClient()
        self.client = client
        self.chunk_days = chunk_days
        self.exog_2_domain = exog_2_domain
        self.exog_2_domain_pre_split = exog_2_domain_pre_split

    # -- transport --
    def _query(self, params: dict) -> str:
        try:
            return self.client.query(params)
        except Exception as exc:  # noqa: BLE001 -- every path may carry the URL
            token = getattr(self.client, "_token", None)
            # drop the request URL entirely (redacted or not, it is noise to a reader)
            text = re.sub(r"\s*for url:\s*\S+", "", str(exc))
            msg = redact(f"{type(exc).__name__}: {text}", token=token)
            raise EntsoeFetchError(f"ENTSO-E request failed ({params.get('documentType')}): {msg[:400]}") from None

    def _fetch_long(self, base: dict, start: str, end: str, generation_only: bool = False,
                    price_sequence: int | None = None) -> pd.DataFrame:
        """Chunked fetch covering local dates [start, end] inclusive, padded a day each side."""
        lo = pd.Timestamp(start) - pd.Timedelta(days=1)
        hi = pd.Timestamp(end) + pd.Timedelta(days=2)
        parts = []
        cur = lo
        while cur < hi:
            nxt = min(cur + pd.Timedelta(days=self.chunk_days), hi)
            xml = self._query({**base, "periodStart": _utc_param(cur), "periodEnd": _utc_param(nxt)})
            parts.append(parse_timeseries(xml, generation_only=generation_only, price_sequence=price_sequence))
            cur = nxt
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else _empty_long()

    @staticmethod
    def _trim(s: pd.Series, start: str, end: str) -> pd.Series:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) + pd.Timedelta(hours=23)
        full = pd.date_range(lo, hi, freq="h", name="timestamp")
        return s.reindex(full)

    def _split_ranges(self, start: str, end: str) -> list[tuple[str, str, bool]]:
        """[(start, end, post_split)] around the 2018-10-01 bidding-zone split."""
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        if e < BZN_SPLIT:
            return [(start, end, False)]
        if s >= BZN_SPLIT:
            return [(start, end, True)]
        before_end = (pd.Timestamp(BZN_SPLIT) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return [(start, before_end, False), (BZN_SPLIT.isoformat(), end, True)]

    # -- series --
    def fetch_prices(self, start: str, end: str) -> pd.Series:
        pieces = []
        for s, e, post in self._split_ranges(start, end):
            dom = DE_LU if post else DE_AT_LU
            long = self._fetch_long({"documentType": "A44", "in_Domain": dom, "out_Domain": dom}, s, e,
                                    price_sequence=SDAC_SEQUENCE)
            pieces.append(self._trim(utc_to_local_hourly(long, fill="interpolate"), s, e))
        return pd.concat(pieces).rename("price")

    def fetch_load_forecast(self, start: str, end: str, domain: str = AMPRION) -> pd.Series:
        long = self._fetch_long(
            {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": domain}, start, end
        )
        return self._trim(utc_to_local_hourly(long, fill="ffill"), start, end).rename("exog_1")

    def fetch_load_actual(self, start: str, end: str, domain: str = AMPRION) -> pd.Series:
        """Realized total load (A65, processType A16). DIAGNOSTIC ONLY.

        A realized value is unknown at the forecast origin; it may only feed the
        perfect-input ceiling (test D), never a served or selected model.
        """
        long = self._fetch_long(
            {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": domain}, start, end
        )
        return self._trim(utc_to_local_hourly(long, fill="ffill"), start, end).rename("load_actual")

    def fetch_wind_solar_actual(self, start: str, end: str, domain: str = DE_LU) -> pd.Series:
        """Realized wind + solar generation (A75, processType A16). DIAGNOSTIC ONLY."""
        per_psr = {}
        for psr in WIND_SOLAR_PSR:
            long = self._fetch_long(
                {"documentType": "A75", "processType": "A16", "in_Domain": domain, "psrType": psr}, start, end,
                generation_only=True,
            )
            if not long.empty:
                per_psr[psr] = self._trim(utc_to_local_hourly(long, fill="interpolate"), start, end)
        if not per_psr:
            return self._trim(pd.Series(dtype=float), start, end).rename("ws_actual")
        frame = pd.DataFrame(per_psr)
        return frame.sum(axis=1, min_count=frame.shape[1]).rename("ws_actual")

    def fetch_wind_solar_forecast(self, start: str, end: str, domain: str | None = None) -> pd.Series:
        pieces = []
        ranges = [(start, end, True)] if domain else self._split_ranges(start, end)
        for s, e, post in ranges:
            dom = domain or (self.exog_2_domain if post else self.exog_2_domain_pre_split)
            long = self._fetch_long({"documentType": "A69", "processType": "A01", "in_Domain": dom}, s, e)
            per_psr = {
                psr: self._trim(utc_to_local_hourly(long[long["psr"] == psr], fill="interpolate"), s, e)
                for psr in WIND_SOLAR_PSR
                if (long["psr"] == psr).any()
            }
            if not per_psr:
                pieces.append(self._trim(pd.Series(dtype=float), s, e))
                continue
            frame = pd.DataFrame(per_psr)
            # every component present, else NaN: a missing offshore hour must
            # not silently shrink the renewables total
            pieces.append(frame.sum(axis=1, min_count=frame.shape[1]))
        return pd.concat(pieces).rename("exog_2")

    def fetch_exog(self, start: str, end: str) -> pd.DataFrame:
        """[price, exog_1, exog_2] for local dates [start, end]. Gaps stay NaN.

        The target day of a live forecast legitimately has NaN prices (not yet
        auctioned) and complete exog (day-ahead forecasts, published before
        gate closure).
        """
        df = pd.concat(
            [
                self.fetch_prices(start, end),
                self.fetch_load_forecast(start, end),
                self.fetch_wind_solar_forecast(start, end),
            ],
            axis=1,
        )
        df.index.name = "timestamp"
        return df[["price", "exog_1", "exog_2"]]
