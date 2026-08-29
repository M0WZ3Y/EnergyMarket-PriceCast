"""Fetch the pinned physical-data snapshot — scripts/fetch_physical_snapshot.py

THE ONLY ENTRY POINT THAT TOUCHES THE NETWORK FOR FEATURE DATA. Run it once;
commit what it writes; every feature afterwards reads the committed files and
never the API.

Why: an API answers differently tomorrow. A thesis result computed against a
live source is not reproducible, and the drift is silent -- a reviewer
re-running the pipeline a year later gets different numbers with no error.
Pinning turns that silent drift into a loud hash mismatch
(`provenance.verify_snapshot`).

Writes to data/raw/physical/:
    ec_price_de.csv          German day-ahead price (DE-AT-LU / DE-LU)
    ec_price_neighbours.csv  FR NL BE AT CH CZ PL DK1 DK2 day-ahead prices
    ec_public_power.csv      realized generation by production type
    ec_cross_border.csv      realized physical cross-border flows
    ec_installed_power.csv   installed capacity by technology, yearly
    eex_eua_auctions.csv     EU ETS primary auction settlement prices
    provenance.json          source, url, licence, fetch time, range, hash

Tier B (ENTSO-E: outages, NTC, reservoir levels) is attempted only when
ENTSOE_API_TOKEN is set. Without it the script reports the skip and exits
successfully -- the dependent features skip too, and are never backfilled
with substitute values.

This script is intentionally NOT ledger-gated: it produces raw input data,
not thesis results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sources import provenance as prov  # noqa: E402
from src.data.sources.clients import (  # noqa: E402
    EC_DATA_START,
    NEIGHBOUR_ZONES,
    EEXCarbonClient,
    EnergyChartsClient,
    EntsoeClient,
    de_bidding_zone,
)

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2018-01-01"


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split a range into calendar-year chunks.

    The API rate-limits long ranges and occasionally truncates them; a year
    at a time is reliable and lets a partial failure be retried cheaply.
    """
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = s
    while cur < e:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1), e)
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def _pin(df: pd.DataFrame, key: str, filename: str, source: str, url: str,
         licence: str, notes: str = "") -> prov.ProvenanceRecord:
    """Write one series to the snapshot and build its provenance record."""
    prov.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = prov.SNAPSHOT_DIR / filename
    df.to_csv(path, index=True, lineterminator="\n")
    idx = df.index
    return prov.ProvenanceRecord(
        key=key,
        source=source,
        url=url,
        licence=licence,
        fetched_at_utc=prov.utc_now(),
        date_start=str(idx.min()),
        date_end=str(idx.max()),
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        sha256=prov.sha256_file(path),
        filename=filename,
        notes=notes,
    )


def fetch_energy_charts(start: str, end: str) -> dict[str, prov.ProvenanceRecord]:
    ec = EnergyChartsClient()
    records: dict[str, prov.ProvenanceRecord] = {}

    # Clamp to the verified coverage floor. Requesting earlier returns HTTP
    # 400 "end must be >= start" -- a message about the wrong thing entirely,
    # which would abort the whole fetch and send the next person debugging
    # the date format rather than the coverage floor.
    floor = pd.Timestamp(EC_DATA_START)
    if pd.Timestamp(start) < floor:
        print(f"    NOTE: requested start {start} precedes Energy-Charts "
              f"coverage; clamping to {floor.date()}.")
        print(f"    Blocks built on these endpoints therefore cover only "
              f"{floor.date()} onward, not the full benchmark window.")
        start = floor.strftime("%Y-%m-%d")

    chunks = _year_chunks(start, end)

    # --- German day-ahead price (zone changes 2018-10-01) ---------------
    parts = []
    for a, b in chunks:
        bzn = de_bidding_zone(pd.Timestamp(a).date())
        s = ec.day_ahead_price(bzn, a, b)
        print(f"    price {bzn:<9} {a}..{b}  {len(s):>6} rows")
        if len(s):
            parts.append(s)
    if parts:
        de = pd.concat(parts).sort_index()
        de = de[~de.index.duplicated(keep="first")].to_frame("price_de")
        records["ec_price_de"] = _pin(
            de, "ec_price_de", "ec_price_de.csv", ec.source,
            "https://api.energy-charts.info/price", ec.licence,
            "bzn DE-AT-LU before 2018-10-01, DE-LU after; tz-naive local time",
        )

    # --- Neighbour day-ahead prices (LAGGED USE ONLY) -------------------
    cols = {}
    for zone in NEIGHBOUR_ZONES:
        parts = []
        for a, b in chunks:
            s = ec.day_ahead_price(zone, a, b)
            if len(s):
                parts.append(s)
        if parts:
            z = pd.concat(parts).sort_index()
            cols[f"price_{zone}"] = z[~z.index.duplicated(keep="first")]
            print(f"    price {zone:<9} {len(cols[f'price_{zone}']):>6} rows")
        else:
            print(f"    price {zone:<9} no data")
    if cols:
        nb = pd.DataFrame(cols).sort_index()
        records["ec_price_neighbours"] = _pin(
            nb, "ec_price_neighbours", "ec_price_neighbours.csv", ec.source,
            "https://api.energy-charts.info/price", ec.licence,
            "COUPLED AUCTION: same-day values leak. Lagged use only.",
        )

    # --- Realized generation mix (LAGGED USE ONLY) ----------------------
    parts = []
    for a, b in chunks:
        d = ec.public_power("de", a, b)
        print(f"    public_power   {a}..{b}  {len(d):>6} rows x {d.shape[1]}")
        if len(d):
            parts.append(d)
    if parts:
        pp = pd.concat(parts).sort_index()
        pp = pp[~pp.index.duplicated(keep="first")]
        records["ec_public_power"] = _pin(
            pp, "ec_public_power", "ec_public_power.csv", ec.source,
            "https://api.energy-charts.info/public_power", ec.licence,
            "REALIZED generation, not a forecast. Lagged use only.",
        )

    # --- Realized cross-border physical flows (LAGGED USE ONLY) ---------
    parts = []
    for a, b in chunks:
        d = ec.cross_border_flows("de", a, b)
        print(f"    cbpf           {a}..{b}  {len(d):>6} rows x {d.shape[1]}")
        if len(d):
            parts.append(d)
    if parts:
        cb = pd.concat(parts).sort_index()
        cb = cb[~cb.index.duplicated(keep="first")]
        records["ec_cross_border"] = _pin(
            cb, "ec_cross_border", "ec_cross_border.csv", ec.source,
            "https://api.energy-charts.info/cbpf", ec.licence,
            "REALIZED flows, not scheduled. Lagged use only.",
        )

    # --- Installed capacity (structural, known ex ante) ------------------
    ip = ec.installed_power("de")
    print(f"    installed_power {len(ip):>5} rows x {ip.shape[1]}")
    if len(ip):
        records["ec_installed_power"] = _pin(
            ip, "ec_installed_power", "ec_installed_power.csv", ec.source,
            "https://api.energy-charts.info/installed_power", ec.licence,
            "Yearly structural capacity; published in advance, no lag needed.",
        )
    return records


def fetch_eex_carbon() -> dict[str, prov.ProvenanceRecord]:
    c = EEXCarbonClient()
    df = c.eua_auction_prices().set_index("date")
    print(f"    EUA auctions   {len(df):>6} rows  "
          f"{df.index.min().date()}..{df.index.max().date()}")
    return {
        "eex_eua_auctions": _pin(
            df, "eex_eua_auctions", "eex_eua_auctions.csv", c.source,
            "https://www.eex.com/en/market-data/environmental-markets/"
            "eua-primary-auction-spot-download", c.licence,
            "EU ETS primary auction settlement, ~11:00 CET on auction days. "
            "Used LAGGED (last auction strictly before the forecast origin).",
        )
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["energy_charts", "eex", "entsoe"])
    args = ap.parse_args(argv)

    want = set(args.only) if args.only else {"energy_charts", "eex", "entsoe"}
    records: dict[str, prov.ProvenanceRecord] = {}

    print("=" * 78)
    print(f"PHYSICAL DATA SNAPSHOT  {args.start} .. {args.end}")
    print("=" * 78)

    if "energy_charts" in want:
        print("\n[Tier A] Energy-Charts (keyless, CC BY 4.0)")
        records.update(fetch_energy_charts(args.start, args.end))

    if "eex" in want:
        print("\n[Tier A] EEX EU ETS primary auction archive (public)")
        records.update(fetch_eex_carbon())

    if "entsoe" in want:
        print("\n[Tier B] ENTSO-E Transparency Platform")
        e = EntsoeClient()
        if not e.configured:
            print("    SKIPPED — ENTSOE_API_TOKEN not set.")
            print("    Outages, NTC and reservoir levels are therefore NOT in")
            print("    this snapshot, and the features depending on them will")
            print("    skip. They are never backfilled with substitute values.")
        else:
            print("    token present — Tier B fetch is implemented in the")
            print("    client; wire the specific queries when the token lands.")

    if records:
        prov.write_manifest(records)

    print("\n" + "=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print(prov.format_provenance_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
