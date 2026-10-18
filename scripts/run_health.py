"""Run health for the scheduled PriceCast jobs — scripts/run_health.py

Called by scripts/run_daily.ps1 after EVERY run, including failed ones:

    python scripts/run_health.py --exit-code N [--refresh-failed exante,shadow] [--next-run-utc ISO]

Writes app/forecast_site/data/status.json (the dashboard's status line) and
prints one "ALERT: ..." line per problem; the wrapper shows each as a Windows
toast. Reads only local files: it never touches the network.

Alert windows (Berlin time) are wide enough that a scheduled run lands in
them in both CEST and CET (run_daily.ps1 schedule table):
  11:00-12:00  tomorrow's ex-ante forecast still missing before the auction
  >= 19:30     tomorrow's shadow forecast still missing (A69 due 18:00)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "app" / "forecast_site" / "data"
FORECASTS = DATA / "forecasts"
LOCAL_TZ = "Europe/Berlin"
EXANTE_WINDOW = (11 * 60, 12 * 60)  # minutes after midnight, Berlin
SHADOW_FROM = 19 * 60 + 30


def issued_forecasts(forecasts_dir: Path = FORECASTS) -> set[str]:
    """{'2026-10-17.exante', ...} from the write-once forecast files."""
    return {p.stem for p in forecasts_dir.glob("????-??-??.*.json")} if forecasts_dir.exists() else set()


def run_errors(path: Path = DATA / "latest.json") -> list[str]:
    """The `errors` daily_run wrote into latest.json this run (already token-redacted)."""
    return list(json.loads(path.read_text(encoding="utf-8")).get("errors") or []) if path.exists() else []


def alerts(now_local: pd.Timestamp, exit_code: int, issued: set[str], refresh_failed: list[str],
           errors: list[str] = ()) -> list[str]:
    out = []
    if exit_code != 0:
        out.append(f"Daily run failed (exit {exit_code}). The page keeps showing the last good data; see the run log (daily_run.log on the PC, the Actions job log in the cloud).")
    # a fetch failure does not fail the run (it still re-serves cached data), so
    # it must alert on its own: a revoked token once ran silently "green" (2026-10-17)
    for e in errors:
        out.append(f"Run finished but {e[:300]}")
    for v in refresh_failed:
        out.append(f"Weekly refresh of the {v} model failed; still serving the previous model.")
    tomorrow = (now_local.normalize() + pd.Timedelta(days=1)).date()
    minute = now_local.hour * 60 + now_local.minute
    if EXANTE_WINDOW[0] <= minute < EXANTE_WINDOW[1] and f"{tomorrow}.exante" not in issued:
        out.append(f"No ex-ante forecast for {tomorrow} at {now_local:%H:%M} Berlin; the auction closes at 12:00.")
    if minute >= SHADOW_FROM and f"{tomorrow}.shadow" not in issued:
        out.append(f"No shadow forecast for {tomorrow} at {now_local:%H:%M} Berlin (wind/solar input A69 late?).")
    return out


def write_status(now_local: pd.Timestamp, exit_code: int, found: list[str], next_run_utc: str | None,
                 path: Path = DATA / "status.json") -> dict:
    prev = json.loads(path.read_text()) if path.exists() else {}
    stamp = now_local.strftime("%Y-%m-%d %H:%M")
    status = {
        "last_run": stamp,
        "last_exit": exit_code,
        "last_success": stamp if exit_code == 0 else prev.get("last_success"),
        "next_run": (pd.Timestamp(next_run_utc).tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
                     if next_run_utc else None),
        "alerts": found,
        "timezone": LOCAL_TZ,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(path)
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--refresh-failed", default="", help="comma-separated variants whose refit failed")
    ap.add_argument("--next-run-utc", default=None)
    args = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC").tz_convert(LOCAL_TZ)
    found = alerts(now, args.exit_code, issued_forecasts(), [v for v in args.refresh_failed.split(",") if v],
                   run_errors())
    write_status(now, args.exit_code, found, args.next_run_utc)
    for a in found:
        print(f"ALERT: {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
