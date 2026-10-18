"""Daily forecast product run — scripts/daily_run.py

Writes the data the static dashboard (app/forecast_site/index.html) renders:

  data/latest.json               tomorrow's forecast + status + recent track record
  data/history.json              one row per scored day (append/upsert by date)
  data/forecasts/YYYY-MM-DD.json every issued forecast, WRITE-ONCE: the track
                                 record cannot be quietly rewritten after the fact

Steps, each failing independently so the page always shows the last good data:
  1. score every issued forecast whose prices have since been published
  2. fetch the latest ENTSO-E days (prices, Amprion load, wind+solar)
  3. forecast the target day if its inputs are published, else mark it pending

STATUS of a forecast (shown on the page, never inferred later):
  ex-ante  generated before 12:00 Europe/Berlin on the day before delivery,
           i.e. before the day-ahead auction -- a genuine forecast
  shadow   generated after the auction: honest as a model test, but the
           prices already existed (A69 wind/solar is only due by 18:00 D-1)

The 80% band is a ROLLING band: 10%/90% quantiles of the served model's errors
over the trailing W days strictly before the target day (all published by
forecast time), from its walk-forward frame plus scored live forecasts. W and
per-hour vs pooled are chosen on the validation window only
(live_eval.select_band; logs/decisions.md 2026-10-16 d). A static band fitted
on validation under-covered 2026 because errors grew.

    ./.venv/Scripts/python.exe scripts/daily_run.py [--target YYYY-MM-DD] [--offline] [--backfill]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.live_service import (  # noqa: E402
    STALE_AFTER_DAYS,
    VARIANTS,
    variant_dir,
    fetch_recent,
    forecast_day,
    history_frame,
    load_served,
    staleness_days,
    v2_context,
)
from src.evaluation import live_eval as E  # noqa: E402
from src.evaluation import live_ladder as LL  # noqa: E402

SITE = REPO_ROOT / "app" / "forecast_site"
DATA = SITE / "data"
FORECASTS = DATA / "forecasts"
LOCAL_TZ = "Europe/Berlin"


def now_local() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert(LOCAL_TZ)


def prices_published(day: pd.Timestamp, now: pd.Timestamp) -> bool:
    """SDAC results for `day` publish ~12:45 Berlin on D-1; never score a day before 13:00 D-1."""
    return now >= (pd.Timestamp(day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13)).tz_localize(LOCAL_TZ)


CROSSCHECK_TOL = 0.01  # EUR/MWh: both sources publish the same SDAC result to the cent


def _energy_charts_price(start: str, end: str) -> pd.Series:
    from src.data.sources.clients import EnergyChartsClient

    return EnergyChartsClient(max_retries=2).day_ahead_price("DE-LU", start, end)


def price_crosscheck(df: pd.DataFrame, today: pd.Timestamp, fetch=_energy_charts_price) -> list[str]:
    """Our last two complete price days vs Energy-Charts (SMARD), an independent
    keyless copy of the SDAC result. A different series stored as "price" went
    unnoticed for months once (2026-10-17 c); this catches it within a day.
    Energy-Charts being unreachable is logged, not alerted: it is the reference,
    not the data."""
    days = [d for d, g in df["price"].groupby(df.index.normalize()) if d < today and g.notna().sum() == 24][-2:]
    if not days:
        return []
    try:
        ref = fetch(days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"))
    except Exception as exc:  # noqa: BLE001 -- reference unavailable: not a data problem
        print(f"price cross-check skipped: Energy-Charts unavailable ({type(exc).__name__})")
        return []
    ref = ref.groupby(ref.index.floor("h")).mean()
    bad = []
    for d in days:
        ours = _day(df, d)["price"]
        diff = (ours - ref.reindex(ours.index)).abs()
        if diff.isna().any() or diff.max() > CROSSCHECK_TOL:
            worst = "missing hours" if diff.isna().all() else f"max |diff| {diff.max():.2f} EUR/MWh"
            bad.append(f"price cross-check failed for {d.date()}: ENTSO-E vs Energy-Charts {worst}")
    return bad


def forecast_status(generated_local: pd.Timestamp, target_day: pd.Timestamp) -> str:
    auction = (pd.Timestamp(target_day).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize(LOCAL_TZ)
    return "ex-ante" if generated_local < auction else "shadow"


def error_history(served, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(walk-forward frame, frame + scored live forecasts) as [origin, hour, y_true, y_pred]."""
    p = REPO_ROOT / "data" / "processed" / "live_v2" / served.rung / f"{LL.slug(served.model)}.csv"
    if not p.exists():
        return None
    f = LL.read_frame(p)[["origin", "hour", "y_true", "y_pred"]]
    live = []
    for fp in sorted(FORECASTS.glob(f"*.{served.variant}.json")):
        fc = json.loads(fp.read_text())
        day = pd.Timestamp(fc["target_day"])
        actual = _day(df, day)["price"]
        if day in set(f["origin"]) or len(actual) < 24 or actual.isna().any():
            continue
        live.append(pd.DataFrame({"origin": day, "hour": range(24), "y_true": actual.to_numpy(dtype=float),
                                  "y_pred": [h["forecast"] for h in fc["hourly"]]}))
    return f, pd.concat([f, *live], ignore_index=True).sort_values(["origin", "hour"])


def band_quantiles(served, df: pd.DataFrame, target: pd.Timestamp) -> tuple[list[float], list[float], dict] | None:
    """Rolling 10%/90% error quantiles over the last W scored days before `target`."""
    h = error_history(served, df)
    if h is None:
        return None
    frame, errs = h
    window, pooled = E.select_band(LL.validation(frame))
    errs = errs[errs["origin"] < pd.Timestamp(target).normalize()]
    days = errs["origin"].drop_duplicates().sort_values().iloc[-window:]
    if len(days) < window:
        return None
    e = errs[errs["origin"].isin(days)].pivot(index="origin", columns="hour", values="y_true")         - errs[errs["origin"].isin(days)].pivot(index="origin", columns="hour", values="y_pred")
    if pooled:
        v = e.to_numpy().ravel()
        lo, hi = [float(np.quantile(v, 0.10))] * 24, [float(np.quantile(v, 0.90))] * 24
    else:
        lo, hi = e.quantile(0.10).tolist(), e.quantile(0.90).tolist()
    meta = {"method": "rolling", "window_days": window, "pooled": pooled,
            "from": str(days.iloc[0].date()), "through": str(days.iloc[-1].date())}
    return lo, hi, meta


def summarise(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "baseload": round(float(v.mean()), 2),
        "peak": round(float(v[8:20].mean()), 2),
        "offpeak": round(float(np.r_[v[:8], v[20:]].mean()), 2),
        "min_hour": int(v.argmin()), "min": round(float(v.min()), 2),
        "max_hour": int(v.argmax()), "max": round(float(v.max()), 2),
        "negative_hours": [int(h) for h in np.where(v < 0)[0]],
    }


def opportunities(values) -> dict:
    """Tomorrow's actionable windows from the forecast: when to run flexible load, when to avoid it."""
    v = np.asarray(values, dtype=float)

    def window(k: int, cheapest: bool) -> dict:
        means = np.convolve(v, np.ones(k) / k, mode="valid")
        i = int(means.argmin() if cheapest else means.argmax())
        return {"start": i, "end": i + k, "mean": round(float(means[i]), 2)}

    out = {
        "cheapest_hour": {"hour": int(v.argmin()), "price": round(float(v.min()), 2)},
        "cheapest_2h": window(2, True), "cheapest_4h": window(4, True), "dearest_2h": window(2, False),
    }
    out["spread_2h"] = round(out["dearest_2h"]["mean"] - out["cheapest_2h"]["mean"], 2)
    return out


def recent_hours(df: pd.DataFrame, target: pd.Timestamp, days: int = 3) -> list[dict]:
    """Published hourly prices for the `days` days before the target day (the chart's context)."""
    lo = pd.Timestamp(target).normalize() - pd.Timedelta(days=days)
    hi = pd.Timestamp(target).normalize() - pd.Timedelta(hours=1)
    s = df.loc[lo:hi, "price"]
    return [{"ts": t.strftime("%Y-%m-%d %H:00"), "price": None if pd.isna(p) else round(float(p), 2)} for t, p in s.items()]


def _day(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return df[df.index.normalize() == pd.Timestamp(day).normalize()]


def write_once(path: Path, payload: dict) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)
    return True


def issue_forecast(df: pd.DataFrame, served, target: pd.Timestamp, provisional: bool) -> dict | None:
    """None until the variant's inputs exist: ex-ante needs only D+1 load (A65);
    shadow also needs D+1 wind+solar (A69, published around 18:15 on D-1)."""
    if not served.fit_end or pd.Timestamp(served.fit_end) >= pd.Timestamp(target).normalize():
        # LEAKAGE GUARD: a model trained on the target day's prices cannot forecast it.
        print(f"  refusing {served.variant}: model fitted through {served.fit_end}, target {pd.Timestamp(target).date()}")
        return None
    day = _day(df, target)
    needed = ["exog_1"] if served.variant == "exante" else ["exog_1", "exog_2"]
    if len(day) < 24 or day[needed].isna().any().any():
        return None
    result = forecast_day(df, target, served)
    fc = result.forecast.to_numpy(dtype=float)
    q = band_quantiles(served, df, target)
    generated = now_local()
    return {
        "target_day": str(target.date()),
        "generated_at": generated.strftime("%Y-%m-%d %H:%M %Z"),
        "status": forecast_status(generated, target),
        "provisional": provisional,
        "variant": served.variant,
        "model": {"rung": served.rung, "name": served.model, "fit_end": served.fit_end},
        "band": q[2] if q else None,
        "hourly": [
            {"hour": h, "forecast": round(float(fc[h]), 2),
             "lo": round(float(fc[h] + q[0][h]), 2) if q else None,
             "hi": round(float(fc[h] + q[1][h]), 2) if q else None}
            for h in range(24)
        ],
        "summary": summarise(fc),
        "opportunities": opportunities(fc),
        "inputs": {"load_mw": [round(float(x), 1) for x in day["exog_1"]],
                   "wind_solar_mw": [None if pd.isna(x) else round(float(x), 1) for x in day["exog_2"]]},
    }


def score_row(fc: dict, df: pd.DataFrame, source: str) -> dict | None:
    day = pd.Timestamp(fc["target_day"])
    actual = _day(df, day)["price"]
    last_week = _day(df, day - pd.Timedelta(days=7))["price"]
    if len(actual) < 24 or actual.isna().any():
        return None
    a = actual.to_numpy(dtype=float)
    f = np.array([h["forecast"] for h in fc["hourly"]])
    row = {"day": fc["target_day"], "variant": fc.get("variant", "shadow"), "source": source,
           "status": fc.get("status", "backtest"),
           "forecast_baseload": round(float(f.mean()), 2), "actual_baseload": round(float(a.mean()), 2),
           "mae": round(float(np.abs(a - f).mean()), 3), "actual": [round(float(x), 2) for x in a]}
    if len(last_week) == 24 and not last_week.isna().any():
        row["naive_mae"] = round(float(np.abs(a - last_week.to_numpy(dtype=float)).mean()), 3)
    return row


def hkey(day: str, variant: str) -> str:
    return f"{day}|{variant}"


def load_history() -> dict:
    p = DATA / "history.json"
    if not p.exists():
        return {}
    return {hkey(r["day"], r.get("variant", "shadow")): r for r in json.loads(p.read_text())}


def save_history(rows: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    out = sorted(rows.values(), key=lambda r: (r["day"], r.get("variant", "shadow")))
    (DATA / "history.json").write_text(json.dumps(out, indent=1))


def backfill(served, df: pd.DataFrame, rows: dict) -> int:
    """Test-window backtest of the served rung/model, labelled 'backtest' (never 'live')."""
    p = REPO_ROOT / "data" / "processed" / "live_v2" / served.rung / f"{LL.slug(served.model)}.csv"
    if not p.exists():
        return 0
    f = LL.test_window(LL.read_frame(p))
    n = 0
    for day, g in f.groupby("origin"):
        key = str(pd.Timestamp(day).date())
        if rows.get(hkey(key, served.variant), {}).get("source") == "live":
            continue
        fc = {"target_day": key, "status": "backtest", "variant": served.variant,
              "hourly": [{"forecast": float(x)} for x in g.sort_values("hour")["y_pred"]]}
        r = score_row(fc, df, "backtest")
        if r:
            rows[hkey(key, served.variant)] = r
            n += 1
    return n


def forecast_path(day, variant: str) -> Path:
    return FORECASTS / f"{day}.{variant}.json"


def migrate_legacy_forecasts() -> None:
    """Pre-variant files (YYYY-MM-DD.json) were all shadow: renamed, never rewritten."""
    for p in FORECASTS.glob("????-??-??.json"):
        dest = forecast_path(p.stem, "shadow")
        if not dest.exists():
            p.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=pd.Timestamp, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="add each served model's 2026 backtest to the track record")
    args = ap.parse_args()
    target = (now_local().tz_localize(None).normalize() + pd.Timedelta(days=1)) if args.target is None else args.target.normalize()

    served = {}
    for v in VARIANTS:
        try:
            served[v] = load_served(variant=v)
        except FileNotFoundError:
            pass
    if not served:
        raise SystemExit("no served model; run scripts/refresh_live_model.py --variant exante|shadow")

    recent, errors = None, []
    if not args.offline:
        try:
            recent = fetch_recent(days_back=10, through=target)
        except Exception as exc:  # noqa: BLE001 -- already token-redacted; page shows it
            errors.append(f"fetch failed: {exc}")
    df = history_frame(recent)
    if not args.offline:
        errors += price_crosscheck(df, target - pd.Timedelta(days=1))

    rows = load_history()
    if args.backfill:
        for v, m in served.items():
            print(f"backfilled {backfill(m, df, rows)} {v} backtest days")

    # 1. score issued forecasts whose prices are now known
    FORECASTS.mkdir(parents=True, exist_ok=True)
    migrate_legacy_forecasts()
    for p in sorted(FORECASTS.glob("*.json")):
        fc = json.loads(p.read_text())
        k = hkey(fc["target_day"], fc.get("variant", "shadow"))
        if rows.get(k, {}).get("source") == "live":
            continue
        if not prices_published(pd.Timestamp(fc["target_day"]), now_local()):
            continue  # a value there now is not the auction price (2026-10-17 c)
        r = score_row(fc, df, "live")
        if r:
            rows[k] = r
            print(f"scored {fc['target_day']} {fc.get('variant', 'shadow')}: MAE {r['mae']:.2f}")
    save_history(rows)

    # 2-3. forecast the target day, each variant write-once
    issued = {}
    for v, m in served.items():
        path = forecast_path(target.date(), v)
        meta = json.loads((variant_dir(v) / "served.json").read_text())
        fc = json.loads(path.read_text()) if path.exists() else issue_forecast(df, m, target, bool(meta.get("provisional")))
        if fc and write_once(path, fc):
            print(f"issued {v} ({fc['status']}) forecast for {target.date()}: baseload {fc['summary']['baseload']:.2f} EUR/MWh")
        issued[v] = fc

    # the headline is the forecast one could have acted on: ex-ante when it exists
    primary = issued.get("exante") or issued.get("shadow")
    primary_model = served["exante"] if issued.get("exante") else served.get("shadow", next(iter(served.values())))
    yesterday_actual = _day(df, target - pd.Timedelta(days=1))["price"]
    target_actual = _day(df, target)["price"]
    latest = {
        "generated_at": now_local().strftime("%Y-%m-%d %H:%M %Z"),
        "target_day": str(target.date()),
        "forecast": primary,
        "forecasts": issued,
        "pending_reason": None if primary else "Tomorrow's load forecast (A65) is not published yet.",
        "stale": staleness_days(primary_model) > STALE_AFTER_DAYS,
        "model_age_days": staleness_days(primary_model),
        "recent": recent_hours(df, target, days=3),
        "yesterday_actual": [round(float(x), 2) for x in yesterday_actual] if yesterday_actual.notna().sum() == 24 else None,
        "target_actual": [round(float(x), 2) for x in target_actual] if target_actual.notna().sum() == 24 else None,
        "accuracy": v2_context(primary_model),
        "history": sorted(rows.values(), key=lambda r: (r["day"], r.get("variant", "shadow")))[-120:],
        "errors": errors,
        "attribution": "Data: ENTSO-E Transparency Platform (transparency.entsoe.eu)",
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "latest.json").write_text(json.dumps(latest, indent=1))
    print(f"wrote {DATA / 'latest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
