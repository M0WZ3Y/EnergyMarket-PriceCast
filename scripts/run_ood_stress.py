"""OOD stress test: frozen benchmark-era models on live 2026 data.

Sanctioned week-5..8 scope (CLAUDE.md gameplan, 2026-07-11): evaluate models
frozen at the end of the benchmark era against live Energy-Charts data, to
measure how a day-ahead price forecaster degrades when the market moves away
from the regime it was trained on. The German market of 2016-17 and of 2026
are not the same market; the benchmark results answer "how good is this
model on its own era", and this answers "what happens when the era changes".

Three stages, deliberately separate so the network is touched exactly once:

  --fit     fit each model on the FINAL calibration window of the benchmark
            data and persist it to models/frozen/. No live data involved.
  --fetch   pull the live window from Energy-Charts and CACHE it to
            data/raw/. Requires network.
  (default) replay: load frozen models + cached live data, predict, score.
            Offline and deterministic.

Why the cache is not optional: the live API returns different data every
day, so an uncached result could never be reproduced — and after the
v1.0-results tag, results must be reproducible forever. The cached CSV is
committed alongside the numbers it produced.

What "frozen" means here: fitted once on the last `calibration_window_days`
of benchmark data (ending at the final benchmark timestamp), which is
exactly the window the walk-forward would have used for one more origin.
The models never see a single hour of live data. Refitting on live data
would measure adaptation, not out-of-distribution degradation.

The frozen artifacts themselves are gitignored (~457 MB, mostly SARIMAX)
and are NOT committed. They are regenerable exactly: --fit is deterministic
under seed 42 and reads only committed benchmark data. The live cache is
the opposite case — it cannot be re-fetched identically, so it IS committed
(see .gitignore).

Usage:
    python scripts/run_ood_stress.py --fit
    python scripts/run_ood_stress.py --fetch --start 2026-01-01 --end 2026-07-31
    python scripts/run_ood_stress.py
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

from src.data.loader import BenchmarkLoader, EnergyChartsLoader, load_config
from src.evaluation.ensemble import combine_forecasts, combine_regime_aware, fit_weights, regime_labels
from src.evaluation.metrics import mae, rmae, rmse, smape
from src.evaluation.results import load_long_frame
from src.evaluation.walk_forward import load_evaluation_config
from src.features.pipeline import build_features
from src.models import (
    LEARLassoModel,
    LightGBMModel,
    LSTMModel,
    NaiveModel,
    SARIMAXModel,
    load_models_config,
)

FROZEN_DIR = REPO_ROOT / "models" / "frozen"
LIVE_CACHE = REPO_ROOT / "data" / "raw" / "live_ood_de.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ood"
VAL_DIR = REPO_ROOT / "data" / "processed" / "validation_preds"

MEMBERS = ["SARIMAX", "LEAR-LASSO", "LightGBM", "LSTM"]
VAL_FILES = {
    "SARIMAX": "sarimax.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "LSTM": "lstm.csv",
}


def _rel(path: Path) -> str:
    """Repo-relative path for display, falling back to the absolute path.

    Path.relative_to raises when the target sits outside the repo — which
    happens whenever these paths are redirected (tests, --cache). A
    cosmetic display call must never abort a function that has already done
    its work.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _build_models(models_cfg: dict) -> dict:
    """Same construction as run_full_baselines.py, so the frozen models are
    the same models the benchmark results describe."""
    return {
        "naive": NaiveModel(models_cfg["naive"]),
        "SARIMAX": SARIMAXModel(models_cfg["sarimax"]),
        "LEAR-LASSO": LEARLassoModel(models_cfg["lear_lasso"]),
        "LightGBM": LightGBMModel(models_cfg["lightgbm"]),
        "LSTM": LSTMModel(models_cfg["lstm"]),
    }


# --------------------------------------------------------------------------
# stage 1: freeze
# --------------------------------------------------------------------------
def fit_frozen() -> None:
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    df = pd.concat([df_train, df_test])
    X, Y = build_features(df)

    eval_cfg = load_evaluation_config()
    calib = int(eval_cfg["walk_forward"]["calibration_window_days"])

    # The trailing calibration window of the benchmark era — what the
    # walk-forward would have trained on for one more origin.
    train_days = Y.index[-calib:]
    print(
        f"freezing on {len(train_days)} days: "
        f"{train_days.min().date()} -> {train_days.max().date()}"
    )

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    models = _build_models(load_models_config())
    for name, model in models.items():
        model.fit(X.loc[train_days], Y.loc[train_days])
        path = FROZEN_DIR / name.lower().replace("-", "_")
        model.save(path)
        print(f"  froze {name} -> {_rel(path)}")

    meta = {
        "frozen_on": str(train_days.max().date()),
        "train_start": str(train_days.min().date()),
        "calibration_window_days": calib,
        "n_train_days": len(train_days),
        "benchmark_price_mean": float(df["price"].loc[: train_days.max()].mean()),
        "benchmark_price_std": float(df["price"].loc[: train_days.max()].std()),
    }
    (FROZEN_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {_rel(FROZEN_DIR / 'metadata.json')}")


# --------------------------------------------------------------------------
# stage 2: fetch + cache
# --------------------------------------------------------------------------
def fetch_live(start: str, end: str, chunk_days: int = 30) -> None:
    """Fetch the live window in chunks and cache the result.

    The API read-times-out on multi-month ranges (each fetch_exog call fans
    out to four endpoints), so the window is walked in chunks and
    concatenated. Chunks overlap by nothing and are de-duplicated on the
    index, so a boundary hour cannot be counted twice.
    """
    cfg = load_config()
    loader = EnergyChartsLoader(cfg)
    bounds = pd.date_range(start, end, freq=f"{chunk_days}D").tolist()
    if pd.Timestamp(end) not in bounds:
        bounds.append(pd.Timestamp(end))

    print(f"fetching {cfg['live']['bzn']} {start} -> {end} in {len(bounds) - 1} chunks")
    parts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        print(f"  {lo.date()} -> {hi.date()}", flush=True)
        try:
            parts.append(loader.fetch_exog(start=str(lo.date()), end=str(hi.date())))
        except Exception as exc:  # noqa: BLE001 - report and continue
            # A failed chunk is a hole in the window, not a reason to lose
            # the chunks already fetched. Report it loudly instead.
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)

    # Merge with whatever is already cached instead of overwriting. Chunks
    # fail independently (429s and transient TLS errors are routine here),
    # so a retry must be able to fill holes without discarding the chunks
    # that already succeeded.
    if LIVE_CACHE.exists():
        existing = pd.read_csv(LIVE_CACHE, index_col=0, parse_dates=True)
        print(f"merging into {len(existing)} already-cached rows")
        parts.append(existing)

    if not parts:
        raise SystemExit("live fetch returned nothing — check connectivity/date range")

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    if df.empty:
        raise SystemExit("live fetch returned no rows — check the date range")

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full.difference(df.index)
    if len(missing):
        days = sorted({t.date() for t in missing})
        print(
            f"WARNING: {len(missing)} hours still missing across {len(days)} day(s), "
            f"{days[0]} .. {days[-1]}. Re-run --fetch for that range to fill them; "
            "incomplete days are dropped by the feature pipeline."
        )

    LIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LIVE_CACHE)
    print(
        f"cached {len(df)} hourly rows "
        f"({df.index.min()} -> {df.index.max()}) -> {_rel(LIVE_CACHE)}"
    )
    print(f"ATTRIBUTION REQUIRED IN THESIS: {loader.attribution}")


# --------------------------------------------------------------------------
# stage 3: replay
# --------------------------------------------------------------------------
def _smape_zero_safe(real: np.ndarray, pred: np.ndarray) -> float:
    """sMAPE with the 0/0 case defined as zero error.

    epftoolbox's sMAPE returns NaN for the whole series if any single hour
    has actual == predicted == 0, because the denominator (|a|+|p|)/2 is
    zero there. That hour is a PERFECT forecast, so its contribution should
    be 0, not undefined — one exactly-zero price should not delete the
    metric for 173 days.

    This is not applied to the frozen benchmark results: no such hour
    occurs in 2016-17, so the shared metrics.smape wrapper is deliberately
    left untouched rather than risk perturbing tagged numbers. Exactly-zero
    day-ahead prices are a live-market phenomenon (2026 solar gluts).
    """
    num = np.abs(real - pred)
    den = (np.abs(real) + np.abs(pred)) / 2.0
    both_zero = den == 0
    ratio = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=~both_zero)
    return float(ratio.mean() * 100)


def _metrics(frame: pd.DataFrame) -> dict:
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    pred = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")

    smape_std = smape(real.values, pred.values) * 100
    smape_safe = _smape_zero_safe(real.values.ravel(), pred.values.ravel())
    if not np.isfinite(smape_std):
        n_zero = int((((np.abs(real.values) + np.abs(pred.values)) / 2) == 0).sum())
        print(
            f"    note: epftoolbox sMAPE is NaN ({n_zero} hour(s) with "
            f"actual == predicted == 0); reporting the zero-safe variant"
        )
    return {
        "MAE": mae(real.values, pred.values),
        "RMSE": rmse(real.values, pred.values),
        "sMAPE": smape_safe,
        "rMAE": rmae(real, pred, m="W"),
    }


def _long(name: str, origins, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day in origins:
        rows.append(
            pd.DataFrame(
                dict(
                    origin=[day] * 24,
                    hour=range(24),
                    y_true=y_true.loc[day].to_numpy(),
                    y_pred=y_pred.loc[day].to_numpy(),
                    model=name,
                )
            )
        )
    return pd.concat(rows, ignore_index=True)


def replay(cache: Path = LIVE_CACHE) -> None:
    if not cache.exists():
        raise SystemExit(f"{cache} not found — run with --fetch first")
    if not (FROZEN_DIR / "metadata.json").exists():
        raise SystemExit("models/frozen/ not populated — run with --fit first")

    meta = json.loads((FROZEN_DIR / "metadata.json").read_text())
    live = pd.read_csv(cache, index_col=0, parse_dates=True)

    # A cache overlapping the frozen models' training window is not an OOD
    # test at all — it would score the models on data they were fitted on
    # and report flattering numbers under an OOD heading. Refuse rather
    # than silently produce a meaningless result.
    frozen_on = pd.Timestamp(meta["frozen_on"], tz=live.index.tz)
    if live.index.min() <= frozen_on:
        raise SystemExit(
            f"cache starts {live.index.min().date()}, on or before the frozen "
            f"models' last training day {frozen_on.date()}. That overlaps the "
            "training window, so it cannot measure out-of-distribution "
            "degradation. Fetch a window strictly after the freeze date."
        )
    X, Y = build_features(live)
    if Y.empty:
        raise SystemExit(
            "no complete days survived feature construction on the live window "
            "— fetch a longer range (lags need 7 days of history)"
        )

    print(
        f"frozen on {meta['frozen_on']} "
        f"({meta['n_train_days']} training days ending there)\n"
        f"live window: {len(Y)} complete days, "
        f"{Y.index.min().date()} -> {Y.index.max().date()}"
    )

    live_mean = float(live["price"].mean())
    print(
        f"\nprice level shift: benchmark train mean "
        f"{meta['benchmark_price_mean']:.2f} EUR/MWh -> live mean {live_mean:.2f} "
        f"({live_mean / meta['benchmark_price_mean']:.2f}x)"
    )

    models = _build_models(load_models_config())
    frames: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        path = FROZEN_DIR / name.lower().replace("-", "_")
        try:
            loaded = model.load(path)
        except (FileNotFoundError, OSError) as exc:
            # metadata.json existing does not prove the model artifacts do
            # (an interrupted --fit leaves a partial freeze). Fail with a
            # instruction rather than a bare filesystem error.
            raise SystemExit(
                f"frozen {name} missing or unreadable at {path}: {exc}. "
                "Re-run with --fit to rebuild the frozen models."
            ) from exc
        # Predict one origin at a time: the wrappers implement the
        # walk-forward contract (one target_day per call), and SARIMAX
        # enforces it. No refitting happens inside the loop — that is the
        # whole point of the frozen evaluation.
        preds = pd.concat([loaded.predict(X.loc[[day]]) for day in Y.index])
        frames[name] = _long(name, Y.index, Y, preds)
        print(f"  scored {name}", flush=True)

    # Ensembles reuse the frozen validation-fitted weights: refitting them on
    # live data would leak the OOD period into its own evaluation.
    val = {m: load_long_frame(VAL_DIR / f) for m, f in VAL_FILES.items()}
    w_static = fit_weights(val)
    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    val_labels = regime_labels(val["LightGBM"], threshold=threshold)
    w_regime = {
        reg: fit_weights({m: f[f["origin"].isin(
            [o for o, lab in val_labels.items() if lab == reg])] for m, f in val.items()})
        for reg in ("calm", "stressed")
    }

    members = {m: frames[m] for m in MEMBERS}
    frames["Ensemble (static)"] = combine_forecasts(members, w_static, name="Ensemble (static)")
    frames["Ensemble (regime-aware)"] = combine_regime_aware(
        members, w_regime, threshold=threshold, name="Ensemble (regime-aware)"
    )

    live_labels = regime_labels(frames["LSTM"], threshold=threshold)
    n_stressed = sum(1 for v in live_labels.values() if v == "stressed")
    print(
        f"\nregime labels on live data at the FROZEN threshold "
        f"{threshold} EUR/MWh: stressed {n_stressed}/{len(live_labels)} "
        f"({100 * n_stressed / len(live_labels):.1f}%)"
    )
    if n_stressed / len(live_labels) > 0.9:
        print(
            "  NOTE: the frozen threshold classifies almost every live day as\n"
            "  stressed, so the regime switch has effectively degenerated to a\n"
            "  single weight set. That is itself an OOD finding — a threshold\n"
            "  calibrated on 2015 prices does not partition a 2026 market."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(OUT_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.csv", index=False)

    rows = [{"model": n, **_metrics(f)} for n, f in frames.items()]
    table = pd.DataFrame(rows).set_index("model")

    bench = _benchmark_reference()
    table = table.join(bench, rsuffix="_benchmark")
    table["MAE ratio"] = table["MAE"] / table["MAE_benchmark"]

    print(f"\nOOD stress test — live DE-LU, {len(Y)} days\n")
    print(table.round(3).to_string())
    print(
        "\nrMAE is the honest column: it rescales by a naive forecast fitted\n"
        "to the live data itself, so it separates 'the market got harder'\n"
        "from 'the frozen model got worse'."
    )

    table.to_csv(OUT_DIR / "ood_summary.csv")
    print(f"\nwrote {_rel(OUT_DIR / 'ood_summary.csv')}")


def _benchmark_reference() -> pd.DataFrame:
    """Benchmark-era test metrics for the same models, for the degradation
    ratio. Read from the committed canonical table, not re-derived."""
    path = REPO_ROOT / "reports" / "tables" / "results_canonical.csv"
    if not path.exists():
        return pd.DataFrame(columns=["MAE"]).rename_axis("model")
    canon = pd.read_csv(path)
    hourly = canon[canon["target"] == "hourly"].set_index("model")
    return hourly[["MAE", "rMAE"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="freeze benchmark-era models")
    parser.add_argument("--fetch", action="store_true", help="fetch + cache live data (network)")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=LIVE_CACHE,
        help="override the cached live-data CSV (for pipeline validation)",
    )
    args = parser.parse_args()

    if args.fit:
        fit_frozen()
    if args.fetch:
        end = args.end or str(pd.Timestamp.today().normalize().date())
        fetch_live(args.start, end)
    if not args.fit and not args.fetch:
        replay(cache=args.cache)


if __name__ == "__main__":
    main()
