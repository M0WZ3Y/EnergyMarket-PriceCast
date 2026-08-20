"""LEAR multi-window sweep — scripts/run_lear_windows.py

Replicates Lago et al. (2021)'s LEAR Ensemble construction on our LEAR-LASSO:
run the model at their four calibration windows and combine by ARITHMETIC
MEAN. Governed by the pre-registration in logs/decisions.md (2026-08-20),
which declares a two-test family and the adoption rule; do not change the
windows or the combiner here without a new dated entry.

WHY THE ARITHMETIC MEAN AND NOT FITTED WEIGHTS. Two reasons, and the second
matters more. First, it is what the paper does, so it is a replication rather
than a new design. Second, it has zero free parameters, which means there is
nothing to fit on validation and therefore nothing to leak — the combination
ladder run earlier today showed exactly what fitted per-hour weights do to
out-of-sample accuracy on this data.

THE WINDOW MUST BE SET IN TWO PLACES. `walk_forward_splits` sizes each
split's `train_days` from evaluation config's
walk_forward.calibration_window_days, while `LEARLassoModel` reads its own
cfg's calibration_window_days and passes it to epftoolbox's LEAR. Setting
only one of them hands LEAR a different amount of history than it believes it
has, silently. `_window_cfg` below sets both from a single value.

NOT ALL OF LAGO'S WINDOWS ARE REACHABLE HERE, for two independent reasons
(logs/decisions.md 2026-08-20):

  56, 84   LassoLarsIC in scikit-learn 1.9 refuses n_samples < n_features,
           and the feature matrix has 247 columns. Structurally infeasible.
  1092     the frozen arm, 728 origins.
  1456     runs, but reaches only 721 of the 728 origins. The raw span starts
           2012-01-09 while build_features consumes 7 days building lags, so
           the usable matrix starts 2012-01-16 and just 1449 days precede the
           first test origin -- seven short of 1456. An earlier version of
           this docstring claimed 1456 fit "with zero slack"; that was
           measured from the RAW start and was wrong.

`_common_origins` therefore intersects the arms rather than assuming they
align, and says so loudly: any metric it produces is NOT comparable to the
frozen 728-origin results table.

SEED 42 unchanged. The 1092 arm is the frozen lear_lasso.csv and is REUSED,
never recomputed.

Usage:
    python scripts/run_lear_windows.py --probe        # time a few origins
    python scripts/run_lear_windows.py --windows 56 84 1456
    python scripts/run_lear_windows.py --combine
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_baselines import run_one
from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.metrics import diebold_mariano_hac, mae, rmae, rmse, smape
from src.evaluation.walk_forward import load_evaluation_config
from src.features.pipeline import build_features
from src.models import LEARLassoModel, load_models_config

OUT_DIR = REPO_ROOT / "data" / "processed" / "lear_windows"
FROZEN_LEAR = REPO_ROOT / "data" / "processed" / "baselines" / "lear_lasso.csv"
PUBLISHED_FC = REPO_ROOT / "data" / "raw" / "Forecasts_DE_DNN_LEAR_ensembles.csv"

# Lago et al. (2021) §4.4: 8 weeks, 12 weeks, 3 years, 4 years.
LAGO_WINDOWS = (56, 84, 1092, 1456)
FROZEN_WINDOW = 1092
N_ORIGINS = 728


def _long(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def _window_cfg(window: int) -> dict:
    """Evaluation config with the walk-forward window overridden.

    Deep-copied: load_evaluation_config may return a cached/shared dict, and
    mutating it would leak this sweep's window into every later caller in the
    same process — including the frozen-window arms of this very script.
    """
    cfg = copy.deepcopy(load_evaluation_config())
    cfg["walk_forward"]["calibration_window_days"] = int(window)
    return cfg


def _path(window: int) -> Path:
    return OUT_DIR / f"lear_w{window}.csv"


def run_windows(windows: list[int], probe: int | None = None) -> None:
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    X, Y = build_features(pd.concat([df_train, df_test]))
    models_cfg = load_models_config()
    first_origin = df_test.index.min().normalize()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for window in windows:
        if window == FROZEN_WINDOW:
            print(f"window {window}: already exists as the frozen lear_lasso.csv, skipping")
            continue
        eval_cfg = _window_cfg(window)
        cfg = {**models_cfg["lear_lasso"], "calibration_window_days": int(window)}
        model = LEARLassoModel(cfg)

        last_origin = None
        if probe:
            origins = pd.DatetimeIndex(sorted(Y.index.normalize().unique()))
            origins = origins[origins >= first_origin]
            last_origin = origins[probe - 1]
            print(f"\n--- PROBE: window {window}, {probe} origins ---", flush=True)

        t0 = time.monotonic()
        run_one(f"LEAR-w{window}", model, X, Y, eval_cfg, first_origin,
                last_origin, OUT_DIR if not probe else OUT_DIR / "_probe")
        dt = time.monotonic() - t0
        if probe:
            per = dt / probe
            print(f"window {window}: {dt:.1f}s for {probe} origins "
                  f"({per:.2f}s/origin) -> full {N_ORIGINS} origins "
                  f"~= {per * N_ORIGINS / 60:.1f} min", flush=True)


def _common_origins(frames: dict[int, pd.DataFrame]) -> pd.DatetimeIndex:
    """Intersect the origin sets and report the restriction loudly.

    Averaging frames over different origins would produce a mean silently
    computed from different days per member — the arithmetic mean gives no
    warning, it just returns a number. So every arm is restricted to the
    common origins instead.

    The restriction is real and not cosmetic: window 1456 cannot reach the
    first 7 test origins, because build_features consumes the first 7 days of
    the raw span building lags and only 1449 days remain before 2016-01-04
    (logs/decisions.md 2026-08-20 correction). Any metric computed here is
    therefore NOT comparable to the frozen 728-origin results table.
    """
    common = None
    for w, f in sorted(frames.items()):
        origins = pd.DatetimeIndex(sorted(f["origin"].unique()))
        counts = f.groupby("origin").size()
        if not (counts == 24).all():
            raise SystemExit(f"window {w}: some origins do not have 24 hours")
        print(f"  window {w}: {len(origins)} origins "
              f"({origins[0].date()} to {origins[-1].date()})")
        common = origins if common is None else common.intersection(origins)
    if len(common) < 2:
        raise SystemExit("windows share too few origins to combine")
    if len(common) != N_ORIGINS:
        print(f"  RESTRICTED to {len(common)} common origins "
              f"(frozen table uses {N_ORIGINS}) — these numbers are NOT "
              "comparable to the frozen results table")
    return common


def _restrict(frame: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    return frame[frame["origin"].isin(origins)].sort_values(["origin", "hour"]).reset_index(drop=True)


def _metrics(frame: pd.DataFrame) -> dict:
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    pred = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")
    return {
        "MAE": float(mae(real.values, pred.values)),
        "RMSE": float(rmse(real.values, pred.values)),
        "sMAPE": float(smape(real.values, pred.values) * 100),
        "rMAE": float(rmae(real, pred, m="W")),
    }


def _piv(frame: pd.DataFrame, col: str = "y_pred") -> np.ndarray:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index().to_numpy()


def _m24(series: pd.Series) -> np.ndarray:
    f = series.to_frame("v")
    f["d"], f["h"] = f.index.normalize(), f.index.hour
    return f.pivot(index="d", columns="h", values="v").sort_index().to_numpy()


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def combine() -> pd.DataFrame:
    frames = {FROZEN_WINDOW: _long(FROZEN_LEAR)}
    for w in LAGO_WINDOWS:
        if w == FROZEN_WINDOW:
            continue
        p = _path(w)
        if not p.exists():
            raise SystemExit(f"missing window {w}: run --windows {w} first")
        frames[w] = _long(p)

    origins = _common_origins(frames)
    frames = {w: _restrict(f, origins) for w, f in frames.items()}
    truth = frames[FROZEN_WINDOW].pivot(index="origin", columns="hour",
                                        values="y_true").sort_index()
    stack = np.stack([_piv(frames[w]) for w in sorted(frames)])
    avg = stack.mean(axis=0)

    ens = pd.DataFrame(avg, index=truth.index, columns=truth.columns)
    ens_long = (
        ens.stack().rename("y_pred").reset_index()
        .merge(truth.stack().rename("y_true").reset_index(), on=["origin", "hour"])
    )
    ens_long["model"] = "LEAR window-ensemble"
    ens_long = ens_long[["origin", "hour", "y_true", "y_pred", "model"]]
    ens_long.to_csv(OUT_DIR / "lear_window_ensemble.csv", index=False)

    rows = []
    for w in sorted(frames):
        tag = " (frozen)" if w == FROZEN_WINDOW else ""
        rows.append({"model": f"LEAR window {w}{tag}", **_metrics(frames[w])})
    rows.append({"model": f"LEAR window-ensemble ({len(frames)} windows)",
                 **_metrics(ens_long)})

    pub = pd.read_csv(PUBLISHED_FC, index_col=0, parse_dates=True)
    for name in ("LEAR 1092", "LEAR Ensemble"):
        if name in pub.columns:
            real = pub["Real price"].to_frame("price").sort_index()
            pred = pub[name].to_frame("price").sort_index()
            rows.append({
                "model": f"Lago {name} (recomputed)",
                "MAE": float(mae(real.values, pred.values)),
                "RMSE": float(rmse(real.values, pred.values)),
                "sMAPE": float(smape(real.values, pred.values) * 100),
                "rMAE": float(rmae(real, pred, m="W")),
            })

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    # The two pre-registered tests, one-sided in the direction that the
    # COMPARATOR is more accurate, so a small p counts against the ensemble.
    pub = pub[pub.index.normalize().isin(origins)]
    real24 = _m24(pub["Real price"])
    ours = _piv(ens_long)
    raw = [
        float(diebold_mariano_hac(p_real=real24, p_pred_1=ours,
                                  p_pred_2=_piv(frames[FROZEN_WINDOW]))),
        float(diebold_mariano_hac(p_real=real24, p_pred_1=ours,
                                  p_pred_2=_m24(pub["LEAR Ensemble"]))),
    ]
    adj = holm(raw)
    print("\npre-registered family (2 tests, Holm-corrected):")
    for label, r, a in zip(
        ("window-ensemble vs our LEAR-1092", "window-ensemble vs their LEAR Ensemble"),
        raw, adj,
    ):
        print(f"  p(comparator better) {label}: raw {r:.4f}  Holm {a:.4f}")

    base = table.loc[table["model"].str.contains("frozen"), "MAE"].iloc[0]
    ens_mae = table.loc[table["model"].str.startswith("LEAR window-ensemble"), "MAE"].iloc[0]
    print(f"\nMAE {base:.4f} -> {ens_mae:.4f}  (delta {base - ens_mae:+.4f}, "
          f"floor 0.02)")
    return table


def main(windows: list[int] | None, do_combine: bool, probe: int | None) -> None:
    if windows:
        run_windows(windows, probe=probe)
    if do_combine:
        combine()


if __name__ == "__main__":
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="*", default=None)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--probe", type=int, default=None,
                        help="run only N origins and report timing")
    args = parser.parse_args()
    main(args.windows, args.combine, args.probe)
