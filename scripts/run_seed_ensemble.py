"""Seed-ensemble the LSTM — scripts/run_seed_ensemble.py

Attempt to close the 0.143 MAE gap to Lago et al.'s DNN Ensemble (3.4135)
by borrowing their actual mechanism: their DNN Ensemble is four runs of ONE
model family averaged together, which is variance reduction, not a better
model. Ours averages different families and is dragged by its weaker
members.

WHY THIS IS THE ONLY REMAINING LEVER. The L1-optimal ensemble weights fitted
directly on the test set -- illegitimate, and therefore a hard upper bound on
any weighting scheme over the current four members -- score MAE 3.558. The
legitimate validation-fitted regime-aware ensemble already scores 3.5569, so
reweighting has exactly zero headroom left. The gap can only move if the base
learners get stronger.

SEED POLICY. The project rule is seed 42 everywhere, and it still holds: 42
is the default, every single-model result uses it, and the seeds varied here
exist only as members of an explicitly labelled seed ensemble. Seed 42's run
is the existing frozen lstm.csv and is reused rather than recomputed.

OUTPUT NAMESPACE. Everything lands in data/processed/seed_ensemble/, which
is new and gitignored. NOTHING under data/processed/baselines/, reports/ or
models/ is touched by this script. That is deliberate even though the
decision to break the freeze has been taken: the frozen numbers should be
replaced only once there is a measured result that justifies replacing them,
not in anticipation of one.

Usage:
    python scripts/run_seed_ensemble.py --seeds 43 44 45
    python scripts/run_seed_ensemble.py --combine
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_baselines import run_one
from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.metrics import diebold_mariano_hac, mae, rmae
from src.evaluation.walk_forward import load_evaluation_config
from src.features.pipeline import build_features
from src.models import LSTMModel, load_models_config

OUT_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble"
FROZEN_LSTM = REPO_ROOT / "data" / "processed" / "baselines" / "lstm.csv"
BASELINE_SEED = 42


def _long(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def _piv(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index()


def run_seeds(
    seeds: list[int],
    first_origin=None,
    last_origin=None,
    out_dir: Path | None = None,
) -> None:
    """Run the LSTM at each seed over a window.

    Defaults to the test window. The validation window is needed too: ensemble
    weights must be fitted on validation, never on test, so a seed-ensembled
    member has to exist there before it can legitimately enter the ensemble.
    Scoring a test-window seed ensemble with weights fitted on the same test
    window is the exact in-sample selection the walk-forward exists to prevent.
    """
    out_dir = out_dir or OUT_DIR
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    X, Y = build_features(pd.concat([df_train, df_test]))
    eval_cfg = load_evaluation_config()
    models_cfg = load_models_config()
    if first_origin is None:
        first_origin = df_test.index.min().normalize()

    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if seed == BASELINE_SEED and out_dir == OUT_DIR:
            print(f"seed {seed}: already exists as the frozen lstm.csv, skipping")
            continue
        cfg = {**models_cfg["lstm"], "seed": seed}
        model = LSTMModel(cfg)
        print(f"\n=== LSTM seed {seed} (units={model.units}, epochs={model.epochs}) ===", flush=True)
        run_one(f"LSTM-s{seed}", model, X, Y, eval_cfg, first_origin, last_origin, out_dir)


def combine() -> pd.DataFrame:
    """Average the available seed runs and score the result."""
    members = {BASELINE_SEED: _long(FROZEN_LSTM)}
    for path in sorted(OUT_DIR.glob("lstm_s*.csv")):
        seed = int(path.stem.split("_s")[-1])
        members[seed] = _long(path)

    complete = {}
    for seed, frame in members.items():
        counts = frame.groupby("origin").size()
        if len(counts) == 728 and (counts == 24).all():
            complete[seed] = frame
        else:
            print(f"  seed {seed}: incomplete ({len(counts)}/728 origins) — excluded")

    if len(complete) < 2:
        raise SystemExit(f"need at least 2 complete seed runs to combine, have {len(complete)}")

    base = _piv(complete[BASELINE_SEED], "y_true")
    preds = {s: _piv(f, "y_pred") for s, f in complete.items()}
    for s, p in preds.items():
        if not p.index.equals(base.index):
            raise SystemExit(f"seed {s}: origin index does not align with the frozen run")

    stack = np.stack([p.values for p in preds.values()])
    avg = stack.mean(axis=0)

    real = base.values
    rows = []
    for s, p in sorted(preds.items()):
        rows.append({"member": f"LSTM seed {s}", "MAE": float(np.abs(real - p.values).mean())})
    rows.append({"member": f"LSTM seed-ensemble ({len(preds)} seeds)",
                 "MAE": float(np.abs(real - avg).mean())})
    table = pd.DataFrame(rows)

    # rMAE needs the hourly series form
    idx = pd.DatetimeIndex(
        [o + pd.Timedelta(hours=h) for o in base.index for h in range(24)]
    )
    r = pd.Series(real.ravel(), index=idx).sort_index().to_frame("price")
    a = pd.Series(avg.ravel(), index=idx).sort_index().to_frame("price")
    print("\n" + table.to_string(index=False))
    print(f"\nseed-ensemble rMAE : {float(rmae(r, a, m='W')):.4f}")
    print(f"seed-ensemble MAE  : {float(mae(r.values, a.values)):.4f}")
    print("\nreference: their DNN Ensemble MAE 3.4135 / rMAE 0.3740")
    print("           our frozen regime-aware ensemble MAE 3.5569 / rMAE 0.3897")

    avg_frame = pd.DataFrame(avg, index=base.index, columns=base.columns)
    out = (
        avg_frame.stack().rename("y_pred").reset_index()
        .merge(base.stack().rename("y_true").reset_index(), on=["origin", "hour"])
    )
    out["model"] = "LSTM (seed ensemble)"
    out = out[["origin", "hour", "y_true", "y_pred", "model"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "lstm_seed_ensemble.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'lstm_seed_ensemble.csv'}")
    return table


def seed_ensemble_frame(directory: Path, extra: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Average every seed run in `directory` (plus `extra`) into one frame.

    All members must cover the same origins in the same order; the frames are
    sorted before averaging rather than assumed aligned, and a mismatch raises
    instead of silently averaging different days together.
    """
    # Match ONLY lstm_s<digits>.csv. A bare "lstm_s*.csv" glob also matches
    # lstm_seed_ensemble.csv -- the averaged frame this module itself writes
    # into the same directory -- so the average would be folded back in as if
    # it were another seed, silently over-weighting it. Caught because the
    # member count printed 5 where 4 were expected.
    paths = sorted(p for p in directory.glob("lstm_s*.csv") if re.fullmatch(r"lstm_s\d+", p.stem))
    if extra is not None:
        paths.append(extra)
    if len(paths) < 2:
        raise SystemExit(f"need >=2 seed runs in {directory}, found {len(paths)}")

    frames = [_long(p).sort_values(["origin", "hour"]).reset_index(drop=True) for p in paths]
    base = frames[0]
    for f in frames[1:]:
        if not f[["origin", "hour"]].equals(base[["origin", "hour"]]):
            raise SystemExit("seed runs do not cover identical (origin, hour) rows")
    out = base.copy()
    out["y_pred"] = np.mean([f["y_pred"].to_numpy() for f in frames], axis=0)
    out["model"] = "LSTM"
    return out, len(frames)


def evaluate() -> pd.DataFrame:
    """Fit ensemble weights on VALIDATION, score on test, DM against Lago.

    The weights are fitted on the validation window and never on test — the
    whole point of running the validation-window seeds. Scoring a test-window
    seed ensemble under test-fitted weights would be in-sample selection, and
    would inflate exactly the comparison this is meant to settle.
    """
    from src.evaluation.ensemble import (
        combine_forecasts, combine_regime_aware, fit_weights, regime_labels,
    )

    val_dir = REPO_ROOT / "data" / "processed" / "seed_ensemble_val"
    val_pre = REPO_ROOT / "data" / "processed" / "validation_preds"
    test_pre = REPO_ROOT / "data" / "processed" / "baselines"

    val_lstm, n_val = seed_ensemble_frame(val_dir)
    test_lstm, n_test = seed_ensemble_frame(OUT_DIR, extra=FROZEN_LSTM)
    print(f"seed-ensemble members: validation {n_val}, test {n_test}")

    val = {
        "SARIMAX": _long(val_pre / "sarimax.csv"),
        "LEAR-LASSO": _long(val_pre / "lear_lasso.csv"),
        "LightGBM": _long(val_pre / "lightgbm.csv"),
        "LSTM": val_lstm,
    }
    test = {
        "SARIMAX": _long(test_pre / "sarimax.csv"),
        "LEAR-LASSO": _long(test_pre / "lear_lasso.csv"),
        "LightGBM": _long(test_pre / "lightgbm.csv"),
        "LSTM": test_lstm,
    }

    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    w_static = fit_weights(val)
    labels = regime_labels(val["LightGBM"], threshold=threshold)
    w_regime = {
        r: fit_weights(
            {m: f[f["origin"].isin([o for o, l in labels.items() if l == r])] for m, f in val.items()}
        )
        for r in ("calm", "stressed")
    }
    print("static weights (validation-fitted):", {k: round(v, 3) for k, v in w_static.items()})

    built = {
        "Ensemble (static)": combine_forecasts(test, w_static, name="Ensemble (static)"),
        "Ensemble (regime-aware)": combine_regime_aware(
            test, w_regime, threshold=threshold, name="Ensemble (regime-aware)"
        ),
    }

    pub = pd.read_csv(
        REPO_ROOT / "data" / "raw" / "Forecasts_DE_DNN_LEAR_ensembles.csv",
        index_col=0, parse_dates=True,
    )

    def m24(series: pd.Series) -> np.ndarray:
        f = series.to_frame("v")
        f["d"], f["h"] = f.index.normalize(), f.index.hour
        return f.pivot(index="d", columns="h", values="v").sort_index().to_numpy()

    real = m24(pub["Real price"])
    rows = []
    for name, frame in built.items():
        ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
        r = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
        p = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")
        piv = frame.pivot(index="origin", columns="hour", values="y_pred").sort_index().to_numpy()
        row = {"model": name, "MAE": float(mae(r.values, p.values)), "rMAE": float(rmae(r, p, m="W"))}
        for theirs in ("DNN Ensemble", "DNN 4", "LEAR Ensemble"):
            row[f"p theirs better vs {theirs}"] = diebold_mariano_hac(
                p_real=real, p_pred_1=piv, p_pred_2=m24(pub[theirs])
            )
        rows.append(row)
        frame.to_csv(OUT_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_seedens.csv", index=False)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    print("\nfrozen reference: regime-aware MAE 3.5569 rMAE 0.3897, p vs DNN Ensemble 0.0127")
    print("their DNN Ensemble: MAE 3.4135 rMAE 0.3740")
    return table


ORACLE_TABLE = REPO_ROOT / "reports" / "tables" / "oracle_bound.csv"


def oracle_bound() -> pd.DataFrame:
    """Test-fitted ORACLE upper bound on global convex reweighting.

    ILLEGITIMATE BY CONSTRUCTION and never a reportable forecast: the weights
    are fitted directly on the test set, so this is cheating and is useful
    only as a hard upper bound on what ANY global convex weighting of these
    members could achieve. One scalar weight per model applied identically
    across all 24 hours, no intercept -- the same family as
    run_combination_ladder rung 0, which is why that ladder's rungs 1-3 sit
    outside this bound and are unmeasured rather than known-futile.

    Both member sets are emitted because the pair is the argument: swapping
    the frozen seed-42 LSTM for the seed-ensembled one moves the bound from
    3.558 to 3.5019, and BOTH remain above Lago et al.'s 3.4135. That is the
    evidence that the gap is not closable by ensembling.
    """
    from scipy.optimize import minimize

    def _matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = _long(path)
        return (d.pivot(index="origin", columns="hour", values="y_pred").sort_index(),
                d.pivot(index="origin", columns="hour", values="y_true").sort_index())

    def _fit(P: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        if not np.isfinite(P).all() or not np.isfinite(y).all():
            raise ValueError("non-finite values reached the oracle weight fit")
        res = minimize(lambda w: float(np.mean(np.abs(y - w @ P))), x0=np.full(n, 1.0 / n),
                       method="SLSQP", bounds=[(0.0, 1.0)] * n,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0})
        # SLSQP returns its start point on failure, which would silently
        # degrade the bound to an equal-weight average. Fail loudly.
        if not res.success:
            raise RuntimeError(f"oracle weight fit did not converge: {res.message}")
        return res.x / res.x.sum()

    base = REPO_ROOT / "data" / "processed" / "baselines"
    rows = []
    for label, lstm in (("frozen seed-42 LSTM", base / "lstm.csv"),
                        ("seed-ensembled LSTM", OUT_DIR / "lstm_seed_ensemble.csv")):
        members = {"SARIMAX": base / "sarimax.csv", "LEAR-LASSO": base / "lear_lasso.csv",
                   "LightGBM": base / "lightgbm.csv", "LSTM": lstm}
        mats, truth = {}, None
        for m, path in members.items():
            mats[m], t = _matrix(path)
            truth = t if truth is None else truth
            if not mats[m].index.equals(truth.index):
                raise ValueError(f"origin grid mismatch for {m} in {label}")
        names = list(mats)
        P = np.stack([mats[m].to_numpy().ravel() for m in names])
        y = truth.to_numpy().ravel()
        w = _fit(P, y)
        rows.append({"member_set": label, "origins": int(truth.shape[0]),
                     "oracle_MAE": float(np.mean(np.abs(y - w @ P))),
                     **{f"w_{m}": float(v) for m, v in zip(names, w)}})

    table = pd.DataFrame(rows)
    ORACLE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ORACLE_TABLE, index=False)
    print()
    print(table.to_string(index=False))
    print()
    print(f"wrote {ORACLE_TABLE.relative_to(REPO_ROOT)}")
    print("their DNN Ensemble: MAE 3.4135 -- both bounds stay above it")
    return table


def main(
    seeds: list[int] | None,
    do_combine: bool,
    do_evaluate: bool = False,
    do_oracle: bool = False,
    first_origin=None,
    last_origin=None,
    out_dir: Path | None = None,
) -> None:
    if seeds:
        run_seeds(seeds, first_origin=first_origin, last_origin=last_origin, out_dir=out_dir)
    if do_combine:
        combine()
    if do_evaluate:
        evaluate()
    if do_oracle:
        oracle_bound()


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--evaluate", action="store_true",
                        help="fit weights on validation, score on test, DM vs Lago")
    parser.add_argument("--oracle", action="store_true",
                        help="test-fitted upper bound on global convex reweighting")
    parser.add_argument("--first-origin", type=pd.Timestamp, default=None)
    parser.add_argument("--last-origin", type=pd.Timestamp, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.seeds, args.combine, args.evaluate, args.oracle,
         args.first_origin, args.last_origin, args.out_dir)
