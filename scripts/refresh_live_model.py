"""Fit the validation-selected v2 model on the latest data — scripts/refresh_live_model.py

Reads data/processed/live_v2/selection.json (written by run_live_ladder.py
score, validation-only), fetches recent ENTSO-E days into the gitignored live
cache, fits every member the served configuration needs on its trailing
calibration window ending at the last day with complete prices, and writes
models/live/ + models/live/served.json. Run it weekly (the app warns after
7 days); daily matches the evaluated R3/R4 protocol most closely.

    ./.venv/Scripts/python.exe scripts/refresh_live_model.py [--offline]
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.live_service import fetch_recent, history_frame, member_dir  # noqa: E402
from src.evaluation import live_ladder as LL  # noqa: E402
from src.evaluation.walk_forward import load_evaluation_config  # noqa: E402
from src.features.pipeline import build_features  # noqa: E402
from src.models import LEARLassoModel, LightGBMModel, LSTMModel, NaiveModel, SARIMAXModel, load_models_config  # noqa: E402

SELECTION = REPO_ROOT / "data" / "processed" / "live_v2" / "selection.json"
LIVE_TUNED = REPO_ROOT / "configs" / "tuned" / "live"
CLASSES = {"naive": (NaiveModel, "naive"), "SARIMAX": (SARIMAXModel, "sarimax"),
           "LEAR-LASSO": (LEARLassoModel, "lear_lasso"), "LightGBM": (LightGBMModel, "lightgbm"),
           "LSTM": (LSTMModel, "lstm")}


def plan(selection: dict, variant: str = "shadow") -> dict:
    """What to fit: members, windows, weights, threshold, bias window."""
    served = selection["served_exante" if variant == "exante" else "served"]
    rung, model = served["rung"], served["model"]
    if rung.rstrip("x") == "R5":
        # leakage review 2026-10-16, F4: R5/R5x were evaluated on NATIONAL load +
        # residual load, but serving builds the default features on Amprion load.
        # A single R5 SARIMAX would silently run on the wrong scale (D2 again).
        raise SystemExit(
            f"served rung {rung} has no serving path yet (national-load inputs, live A65 DE-LU fetch); "
            "refusing to fit it on the default Amprion-load features."
        )
    if rung in ("R1b", "R2"):
        raise SystemExit(
            f"served rung {rung} uses the FROZEN benchmark-era models; there is nothing to refit. "
            "Copy models/frozen/ members into models/live/ manually if this is really intended."
        )
    if model in LL.ENSEMBLES:
        spec = selection["ensembles"][rung]
        kind = "regime" if model == "Ensemble (regime-aware)" else "static"
        names = LL.MEMBERS
        if model == "Ensemble (mean)":
            weights = {m: 1.0 for m in names}
        else:
            weights = spec["static"] if kind == "static" else spec["regime"]
        threshold = spec["threshold"]
    else:
        kind, names, weights, threshold = "single", [model], {}, None
    return {"rung": rung, "model": model, "kind": kind, "names": names, "weights": weights, "threshold": threshold}


def fit_member(name: str, rung: str, X, Y, last_day, window: int | None, variant: str = "shadow"):
    cls, key = CLASSES[name]
    cfg = copy.deepcopy(load_models_config()[key])
    if variant == "exante" and name == "SARIMAX":
        cfg["exog_columns_prefix"] = list(LL.EXANTE_SARIMAX_EXOG)
    if rung.rstrip("x") == "R4" and name in ("LightGBM", "LSTM"):
        cfg["tuned_params_file"] = str(LIVE_TUNED / f"{key}_params.yaml")
    calib = window or int(load_evaluation_config()["walk_forward"]["calibration_window_days"])
    if window is not None:
        cfg["calibration_window_days"] = window
    days = Y.index[Y.index <= last_day][-calib:]
    if len(days) < calib:
        raise SystemExit(f"{name}: only {len(days)} of {calib} training days available")
    print(f"  fit {name}{'' if window is None else f' w={window}'} on {days.min().date()} -> {days.max().date()}", flush=True)
    return cls(cfg).fit(X.loc[days], Y.loc[days])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the live fetch; use snapshot + cache only")
    ap.add_argument("--variant", choices=["exante", "shadow"], default="shadow",
                    help="exante: no D+1 wind/solar input, can run before the noon auction")
    ap.add_argument("--provisional", choices=list(CLASSES),
                    help="serve this model with the R3 protocol BEFORE selection.json exists; "
                         "labelled provisional everywhere it is shown")
    args = ap.parse_args()

    rung0 = "R3x" if args.variant == "exante" else "R3"
    if args.provisional:
        selection = {"served": {"rung": rung0, "model": args.provisional, "val_mae": None}}
        selection["served_exante"] = selection["served"]
        p = {"rung": rung0, "model": args.provisional, "kind": "single", "names": [args.provisional],
             "weights": {}, "threshold": None}
    else:
        selection = json.loads(SELECTION.read_text())
        p = plan(selection, args.variant)
    print(f"serving {p['rung']} / {p['model']} ({p['kind']})")
    if not args.offline:
        fetch_recent(days_back=21)
    df = history_frame()
    X, Y = build_features(df)
    if args.variant == "exante":
        X = LL.exante(X)
    last_day = Y.index.max()  # last day whose 24 prices are all known
    print(f"history through {df.index.max()}, last complete price day {last_day.date()}")

    out_dir = REPO_ROOT / "models" / "live" / args.variant
    tmp = out_dir.with_name(f"{args.variant}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    member_dirs = {}
    for name in p["names"]:
        windows = list(LL.LEAR_WINDOWS) if name == "LEAR-LASSO" else [None]
        member_dirs[name] = []
        for w in windows:
            m = fit_member(name, p["rung"], X, Y, last_day, w, args.variant)
            d = member_dir(name, w)
            m.save(tmp / d)
            member_dirs[name].append(d)
    meta = {**{k: p[k] for k in ("rung", "model", "kind", "weights", "threshold")},
            "member_dirs": member_dirs, "fit_end": str(last_day.date()),
            "fitted_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
            "selection_val_mae": selection["served_exante" if args.variant == "exante" else "served"]["val_mae"],
            "variant": args.variant,
            "provisional": bool(args.provisional)}
    (tmp / "served.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
