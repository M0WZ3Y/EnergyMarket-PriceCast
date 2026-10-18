"""v2 live ladder — scripts/run_live_ladder.py

Spec: docs/superpowers/specs/2026-10-15-live-entsoe-recalibration-design.md §6.

    frozen                     R0 (Energy-Charts as served, UTC), R1a (same
                               data, local time), R1b (ENTSO-E snapshot).
                               Frozen benchmark-era models, no refit.
    recal --rung R3|R4 --model NAME [--window W]
                               rolling recalibration over validation + test
                               (run_full_baselines.run_one; resumable).
    combine-lear --rung R3     arithmetic mean of the LEAR window arms.
    score                      R2, R3/R4 ensembles, served-model selection
                               (validation only, written FIRST), then the
                               test-window ladder table and DM tests.

All outputs go under data/processed/live_v2/ and reports/tables/live_ladder*.
Nothing frozen is written: models/frozen/ only gains files that were missing
(naive, SARIMAX), fitted exactly as run_ood_stress.fit_frozen fits them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.entsoe import utc_hourly_frame_to_local  # noqa: E402
from src.data.sources.provenance import verify_snapshot  # noqa: E402
from src.evaluation import live_ladder as LL  # noqa: E402
from src.evaluation.walk_forward import load_evaluation_config  # noqa: E402
from src.features.pipeline import build_features  # noqa: E402
from src.models import load_models_config  # noqa: E402

LIVE_V2_RAW = REPO_ROOT / "data" / "raw" / "live_v2"
OUT = REPO_ROOT / "data" / "processed" / "live_v2"
TABLES = REPO_ROOT / "reports" / "tables"
LIVE_TUNED = REPO_ROOT / "configs" / "tuned" / "live"
FILL = {"price": "interpolate", "exog_1": "ffill", "exog_2": "interpolate"}

MODEL_CFG_KEY = {"naive": "naive", "SARIMAX": "sarimax", "LEAR-LASSO": "lear_lasso",
                 "LightGBM": "lightgbm", "LSTM": "lstm"}


# -- data ---------------------------------------------------------------------------
def load_entsoe() -> pd.DataFrame:
    rec = verify_snapshot("entsoe_de_2019_2026_sdac", snapshot_dir=LIVE_V2_RAW)
    return pd.read_csv(LIVE_V2_RAW / rec.filename, index_col=0, parse_dates=True)


def load_extra() -> pd.DataFrame:
    rec = verify_snapshot("entsoe_extra_2019_2026", snapshot_dir=LIVE_V2_RAW)
    return pd.read_csv(LIVE_V2_RAW / rec.filename, index_col=0, parse_dates=True)


def rung_inputs(base_rung: str) -> tuple[pd.DataFrame, dict | None]:
    """(hourly frame, feature config) for a rung's information set.

    D   REALIZED Amprion load and wind/solar in place of the forecasts: the
        perfect-input ceiling. Diagnostic only; never a selection candidate.
    R5  national load forecast as exog_1, plus the residual-load block
        (national load - national wind/solar), now physically consistent.
    """
    df = load_entsoe()
    if base_rung == "D":
        ex = load_extra()
        df = df.assign(exog_1=ex["load_act_amprion"], exog_2=ex["ws_act"])
        return df, None
    if base_rung == "R5":
        from src.features.pipeline import load_feature_config

        ex = load_extra()
        df = df.assign(exog_1=ex["load_fc_national"])
        cfg = copy.deepcopy(load_feature_config())
        cfg["physical_blocks"] = {**(cfg.get("physical_blocks") or {}), "residual_load_block": True}
        return df, cfg
    return df, None


def load_ec_as_served() -> pd.DataFrame:
    """v1.1-ood cache + the pinned extension, UTC, original rows win on overlap."""
    orig = pd.read_csv(REPO_ROOT / "data" / "raw" / "live_ood_de.csv", index_col=0, parse_dates=True)
    rec = verify_snapshot("ec_as_served_ext", snapshot_dir=LIVE_V2_RAW)
    ext = pd.read_csv(LIVE_V2_RAW / rec.filename, index_col=0, parse_dates=True)
    df = pd.concat([orig, ext])
    return df[~df.index.duplicated(keep="first")].sort_index()[["price", "exog_1", "exog_2"]]


def _write(rung: str, frames: dict[str, pd.DataFrame]) -> None:
    d = OUT / rung
    d.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_csv(d / f"{LL.slug(name)}.csv", index=False)


def _read(rung: str, name: str) -> pd.DataFrame | None:
    p = OUT / rung / f"{LL.slug(name)}.csv"
    if not p.exists():
        return None
    f = LL.read_frame(p)
    f["model"] = name
    return f


# -- frozen rungs ------------------------------------------------------------------------
def _ensure_frozen() -> dict:
    """Load frozen models; fit ONLY missing ones, exactly as fit_frozen does."""
    from scripts.run_ood_stress import FROZEN_DIR, _build_models
    from src.data.loader import BenchmarkLoader, load_config

    models = _build_models(load_models_config())
    missing = [n for n in models if not (FROZEN_DIR / n.lower().replace("-", "_")).exists()]
    if missing:
        print(f"fitting missing frozen models {missing} on the fit_frozen window")
        df_train, df_test = BenchmarkLoader(load_config()).load()
        X, Y = build_features(pd.concat([df_train, df_test]))
        calib = int(load_evaluation_config()["walk_forward"]["calibration_window_days"])
        train_days = Y.index[-calib:]
        meta = json.loads((FROZEN_DIR / "metadata.json").read_text())
        assert str(train_days.max().date()) == meta["frozen_on"], "frozen window drifted"
        for n in missing:
            models[n].fit(X.loc[train_days], Y.loc[train_days])
            models[n].save(FROZEN_DIR / n.lower().replace("-", "_"))
            print(f"  froze {n}")
    return {n: m.load(FROZEN_DIR / n.lower().replace("-", "_")) for n, m in models.items()}


def _frozen_predict(models: dict, df: pd.DataFrame, days_from, days_to) -> dict[str, pd.DataFrame]:
    X, Y = build_features(df)
    days = [d for d in Y.index if days_from <= d <= days_to]
    out = {}
    for name, model in models.items():
        preds = pd.concat([model.predict(X.loc[[d]]) for d in days])
        out[name] = LL.long_frame(name, Y, preds, days)
        print(f"    {name}: {len(days)} days", flush=True)
    return out


def _frozen_ensembles(members: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """v1 weights and v1 threshold, exactly as run_ood_stress.replay builds them."""
    from scripts.run_ood_stress import VAL_DIR, VAL_FILES
    from src.evaluation.ensemble import fit_weights, regime_labels
    from src.evaluation.results import load_long_frame

    val = {m: load_long_frame(VAL_DIR / f) for m, f in VAL_FILES.items()}
    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    labels = regime_labels(val["LightGBM"], threshold=threshold)
    spec = {
        "static": fit_weights(val),
        "regime": {reg: fit_weights({m: f[f["origin"].isin([o for o, lab in labels.items() if lab == reg])]
                                     for m, f in val.items()}) for reg in ("calm", "stressed")},
        "threshold": threshold,
    }
    return LL.apply_ensembles({m: members[m] for m in LL.MEMBERS}, spec)


def cmd_frozen(args) -> None:
    models = _ensure_frozen()
    arms = {  # lazy: each rung loads only its own inputs
        "R0": lambda: (load_ec_as_served(), LL.TEST_START, LL.TEST_END),
        "R1a": lambda: (utc_hourly_frame_to_local(load_ec_as_served(), FILL), LL.TEST_START, LL.TEST_END),
        "R1b": lambda: (load_entsoe(), LL.VAL_START, LL.TEST_END),
    }
    for rung in args.rungs:
        df, lo, hi = arms[rung]()
        if rung == "R0":
            # v1 built days from the UTC index; match it exactly (tz-aware day stamps)
            lo, hi = lo.tz_localize("UTC"), (hi + pd.Timedelta(hours=23)).tz_localize("UTC")
        print(f"[{rung}] {len(df)} rows", flush=True)
        frames = _frozen_predict(models, df, lo, hi)
        if rung == "R0":
            for f in frames.values():
                f["origin"] = pd.to_datetime(f["origin"]).dt.tz_localize(None)
        frames.update(_frozen_ensembles(frames))
        _write(rung, frames)
    if "R0" in args.rungs:
        reproduction_check()


def reproduction_check() -> None:
    """R0 on the original window must equal the frozen v1.1-ood forecasts."""
    ok = True
    for name in ["naive", "LEAR-LASSO", "LightGBM", "SARIMAX", "LSTM"]:
        new = _read("R0", name)
        old_p = REPO_ROOT / "data" / "processed" / "ood" / f"{LL.slug(name)}.csv"
        old = pd.read_csv(old_p, parse_dates=["origin"])
        old["origin"] = pd.to_datetime(old["origin"], utc=True).dt.tz_localize(None)
        m = old.merge(new, on=["origin", "hour"], suffixes=("_old", "_new"))
        diff = float((m.y_pred_old - m.y_pred_new).abs().max())
        same_truth = float((m.y_true_old - m.y_true_new).abs().max())
        status = "OK" if diff < 1e-6 and same_truth < 1e-9 else "DIFFERS"
        if status != "OK" and name not in ("LSTM", "SARIMAX"):
            ok = False
        print(f"  reproduction {name}: {m.origin.nunique()} common days, max |dpred|={diff:.2e} {status}")
    if not ok:
        raise SystemExit("R0 does not reproduce v1.1-ood for a deterministic model -- stop.")


# -- rolling recalibration -------------------------------------------------------------
def _recal_model(name: str, rung: str, window: int | None, exante: bool = False):
    from src.models import LEARLassoModel, LightGBMModel, LSTMModel, NaiveModel, SARIMAXModel

    cfg = copy.deepcopy(load_models_config()[MODEL_CFG_KEY[name]])
    if rung == "R4" and name in ("LightGBM", "LSTM"):
        tuned = LIVE_TUNED / f"{MODEL_CFG_KEY[name]}_params.yaml"
        if not tuned.exists():
            raise SystemExit(f"{tuned} missing -- run scripts/tune_live.py --model {MODEL_CFG_KEY[name]} first")
        cfg["tuned_params_file"] = str(tuned)
    if window is not None:
        cfg["calibration_window_days"] = window
    if exante and name == "SARIMAX":
        cfg["exog_columns_prefix"] = list(LL.EXANTE_SARIMAX_EXOG)
    cls = {"naive": NaiveModel, "SARIMAX": SARIMAXModel, "LEAR-LASSO": LEARLassoModel,
           "LightGBM": LightGBMModel, "LSTM": LSTMModel}[name]
    return cls(cfg)


def cmd_recal(args) -> None:
    from scripts.run_full_baselines import run_one
    from src.runtime import keep_awake

    exante = args.rung.endswith("x")
    base_rung = args.rung.rstrip("x")
    if base_rung == "R4" and args.model not in ("LightGBM", "LSTM"):
        raise SystemExit("R4 only reruns the Optuna-tuned models; the others are R3's")
    df, feat_cfg = rung_inputs(base_rung)
    X, Y = build_features(df, cfg=feat_cfg)
    if exante:
        X = LL.exante(X)
    eval_cfg = copy.deepcopy(load_evaluation_config())
    window = args.window
    if args.model == "LEAR-LASSO" and window is None:
        raise SystemExit(f"LEAR-LASSO needs --window, one of {LL.LEAR_WINDOWS}")
    if window is not None:
        eval_cfg["walk_forward"]["calibration_window_days"] = window  # set in BOTH places
    model = _recal_model(args.model, base_rung, window, exante=exante)
    label = args.model if window is None else f"{args.model}_w{window}"
    out_dir = OUT / args.rung
    out_dir.mkdir(parents=True, exist_ok=True)
    with keep_awake():
        run_one(label, model, X, Y, eval_cfg, LL.VAL_START, LL.TEST_END, out_dir)


def cmd_combine_lear(args) -> None:
    arms = []
    for w in LL.LEAR_WINDOWS:
        p = OUT / args.rung / f"lear_lasso_w{w}.csv"
        if not p.exists():
            raise SystemExit(f"missing LEAR arm {p}")
        arms.append(LL.read_frame(p))
    days = LL.common_days(arms)
    arms = [LL.restrict(a, days) for a in arms]
    out = arms[0][["origin", "hour", "y_true"]].copy()
    out["y_pred"] = np.mean([a["y_pred"].to_numpy() for a in arms], axis=0)
    out["model"] = "LEAR-LASSO"
    out.to_csv(OUT / args.rung / "lear_lasso.csv", index=False)
    print(f"LEAR-LASSO {args.rung}: mean of windows {LL.LEAR_WINDOWS} over {len(days)} days")


# -- scoring --------------------------------------------------------------------------
def _rung_frames(rung: str) -> dict[str, pd.DataFrame]:
    """Where each member of a rung lives on disk.

    R4/R4x re-tune only LightGBM and LSTM; their other members ARE R3's/R3x's
    forecasts. naive reads no exogenous input, so the ex-ante rungs reuse R3's.
    """
    base = {"R4": "R3", "R4x": "R3x"}.get(rung, rung)
    src = {name: base for name in LL.ALL_MODELS}
    if rung in ("R4", "R4x"):
        src.update({"LightGBM": rung, "LSTM": rung})
    if rung in ("R3x", "R4x", "R5", "R5x", "D"):
        src["naive"] = "R3"
    frames = {}
    for name in LL.ALL_MODELS:
        f = _read(src[name], name)
        if f is not None and _complete(f, f"{rung}/{name}"):
            frames[name] = f
    return frames


# A walk-forward still running leaves a partial CSV; scoring it would shrink
# every common-day intersection (and the test table) to its few days. Runs are
# written in date order, so an in-progress file always lacks its TAIL: the
# day-share test alone let a 96%-done file through (2026-10-16 g).
FULL_ORIGINS = None
# rungs whose inputs can leave days uncovered; never allowed to shrink the core day set
EXTRA_RUNGS = ("R5", "R5x", "D")
# selection candidates per information set. D (realized inputs) is NEVER one:
# tests/test_ladder_inputs.py checks these lists, which cmd_score uses directly.
SHADOW_CANDIDATES = ("R1b", "R2", "R3", "R4", "R5")
EXANTE_CANDIDATES = ("R3x", "R4x", "R5x")
# rungs refresh_live_model can actually serve today. R5/R5x need national-load
# inputs that serving does not build yet (leakage review F4): they are still
# scored and reported as best_overall, but never written as the served pick.
NOT_YET_SERVABLE = ("R5", "R5x")


def _complete(frame: pd.DataFrame, label: str, share: float = 0.95) -> bool:
    global FULL_ORIGINS
    if FULL_ORIGINS is None:
        FULL_ORIGINS = len(LL.common_days([_read("R3", "naive")])) if _read("R3", "naive") is not None else 0
    n = frame["origin"].nunique()
    last = LL.origins(frame).max()
    if (FULL_ORIGINS and n < share * FULL_ORIGINS) or last < LL.TEST_END:
        print(f"  [{label}] partial ({n}/{FULL_ORIGINS} days, last {last.date()}), excluded from scoring")
        return False
    return True


def cmd_score(args) -> None:
    from scripts.run_ood_stress import _metrics
    from src.evaluation.metrics import diebold_mariano_hac

    ladder: dict[str, dict[str, pd.DataFrame]] = {}
    for rung in ("R0", "R1a", "R1b"):
        ladder[rung] = {n: f for n in LL.ALL_MODELS + LL.ENSEMBLES if (f := _read(rung, n)) is not None}

    # R2: bias window per model chosen on validation, applied causally over val+test
    r2, r2_windows = {}, {}
    cv_val: dict[tuple[str, str], pd.DataFrame] = {}  # cross-fitted validation frames (F1)
    for name, f in ladder["R1b"].items():
        w, _ = LL.select_bias_window(LL.validation(f))
        r2_windows[name] = w
        r2[name] = LL.apply_bias_correction(f, w)
        cv_val[("R2", name)] = LL.crossfit_bias(LL.validation(f))
    ladder["R2"] = r2
    _write("R2", r2)

    # R3/R4: ensembles fitted on validation, threshold from the v2 history
    # Regime reference = the 364 days before validation (decision 2026-10-16 a): the
    # full 2019-25 history's std is inflated by the 2022 crisis and leaves no
    # feasible k. Same k grid and 20-day rule as v1; validation-only.
    hist = load_entsoe()["price"]
    hist = hist[(hist.index >= LL.VAL_START - pd.Timedelta(days=LL.REGIME_REFERENCE_DAYS)) & (hist.index <= LL.HISTORY_END)].dropna()
    ens_specs = {}
    for rung in ("R3", "R4", "R3x", "R4x", "R5", "R5x", "D"):
        frames = _rung_frames(rung)
        if not all(m in frames for m in LL.MEMBERS):
            # single models still count; only the ensembles need every member
            print(f"[{rung}] members {sorted(frames)}: ensembles skipped, single models kept")
            if frames:
                ladder[rung] = frames
            continue
        days = LL.common_days([frames[m] for m in LL.MEMBERS])
        members = {m: LL.restrict(frames[m], days) for m in LL.MEMBERS}
        val_members = {m: LL.validation(f) for m, f in members.items()}
        try:
            k, thr = LL.regime_threshold(hist, val_members["LightGBM"])
        except ValueError as exc:
            # no feasible regime split: the regime-aware ensemble is skipped for
            # this rung (mean and static still run), never silently degenerated
            print(f"[{rung}] regime-aware ensemble skipped: {exc}")
            k, thr = None, float("inf")
        if k is not None:
            spec = LL.fit_ensembles(val_members, threshold=thr)
        else:
            from src.evaluation.ensemble import fit_weights

            static = fit_weights(val_members, test_days=pd.DatetimeIndex([LL.TEST_START]))
            spec = {"static": static, "regime": {"calm": static, "stressed": static},
                    "threshold": thr, "skip_regime": True}
        spec["k"] = k
        ens_specs[rung] = spec
        ens = LL.apply_ensembles(members, spec)
        for n, fr in LL.crossfit_ensembles(val_members, thr).items():
            cv_val[(rung, n)] = fr
        if spec.get("skip_regime"):
            ens.pop("Ensemble (regime-aware)")
            cv_val.pop((rung, "Ensemble (regime-aware)"), None)
        frames.update(ens)
        ladder[rung] = frames
        _write(rung, {n: frames[n] for n in LL.ENSEMBLES if n in frames})

    # ---- SELECTION: validation only, written before any test metric exists
    # Within each information set every candidate is scored on the SAME validation
    # days (those all candidates cover), so a rung that misses a day neither gains
    # nor is silently dropped by select_served's full-coverage filter.
    # Fitted candidates (ensembles, R2) are scored on their CROSS-FITTED validation
    # frames, out-of-sample like every single model's (leakage review F1); the
    # in-sample score is kept alongside for transparency only.
    def val_table(rungs: list[str]) -> pd.DataFrame:
        cands = {(r, n): LL.validation(f) for r in rungs for n, f in ladder.get(r, {}).items()}
        if not cands:
            return pd.DataFrame(columns=["rung", "model", "val_mae", "val_mae_insample", "crossfit", "n_days"])
        honest = {k: cv_val.get(k, v) for k, v in cands.items()}
        vdays = LL.common_days(list(honest.values()))
        return pd.DataFrame([{"rung": r, "model": n, "val_mae": LL.mae_of(LL.restrict(honest[(r, n)], vdays)),
                              "val_mae_insample": LL.mae_of(LL.restrict(v, vdays)),
                              "crossfit": (r, n) in cv_val, "n_days": len(vdays)}
                             for (r, n), v in cands.items()])

    shadow_scores = val_table(list(SHADOW_CANDIDATES))
    exante_scores = val_table(list(EXANTE_CANDIDATES))
    val_scores = pd.concat([shadow_scores, exante_scores], ignore_index=True)
    # two information sets, each selected on validation only
    servable = lambda t: t[~t.rung.isin(NOT_YET_SERVABLE)]
    served = LL.select_served(servable(shadow_scores))
    served_exante = LL.select_served(servable(exante_scores)) if len(servable(exante_scores)) else None
    best_overall = {"shadow": LL.select_served(shadow_scores),
                    "exante": LL.select_served(exante_scores) if len(exante_scores) else None}
    selection = {
        "served": served,
        "served_exante": served_exante,
        "best_overall": best_overall,  # incl. not-yet-servable rungs; informational
        "selection_scores": "cross-fitted for fitted candidates (decision 2026-10-16 j)",
        "bias_windows_R2": r2_windows,
        "ensembles": {r: {"k": s["k"], "threshold": s["threshold"], "static": s["static"], "regime": s["regime"]}
                      for r, s in ens_specs.items()},
        "validation_scores": val_scores.to_dict(orient="records"),
        "windows": {"validation": [str(LL.VAL_START.date()), str(LL.VAL_END.date())],
                    "test": [str(LL.TEST_START.date()), str(LL.TEST_END.date())]},
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    print(f"SELECTED shadow  (validation MAE {served['val_mae']:.3f}): {served['rung']} / {served['model']}")
    if served_exante:
        print(f"SELECTED ex-ante (validation MAE {served_exante['val_mae']:.3f}): {served_exante['rung']} / {served_exante['model']}")
    for v, b in best_overall.items():
        if b and b["rung"] in NOT_YET_SERVABLE:
            print(f"  best overall {v}: {b['rung']} / {b['model']} (val MAE {b['val_mae']:.3f}) -- not yet servable")

    # ---- TEST: scored once. CORE rungs on the days they all cover; an extra rung
    # (R5, R5x, D) is scored on its overlap with those days (n_days per row) and can
    # never shrink the core set; DM pairs use each pair's own common days.
    test_frames = {(r, n): LL.test_window(f) for r, fr in ladder.items() for n, f in fr.items()}
    days = LL.common_days([f for (r, _), f in test_frames.items() if r not in EXTRA_RUNGS])
    table = []
    for (rung, name), f in test_frames.items():
        d = days if rung not in EXTRA_RUNGS else LL.common_days([LL.restrict(f, days)])
        table.append({"rung": rung, "model": name, "n_days": len(d), **_metrics(LL.restrict(f, d))})
    table = pd.DataFrame(table)
    order = {r: i for i, r in enumerate(["R0", "R1a", "R1b", "R2", "R3", "R4", "R5", "R3x", "R4x", "R5x", "D"])}
    table = table.sort_values(["model", "rung"], key=lambda s: s.map(order) if s.name == "rung" else s)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "live_ladder.csv", index=False)
    pivot = table.pivot(index="model", columns="rung", values="rMAE")[[r for r in order if r in set(table.rung)]]
    pivot.to_latex(TABLES / "live_ladder.tex", float_format="%.3f",
                   caption=f"v2 live ladder, rMAE on {len(days)} test days "
                           f"({days.min().date()} to {days.max().date()}); R5, R5x and D on their overlap with "
                           f"those days (see n\_days). Lower is better; < 1 beats weekly naive.",
                   label="tab:live_ladder")
    with pd.option_context("display.width", 200):
        print(pivot.round(3).to_string())

    # DM-HAC between adjacent rungs (small p: the later rung is more accurate)
    dm_rows = []
    # each pair changes ONE thing; R3 vs R3x is the price of not knowing D+1 renewables
    pairs = [("R0", "R1a"), ("R1a", "R1b"), ("R1b", "R2"), ("R2", "R3"), ("R3", "R4"), ("R3", "R5"),
             ("R3", "R3x"), ("R3x", "R4x"), ("R3x", "R5x"), ("R3", "D")]
    for name in LL.ALL_MODELS + LL.ENSEMBLES:
        for a, b in pairs:
            if (a, name) in test_frames and (b, name) in test_frames:
                pd_ = LL.common_days([LL.restrict(test_frames[(a, name)], days), test_frames[(b, name)]])
                fa, fb = LL.restrict(test_frames[(a, name)], pd_), LL.restrict(test_frames[(b, name)], pd_)
                real = fa["y_true"].to_numpy().reshape(-1, 24)
                p = diebold_mariano_hac(real, fa["y_pred"].to_numpy().reshape(-1, 24),
                                        fb["y_pred"].to_numpy().reshape(-1, 24))
                dm_rows.append({"model": name, "from": a, "to": b, "p_value": p})
    pd.DataFrame(dm_rows).to_csv(TABLES / "live_ladder_dm.csv", index=False)
    print(f"wrote {TABLES / 'live_ladder.csv'}, live_ladder.tex, live_ladder_dm.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frozen")
    f.add_argument("--rungs", nargs="+", default=["R0", "R1a", "R1b"], choices=["R0", "R1a", "R1b"])
    r = sub.add_parser("recal")
    r.add_argument("--rung", choices=["R3", "R4", "R3x", "R4x", "R5", "R5x", "D"], required=True)
    r.add_argument("--model", choices=list(MODEL_CFG_KEY), required=True)
    r.add_argument("--window", type=int, choices=LL.LEAR_WINDOWS)
    c = sub.add_parser("combine-lear")
    c.add_argument("--rung", choices=["R3", "R3x", "R5", "R5x", "D"], default="R3")
    sub.add_parser("score")
    args = ap.parse_args()
    {"frozen": cmd_frozen, "recal": cmd_recal, "combine-lear": cmd_combine_lear, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
