"""Optuna hyperparameter search for the DIRECT daily-baseload models —
scripts/tune_daily.py

Same protocol as scripts/tune_lightgbm.py and scripts/tune_lstm.py, with the
daily baseload (features.pipeline.daily_target) as the objective instead of
the 24-hour vector: 50 trials (configs/evaluation.yaml optuna.n_trials), TPE
seeded 42, one static fit per trial on the calibration window ending right
before the validation window, validation window hard-asserted to end strictly
before the test period.

Why this exists (logs/decisions.md 2026-08-02): the daily-direct wrappers
initially inherited the hourly models' tuned hyperparameters. Reusing them
means any direct-vs-aggregated difference is partly a difference in tuning
effort rather than in the target, which is exactly the confound RQ4 must
avoid. Tuning against the daily target removes it.

Only LightGBM and LSTM are tuned, because only they have an Optuna search in
this project at all:
  * LEAR-LASSO selects its own lambda per fit via LassoLarsIC (that IS its
    tuning, and it re-runs on the daily target automatically),
  * SARIMAX's (p,d,q)(P,D,Q,s) order is fixed by configs/models.yaml on both
    routes, so both arms already share one convention.

Outputs (one per model):
  configs/tuned/daily_lightgbm_params.yaml   /  daily_lstm_params.yaml
  data/processed/tuning/daily_<model>_trials.csv
  data/processed/tuning/daily_<model>_study.db   (resumable, SQLite)

Usage:
    python scripts/tune_daily.py                 # both models
    python scripts/tune_daily.py DailyLightGBM
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb

from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.walk_forward import (
    assert_validation_before_test,
    carve_validation_from_train,
    load_evaluation_config,
)
from src.features.pipeline import build_features, daily_target
from src.models.daily import FORCED_PARAMS, DailyLSTMModel

TUNED_DIR = REPO_ROOT / "configs" / "tuned"
TUNING_DIR = REPO_ROOT / "data" / "processed" / "tuning"


def lgbm_space(trial: optuna.Trial) -> dict:
    """Identical bounds to scripts/tune_lightgbm.py — the two routes must
    search the same space, or a difference in results could be a difference
    in how hard each was allowed to look."""
    return dict(
        objective="regression_l1",
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    )


def lstm_space(trial: optuna.Trial) -> dict:
    """Identical bounds to scripts/tune_lstm.py, for the same reason."""
    return dict(
        units=trial.suggest_int("units", 16, 256, log=True),
        epochs=trial.suggest_int("epochs", 20, 100),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    )


def lgbm_objective(trial, X_fit, y_fit, X_val, y_val) -> float:
    params = {**lgbm_space(trial), **FORCED_PARAMS}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_fit.values, y_fit.values)
    pred = model.predict(X_val.values)
    return float(np.mean(np.abs(y_val.values - pred)))


def lstm_objective(trial, X_fit, y_fit, X_val, y_val) -> float:
    cfg = dict(
        **lstm_space(trial),
        refit_every_n_days=10**6,  # single static fit per trial
        tuned_params_file=None,
    )
    model = DailyLSTMModel(cfg).fit(X_fit, y_fit)

    # Batch-predict the validation year through the wrapper's OWN fitted
    # scalers/net (audited code), bypassing only the single-row-per-call
    # harness guard — same approach as scripts/tune_lstm.py.
    seq_scaler, static_scaler, y_scaler = model._scalers
    seq, static = model._split(X_val)
    n = len(X_val)
    seq_s = seq_scaler.transform(seq.reshape(n, -1)).reshape(seq.shape)
    static_s = static_scaler.transform(static)
    pred = y_scaler.inverse_transform(model._net.predict([seq_s, static_s], verbose=0))
    return float(np.mean(np.abs(y_val.to_numpy() - pred.ravel())))


MODELS = {
    "DailyLightGBM": dict(
        objective=lgbm_objective,
        study_name="lightgbm_daily_v1",
        slug="daily_lightgbm",
        extra_params={"objective": "regression_l1"},
    ),
    "DailyLSTM": dict(
        objective=lstm_objective,
        study_name="lstm_daily_v1",
        slug="daily_lstm",
        extra_params={},
    ),
}


def tune_one(name: str, spec: dict, X_fit, y_fit, X_val, y_val, eval_cfg, windows) -> None:
    slug = spec["slug"]
    study_db = TUNING_DIR / f"{slug}_study.db"
    study_db.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        t0 = time.monotonic()
        mae = spec["objective"](trial, X_fit, y_fit, X_val, y_val)
        print(
            f"[{name}] trial {trial.number:2d}: MAE={mae:.4f} "
            f"({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
        return mae

    sampler = optuna.samplers.TPESampler(seed=eval_cfg["random_seed"])
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{study_db.as_posix()}",
        study_name=spec["study_name"],
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, eval_cfg["optuna"]["n_trials"] - done)
    print(f"[{name}] {done} finished trials in storage, {remaining} to run", flush=True)
    study.optimize(objective, n_trials=remaining)

    study.trials_dataframe().to_csv(TUNING_DIR / f"{slug}_trials.csv", index=False)

    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNED_DIR / f"{slug}_params.yaml"
    fit_window, val_days = windows
    with open(out, "w") as f:
        yaml.safe_dump(
            dict(
                params={**spec["extra_params"], **study.best_params},
                validation_mae=round(study.best_value, 4),
                n_trials=len(study.trials),
                target="daily_baseload",
                validation_window=f"{val_days.min().date()} -> {val_days.max().date()}",
                fit_window=f"{fit_window.min().date()} -> {fit_window.max().date()}",
                sampler_seed=eval_cfg["random_seed"],
            ),
            f,
            sort_keys=False,
        )
    print(f"[{name}] best MAE={study.best_value:.4f} -> {out}", flush=True)


def main(only: list[str] | None = None) -> None:
    df_train, df_test = BenchmarkLoader(load_config()).load()
    X, Y = build_features(pd.concat([df_train, df_test]))
    y = daily_target(Y)
    eval_cfg = load_evaluation_config()

    first_test_origin = df_test.index.min().normalize()
    days = X.index
    train_calendar = days[days < first_test_origin]
    test_calendar = days[days >= first_test_origin]

    fit_days, val_days = carve_validation_from_train(train_calendar, cfg=eval_cfg)
    # The non-negotiable ordering rule, checked rather than assumed.
    assert_validation_before_test(val_days, test_calendar)

    window = eval_cfg["walk_forward"]["calibration_window_days"]
    fit_window = fit_days[-window:]
    print(
        f"target: daily baseload\n"
        f"fit window: {fit_window.min().date()} -> {fit_window.max().date()} "
        f"({len(fit_window)} days)\n"
        f"validation: {val_days.min().date()} -> {val_days.max().date()} "
        f"({len(val_days)} days)\n"
        f"test period untouched, starts {test_calendar.min().date()}\n",
        flush=True,
    )

    X_fit, y_fit = X.loc[fit_window], y.loc[fit_window]
    X_val, y_val = X.loc[val_days], y.loc[val_days]

    selected = MODELS if not only else {k: v for k, v in MODELS.items() if k in only}
    if not selected:
        raise SystemExit(f"no matching models in {only} (choose from {list(MODELS)})")

    for name, spec in selected.items():
        tune_one(name, spec, X_fit, y_fit, X_val, y_val, eval_cfg, (fit_window, val_days))


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help=f"subset of {list(MODELS)} (default: all)")
    main(only=parser.parse_args().models or None)
