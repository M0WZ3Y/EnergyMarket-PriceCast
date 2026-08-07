"""Optuna hyperparameter search for LightGBM — scripts/tune_lightgbm.py

50 trials (configs/evaluation.yaml optuna.n_trials), TPE sampler seeded 42.
The validation window is carved from the trailing days of the train
calendar (carve_validation_from_train) and hard-asserted to end strictly
before the test period (assert_validation_before_test) — the project's
non-negotiable ordering rule.

Tuning protocol (logged in logs/decisions.md): each trial fits the 24
per-hour models ONCE on the trailing calibration window ending right
before the validation window, then predicts all validation days
statically. Daily recalibration inside the tuning loop would multiply
cost ~360x for a hyperparameter *ranking* signal; the final walk-forward
evaluation (scripts/run_full_baselines.py) still recalibrates daily.
Objective: MAE over all (validation day, hour) pairs.

Outputs:
  configs/tuned/lightgbm_params.yaml   best params (committed; the wrapper
                                       merges this over models.yaml defaults)
  data/processed/tuning/lightgbm_trials.csv   full trial history
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
from src.features.pipeline import build_features
from src.models.base import HOURS
from src.models.lgbm import FORCED_PARAMS

BEST_PARAMS_FILE = REPO_ROOT / "configs" / "tuned" / "lightgbm_params.yaml"
TRIALS_FILE = REPO_ROOT / "data" / "processed" / "tuning" / "lightgbm_trials.csv"


def search_space(trial: optuna.Trial) -> dict:
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


def main() -> None:
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    df = pd.concat([df_train, df_test])
    X, Y = build_features(df)
    eval_cfg = load_evaluation_config()

    first_test_origin = df_test.index.min().normalize()
    days = X.index
    train_calendar = days[days < first_test_origin]
    test_calendar = days[days >= first_test_origin]

    fit_days, val_days = carve_validation_from_train(train_calendar, cfg=eval_cfg)
    assert_validation_before_test(val_days, test_calendar)

    window = eval_cfg["walk_forward"]["calibration_window_days"]
    fit_window = fit_days[-window:]
    print(
        f"fit window: {fit_window.min().date()} -> {fit_window.max().date()} "
        f"({len(fit_window)} days)\n"
        f"validation: {val_days.min().date()} -> {val_days.max().date()} "
        f"({len(val_days)} days)\n"
        f"test period untouched, starts {test_calendar.min().date()}",
        flush=True,
    )

    X_fit, Y_fit = X.loc[fit_window], Y.loc[fit_window]
    X_val, Y_val = X.loc[val_days], Y.loc[val_days]

    def objective(trial: optuna.Trial) -> float:
        params = {**search_space(trial), **FORCED_PARAMS}
        t0 = time.monotonic()
        abs_errors = []
        for hour in HOURS:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_fit.values, Y_fit[f"y_{hour}"].values)
            pred = model.predict(X_val.values)
            abs_errors.append(np.abs(Y_val[f"y_{hour}"].values - pred))
        mae = float(np.mean(abs_errors))
        print(
            f"trial {trial.number:2d}: MAE={mae:.4f} "
            f"({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
        return mae

    # SQLite-backed study so a killed run resumes instead of restarting
    # (the first attempt lost 20 in-memory trials to a machine shutdown).
    storage_path = TRIALS_FILE.parent / "lightgbm_study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=eval_cfg["random_seed"])
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{storage_path.as_posix()}",
        study_name="lightgbm_hourly_v1",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, eval_cfg["optuna"]["n_trials"] - done)
    print(f"{done} finished trials in storage, {remaining} to run", flush=True)
    study.optimize(objective, n_trials=remaining)

    TRIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(TRIALS_FILE, index=False)

    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_FILE, "w") as f:
        yaml.safe_dump(
            dict(
                params={"objective": "regression_l1", **study.best_params},
                validation_mae=round(study.best_value, 4),
                n_trials=len(study.trials),
                validation_window=f"{val_days.min().date()} -> {val_days.max().date()}",
                fit_window=f"{fit_window.min().date()} -> {fit_window.max().date()}",
                sampler_seed=eval_cfg["random_seed"],
            ),
            f,
            sort_keys=False,
        )
    print(
        f"\nbest MAE={study.best_value:.4f}\nbest params -> {BEST_PARAMS_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    main()
