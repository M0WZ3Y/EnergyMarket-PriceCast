"""Optuna hyperparameter search for the LSTM — scripts/tune_lstm.py

Mirrors scripts/tune_lightgbm.py: 50 trials (configs/evaluation.yaml),
TPE sampler seeded 42, SQLite-backed resumable study, validation window
carved strictly before the test period and hard-asserted
(assert_validation_before_test).

Tuning protocol (same static-fit rationale as LightGBM, logged in
logs/decisions.md): each trial trains ONE network on the trailing
calibration window ending right before the validation window, then
predicts all validation days statically. The trial reuses LSTMModel
itself (fit + fitted scalers + net) so the scaling/split logic under
test is exactly the audited wrapper code, not a parallel copy.

DO NOT run concurrently with a walk-forward run — a trial trains a
network and competes for CPU (see the task monitor's 'avoid' list).

Outputs:
  configs/tuned/lstm_params.yaml
  data/processed/tuning/lstm_trials.csv (+ lstm_study.db)
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

from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.walk_forward import (
    assert_validation_before_test,
    carve_validation_from_train,
    load_evaluation_config,
)
from src.features.pipeline import build_features
from src.models import LSTMModel
from src.runtime import keep_awake

BEST_PARAMS_FILE = REPO_ROOT / "configs" / "tuned" / "lstm_params.yaml"
TRIALS_FILE = REPO_ROOT / "data" / "processed" / "tuning" / "lstm_trials.csv"
STUDY_DB = REPO_ROOT / "data" / "processed" / "tuning" / "lstm_study.db"


def search_space(trial: optuna.Trial) -> dict:
    return dict(
        units=trial.suggest_int("units", 16, 256, log=True),
        epochs=trial.suggest_int("epochs", 20, 100),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
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
        t0 = time.monotonic()
        cfg = dict(
            **search_space(trial),
            refit_every_n_days=10**6,  # single static fit per trial
            tuned_params_file=None,
        )
        model = LSTMModel(cfg).fit(X_fit, Y_fit)

        # Batch-predict the validation year through the wrapper's OWN
        # fitted scalers/net (audited code), bypassing only the
        # single-row-per-call harness guard.
        seq_scaler, static_scaler, y_scaler = model._scalers
        seq, static = model._split(X_val)
        n = len(X_val)
        seq_s = seq_scaler.transform(seq.reshape(n, -1)).reshape(seq.shape)
        static_s = static_scaler.transform(static)
        pred = y_scaler.inverse_transform(model._net.predict([seq_s, static_s], verbose=0))

        mae = float(np.mean(np.abs(Y_val.to_numpy() - pred)))
        print(f"trial {trial.number:2d}: MAE={mae:.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
        return mae

    STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=eval_cfg["random_seed"])
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        study_name="lstm_hourly_v1",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, eval_cfg["optuna"]["n_trials"] - done)
    print(f"{done} finished trials in storage, {remaining} to run", flush=True)
    with keep_awake():
        study.optimize(objective, n_trials=remaining)

    study.trials_dataframe().to_csv(TRIALS_FILE, index=False)
    BEST_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_FILE, "w") as f:
        yaml.safe_dump(
            dict(
                params=study.best_params,
                validation_mae=round(study.best_value, 4),
                n_trials=len(study.trials),
                validation_window=f"{val_days.min().date()} -> {val_days.max().date()}",
                fit_window=f"{fit_window.min().date()} -> {fit_window.max().date()}",
                sampler_seed=eval_cfg["random_seed"],
            ),
            f,
            sort_keys=False,
        )
    print(f"\nbest MAE={study.best_value:.4f}\nbest params -> {BEST_PARAMS_FILE}", flush=True)


if __name__ == "__main__":
    main()
