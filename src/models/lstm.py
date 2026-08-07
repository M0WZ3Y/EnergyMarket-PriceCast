"""LSTM model — src/models/lstm.py

Recurrent model over the SAME leakage-audited build_features() X every
other wrapper consumes — no separate sequence-building path that would
create a second leakage surface. The X row is reshaped into:

  * sequence branch: one timestep per configured price lag (default
    [D-7, D-3, D-2, D-1], oldest -> newest), each timestep the 24-hour
    price vector of that lag day -> LSTM layer.
  * static branch: every remaining X column (exog lags, target-day D0
    exog forecasts, weekday dummies) -> concatenated with the LSTM
    output -> dense head -> 24 outputs (the D+1 hourly price vector).

Scaling: StandardScaler fit on the training slice only (per refit), for
sequence, static, and Y; predictions are inverse-transformed. MAE loss
(consistent with the headline metric and LightGBM's objective).

Recalibration cadence: same mechanism as LightGBMModel — full refit
every `refit_every_n_days` (default 7: retraining a network daily for
728 origins is impractical on this machine; deviation logged in
logs/decisions.md), model reused between refits, forecasts still
produced every origin.

Determinism: keras.utils.set_random_seed(42) + op determinism enabled
before every build/fit. TF thread count pinned (fixed constant) so runs
don't destabilize concurrent walk-forward jobs and stay reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from src.models.base import HOURS, Y_COLUMNS, BaseModel

DEFAULT_SEED = 42
DEFAULT_SEQUENCE_LAGS = (7, 3, 2, 1)  # oldest -> newest
DEFAULT_REFIT_EVERY_N_DAYS = 7
TF_THREADS = 2  # fixed constant: reproducible and polite to concurrent runs

_TF_CONFIGURED = False


def _tf():
    """Import tensorflow lazily (heavy) and pin threads exactly once."""
    global _TF_CONFIGURED
    import tensorflow as tf

    if not _TF_CONFIGURED:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(TF_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(TF_THREADS)
        except RuntimeError:
            pass  # already initialized elsewhere; keep going
        _TF_CONFIGURED = True
    return tf


class LSTMModel(BaseModel):
    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)

        # Merge tuned params over config defaults (same channel and rules
        # as LightGBMModel: repo-root-resolved path, loud failure on a
        # malformed file). The leakage review caught that models.yaml
        # declared this file but the wrapper silently ignored it.
        tuned_file = self.cfg.get("tuned_params_file")
        if tuned_file:
            tuned_path = Path(tuned_file)
            if not tuned_path.is_absolute():
                tuned_path = Path(__file__).resolve().parents[2] / tuned_path
            if tuned_path.exists():
                import yaml

                with open(tuned_path) as f:
                    tuned = yaml.safe_load(f) or {}
                if "params" not in tuned:
                    raise ValueError(
                        f"tuned params file {tuned_path} has no 'params' key -- "
                        "refusing to merge file metadata into model hyperparameters"
                    )
                self.cfg = {**self.cfg, **tuned["params"]}

        self.sequence_lags = list(self.cfg.get("sequence_lags", DEFAULT_SEQUENCE_LAGS))
        self.units = int(self.cfg.get("units", 64))
        self.epochs = int(self.cfg.get("epochs", 50))
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self.learning_rate = float(self.cfg.get("learning_rate", 1e-3))
        self.refit_every_n_days = int(
            self.cfg.get("refit_every_n_days", DEFAULT_REFIT_EVERY_N_DAYS)
        )
        # Seed is configurable ONLY to make seed-ensembling possible (averaging
        # several runs of this same model, the variance-reduction trick Lago et
        # al.'s DNN Ensemble uses). The project rule remains seed 42: that is
        # the default, every single-model result uses it, and any run with a
        # different seed exists only as a member of an explicitly labelled
        # seed ensemble. Logged in decisions.md 2026-08-07.
        self.seed = int(self.cfg.get("seed", DEFAULT_SEED))
        self._net = None
        self._scalers = None
        self._feature_columns: list[str] | None = None
        self._last_refit_end: pd.Timestamp | None = None

    # -- feature split ----------------------------------------------------

    def _sequence_columns(self, columns) -> list[list[str]]:
        """Per-lag 24-column groups (h00..h23), oldest lag first, selected
        by explicit `price_D-<lag>_` prefix — never positionally."""
        groups = []
        for lag in self.sequence_lags:
            cols = [f"price_D-{lag}_{h}" for h in HOURS]
            missing = [c for c in cols if c not in columns]
            if missing:
                raise ValueError(f"X is missing sequence columns for lag D-{lag}: {missing[:3]}")
            groups.append(cols)
        return groups

    def _static_columns(self, columns) -> list[str]:
        seq_flat = {c for group in self._sequence_columns(columns) for c in group}
        return [c for c in columns if c not in seq_flat]

    def _split(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        seq_groups = self._sequence_columns(X.columns)
        seq = np.stack([X[cols].to_numpy(dtype=np.float32) for cols in seq_groups], axis=1)
        static = X[self._static_columns(X.columns)].to_numpy(dtype=np.float32)
        return seq, static

    # -- fit / predict ----------------------------------------------------

    def _build_net(self, n_static: int):
        tf = _tf()
        keras = tf.keras
        seq_in = keras.Input(shape=(len(self.sequence_lags), 24))
        static_in = keras.Input(shape=(n_static,))
        h = keras.layers.LSTM(self.units)(seq_in)
        h = keras.layers.Concatenate()([h, static_in])
        h = keras.layers.Dense(self.units, activation="relu")(h)
        out = keras.layers.Dense(24)(h)
        net = keras.Model([seq_in, static_in], out)
        net.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mae"
        )
        return net

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LSTMModel":
        from sklearn.preprocessing import StandardScaler

        if self._feature_columns is None:
            self._feature_columns = list(X.columns)
        elif list(X.columns) != self._feature_columns:
            raise ValueError(
                "LSTMModel.fit: X column order changed between calls -- "
                "refusing to fit on silently reordered features"
            )

        train_days = Y.index.sort_values()
        train_end = train_days.max()
        if self._last_refit_end is not None and train_end < self._last_refit_end:
            # A backward-moving window would answer past origins with a
            # network trained on later data -- genuine leakage. The harness
            # yields monotone origins, so any hit here is a caller bug.
            raise ValueError(
                f"LSTMModel.fit: train window ends {train_end}, backward of "
                f"the last full refit ({self._last_refit_end}) -- refusing "
                "to reuse a model trained on later data"
            )
        needs_full_refit = (
            self._net is None
            or self._last_refit_end is None
            or (train_end - self._last_refit_end).days >= self.refit_every_n_days
        )

        if needs_full_refit:
            tf = _tf()
            tf.keras.utils.set_random_seed(self.seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception as exc:
                import warnings

                warnings.warn(
                    f"LSTMModel: TF op determinism unavailable ({exc}); seed 42 "
                    "is set but bit-reproducibility is not guaranteed",
                    RuntimeWarning,
                )

            seq, static = self._split(X.loc[train_days])
            y = Y.loc[train_days].to_numpy(dtype=np.float32)

            n_days, n_steps, n_hours = seq.shape
            seq_scaler = StandardScaler().fit(seq.reshape(n_days, -1))
            static_scaler = StandardScaler().fit(static)
            y_scaler = StandardScaler().fit(y)
            self._scalers = (seq_scaler, static_scaler, y_scaler)

            seq_s = seq_scaler.transform(seq.reshape(n_days, -1)).reshape(seq.shape)
            static_s = static_scaler.transform(static)
            y_s = y_scaler.transform(y)

            self._net = self._build_net(static.shape[1])
            self._net.fit(
                [seq_s, static_s],
                y_s,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=0,
            )
            self._last_refit_end = train_end

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._net is None or self._scalers is None:
            raise RuntimeError(
                "LSTMModel.predict: model is not fitted (no network/scalers)"
            )
        if len(X) != 1:
            raise ValueError(
                f"LSTMModel.predict expects exactly one target_day row "
                f"(the walk-forward origin), got {len(X)}"
            )
        if list(X.columns) != self._feature_columns:
            raise ValueError(
                "LSTMModel.predict: X columns do not match the columns the "
                "model was fitted on"
            )

        seq_scaler, static_scaler, y_scaler = self._scalers
        seq, static = self._split(X)
        seq_s = seq_scaler.transform(seq.reshape(1, -1)).reshape(seq.shape)
        static_s = static_scaler.transform(static)
        pred_s = self._net.predict([seq_s, static_s], verbose=0)
        pred = y_scaler.inverse_transform(pred_s)
        return pd.DataFrame(pred, index=X.index, columns=Y_COLUMNS)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        net, self._net = self._net, None
        try:
            if net is not None:
                net.save(path.with_suffix(".keras"))
            self._pickle_save(path)
        finally:
            self._net = net

    def load(self, path: str | Path) -> "LSTMModel":
        path = Path(path)
        self._pickle_load(path)
        keras_file = path.with_suffix(".keras")
        if keras_file.exists():
            self._net = _tf().keras.models.load_model(keras_file)
        elif self.is_fitted:
            # Pickle says fitted but the network artifact is gone -- fail
            # here, not with an opaque AttributeError at predict time (this
            # load path is exactly what the OOD stress test will exercise).
            raise RuntimeError(
                f"LSTMModel.load: {path} marks the model as fitted but the "
                f"companion network file {keras_file} is missing"
            )
        return self
