"""Wire model wrappers onto the walk-forward harness — src/evaluation/run_baselines.py

loader -> build_features -> walk_forward_splits -> model.fit/predict ->
long-format results frame. Metrics/results-table export is a separate,
not-yet-built module; this file stops at producing the per-(origin, hour)
long DataFrame every downstream consumer needs.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.walk_forward import load_evaluation_config, walk_forward_splits
from src.features.pipeline import build_features
from src.models import BaseModel, LEARLassoModel, NaiveModel, SARIMAXModel, load_models_config


def run_model(
    model_name: str,
    model: BaseModel,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    eval_cfg: dict | None = None,
    first_origin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run one model over the full walk-forward harness.

    Returns a long DataFrame with columns [origin, hour, y_true, y_pred,
    model] -- one row per (origin day, hour), the atomic unit both the
    metrics layer (mae/rmse/smape/rmae take flat arrays) and a later
    results-table export need.
    """
    records = []
    for split in walk_forward_splits(Y.index, cfg=eval_cfg, first_origin=first_origin):
        X_train, Y_train = X.loc[split.train_days], Y.loc[split.train_days]
        X_test = X.loc[split.test_days]
        y_true = Y.loc[split.test_days]

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        for h in range(24):
            records.append(
                dict(
                    origin=split.origin,
                    hour=h,
                    y_true=y_true.iloc[0, h],
                    y_pred=y_pred.iloc[0, h],
                    model=model_name,
                )
            )
    # An empty run must still carry the documented schema. from_records([])
    # yields a (0, 0) frame with NO columns, so load_long_frame, _pivot_24 and
    # daily_baseload all die later with an opaque KeyError: 'origin', far from
    # the real cause. Zero origins is reachable whenever Y is shorter than
    # calibration_window_days or first_origin sits past the end of the data --
    # a configuration mistake that should surface as an empty result, not as a
    # missing column three modules downstream.
    if not records:
        return pd.DataFrame(columns=["origin", "hour", "y_true", "y_pred", "model"])
    return pd.DataFrame.from_records(records)


def main(
    models: dict[str, BaseModel] | None = None,
    first_origin: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    # walk_forward_splits carves train/test by its own trailing calibration
    # window, not by the loader's years_test split -- training history for
    # any origin is drawn from everything before it (see walk_forward.py's
    # docstring), so train and test frames are concatenated up front.
    df = pd.concat([df_train, df_test])

    X, Y = build_features(df)
    eval_cfg = load_evaluation_config()
    models_cfg = load_models_config()

    if models is None:
        models = {
            "naive": NaiveModel(models_cfg["naive"]),
            "SARIMAX": SARIMAXModel(models_cfg["sarimax"]),
            "LEAR-LASSO": LEARLassoModel(models_cfg["lear_lasso"]),
        }

    if first_origin is None:
        first_origin = df_test.index.min().normalize()

    return {
        name: run_model(name, model, X, Y, eval_cfg=eval_cfg, first_origin=first_origin)
        for name, model in models.items()
    }


if __name__ == "__main__":
    results = main()
    for name, frame in results.items():
        print(name, frame.shape)
