"""Walk-forward runs for the DIRECT daily-baseload route.

Mirrors scripts/run_full_baselines.py — same splits, same resume-on-restart
checkpointing, same keep_awake — but the target is the daily baseload
(features.pipeline.daily_target) and each origin contributes ONE row, not
24.

Output schema is deliberately identical to what
evaluation.results.daily_baseload() produces for the aggregated route
([origin, y_true, y_pred, model]), so the RQ4 comparison is a direct diff
of two frames rather than a translation between schemas.

Usage:
    python scripts/run_daily_direct.py                    # all daily models
    python scripts/run_daily_direct.py DailyLightGBM
    python scripts/run_daily_direct.py naive --first-origin 2015-01-05 \
        --last-origin 2016-01-03 --out-dir data/processed/daily_validation
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.walk_forward import load_evaluation_config, walk_forward_splits
from src.features.pipeline import build_features, daily_target
from src.models import load_models_config
from src.models.daily import (
    DailyLEARLassoModel,
    DailyLightGBMModel,
    DailyLSTMModel,
    DailyNaiveModel,
    DailySARIMAXModel,
)
from src.runtime import keep_awake

OUT_DIR = REPO_ROOT / "data" / "processed" / "daily_direct"
COLUMNS = ["origin", "y_true", "y_pred", "model"]


def completed_origins(out_path: Path) -> set[pd.Timestamp]:
    if not out_path.exists():
        return set()
    done = pd.read_csv(out_path, usecols=["origin"])
    return {pd.Timestamp(o) for o in done["origin"]}


def run_one(model_name, model, X, y, eval_cfg, first_origin, last_origin, out_dir) -> None:
    out_path = out_dir / f"{model_name.lower().replace('-', '_')}.csv"
    done = completed_origins(out_path)
    if done:
        print(f"[{model_name}] resuming: {len(done)} origins already complete", flush=True)

    splits = list(walk_forward_splits(y.index, cfg=eval_cfg, first_origin=first_origin))
    if last_origin is not None:
        splits = [s for s in splits if s.origin <= last_origin]
    todo = [s for s in splits if s.origin not in done]
    print(f"[{model_name}] {len(todo)} of {len(splits)} origins to run", flush=True)

    t0 = time.monotonic()
    for i, split in enumerate(todo, 1):
        model.fit(X.loc[split.train_days], y.loc[split.train_days])
        y_pred = model.predict(X.loc[split.test_days])

        pd.DataFrame(
            dict(
                origin=[split.origin],
                y_true=[y.loc[split.test_days].iloc[0]],
                y_pred=[y_pred["y_daily"].iloc[0]],
                model=[model_name],
            )
        ).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

        if i % 25 == 0 or i == len(todo):
            rate = (time.monotonic() - t0) / i
            print(
                f"[{model_name}] {i}/{len(todo)} origins "
                f"({rate:.1f}s/origin, ~{rate * (len(todo) - i) / 60:.0f} min left)",
                flush=True,
            )
    print(f"[{model_name}] DONE -> {out_path}", flush=True)


def main(only=None, first_origin=None, last_origin=None, out_dir=OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train, df_test = BenchmarkLoader(load_config()).load()
    X, Y = build_features(pd.concat([df_train, df_test]))
    y = daily_target(Y)
    eval_cfg = load_evaluation_config()
    models_cfg = load_models_config()
    if first_origin is None:
        first_origin = df_test.index.min().normalize()

    # Same five models as the hourly route -- RQ4 compares the two routes,
    # so a different model list on one side would confound it. LightGBM and
    # LSTM read their own `daily_*` config entries, tuned against the daily
    # target by scripts/tune_daily.py; SARIMAX and LEAR-LASSO reuse the
    # hourly entries because neither has a separate Optuna search (see the
    # comment above `daily_lightgbm:` in configs/models.yaml).
    models = {
        "naive": DailyNaiveModel(models_cfg["naive"]),
        "DailySARIMAX": DailySARIMAXModel(models_cfg["sarimax"]),
        "DailyLEAR-LASSO": DailyLEARLassoModel(models_cfg["lear_lasso"]),
        "DailyLightGBM": DailyLightGBMModel(models_cfg["daily_lightgbm"]),
        "DailyLSTM": DailyLSTMModel(models_cfg["daily_lstm"]),
    }
    if only:
        models = {k: v for k, v in models.items() if k in only}
        if not models:
            raise SystemExit(f"no matching models in {only}")

    print(
        f"origins from {first_origin}"
        + (f" to {last_origin}" if last_origin is not None else " (to end)")
        + f", models: {list(models)}, out: {out_dir}",
        flush=True,
    )
    with keep_awake():
        for name, model in models.items():
            run_one(name, model, X, y, eval_cfg, first_origin, last_origin, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="subset of model names (default: all)")
    parser.add_argument("--first-origin", type=pd.Timestamp, default=None)
    parser.add_argument("--last-origin", type=pd.Timestamp, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(
        only=args.models or None,
        first_origin=args.first_origin,
        last_origin=args.last_origin,
        out_dir=args.out_dir,
    )
