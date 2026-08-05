"""Full walk-forward baseline run over the benchmark test period.

Same wiring as src/evaluation/run_baselines.py, plus what a multi-hour
run needs: per-origin checkpointing to CSV (append + resume), progress
logging, and one output file per model under data/processed/baselines/.

Usage:
    python scripts/run_full_baselines.py              # all three baselines
    python scripts/run_full_baselines.py naive LEAR-LASSO
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
from src.runtime import keep_awake
from src.features.pipeline import build_features
from src.models import (
    LEARLassoModel,
    LightGBMModel,
    LSTMModel,
    NaiveModel,
    SARIMAXModel,
    load_models_config,
)

OUT_DIR = REPO_ROOT / "data" / "processed" / "baselines"
COLUMNS = ["origin", "hour", "y_true", "y_pred", "model"]


def completed_origins(out_path: Path) -> set[pd.Timestamp]:
    if not out_path.exists():
        return set()
    done = pd.read_csv(out_path, usecols=["origin", "hour"])
    # only trust origins with all 24 hours written (guards against a row
    # written mid-crash)
    counts = done.groupby("origin").size()
    return {pd.Timestamp(o) for o in counts[counts == 24].index}


def repair_partial_origins(out_path: Path) -> int:
    """Drop rows belonging to any origin that does not have all 24 hours.

    completed_origins() distrusts a torn origin, so it goes back into todo
    and 24 fresh rows are appended -- but the torn rows are never removed,
    leaving e.g. 34 rows for that day permanently. Every downstream
    consumer then breaks or lies: daily_baseload raises, pivot() raises on
    the duplicate (origin, hour) entries, and the monitor's row count
    reports inflated progress. Repairing on resume keeps the file
    self-consistent without hand-editing frozen outputs.

    Returns the number of rows removed. No-op when the file is absent.
    """
    if not out_path.exists():
        return 0
    df = pd.read_csv(out_path)
    if df.empty:
        return 0
    counts = df.groupby("origin")["hour"].size()
    complete = set(counts[counts == 24].index)
    if len(complete) == len(counts):
        return 0
    kept = df[df["origin"].isin(complete)]
    removed = len(df) - len(kept)
    kept.to_csv(out_path, index=False)
    print(
        f"repaired {out_path.name}: dropped {removed} row(s) from "
        f"{len(counts) - len(complete)} incomplete origin(s)",
        flush=True,
    )
    return removed


def validate_origin_range(first_origin, last_origin) -> None:
    """Reject an inverted --first-origin/--last-origin pair.

    Without this, the range filter removes every split, the loop body never
    executes, and the script prints "0 of 0 origins" and exits 0 -- having
    produced nothing while reporting success. An exit code of 0 meaning
    "nothing failed" rather than "everything ran" already cost this project
    a deliverable once (logs/decisions.md, 2026-08-04). Both bounds are
    optional, so only a genuinely inverted pair is an error.
    """
    if first_origin is None or last_origin is None:
        return
    if pd.Timestamp(last_origin) < pd.Timestamp(first_origin):
        raise ValueError(
            f"origin range is inverted: --last-origin {pd.Timestamp(last_origin).date()} "
            f"is before --first-origin {pd.Timestamp(first_origin).date()}; "
            "this would silently run zero origins and still exit 0"
        )


def run_one(
    model_name: str, model, X, Y, eval_cfg, first_origin, last_origin=None, out_dir=OUT_DIR
) -> None:
    validate_origin_range(first_origin, last_origin)
    out_path = out_dir / f"{model_name.lower().replace('-', '_')}.csv"
    # Before anything is read or appended: a crash mid-origin must not
    # leave partial rows to be duplicated by this run's append path.
    repair_partial_origins(out_path)
    done = completed_origins(out_path)
    if done:
        print(f"[{model_name}] resuming: {len(done)} origins already complete", flush=True)

    splits = list(walk_forward_splits(Y.index, cfg=eval_cfg, first_origin=first_origin))
    if last_origin is not None:
        splits = [s for s in splits if s.origin <= last_origin]
    todo = [s for s in splits if s.origin not in done]
    print(f"[{model_name}] {len(todo)} of {len(splits)} origins to run", flush=True)

    t0 = time.monotonic()
    for i, split in enumerate(todo, 1):
        model.fit(X.loc[split.train_days], Y.loc[split.train_days])
        y_pred = model.predict(X.loc[split.test_days])
        y_true = Y.loc[split.test_days]

        rows = pd.DataFrame(
            dict(
                origin=[split.origin] * 24,
                hour=range(24),
                y_true=y_true.iloc[0].to_numpy(),
                y_pred=y_pred.iloc[0].to_numpy(),
                model=model_name,
            )
        )
        rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

        if i % 25 == 0 or i == len(todo):
            rate = (time.monotonic() - t0) / i
            eta_min = rate * (len(todo) - i) / 60
            print(
                f"[{model_name}] {i}/{len(todo)} origins "
                f"({rate:.1f}s/origin, ~{eta_min:.0f} min left)",
                flush=True,
            )
    print(f"[{model_name}] DONE -> {out_path}", flush=True)


def main(
    only: list[str] | None = None,
    first_origin: pd.Timestamp | None = None,
    last_origin: pd.Timestamp | None = None,
    out_dir=OUT_DIR,
) -> None:
    """Default: full benchmark test period into data/processed/baselines/.

    Validation-period runs (week-7 ensemble weight fitting) override the
    window and output dir, e.g.:
        run_full_baselines.py naive --first-origin 2015-01-05 \
            --last-origin 2016-01-03 --out-dir data/processed/validation_preds
    The validation window must end strictly before the test period; the
    ensemble weight-fitting code never sees test-period predictions.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    df = pd.concat([df_train, df_test])
    X, Y = build_features(df)
    eval_cfg = load_evaluation_config()
    models_cfg = load_models_config()
    if first_origin is None:
        first_origin = df_test.index.min().normalize()

    models = {
        "naive": NaiveModel(models_cfg["naive"]),
        "LEAR-LASSO": LEARLassoModel(models_cfg["lear_lasso"]),
        "SARIMAX": SARIMAXModel(models_cfg["sarimax"]),
        "LightGBM": LightGBMModel(models_cfg["lightgbm"]),
        "LSTM": LSTMModel(models_cfg["lstm"]),
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
            run_one(name, model, X, Y, eval_cfg, first_origin, last_origin, out_dir)


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
