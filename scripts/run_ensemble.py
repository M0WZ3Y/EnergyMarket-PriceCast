"""Fit and evaluate the static and regime-aware ensembles —
scripts/run_ensemble.py

Week-7 contribution (Plan B, sanctioned 2026-07-11). The machinery lives in
src/evaluation/ensemble.py; this script is the runner that points it at real
data and enforces the protocol around it:

  weights fitted on   data/processed/validation_preds/   (2015-01-05 .. 2016-01-03)
  weights applied to  data/processed/baselines/          (2016-01-04 .. 2017-12-31)

The two windows never overlap, and `fit_weights(..., test_days=...)` is always
given the test calendar so the leakage contract is checked rather than assumed.

Members are SARIMAX, LEAR-LASSO, LightGBM and LSTM (decision 2026-07-31).
naive is a reference row only — the rMAE denominator — never a weighted member.

Outputs
  data/processed/baselines/ensemble_static.csv
  data/processed/baselines/ensemble_regime.csv
  (long frames in the same schema as every model, so results.dm_matrix() and
  the canonical results table consume them without special-casing)

Usage:
    python scripts/run_ensemble.py            # fit, evaluate, write frames
    python scripts/run_ensemble.py --dry-run  # report only, write nothing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ensemble import (
    combine_forecasts,
    combine_regime_aware,
    fit_weights,
    regime_labels,
)
from src.evaluation.metrics import mae, rmae, rmse, smape
from src.evaluation.results import load_long_frame
from src.evaluation.walk_forward import load_evaluation_config

VAL_DIR = REPO_ROOT / "data" / "processed" / "validation_preds"
TEST_DIR = REPO_ROOT / "data" / "processed" / "baselines"

# Weighted members only. naive is deliberately absent (see module docstring).
MEMBERS = {
    "SARIMAX": "sarimax.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "LSTM": "lstm.csv",
}
REFERENCE = {"naive": "naive.csv"}

# A weight set fitted on a handful of days is noise, not a regime model.
MIN_DAYS_PER_REGIME = 20


def _load_frames(directory: Path, files: dict[str, str], label: str) -> dict[str, pd.DataFrame]:
    """Load one long frame per model, refusing anything incomplete.

    An in-progress walk-forward run leaves a short CSV behind. Fitting
    weights on a truncated validation set would silently produce weights
    for a different window than the one reported, so a short file is an
    error here, not a warning.
    """
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name, fname in files.items():
        path = directory / fname
        if not path.exists():
            missing.append(f"{name}: {path.name} not found in {directory.name}/")
            continue
        frames[name] = load_long_frame(path)
    if missing:
        raise SystemExit(
            f"{label}: cannot proceed — these members are not ready:\n  "
            + "\n  ".join(missing)
            + "\n\nFinish their walk-forward runs first."
        )

    # Every member must cover exactly the same origins, each with 24 hours.
    sizes = {}
    for name, f in frames.items():
        counts = f.groupby("origin").size()
        bad = counts[counts != 24]
        if len(bad):
            raise SystemExit(
                f"{label}: {name} has {len(bad)} origin(s) without 24 rows "
                f"(first: {bad.index[0]}) — the file is mid-run or corrupted."
            )
        if f.duplicated(["origin", "hour"]).any():
            raise SystemExit(f"{label}: {name} has duplicate (origin, hour) rows.")
        sizes[name] = set(counts.index)

    base_name, base = next(iter(sizes.items()))
    for name, days in sizes.items():
        if days != base:
            raise SystemExit(
                f"{label}: {name} covers {len(days)} origins but {base_name} "
                f"covers {len(base)} — members must span an identical window.\n"
                f"  only in {name}: {len(days - base)}, only in {base_name}: {len(base - days)}"
            )
    return frames


def _metrics_row(name: str, frame: pd.DataFrame) -> dict:
    """MAE/RMSE/sMAPE/rMAE for one long frame, via the epftoolbox wrappers.

    rMAE needs an hourly DatetimeIndex (its naive denominator is a weekly
    lag), so the long frame is rebuilt into an hourly series first — the
    same conversion week5_checkpoint.py uses.
    """
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    pred = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")
    return dict(
        model=name,
        MAE=mae(real.values, pred.values),
        RMSE=rmse(real.values, pred.values),
        sMAPE=smape(real.values, pred.values) * 100,
        rMAE=rmae(real, pred, m="W"),
    )


def _subset_by_regime(
    frames: dict[str, pd.DataFrame], labels: dict[pd.Timestamp, str], regime: str
) -> dict[str, pd.DataFrame]:
    days = [o for o, lab in labels.items() if lab == regime]
    return {m: f[f["origin"].isin(days)].copy() for m, f in frames.items()}


def main(dry_run: bool = False) -> None:
    eval_cfg = load_evaluation_config()
    threshold = float(eval_cfg["regime"]["stress_threshold_eur_mwh"])

    val = _load_frames(VAL_DIR, MEMBERS, "validation")
    test = _load_frames(TEST_DIR, MEMBERS, "test")
    test_ref = _load_frames(TEST_DIR, REFERENCE, "test (reference)")

    val_days = pd.DatetimeIndex(sorted(val["LightGBM"]["origin"].unique()))
    test_days = pd.DatetimeIndex(sorted(test["LightGBM"]["origin"].unique()))
    print(
        f"validation: {len(val_days)} origins "
        f"{val_days.min().date()} -> {val_days.max().date()}\n"
        f"test:       {len(test_days)} origins "
        f"{test_days.min().date()} -> {test_days.max().date()}\n"
        f"members:    {list(MEMBERS)}\n"
        f"stress threshold: {threshold} EUR/MWh (previous-day rule)\n"
    )

    # ---- static ensemble -------------------------------------------------
    # test_days is passed so the leakage contract is enforced by
    # assert_validation_before_test, not merely documented.
    w_static = fit_weights(val, test_days=test_days)
    print("static weights (fitted on validation):")
    for m, wi in sorted(w_static.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<12} {wi:6.3f}")

    # ---- regime-aware ensemble ------------------------------------------
    # Regimes are labeled on the VALIDATION frames to fit the two weight
    # sets; combine_regime_aware() re-labels the test frames itself, using
    # the same previous-day-only rule.
    labels = regime_labels(val["LightGBM"], threshold=threshold)
    counts = pd.Series(list(labels.values())).value_counts().to_dict()
    print(
        f"\nvalidation regime split: "
        f"calm {counts.get('calm', 0)}, stressed {counts.get('stressed', 0)}"
    )

    w_regime: dict[str, dict[str, float]] = {}
    for regime in ("calm", "stressed"):
        n = counts.get(regime, 0)
        if n < MIN_DAYS_PER_REGIME:
            raise SystemExit(
                f"\nregime-aware ensemble not defensible: only {n} '{regime}' day(s) "
                f"in the validation window (minimum {MIN_DAYS_PER_REGIME}).\n"
                "Weights fitted on that few days would be noise. Widening the\n"
                "validation window does NOT help on this data (a 730-day window\n"
                "still holds only 4 stressed days at 3*std -- decision 2026-08-04);\n"
                "the lever is regime.stress_threshold_eur_mwh in\n"
                "configs/evaluation.yaml. Re-pick k by the validation-only rule\n"
                "documented there -- never by which value looks best on test."
            )
        sub = _subset_by_regime(val, labels, regime)
        w_regime[regime] = fit_weights(sub, test_days=test_days)

    print("\nregime weights (fitted on validation, per regime):")
    header = "  " + " ".join(f"{m:>12}" for m in MEMBERS)
    print(header)
    for regime in ("calm", "stressed"):
        print(f"  {regime:<9}" + " ".join(f"{w_regime[regime][m]:12.3f}" for m in MEMBERS))

    # ---- apply to the test period ---------------------------------------
    ens_static = combine_forecasts(test, w_static, name="Ensemble (static)")
    ens_regime = combine_regime_aware(
        test, w_regime, threshold=threshold, name="Ensemble (regime-aware)"
    )

    test_labels = regime_labels(test["LightGBM"], threshold=threshold)
    t_counts = pd.Series(list(test_labels.values())).value_counts().to_dict()
    print(
        f"\ntest regime split: calm {t_counts.get('calm', 0)}, "
        f"stressed {t_counts.get('stressed', 0)}"
    )

    # ---- results ---------------------------------------------------------
    rows = [_metrics_row(n, f) for n, f in test_ref.items()]
    rows += [_metrics_row(n, f) for n, f in test.items()]
    rows.append(_metrics_row("Ensemble (static)", ens_static))
    rows.append(_metrics_row("Ensemble (regime-aware)", ens_regime))
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    pd.set_option("display.float_format", lambda v: f"{v:8.3f}")
    print(f"\nDE test period {test_days.min().date()} -> {test_days.max().date()}\n")
    print(table.to_string(index=False))

    best_single = min(r["MAE"] for r in rows if r["model"] not in ("naive",) and "Ensemble" not in r["model"])
    for label, frame_name in (("static", "Ensemble (static)"), ("regime-aware", "Ensemble (regime-aware)")):
        m = next(r["MAE"] for r in rows if r["model"] == frame_name)
        print(f"\n{label:<12} vs best single model: {m:.4f} vs {best_single:.4f} ({m - best_single:+.4f})")

    if dry_run:
        print("\n--dry-run: no files written")
        return

    for frame, fname in ((ens_static, "ensemble_static.csv"), (ens_regime, "ensemble_regime.csv")):
        out = TEST_DIR / fname
        frame.to_csv(out, index=False)
        print(f"wrote {out.relative_to(REPO_ROOT)} ({len(frame)} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="print weights and metrics, write no files"
    )
    main(dry_run=parser.parse_args().dry_run)
