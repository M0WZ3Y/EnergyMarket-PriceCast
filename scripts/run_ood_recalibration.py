"""OOD bias recalibration — post-hoc, supplementary, NON-frozen analysis.

Sits on top of the v1.1-ood addendum and answers one question about it:
when every frozen model came out worse than naive on live 2026 data
(rMAE 1.087 to 1.828), was that a loss of forecasting SKILL, or only a loss
of LEVEL?

The distinction matters for the thesis. "The models fail out of
distribution" and "the models still rank hours correctly but sit ~3x too
low" are different findings with different implications: the second says the
learned shape survives a regime change and only the intercept has to be
re-anchored, which is a cheap fix in deployment. The PriceCast demo showing
48.42 predicted against 143.21 realized is what motivates the question.

METHOD. For each model and each live day d, a correction is the mean SIGNED
error (y_true - y_pred) over the `window` days STRICTLY BEFORE d, added to
all 24 hours of day d. Nothing from day d or later enters its own
correction — the same causal discipline the walk-forward uses everywhere
else in this repo, and the property tests/test_ood_recalibration.py pins
first, because a leak here would manufacture exactly the recovery the
hypothesis is hoping for.

COLD START. The first days have no prior window to average. They are
DROPPED, never filled — see `--cold-start`, and the note in `compare_one`
about why the raw arm is re-scored on the same reduced day set.

WHAT THIS IS NOT. It does not retrain, refit or reweight any model, it
reads the frozen v1.1-ood predictions read-only, and it writes only to
data/processed/ood_recalibrated/ and reports/tables/ood_recalibration.*.
The correction uses realized prices from previous days, which a real
operator would also have; it is nonetheless a POST-HOC analysis of a
finished experiment, not a new forecasting result, and it belongs in
thesis section 5-2 (limitations) — not chapter 4, whose page budget is
fixed and whose numbers are frozen.

Usage:
    python scripts/run_ood_recalibration.py
    python scripts/run_ood_recalibration.py --window 7 14 30
    python scripts/run_ood_recalibration.py --cold-start expanding
    python scripts/run_ood_recalibration.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import mae, rmae, rmse
from src.evaluation.results import load_long_frame

OOD_DIR = REPO_ROOT / "data" / "processed" / "ood"
OUT_DIR = REPO_ROOT / "data" / "processed" / "ood_recalibrated"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

# Fixed presentation order (export-results skill).
OOD_FILES = {
    "naive": "naive.csv",
    "SARIMAX": "sarimax.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "LSTM": "lstm.csv",
    "Ensemble (static)": "ensemble_static.csv",
    "Ensemble (regime-aware)": "ensemble_regime_aware.csv",
}

COLD_START_MODES = ("exclude", "expanding")
DEFAULT_WINDOWS = [7, 14, 30]


def _rel(path: Path) -> str:
    """Repo-relative path for display, falling back to the absolute path.

    Deliberately a local copy of the same helper in run_ood_stress.py and
    export_tables.py: scripts in this repo do not import from each other.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_ood_frame(path: str | Path) -> pd.DataFrame:
    """Read a frozen v1.1-ood long frame. Read-only by construction."""
    return load_long_frame(path)


# --------------------------------------------------------------------------
# the correction
# --------------------------------------------------------------------------
def daily_signed_error(frame: pd.DataFrame) -> pd.Series:
    """Mean signed error (y_true - y_pred) per origin day, sorted by day.

    SIGNED, not absolute: the sign is the entire hypothesis. A model that is
    uniformly too low has a positive mean signed error, and adding it back
    is what tests whether the miss was a level shift.
    """
    per_day = (
        frame.assign(signed=frame["y_true"] - frame["y_pred"])
        .groupby("origin")["signed"]
        .mean()
        .sort_index()
    )
    return per_day


def rolling_signed_correction(
    frame: pd.DataFrame, window: int, cold_start: str = "exclude"
) -> pd.Series:
    """Correction to ADD to each day's forecast, from PAST days only.

    Returns a Series indexed by origin. Days with no usable history hold
    NaN — deliberately not 0.0, which would silently pass an uncorrected
    day off as a corrected one and quietly dilute the result.

    The `.shift(1)` is the causal step and the reason this function exists
    separately from `recalibrate`: it is the single line a future reader
    must be able to check. Without it, day d's own error would enter its own
    correction and the experiment would be measuring hindsight.
    """
    if cold_start not in COLD_START_MODES:
        raise ValueError(
            f"rolling_signed_correction: cold_start must be one of "
            f"{COLD_START_MODES}, got {cold_start!r}"
        )
    if window < 1:
        raise ValueError(f"rolling_signed_correction: window must be >= 1, got {window}")

    signed = daily_signed_error(frame)
    # 'expanding' applies ONLY to the cold start: it relaxes min_periods so
    # the early days average whatever history exists, then becomes the same
    # rolling window as 'exclude' once `window` days are available. It is
    # deliberately not `.expanding()` over the whole series -- that would
    # ignore `window` entirely, make every window in a sweep produce
    # identical numbers, and quietly change the estimator rather than just
    # the cold-start policy.
    min_periods = window if cold_start == "exclude" else 1
    rolled = signed.rolling(window, min_periods=min_periods).mean()
    return rolled.shift(1)


def recalibrate(frame: pd.DataFrame, window: int, cold_start: str = "exclude") -> pd.DataFrame:
    """Apply the causal correction; drop days that have none.

    Cold-start days are removed from the returned frame rather than passed
    through uncorrected. Passing them through would mix corrected and
    uncorrected days under one heading — the metric would then describe
    neither arm.
    """
    if cold_start not in COLD_START_MODES:
        raise ValueError(
            f"recalibrate: cold_start must be one of {COLD_START_MODES}, got {cold_start!r}"
        )
    corr = rolling_signed_correction(frame, window=window, cold_start=cold_start)
    out = frame.copy()
    out["correction"] = out["origin"].map(corr)
    out = out[out["correction"].notna()].copy()
    out["y_pred"] = out["y_pred"] + out["correction"]
    return out.sort_values(["origin", "hour"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# metrics — the repo's own implementations, never reimplemented
# --------------------------------------------------------------------------
def metrics(frame: pd.DataFrame) -> dict:
    """MAE / RMSE / rMAE via src.evaluation.metrics.

    Same long-frame -> hourly-series conversion as run_ood_stress._metrics
    and export_tables._hourly_metrics, so the numbers are comparable to the
    frozen OOD table row for row.

    sMAPE is deliberately absent: the live window contains an hour with
    actual == predicted == 0 that makes epftoolbox's sMAPE NaN for the whole
    series (decisions.md 2026-08-04), and the zero-safe variant lives inside
    run_ood_stress.py, which this script must not import from. rMAE is the
    honest column here anyway.
    """
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    pred = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")
    return {
        "MAE": float(mae(real.values, pred.values)),
        "RMSE": float(rmse(real.values, pred.values)),
        "rMAE": float(rmae(real, pred, m="W")),
    }


def compare_one(frame: pd.DataFrame, window: int, cold_start: str = "exclude") -> dict:
    """Raw vs recalibrated for one model at one window.

    The raw arm is re-scored on the SAME days that survive the cold start,
    not on all 173. Comparing a 173-day raw rMAE against a 159-day
    recalibrated one would let the cold-start exclusion move the headline by
    itself — the improvement would partly be a change of sample, and on a
    window where early days happened to be bad it could look like a fix
    while nothing had been fixed. The full-window raw metric is still
    reported alongside, as the link back to the published v1.1-ood row.
    """
    recalibrated = recalibrate(frame, window=window, cold_start=cold_start)
    days = recalibrated["origin"].unique()
    raw_subset = frame[frame["origin"].isin(days)]

    raw_all = metrics(frame)
    raw = metrics(raw_subset)
    new = metrics(recalibrated)
    return {
        "window": window,
        # No underscore in the column name: a bare '_' is a LaTeX error in
        # text mode (same trap as export_tables.py's 'days' column).
        "n days": int(len(days)),
        "n_days_raw": int(raw_subset["origin"].nunique()),
        "rMAE raw (all days)": raw_all["rMAE"],
        "MAE raw": raw["MAE"],
        "MAE recal": new["MAE"],
        "rMAE raw": raw["rMAE"],
        "rMAE recal": new["rMAE"],
        "rMAE change": new["rMAE"] - raw["rMAE"],
        "_frame": recalibrated,
    }


def sweep(windows: list[int], cold_start: str = "exclude") -> tuple[pd.DataFrame, dict]:
    """Every model x every window. Nothing is filtered out afterwards —
    the whole sweep is reported, so a favourable window cannot be selected
    after the fact."""
    rows, frames = [], {}
    for model, filename in OOD_FILES.items():
        frame = load_ood_frame(OOD_DIR / filename)
        for window in windows:
            row = compare_one(frame, window=window, cold_start=cold_start)
            frames[(model, window)] = row.pop("_frame")
            rows.append({"model": model, **row})

    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table["model"], categories=list(OOD_FILES), ordered=True)
    return table.sort_values(["model", "window"]).set_index(["model", "window"]), frames


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def _fmt_cell(v) -> str:
    """3 decimals, integers as integers, never a signed zero.

    Same convention as export_tables._fmt_cell minus the p-value branch —
    this table holds no probabilities. "-0.000" reads as a signed zero and
    means nothing in a thesis table.
    """
    if pd.isna(v):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    out = f"{float(v):.3f}"
    return "0.000" if out == "-0.000" else out


def export(table: pd.DataFrame, caption: str, dry_run: bool = False) -> None:
    if dry_run:
        print("\n--- ood_recalibration (dry run, nothing written) ---")
        print(table.to_string())
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "ood_recalibration.csv"
    tex_path = TABLES_DIR / "ood_recalibration.tex"
    table.to_csv(csv_path)
    table.apply(lambda col: col.map(_fmt_cell)).to_latex(
        tex_path,
        escape=False,
        caption=caption,
        label="tab:ood-recalibration",
        position="htbp",
        column_format="l" * table.index.nlevels + "r" * len(table.columns),
    )
    print(f"wrote {_rel(csv_path)} and {_rel(tex_path)}")


def crossings(table: pd.DataFrame) -> pd.DataFrame:
    """Model/window combinations that cross from rMAE >= 1.0 to < 1.0.

    naive is excluded deliberately. rMAE is normalised BY a naive forecast,
    so naive sits below 1.0 to begin with; counting it among the rows that
    "beat naive" would be circular, and here it would also be false —
    correcting naive makes it worse, which is the sanity check that this
    correction is not a free lunch.
    """
    trained = table.drop(index="naive", level="model", errors="ignore")
    return trained[(trained["rMAE raw"] >= 1.0) & (trained["rMAE recal"] < 1.0)]


def _caption(table: pd.DataFrame, windows: list[int], cold_start: str) -> str:
    crossed = crossings(table)
    n_trained = len(table.drop(index="naive", level="model", errors="ignore"))
    cold = (
        f"the first {min(windows)}--{max(windows)} days (one per window) are excluded "
        "for want of prior history"
        if cold_start == "exclude"
        else "an expanding window is used until the nominal window fills, so only the "
        "first day is excluded"
    )
    return (
        "Post-hoc bias recalibration of the OOD stress test (supplementary to "
        "addendum v1.1-ood; not a benchmark result). Each day's forecast is "
        "shifted by the mean signed error of the preceding $w$ days only, so "
        "no future information enters a correction. Raw and recalibrated "
        "columns are scored on the identical day subset; the "
        "\\emph{rMAE raw (all days)} column is the published 173-day figure, "
        f"for reference. Cold start: {cold}. All tested windows are reported "
        f"({', '.join(str(w) for w in windows)} days), none selected post hoc. "
        + (
            f"{len(crossed)} of {n_trained} trained model--window combinations "
            "cross from rMAE $\\geq$ 1.0 to below it ("
            + ", ".join(sorted({str(m) for m, _ in crossed.index}))
            + "); the rest improve without reaching it. naive is a reference "
            "row and is excluded from that count: rMAE is normalised by a "
            "naive forecast, and correcting naive makes it worse, which is "
            "the sanity check that the correction is not a free lunch."
            if len(crossed)
            else "No trained model--window combination reaches rMAE 1.0: "
            "correcting the level does not by itself restore skill relative "
            "to naive."
        )
        + " Data: Energy-Charts (Fraunhofer ISE) / SMARD.de, CC BY 4.0."
    )


def main(
    windows: list[int] | None = None,
    cold_start: str = "exclude",
    dry_run: bool = False,
) -> pd.DataFrame:
    windows = list(windows or DEFAULT_WINDOWS)
    if cold_start not in COLD_START_MODES:
        raise ValueError(f"main: cold_start must be one of {COLD_START_MODES}, got {cold_start!r}")

    table, frames = sweep(windows, cold_start=cold_start)

    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for (model, window), frame in frames.items():
            stem = model.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            frame.to_csv(OUT_DIR / f"{stem}_w{window}.csv", index=False)
        print(f"wrote {len(frames)} recalibrated frames -> {_rel(OUT_DIR)}")

    shown = table.drop(columns=["n_days_raw"])
    print(f"\nOOD bias recalibration — cold start: {cold_start}\n")
    print(shown.round(3).to_string())

    crossed = crossings(table)
    print("\n" + "-" * 70)
    if crossed.empty:
        print(
            "No TRAINED model/window combination crosses below rMAE 1.0.\n"
            "Correcting the level does NOT by itself restore skill relative to\n"
            "naive: the frozen models remain worse than a naive forecast even\n"
            "after their systematic offset is removed."
        )
    else:
        print("Crossed from rMAE >= 1.0 to < 1.0 after correction (trained models only):")
        print(crossed[["rMAE raw", "rMAE recal"]].round(3).to_string())
    if "naive" in table.index.get_level_values("model"):
        naive = table.loc["naive"]
        print(
            "\nnaive (reference, not a competitor): rMAE "
            f"{naive['rMAE raw'].min():.3f}-{naive['rMAE raw'].max():.3f} raw -> "
            f"{naive['rMAE recal'].min():.3f}-{naive['rMAE recal'].max():.3f} corrected. "
            "It gets WORSE,\nwhich is the sanity check that this correction is not a free lunch."
        )
    print("-" * 70)

    export(shown, _caption(table, windows, cold_start), dry_run=dry_run)
    return table


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window", type=int, nargs="+", default=DEFAULT_WINDOWS,
        help="rolling window length(s) in days to sweep",
    )
    parser.add_argument(
        "--cold-start", choices=COLD_START_MODES, default="exclude",
        help="how to treat days with no full prior window (never filled)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(windows=args.window, cold_start=args.cold_start, dry_run=args.dry_run)
