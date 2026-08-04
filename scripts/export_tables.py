"""Export the canonical results table and the DM tables — scripts/export_tables.py

Feeds the thesis directly (thesis/outline.md):
  reports/tables/results_canonical.{tex,csv}  -> sections 4-2, 4-3, 4-4
  reports/tables/dm_tests.{tex,csv}           -> section 4-5
  reports/tables/dm_regime_split.{tex,csv}    -> section 4-5

Targets
  hourly            hourly 24-price vector, data/processed/baselines/
  daily-direct      models trained on the daily baseload directly,
                    data/processed/daily_direct/
  daily-aggregated  hourly forecasts averaged to daily baseload
The direct-vs-aggregated pair is RQ4 (section 4-4); the comparison only
means anything because both arms cover the identical 728 origins.

rMAE note: epftoolbox's rMAE accepts only sub-daily frequencies ('1h',
'30T', '15T', '5T'), so it cannot be called on the daily targets. For those
the same definition is applied explicitly — MAE divided by the MAE of a
weekly-lag naive forecast (previous week, same weekday), which is what
m='W' means on the hourly arm. The hourly rows still use epftoolbox's own
implementation, so the benchmark-comparable numbers stay untouched.

Export once, in final form (export-results skill). Run BEFORE tagging
v1.0-results — the PreToolUse hook blocks exports once the tag exists.

Usage:
    python scripts/export_tables.py
    python scripts/export_tables.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ensemble import regime_labels
from src.evaluation.metrics import diebold_mariano_hac, mae, rmae, rmse, smape
from src.evaluation.results import daily_baseload, dm_matrix, load_long_frame
from src.evaluation.walk_forward import load_evaluation_config

OUT_DIR = REPO_ROOT / "reports" / "tables"
HOURLY_DIR = REPO_ROOT / "data" / "processed" / "baselines"
DAILY_DIR = REPO_ROOT / "data" / "processed" / "daily_direct"

# Fixed presentation order (export-results skill).
HOURLY_FILES = {
    "naive": "naive.csv",
    "SARIMAX": "sarimax.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "LSTM": "lstm.csv",
    "Ensemble (static)": "ensemble_static.csv",
    "Ensemble (regime-aware)": "ensemble_regime.csv",
}
DAILY_FILES = {
    "naive": "naive.csv",
    "SARIMAX": "dailysarimax.csv",
    "LEAR-LASSO": "dailylear_lasso.csv",
    "LightGBM": "dailylightgbm.csv",
    "LSTM": "dailylstm.csv",
}
MODEL_ORDER = list(HOURLY_FILES)
METRICS = ["MAE", "RMSE", "sMAPE", "rMAE"]
TARGETS = ["hourly", "daily-direct", "daily-aggregated"]


def _hourly_metrics(frame: pd.DataFrame) -> dict:
    """Metrics for an hourly long frame, via the epftoolbox wrappers.

    rMAE needs an hourly DatetimeIndex (weekly-lag naive denominator), so
    the long frame is rebuilt into an hourly series first — same conversion
    as run_ensemble.py::_metrics_row.
    """
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    pred = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")
    return {
        "MAE": mae(real.values, pred.values),
        "RMSE": rmse(real.values, pred.values),
        "sMAPE": smape(real.values, pred.values) * 100,
        "rMAE": rmae(real, pred, m="W"),
    }


def _daily_metrics(frame: pd.DataFrame) -> dict:
    """Metrics for a daily frame (one baseload value per origin).

    rMAE is computed explicitly against a weekly-lag naive forecast because
    epftoolbox's rMAE rejects daily frequency (see module docstring).
    """
    f = frame.sort_values("origin").reset_index(drop=True)
    real = f["y_true"].to_numpy(dtype=float)
    pred = f["y_pred"].to_numpy(dtype=float)

    naive = real[:-7]
    target = real[7:]
    naive_mae = float(np.abs(target - naive).mean())
    if naive_mae <= 0:
        raise ValueError("daily rMAE: weekly-lag naive MAE is zero")

    return {
        "MAE": float(np.abs(real - pred).mean()),
        "RMSE": float(np.sqrt(((real - pred) ** 2).mean())),
        "sMAPE": float((2 * np.abs(real - pred) / (np.abs(real) + np.abs(pred))).mean() * 100),
        "rMAE": float(np.abs(real - pred).mean() / naive_mae),
    }


def _fmt_cell(v) -> str:
    """Format a table cell for LaTeX.

    p-values underflow to exactly 0.0 for hopelessly-beaten comparisons
    (e.g. anything versus naive). Printing "0.0000" in a thesis table
    asserts an exact zero probability, which is indefensible — report a
    bound instead. Counts stay integers; other floats get 3 decimals.
    """
    if pd.isna(v):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if 0.0 <= v < 0.001:
        return r"$<$0.001"
    if v > 0.999 and v <= 1.0:
        return r"$>$0.999"
    return f"{v:.3f}"


def _bold_best(col: pd.Series) -> pd.Series:
    """Bold the best (lowest) value in a metric column, WITHIN each target.

    Bolding down the whole column would compare hourly against daily
    numbers, which live on different scales and answer different questions —
    a spurious "winner" a reader would take at face value. naive is a
    reference row, not a competitor, so it is never bolded.
    """
    formatted = col.map(lambda v: "--" if pd.isna(v) else f"{v:.2f}")
    for target in col.index.get_level_values("target").unique():
        block = col[col.index.get_level_values("target") == target]
        block = block.drop(index=[i for i in block.index if i[0] == "naive"], errors="ignore")
        if block.notna().any():
            best = block.idxmin()
            formatted.loc[best] = rf"\textbf{{{col.loc[best]:.2f}}}"
    return formatted


def build_canonical() -> pd.DataFrame:
    hourly = {m: load_long_frame(HOURLY_DIR / f) for m, f in HOURLY_FILES.items()}
    daily_direct = {m: load_long_frame(DAILY_DIR / f) for m, f in DAILY_FILES.items()}

    origins = {m: set(f["origin"].unique()) for m, f in daily_direct.items()}
    base = next(iter(origins.values()))
    if any(o != base for o in origins.values()):
        raise ValueError("daily-direct frames cover different origin sets")

    rows = []
    for model in MODEL_ORDER:
        rows.append({"model": model, "target": "hourly", **_hourly_metrics(hourly[model])})
        rows.append(
            {
                "model": model,
                "target": "daily-aggregated",
                **_daily_metrics(daily_baseload(hourly[model])),
            }
        )
        if model in daily_direct:
            rows.append(
                {"model": model, "target": "daily-direct", **_daily_metrics(daily_direct[model])}
            )
        else:
            # No daily-direct ensemble was built: the ensemble members are
            # hourly, so its daily arm exists only by aggregation. Recorded
            # as absent rather than silently omitted.
            rows.append({"model": model, "target": "daily-direct", **dict.fromkeys(METRICS, np.nan)})

    table = pd.DataFrame(rows)
    table["target"] = pd.Categorical(table["target"], categories=TARGETS, ordered=True)
    table["model"] = pd.Categorical(table["model"], categories=MODEL_ORDER, ordered=True)
    return table.sort_values(["model", "target"]).set_index(["model", "target"])[METRICS]


def build_ood() -> pd.DataFrame | None:
    """OOD addendum table (v1.1-ood), if the OOD run has been performed.

    Deliberately a separate table from results_canonical: these numbers come
    from a different data source (live Energy-Charts) and a different
    protocol (frozen models, no recalibration), and merging them into the
    canonical table would invite reading a 2026 MAE against a 2016-17 MAE as
    though they were comparable. They are not — the market level differs by
    ~2.8x, which is the whole point.
    """
    summary = REPO_ROOT / "data" / "processed" / "ood" / "ood_summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary, index_col=0)
    keep = [c for c in ["MAE", "RMSE", "sMAPE", "rMAE", "MAE ratio"] if c in df.columns]
    df = df[keep].rename(columns={"MAE ratio": "MAE vs benchmark"})
    order = [m for m in MODEL_ORDER if m in df.index]
    return df.loc[order].sort_values("rMAE")


def export(df: pd.DataFrame, stem: str, caption: str, label: str, dry_run: bool) -> None:
    csv_path, tex_path = OUT_DIR / f"{stem}.csv", OUT_DIR / f"{stem}.tex"
    if dry_run:
        print(f"\n--- {stem} ---")
        print(df.to_string())
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    if df.columns.tolist() == METRICS:
        shown = df.apply(_bold_best)
    else:
        shown = df.map(_fmt_cell)

    shown.to_latex(
        tex_path,
        escape=False,
        caption=caption,
        label=label,
        position="htbp",
        column_format="l" * shown.index.nlevels + "r" * len(shown.columns),
    )
    print(f"wrote {csv_path.relative_to(REPO_ROOT)} and {tex_path.relative_to(REPO_ROOT)}")


def main(dry_run: bool = False) -> None:
    canonical = build_canonical()
    export(
        canonical,
        "results_canonical",
        caption=(
            "Forecast accuracy by model and target, EPEX-DE test period "
            "2016-01-04 to 2017-12-31 (728 origins). Best value per column "
            "in bold; naive is a reference row, not a competitor. "
            "Daily-direct is unavailable for the ensembles, whose members "
            "are hourly."
        ),
        label="tab:results-canonical",
        dry_run=dry_run,
    )

    hourly = {m: load_long_frame(HOURLY_DIR / f) for m, f in HOURLY_FILES.items()}
    dm = dm_matrix(hourly, method="hac")
    export(
        dm,
        "dm_tests",
        caption=(
            "Pairwise one-sided Diebold-Mariano p-values, multivariate "
            "(24-hour vector, $L_1$ norm), Newey-West HAC corrected. "
            "Cell (row, column) tests the alternative that the row model is "
            "more accurate than the column model."
        ),
        label="tab:dm-tests",
        dry_run=dry_run,
    )

    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    labels = regime_labels(hourly["LSTM"], threshold=threshold)
    piv = {
        m: f.pivot(index="origin", columns="hour", values="y_pred").sort_index()
        for m, f in hourly.items()
    }
    truth = hourly["LSTM"].pivot(index="origin", columns="hour", values="y_true").sort_index()

    split_rows = []
    for subset, days in (
        ("all", sorted(labels)),
        ("stressed", sorted(o for o, v in labels.items() if v == "stressed")),
        ("calm", sorted(o for o, v in labels.items() if v == "calm")),
    ):
        idx = pd.DatetimeIndex(days)
        real = truth.loc[idx]
        pr = piv["Ensemble (regime-aware)"].loc[idx]
        ps = piv["Ensemble (static)"].loc[idx]
        split_rows.append(
            {
                "subset": subset,
                # No underscore: a bare '_' is a LaTeX error in text mode.
                "days": len(idx),
                "MAE regime-aware": float(np.abs(real.values - pr.values).mean()),
                "MAE static": float(np.abs(real.values - ps.values).mean()),
                "DM p (HAC)": diebold_mariano_hac(
                    p_real=real.values, p_pred_1=ps.values, p_pred_2=pr.values
                ),
            }
        )
    split = pd.DataFrame(split_rows).set_index("subset")
    export(
        split,
        "dm_regime_split",
        caption=(
            "Regime-aware versus static ensemble by regime, with "
            "Newey-West HAC corrected Diebold-Mariano p-values. The "
            "improvement is concentrated on stressed days, where it is "
            "significant under every dependence correction applied "
            "($p$ between 0.006 and 0.044). Over the full test period it "
            "is significant under HAC ($p=0.023$) but not robust across "
            "block-bootstrap block lengths (0.013 to 0.057), and is "
            "therefore not claimed. The calm subset is a sanity check that "
            "the regime switch does not fire where it should not, not "
            "independent corroboration. Block-bootstrap sensitivity is "
            "reported by scripts/run\\_dm\\_ensembles.py."
        ),
        label="tab:dm-regime-split",
        dry_run=dry_run,
    )

    ood = build_ood()
    if ood is None:
        print("\n(no OOD summary found — skipping the addendum table)")
        return
    export(
        ood,
        "ood_stress",
        caption=(
            "OOD stress test (addendum v1.1-ood): models frozen on the "
            "benchmark era (2015-01-05 to 2017-12-31) and applied without "
            "recalibration to live DE-LU data, 173 days from 2026-01-08 to "
            "2026-06-29. Mean price 98.67 versus 34.69 EUR/MWh at training "
            "time. Every trained model exceeds rMAE 1.0, i.e. performs worse "
            "than a naive forecast; naive alone stays below it because it "
            "carries no frozen parameters. The in-era ranking inverts: "
            "LightGBM and LSTM degrade most, SARIMAX and LEAR-LASSO least. "
            "Sorted by rMAE. Not comparable row-for-row with the canonical "
            "results table, which uses a different market period. "
            "Data: Energy-Charts (Fraunhofer ISE) / SMARD.de, CC BY 4.0."
        ),
        label="tab:ood-stress",
        dry_run=dry_run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
