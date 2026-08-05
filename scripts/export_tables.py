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
import json
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
OOD_DIR = REPO_ROOT / "data" / "processed" / "ood"
FROZEN_META = REPO_ROOT / "models" / "frozen" / "metadata.json"

# Block-bootstrap sensitivity ranges for the regime-split DM comparison,
# (min p, max p) across block lengths. Produced by
# scripts/run_dm_ensembles.py, which this script never runs — so they
# cannot be interpolated here and are pinned as literals instead
# (tests/test_regressions_scripts.py fails loudly if they drift).
#
# They are not optional decoration: CLAUDE.md's claim discipline requires
# the stressed-day result and its not-robust-over-all-days counterpart to
# always be reported together. Quoting the stressed range without the
# all-days range would turn a hedged finding into an unqualified claim, so
# neither may be dropped from the caption.
BOOTSTRAP_P_RANGE_STRESSED = (0.006, 0.044)
BOOTSTRAP_P_RANGE_ALL = (0.013, 0.057)

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


def _fmt_cell(v, *, is_pvalue: bool = False) -> str:
    """Format a table cell for LaTeX.

    p-values underflow to exactly 0.0 for hopelessly-beaten comparisons
    (e.g. anything versus naive). Printing "0.0000" in a thesis table
    asserts an exact zero probability, which is indefensible — report a
    bound instead. That reasoning applies to PROBABILITIES ONLY: a metric
    of exactly 1.000 is a measured value, not an underflowed one, and
    printing "$>$0.999" for it (rMAE 1.000 is the load-bearing OOD number)
    would replace a fact with a bound. Hence `is_pvalue`.

    Counts stay integers; other floats get 3 decimals, never as "-0.000",
    which reads as a signed zero and means nothing in a thesis table.
    """
    if pd.isna(v):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if is_pvalue:
        if 0.0 <= v < 0.001:
            return r"$<$0.001"
        if v > 0.999 and v <= 1.0:
            return r"$>$0.999"
    out = f"{v:.3f}"
    return "0.000" if out == "-0.000" else out


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
    summary = OOD_DIR / "ood_summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary, index_col=0)
    keep = [c for c in ["MAE", "RMSE", "sMAPE", "rMAE", "MAE ratio"] if c in df.columns]
    df = df[keep].rename(columns={"MAE ratio": "MAE vs benchmark"})
    order = [m for m in MODEL_ORDER if m in df.index]
    return df.loc[order].sort_values("rMAE")


def _origin_span(frame: pd.DataFrame) -> tuple[int, str, str]:
    """(number of origins, first date, last date) for a long frame.

    Every caption that quotes a coverage figure gets it from here rather
    than from a literal, so re-running after a window or data change can
    never leave a table contradicting its own caption.
    """
    origins = pd.DatetimeIndex(pd.unique(frame["origin"])).sort_values()
    return len(origins), origins[0].strftime("%Y-%m-%d"), origins[-1].strftime("%Y-%m-%d")


def _ood_facts() -> dict:
    """Facts the OOD caption states, read back from the OOD artifacts.

    The live mean price, the day count and the date range come from the
    evaluated frame itself; the training-era price level and the freeze
    dates come from models/frozen/metadata.json, which is what the frozen
    models were actually calibrated on.
    """
    frame = pd.read_csv(OOD_DIR / "naive.csv", parse_dates=["origin"])
    n_days, first, last = _origin_span(frame)
    meta = json.loads(FROZEN_META.read_text(encoding="utf-8"))
    return {
        "n_days": n_days,
        "first": first,
        "last": last,
        "live_mean": float(frame["y_true"].mean()),
        "benchmark_mean": float(meta["benchmark_price_mean"]),
        "train_start": meta["train_start"],
        "frozen_on": meta["frozen_on"],
    }


def _rel(path: Path) -> str:
    """Repo-relative path for display, falling back to the absolute path.

    Path.relative_to raises when the target sits outside the repo — which
    happens whenever OUT_DIR is redirected (tests). A cosmetic display call
    must never abort a function that has already written its files.
    Deliberately a local copy of the same helper in run_ood_stress.py:
    scripts do not import from each other.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def export(
    df: pd.DataFrame,
    stem: str,
    caption: str,
    label: str,
    dry_run: bool,
    pvalue_cols: list[str] | None = None,
) -> None:
    """Write <stem>.csv and <stem>.tex.

    `pvalue_cols` names the columns whose values are probabilities, per
    column rather than per table: dm_regime_split mixes one p-value column
    with a day count and two MAE columns, so a single table-wide flag would
    misformat three of its four columns.
    """
    csv_path, tex_path = OUT_DIR / f"{stem}.csv", OUT_DIR / f"{stem}.tex"
    if dry_run:
        print(f"\n--- {stem} ---")
        print(df.to_string())
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    # _bold_best bolds within each target, so it needs a 'target' index
    # level — which only build_canonical() produces. Dispatching on the
    # column names instead used to send any (model)-indexed table whose
    # columns happen to collapse to exactly METRICS (an OOD summary without
    # 'MAE ratio') into _bold_best, where it died on the missing level —
    # after the earlier tables had already been overwritten.
    pvalues = set(pvalue_cols or ())
    if "target" in (df.index.names or []):
        shown = df.apply(_bold_best)
    else:
        shown = df.apply(
            lambda col: col.map(lambda v: _fmt_cell(v, is_pvalue=col.name in pvalues))
        )

    shown.to_latex(
        tex_path,
        escape=False,
        caption=caption,
        label=label,
        position="htbp",
        column_format="l" * shown.index.nlevels + "r" * len(shown.columns),
    )
    print(f"wrote {_rel(csv_path)} and {_rel(tex_path)}")


def main(dry_run: bool = False) -> None:
    canonical = build_canonical()

    hourly = {m: load_long_frame(HOURLY_DIR / f) for m, f in HOURLY_FILES.items()}
    n_origins, first_origin, last_origin = _origin_span(hourly["LSTM"])

    export(
        canonical,
        "results_canonical",
        caption=(
            "Forecast accuracy by model and target, EPEX-DE test period "
            f"{first_origin} to {last_origin} ({n_origins} origins). Best "
            "value per column bolded separately within each target — hourly "
            "and daily numbers live on different scales, so a single "
            "column-wide winner would be spurious; naive is a reference row, "
            "not a competitor. Daily-direct is unavailable for the "
            "ensembles, whose members are hourly."
        ),
        label="tab:results-canonical",
        dry_run=dry_run,
    )

    dm = dm_matrix(hourly, method="hac")
    export(
        dm,
        "dm_tests",
        caption=(
            "Pairwise one-sided Diebold-Mariano p-values, multivariate "
            "(24-hour vector, $L_1$ norm), Newey-West HAC corrected, over "
            f"the {n_origins} test origins {first_origin} to {last_origin}. "
            "Cell (row, column) tests the alternative that the row model is "
            "more accurate than the column model."
        ),
        label="tab:dm-tests",
        dry_run=dry_run,
        pvalue_cols=dm.columns.tolist(),
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
    n_all = int(split.loc["all", "days"])
    n_stressed = int(split.loc["stressed", "days"])
    n_calm = int(split.loc["calm", "days"])
    p_all = float(split.loc["all", "DM p (HAC)"])
    export(
        split,
        "dm_regime_split",
        caption=(
            "Regime-aware versus static ensemble by regime (stress "
            f"threshold {threshold:.2f} EUR/MWh), with Newey-West HAC "
            "corrected Diebold-Mariano p-values, over "
            f"{n_all} test origins ({n_stressed} stressed, {n_calm} calm). "
            "The improvement is concentrated on stressed days, where it is "
            "significant under every dependence correction applied ($p$ "
            f"between {BOOTSTRAP_P_RANGE_STRESSED[0]:.3f} and "
            f"{BOOTSTRAP_P_RANGE_STRESSED[1]:.3f}). Over the full test "
            f"period it is significant under HAC ($p={p_all:.3f}$) but not "
            "robust across block-bootstrap block lengths "
            f"({BOOTSTRAP_P_RANGE_ALL[0]:.3f} to "
            f"{BOOTSTRAP_P_RANGE_ALL[1]:.3f}), and is therefore not "
            "claimed. The calm subset is a sanity check that the regime "
            "switch does not fire where it should not, not independent "
            "corroboration. Block-bootstrap sensitivity is reported by "
            "scripts/run\\_dm\\_ensembles.py."
        ),
        label="tab:dm-regime-split",
        dry_run=dry_run,
        pvalue_cols=["DM p (HAC)"],
    )

    ood = build_ood()
    if ood is None:
        print("\n(no OOD summary found — skipping the addendum table)")
        return
    facts = _ood_facts()
    export(
        ood,
        "ood_stress",
        caption=(
            "OOD stress test (addendum v1.1-ood): models frozen on the "
            f"benchmark era ({facts['train_start']} to {facts['frozen_on']}) "
            "and applied without recalibration to live DE-LU data, "
            f"{facts['n_days']} days from {facts['first']} to "
            f"{facts['last']}. Mean price {facts['live_mean']:.2f} versus "
            f"{facts['benchmark_mean']:.2f} EUR/MWh at training "
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
