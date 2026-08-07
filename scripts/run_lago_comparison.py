"""Formal comparison against Lago et al. (2021), plus DM tests — scripts/run_lago_comparison.py

Two deliverables, both NEW files (nothing frozen is touched):

  reports/tables/lago_comparison.{csv,tex}   our hourly results vs the paper
  reports/tables/dm_vs_lago.{csv,tex}        DM tests vs their own forecasts

WHY TWO SOURCES FOR THE SAME PAPER. The paper's Tables 2/3 report metrics,
and the toolbox separately ships the day-ahead forecasts those models
produced. Scoring the shipped forecasts with our own metric code reproduces
the paper EXACTLY for all four DNNs and the DNN Ensemble — which is the
control proving our alignment and metrics are right — but NOT for any LEAR
variant, where the shipped forecasts score consistently better than the
printed table (LEAR 1092: 4.108 printed vs 3.930 recomputed; LEAR Ensemble:
3.955 vs 3.609).

Both are therefore reported, as separate rows with an explicit source. This
is not a detail: comparing our LEAR-LASSO (3.899) against the printed 4.108
would show a comfortable win, while against the shipped forecasts (3.930)
the same model is barely ahead. The second is the defensible comparison —
identical data, identical metric code, no transcription — and is the one
logs/decisions.md 2026-07-31 already used for the week-5 checkpoint.

PROTOCOL EQUIVALENCE. Recorded in logs/decisions.md 2026-08-07. Verified
match: dataset, 728-origin test window, rMAE naive2 denominator, and the
asinh-median VST (epftoolbox LEAR.recalibrate applies scaling(...,
'Invariant'), which our wrapper inherits by calling it directly). Verified
difference: their LEAR spans four calibration windows (56/84/1092/1456) with
the LEAR Ensemble as their arithmetic mean, while ours runs the 1092 window
only — so our LEAR-LASSO is comparable to their LEAR 1092 variant, not to
their LEAR Ensemble. Their DNN Ensemble averages four runs of ONE model
family; our ensembles average different families.

Usage:
    python scripts/run_lago_comparison.py
    python scripts/run_lago_comparison.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import diebold_mariano_hac, mae, rmae, rmse, smape

PUBLISHED_FC = REPO_ROOT / "data" / "raw" / "Forecasts_DE_DNN_LEAR_ensembles.csv"
CANONICAL = REPO_ROOT / "reports" / "tables" / "results_canonical.csv"
BASELINES = REPO_ROOT / "data" / "processed" / "baselines"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

# Lago et al. (2021), Applied Energy 293:116983, Tables 2 and 3, EPEX-DE rows.
# Transcribed from the paper; pinned by tests/test_lago_comparison.py so a
# silent edit here fails loudly.
PAPER = {
    "DNN 1": dict(rMAE=0.407, MAE=3.716, MAPE=77.145, sMAPE=14.970, RMSE=6.796),
    "DNN 2": dict(rMAE=0.422, MAE=3.850, MAPE=137.449, sMAPE=15.356, RMSE=7.304),
    "DNN 3": dict(rMAE=0.406, MAE=3.706, MAPE=100.214, sMAPE=15.508, RMSE=6.271),
    "DNN 4": dict(rMAE=0.394, MAE=3.592, MAPE=90.578, sMAPE=14.680, RMSE=6.080),
    "LEAR 56": dict(rMAE=0.506, MAE=4.619, MAPE=129.763, sMAPE=17.600, RMSE=8.122),
    "LEAR 84": dict(rMAE=0.499, MAE=4.555, MAPE=133.580, sMAPE=17.491, RMSE=7.923),
    "LEAR 1092": dict(rMAE=0.450, MAE=4.108, MAPE=128.295, sMAPE=16.984, RMSE=6.996),
    "LEAR 1456": dict(rMAE=0.451, MAE=4.118, MAPE=124.191, sMAPE=17.054, RMSE=6.987),
    "DNN Ensemble": dict(rMAE=0.374, MAE=3.413, MAPE=94.434, sMAPE=14.078, RMSE=5.927),
    "LEAR Ensemble": dict(rMAE=0.433, MAE=3.955, MAPE=122.412, sMAPE=15.747, RMSE=7.079),
}

OUR_FILES = {
    "naive": "naive.csv",
    "SARIMAX": "sarimax.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "LSTM": "lstm.csv",
    "Ensemble (static)": "ensemble_static.csv",
    "Ensemble (regime-aware)": "ensemble_regime.csv",
}

# Ours vs theirs, for the DM matrix. Our four strongest by rMAE, against
# both of their ensembles and their best individual model.
OUR_DM = ["Ensemble (regime-aware)", "Ensemble (static)", "LSTM", "LEAR-LASSO"]
THEIR_DM = ["DNN Ensemble", "DNN 4", "LEAR Ensemble", "LEAR 1092"]

MAPE_NOTE = (
    "paper reports MAPE ~10x sMAPE on this market (negative and near-zero "
    "prices); the paper itself flags MAPE as unreliable here, which is why "
    "this thesis excludes it (section 3-5)"
)


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_published() -> pd.DataFrame:
    if not PUBLISHED_FC.exists():
        raise SystemExit(
            f"published forecasts not found at {_rel(PUBLISHED_FC)}. Download from "
            "https://github.com/jeslago/epftoolbox (forecasts of the benchmark "
            "models, DE)."
        )
    return pd.read_csv(PUBLISHED_FC, index_col=0, parse_dates=True)


def our_hourly(name: str) -> pd.Series:
    df = pd.read_csv(BASELINES / OUR_FILES[name], parse_dates=["origin"])
    ts = pd.DatetimeIndex(df["origin"] + pd.to_timedelta(df["hour"], unit="h"))
    return pd.Series(df["y_pred"].values, index=ts).sort_index()


def _metrics(real: pd.Series, pred: pd.Series) -> dict:
    r = real.to_frame("price")
    p = pred.reindex(real.index).to_frame("price")
    return {
        "rMAE": float(rmae(r, p, m="W")),
        "MAE": float(mae(r.values, p.values)),
        "sMAPE": float(smape(r.values, p.values) * 100),
        "RMSE": float(rmse(r.values, p.values)),
    }


def as_matrix(series: pd.Series) -> np.ndarray:
    """Hourly series -> [n_days, 24], the multivariate DM shape."""
    frame = series.to_frame("v")
    frame["day"] = frame.index.normalize()
    frame["hour"] = frame.index.hour
    piv = frame.pivot(index="day", columns="hour", values="v").sort_index()
    return piv.to_numpy(dtype=float)


def verify_alignment(published: pd.DataFrame) -> pd.DatetimeIndex:
    """Fail loudly on any index mismatch before a single p-value is computed.

    An off-by-one day or a DST-shifted hour would not raise anywhere
    downstream — it would quietly pair each forecast with the wrong truth and
    produce p-values that look entirely reasonable. This is the one check
    that has to happen before the tests, not after.
    """
    ours = our_hourly("LEAR-LASSO")
    theirs = published.index

    if len(ours) != len(theirs):
        raise SystemExit(f"length mismatch: ours {len(ours)} hours, theirs {len(theirs)}")
    if not ours.index.equals(theirs):
        first = next((i for i in range(len(ours)) if ours.index[i] != theirs[i]), None)
        raise SystemExit(
            f"index mismatch at position {first}: ours {ours.index[first]}, "
            f"theirs {theirs[first]}"
        )

    # And the realized prices must agree, else the two pipelines are not
    # looking at the same market (the check week5_checkpoint.py also makes).
    our_true = pd.read_csv(BASELINES / OUR_FILES["LEAR-LASSO"], parse_dates=["origin"])
    ts = pd.DatetimeIndex(our_true["origin"] + pd.to_timedelta(our_true["hour"], unit="h"))
    y_true = pd.Series(our_true["y_true"].values, index=ts).sort_index()
    max_diff = float((y_true - published["Real price"].reindex(y_true.index)).abs().max())
    if max_diff > 1e-6:
        raise SystemExit(f"realized prices differ by up to {max_diff:.6f} — not the same data")

    print(
        f"alignment OK: {len(theirs)} hours, {len(theirs) // 24} days, "
        f"{theirs.min().date()} -> {theirs.max().date()}, "
        f"max |y_true difference| = {max_diff:.6f}"
    )
    return theirs


# --------------------------------------------------------------------------
# task 1: comparison table
# --------------------------------------------------------------------------
def build_comparison(published: pd.DataFrame) -> pd.DataFrame:
    canon = pd.read_csv(CANONICAL)
    hourly = canon[canon["target"] == "hourly"].set_index("model")
    real = published["Real price"]

    rows = []
    for name in OUR_FILES:
        row = hourly.loc[name]
        rows.append(
            {
                "model": name,
                "source": "this thesis",
                "rMAE": float(row["rMAE"]),
                "MAE": float(row["MAE"]),
                "sMAPE": float(row["sMAPE"]),
                "RMSE": float(row["RMSE"]),
                "MAPE": np.nan,
                "note": (
                    "MAPE deliberately not computed (negative prices); "
                    "ensemble averages different model FAMILIES, unlike theirs"
                    if "Ensemble" in name
                    else "MAPE deliberately not computed (negative prices)"
                ),
            }
        )

    for name, vals in PAPER.items():
        recomputed = _metrics(real, published[name])
        drift = abs(recomputed["MAE"] - vals["MAE"])
        note = MAPE_NOTE
        if drift > 0.005:
            note = (
                f"PAPER TABLE AND SHIPPED FORECASTS DISAGREE: shipped score "
                f"rMAE {recomputed['rMAE']:.3f} / MAE {recomputed['MAE']:.3f}. " + note
            )
        rows.append(
            {"model": name, "source": "Lago2021 (paper Tables 2/3)", **vals, "note": note}
        )
        rows.append(
            {
                "model": name,
                "source": "Lago2021 (shipped forecasts, our metric code)",
                **recomputed,
                "MAPE": np.nan,
                "note": (
                    "reproduces the paper table exactly"
                    if drift <= 0.005
                    else f"does NOT match the paper table (MAE {vals['MAE']:.3f} printed)"
                ),
            }
        )

    cols = ["model", "source", "rMAE", "MAE", "sMAPE", "RMSE", "MAPE", "note"]
    return pd.DataFrame(rows)[cols].sort_values("rMAE").reset_index(drop=True)


# --------------------------------------------------------------------------
# task 2: DM against their forecasts
# --------------------------------------------------------------------------
def build_dm(published: pd.DataFrame) -> pd.DataFrame:
    real = as_matrix(published["Real price"])
    ours = {n: as_matrix(our_hourly(n)) for n in OUR_DM}
    theirs = {n: as_matrix(published[n]) for n in THEIR_DM}

    rows = []
    for on, om in ours.items():
        for tn, tm in theirs.items():
            # epftoolbox convention, shared by diebold_mariano_hac: a small
            # p supports "p_pred_2 is more accurate than p_pred_1".
            p_ours_better = diebold_mariano_hac(p_real=real, p_pred_1=tm, p_pred_2=om)
            p_theirs_better = diebold_mariano_hac(p_real=real, p_pred_1=om, p_pred_2=tm)
            if p_ours_better < 0.05:
                verdict = "ours better (p<0.05)"
            elif p_theirs_better < 0.05:
                verdict = "theirs better (p<0.05)"
            else:
                verdict = "no significant difference"
            rows.append(
                {
                    "ours": on,
                    "theirs": tn,
                    "MAE ours": float(np.abs(real - om).mean()),
                    "MAE theirs": float(np.abs(real - tm).mean()),
                    "DM p (ours better)": p_ours_better,
                    "DM p (theirs better)": p_theirs_better,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "--"
    v = float(v)
    if 0.0 <= v < 0.001:
        return r"$<$0.001"
    out = f"{v:.3f}"
    return "0.000" if out == "-0.000" else out


def export(df: pd.DataFrame, stem: str, caption: str, label: str, dry_run: bool) -> None:
    if dry_run:
        print(f"\n--- {stem} (dry run) ---")
        print(df.to_string(index=False))
        return
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{stem}.csv", index=False)
    df.map(_fmt).to_latex(
        TABLES_DIR / f"{stem}.tex",
        escape=True,
        index=False,
        caption=caption,
        label=label,
        position="htbp",
    )
    print(f"wrote {_rel(TABLES_DIR / f'{stem}.csv')} and {_rel(TABLES_DIR / f'{stem}.tex')}")


def main(dry_run: bool = False) -> None:
    published = load_published()
    verify_alignment(published)

    comparison = build_comparison(published)
    print("\n=== comparison (sorted by rMAE) ===")
    print(comparison.drop(columns=["note"]).to_string(index=False))
    export(
        comparison,
        "lago_comparison",
        caption=(
            "This thesis versus Lago et al. (2021) on EPEX-DE, identical test "
            "period (728 origins, 2016-01-04 to 2017-12-31) and identical rMAE "
            "naive2 denominator. Published models appear twice: as printed in "
            "the paper's Tables 2--3, and as recomputed from the forecasts the "
            "toolbox ships, using this thesis's own metric code. The two agree "
            "exactly for every DNN but for no LEAR variant, so the source of "
            "each number is stated rather than merged. MAPE is reported only "
            "for the published rows; this thesis excludes it because negative "
            "and near-zero prices make it unreliable on this market, a point "
            "the paper's own MAPE column (roughly ten times its sMAPE) "
            "illustrates."
        ),
        label="tab:lago-comparison",
        dry_run=dry_run,
    )

    dm = build_dm(published)
    print("\n=== DM vs published forecasts (HAC, multivariate, one-sided both ways) ===")
    print(dm.to_string(index=False))
    export(
        dm,
        "dm_vs_lago",
        caption=(
            "Diebold--Mariano tests against the forecasts published by Lago et "
            "al. (2021), multivariate (24-hour vector, $L_1$ norm), "
            "Newey--West HAC corrected, over the identical 728 test origins. "
            "Both one-sided directions are reported; a comparison is called "
            "only when one of them falls below 0.05. Computed against their "
            "shipped forecasts, not their printed table."
        ),
        label="tab:dm-vs-lago",
        dry_run=dry_run,
    )


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
