"""Supplementary seed-ensemble table — scripts/export_seed_ensemble.py

Writes ONE new file pair, and touches nothing frozen:

  reports/tables/seed_ensemble.{csv,tex}   -> thesis section 4-5-2

WHAT THIS TABLE IS. On 2026-08-07 the LSTM was re-run at seeds 43--45 and
averaged with the frozen seed-42 run, borrowing the mechanism behind Lago et
al.'s DNN Ensemble (four runs of ONE family, i.e. variance reduction rather
than a better model). The attempt to close the gap to their DNN Ensemble
FAILED and was shown to be unclosable by ensembling. A narrower result held:
the seed-ensembled LSTM arm improves, and the regime-aware ensemble built on
it is no longer significantly worse than their DNN Ensemble.

WHAT THIS TABLE IS NOT. It is SUPPLEMENTARY, outside v1.0-results. The
headline numbers in sections 4-2/4-3, the SHAP analysis in 4-6 and the OOD
addendum are all computed on the frozen seed-42 LSTM and are unchanged. The
two sets of numbers must never be read as one series; the caption says so.

SEED POLICY. Unchanged: 42 is the default and every single-model result uses
it. Seeds 43--45 exist only as members of an explicitly labelled seed
ensemble, and seed 42's member IS the frozen lstm.csv, reused not recomputed.

PROVENANCE. The ensemble frames read here were produced by
scripts/run_seed_ensemble.py --evaluate, which fits the ensemble weights on
the VALIDATION window and scores on test. That direction matters: scoring a
test-window seed ensemble under test-fitted weights would be the in-sample
selection the walk-forward exists to prevent. The weights are not refitted
here -- this script only reads, scores and formats.

Usage:
    python scripts/export_seed_ensemble.py
    python scripts/export_seed_ensemble.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_seed_ensemble import seed_ensemble_frame
from src.evaluation.metrics import diebold_mariano_hac, mae, rmae

TABLES_DIR = REPO_ROOT / "reports" / "tables"
BASELINES = REPO_ROOT / "data" / "processed" / "baselines"
SEED_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble"
PUBLISHED_FC = REPO_ROOT / "data" / "raw" / "Forecasts_DE_DNN_LEAR_ensembles.csv"

FROZEN_LSTM = BASELINES / "lstm.csv"
BASELINE_SEED = 42

# Their columns we test against. "DNN Ensemble" is the target that motivated
# the whole experiment; the other two are context, not cherry-picked.
THEIRS = ("DNN Ensemble", "DNN 4", "LEAR Ensemble")


def _long(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def _piv(frame: pd.DataFrame, col: str) -> np.ndarray:
    return frame.pivot(index="origin", columns="hour", values=col).sort_index().to_numpy()


def _series(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long frame -> hourly-indexed single-column frame.

    rMAE needs an hourly DatetimeIndex for its weekly-lag naive denominator,
    so the long frame is rebuilt into an hourly series first -- the same
    conversion export_tables.py::_hourly_metrics performs.
    """
    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    return pd.Series(frame[col].values, index=ts).sort_index().to_frame("price")


def _scores(frame: pd.DataFrame) -> dict:
    real, pred = _series(frame, "y_true"), _series(frame, "y_pred")
    return {"MAE": float(mae(real.values, pred.values)), "rMAE": float(rmae(real, pred, m="W"))}


def load_published() -> pd.DataFrame:
    return pd.read_csv(PUBLISHED_FC, index_col=0, parse_dates=True)


def _m24(series: pd.Series) -> np.ndarray:
    f = series.to_frame("v")
    f["d"], f["h"] = f.index.normalize(), f.index.hour
    return f.pivot(index="d", columns="h", values="v").sort_index().to_numpy()


def build() -> pd.DataFrame:
    """Assemble the supplementary table from the committed prediction frames."""
    pub = load_published()
    real24 = _m24(pub["Real price"])

    # --- member block -------------------------------------------------
    members = {BASELINE_SEED: _long(FROZEN_LSTM)}
    for path in sorted(SEED_DIR.glob("lstm_s[0-9]*.csv")):
        members[int(path.stem.split("_s")[-1])] = _long(path)

    # Average via the shared helper so this table and run_seed_ensemble.py
    # --evaluate can never disagree about what "the seed ensemble" means,
    # including its digit-only glob (a bare lstm_s*.csv also matches the
    # averaged output file and folds it back in as a member).
    avg, n_members = seed_ensemble_frame(SEED_DIR, extra=FROZEN_LSTM)

    saved = SEED_DIR / "lstm_seed_ensemble.csv"
    if saved.exists():
        # Integrity check, not decoration: combine() and evaluate() build the
        # same average by different routes, and a silent divergence between
        # them would put a number in the thesis that no longer matches the
        # frames it claims to come from.
        ref = _long(saved).sort_values(["origin", "hour"]).reset_index(drop=True)
        got = avg.sort_values(["origin", "hour"]).reset_index(drop=True)
        if not np.allclose(ref["y_pred"].to_numpy(), got["y_pred"].to_numpy(), atol=1e-9):
            raise SystemExit(
                "the recomputed seed average disagrees with lstm_seed_ensemble.csv"
            )

    rows = []
    for seed, frame in sorted(members.items()):
        tag = " (frozen)" if seed == BASELINE_SEED else ""
        rows.append({"model": f"LSTM seed {seed}{tag}", **_scores(frame)})
    rows.append({"model": f"LSTM {n_members}-seed ensemble", **_scores(avg)})

    # --- ensemble block -----------------------------------------------
    pairs = [
        ("Ensemble (static)", BASELINES / "ensemble_static.csv",
         SEED_DIR / "ensemble_static_seedens.csv"),
        ("Ensemble (regime-aware)", BASELINES / "ensemble_regime.csv",
         SEED_DIR / "ensemble_regime_aware_seedens.csv"),
    ]
    for name, frozen_path, seedens_path in pairs:
        for label, path in ((f"{name}, frozen LSTM", frozen_path),
                            (f"{name}, seed-ensembled LSTM", seedens_path)):
            frame = _long(path)
            row = {"model": label, **_scores(frame)}
            ours = _piv(frame, "y_pred")
            for theirs in THEIRS:
                row[f"DM p vs {theirs}"] = float(
                    diebold_mariano_hac(
                        p_real=real24, p_pred_1=ours, p_pred_2=_m24(pub[theirs])
                    )
                )
            rows.append(row)

    # --- reference row -------------------------------------------------
    # Recomputed from their shipped forecasts with this thesis's own metric
    # code, never transcribed from their printed table: the two agree exactly
    # for every DNN, and only the recomputed form is a like-for-like number.
    ref_real = pub["Real price"].to_frame("price").sort_index()
    ref_pred = pub["DNN Ensemble"].to_frame("price").sort_index()
    rows.append({
        "model": "Lago et al. DNN Ensemble (reference)",
        "MAE": float(mae(ref_real.values, ref_pred.values)),
        "rMAE": float(rmae(ref_real, ref_pred, m="W")),
    })

    cols = ["model", "MAE", "rMAE"] + [f"DM p vs {t}" for t in THEIRS]
    return pd.DataFrame(rows).reindex(columns=cols)


def _fmt(v) -> str:
    """Format a cell. p-values never print as an exact 0.000."""
    if pd.isna(v):
        return "--"
    if isinstance(v, str):
        return v
    v = float(v)
    if 0.0 <= v < 0.001:
        return r"$<$0.001"
    out = f"{v:.4f}"
    return "0.0000" if out == "-0.0000" else out


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


CAPTION = (
    "SUPPLEMENTARY result, outside the v1.0-results freeze: LSTM seed "
    "ensemble (seeds 42--45) and the ensembles built on it, EPEX-DE test "
    "period, 728 origins 2016-01-04 to 2017-12-31. Averaging four seeds of "
    "one model family is variance reduction, the same mechanism behind Lago "
    "et al.'s DNN Ensemble, not a better model. The attempt to close the gap "
    "to their DNN Ensemble FAILED: even an oracle ensemble including this "
    "member -- weights fitted directly on the test set, therefore cheating "
    "and a hard upper bound -- scores 3.5019, still short of their 3.4135, "
    "because their best single model (3.592) beats this thesis's best single "
    "model (3.873) by 0.28 and averaging cannot manufacture that difference. "
    "What does change is the significance verdict, and it must be read as a "
    "pair: on the seed-ensembled LSTM the regime-aware ensemble is no longer "
    "significantly worse than their DNN Ensemble, while the static ensemble "
    "still is. The two straddle the 0.05 level, so both are reported "
    "together -- the same rule already in force for regime-aware versus "
    "static. A p-value above 0.05 is a FAILURE TO REJECT, not evidence of "
    "equivalence. Ensemble weights are fitted on the validation window and "
    "never on the test window. DM tests are multivariate (24-hour vector, "
    "$L_1$ norm), Newey--West HAC corrected, one-sided in the direction "
    "that THEIR forecast is more accurate, so a small value counts against "
    "this thesis. Reference row recomputed from their shipped forecasts "
    "using this thesis's own metric code, not transcribed from their printed "
    "table. The headline results in sections 4-2 and 4-3, the SHAP analysis "
    "in 4-6 and the OOD addendum are all computed on the frozen seed-42 LSTM "
    "and are unaffected by this table; the two sets of numbers are not one "
    "series."
)


def main(dry_run: bool = False) -> None:
    table = build()
    print("\n=== seed-ensemble supplementary table ===")
    print(table.to_string(index=False))
    export(table, "seed_ensemble", caption=CAPTION,
           label="tab:seed-ensemble", dry_run=dry_run)


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
