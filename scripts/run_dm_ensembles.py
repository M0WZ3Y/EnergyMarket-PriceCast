"""Diebold-Mariano tests on the ensembles — scripts/run_dm_ensembles.py

Answers the week-7 question the aggregate MAE cannot: is the regime-aware
ensemble's edge over the static one real, and where does it come from?

Reports three things (decision 2026-08-04):
  1. the full pairwise DM matrix over every model + both ensembles;
  2. regime-aware vs static over all test days;
  3. the same split by regime.

(3) is the load-bearing one. The two ensembles differ only slightly on calm
days (their calm weights are close to the static weights), so 651 near-
identical days dilute the aggregate differential toward non-significance.
Splitting by regime shows whether the gain is localized to the days the
mechanism is designed to act on -- a diffuse noise advantage would not be.

Reads only committed artifacts, so it is reproducible after the freeze:
  data/processed/baselines/*.csv  (test period, 2016-01-04 -> 2017-12-31)

The regime threshold comes from configs/evaluation.yaml, never hardcoded —
the labeling rule is previous-day-only (see regime_labels()).

Usage:
    python scripts/run_dm_ensembles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SEED = 42
N_BOOT = 20000

from src.evaluation.ensemble import regime_labels
from src.evaluation.metrics import (
    diebold_mariano,
    diebold_mariano_hac,
    loss_differential,
    mae,
)
from src.evaluation.results import dm_matrix, load_long_frame
from src.evaluation.walk_forward import load_evaluation_config

TEST_DIR = REPO_ROOT / "data" / "processed" / "baselines"

FILES = {
    "Ensemble (regime-aware)": "ensemble_regime.csv",
    "Ensemble (static)": "ensemble_static.csv",
    "LSTM": "lstm.csv",
    "LEAR-LASSO": "lear_lasso.csv",
    "LightGBM": "lightgbm.csv",
    "SARIMAX": "sarimax.csv",
    "naive": "naive.csv",
}

REGIME = "Ensemble (regime-aware)"
STATIC = "Ensemble (static)"


def _block_bootstrap_p(d: np.ndarray, block: int) -> float:
    """One-sided CIRCULAR block bootstrap p-value for H0: mean(d) <= 0.

    Resampling contiguous blocks preserves the serial dependence the plain
    DM statistic ignores. The null is imposed by centring d. Wrapping the
    index (`% n`) makes every observation appear in equally many blocks,
    removing the edge under-weighting of the non-circular variant.

    p is (1 + #{resamples >= observed}) / (N_BOOT + 1) so it can never be
    exactly 0 — a literal "p = 0.0000" in a thesis table is indefensible.
    Deterministic under SEED.
    """
    n = len(d)
    block = max(1, min(block, n - 1))
    rng = np.random.default_rng(SEED)
    d0 = d - d.mean()
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(N_BOOT, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :n] % n
    hits = int((d0[idx].mean(axis=1) >= d.mean()).sum())
    return (1.0 + hits) / (N_BOOT + 1.0)


def _bandwidth(n: int) -> int:
    """Newey-West 1994 rule, mirrored from metrics.diebold_mariano_hac so
    the report can print the bandwidth it used."""
    return min(max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))), n - 1)


def _report(piv, truth, days, name: str) -> tuple[dict[int, float], float]:
    """MAE of both ensembles on `days`, the uncorrected DM p-value, and the
    block-bootstrap p-value that is the reported statistic.

    Why a RANGE rather than one number (decision 2026-08-04, revised after
    code review): epftoolbox's DM is mean(d)/sqrt(var(d)/N) with no HAC
    correction, but the loss differential is strongly autocorrelated —
    stressed days are DEFINED by their predecessor breaching the threshold,
    so they arrive in runs. Treating clustered days as independent
    understates the standard error.

    Two independent corrections are reported because they materially
    disagree on the stressed subset, and collapsing them to a single
    "corrected p" would hide that disagreement. The block bootstrap is also
    sensitive to block length, and no block rule is authoritative here: the
    n**(1/3) rule would hand the stressed subset a SHORTER block (4) than
    the full sample (9) despite its STRONGER dependence, purely because it
    is shorter. Block lengths beyond ~7 also exceed the longest observed run
    of stressed days, over-correcting by treating months-apart runs as
    dependent. So the sweep is reported and the claim is made against its
    worst case, not its best.

    NOTE on the subsets: for the stressed/calm calls `d` is a FILTERED
    series, so adjacent entries are not always calendar-adjacent. Blocks
    then preserve within-run dependence (the real structure) but not a
    clean time series — another reason to read the sweep as a range.
    """
    idx = pd.DatetimeIndex(days)
    real = truth.loc[idx]
    pr, ps = piv[REGIME].loc[idx], piv[STATIC].loc[idx]

    mae_r = mae(real.values.ravel(), pr.values.ravel())
    mae_s = mae(real.values.ravel(), ps.values.ravel())
    # epftoolbox DM: small p supports "p_pred_2 more accurate than p_pred_1",
    # so the regime-aware forecasts go in as p_pred_2.
    p_naive = diebold_mariano(p_real=real.values, p_pred_1=ps.values, p_pred_2=pr.values)
    p_hac = diebold_mariano_hac(p_real=real.values, p_pred_1=ps.values, p_pred_2=pr.values)

    # Per-day multivariate L1 loss differential: positive => regime better.
    d = loss_differential(real.values, ps.values, pr.values, norm=1)
    n = len(d)
    bandwidth = _bandwidth(n)
    blocks = [b for b in (3, 4, 5, 7, 9, 10) if b < n]
    sweep = {b: _block_bootstrap_p(d, b) for b in blocks}

    diff = np.abs(pr.values - ps.values)
    print(f"\n{name}  (n={n} days)")
    print(
        f"  MAE regime-aware {mae_r:.4f} | static {mae_s:.4f} | "
        f"delta {mae_r - mae_s:+.4f} ({100 * (mae_r - mae_s) / mae_s:+.2f}%)"
    )
    print(f"  mean |pred difference| {diff.mean():.4f} EUR/MWh, max {diff.max():.4f}")
    print(
        f"  lag-1 autocorr {pd.Series(d).autocorr(1):+.4f} | "
        f"skewness {float(pd.Series(d).skew()):+.2f}"
    )
    print(f"  p, uncorrected DM (epftoolbox; ignores dependence): {p_naive:.4f}")
    print(f"  p, Newey-West HAC DM (bandwidth {bandwidth}):        {p_hac:.4f}")
    if sweep:
        print("  p, circular block bootstrap by block length:")
        print("      " + "  ".join(f"b={b}:{sweep[b]:.4f}" for b in blocks))
        lo, hi = min(sweep.values()), max(sweep.values())
        print(f"  >> REPORTED RANGE across dependence corrections: {min(lo, p_hac):.4f} - {hi:.4f}")
    else:
        # Every candidate block length must be shorter than the series, so a
        # subset of <= 3 days admits no block bootstrap at all. Previously
        # min() on the empty sweep raised mid-report, killing the run after
        # the MAE lines and before anything was written. The HAC statistic is
        # still valid here, so report it and say plainly what is missing --
        # a reachable state, since the stress threshold is the documented
        # tuning lever and raising it shrinks the stressed subset.
        print(
            f"  p, circular block bootstrap: SKIPPED -- only {n} day(s), "
            f"fewer than the shortest block length ({min((3, 4, 5, 7, 9, 10))})"
        )
        print(f"  >> REPORTED: HAC only, p={p_hac:.4f} (no bootstrap range available)")
    return sweep, p_hac


def main() -> None:
    frames = {name: load_long_frame(TEST_DIR / f) for name, f in FILES.items()}

    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    labels = regime_labels(frames["LSTM"], threshold=threshold)
    stressed = sorted(o for o, lab in labels.items() if lab == "stressed")
    calm = sorted(o for o, lab in labels.items() if lab == "calm")
    print(
        f"test split: calm {len(calm)}, stressed {len(stressed)} "
        f"(threshold {threshold} EUR/MWh, previous-day rule)"
    )

    print("\n" + "=" * 78)
    print("Pairwise one-sided DM p-values, multivariate (24-h vector, L1 norm)")
    print("cell[row, col] = P(row MORE accurate than col); small => row wins")
    print("=" * 78)
    print(dm_matrix(frames).round(4).to_string(na_rep="  --  "))

    piv = {
        m: f.pivot(index="origin", columns="hour", values="y_pred").sort_index()
        for m, f in frames.items()
    }
    truth = frames["LSTM"].pivot(index="origin", columns="hour", values="y_true").sort_index()

    print("\n" + "=" * 78)
    print("FOCUSED: regime-aware vs static")
    print("=" * 78)
    _report(piv, truth, sorted(labels), "ALL test days")
    _report(piv, truth, stressed, "STRESSED days only (where the mechanism acts)")
    _report(piv, truth, calm, "CALM days only (near-null check)")


if __name__ == "__main__":
    # Writing is the binding constraint; new technical output does not
    # move it. Gate is inside the guard on purpose -- at module level it
    # would fire on import and gate the test suite too.
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    main()
