"""Feature-support diagnostic: train/serve skew on the OOD window (§10.3).

Answers the question `docs/HANDOFF_new_model_design.md` §10.3 left open: are
the exogenous features served to the frozen models in 2026 inside the support
they were trained on in 2012-2016?

READ-ONLY BY CONSTRUCTION. It loads two already-committed datasets, compares
their distributions, and writes one table. It fits nothing, refits nothing and
re-evaluates nothing, so it cannot disturb `v1.0-results` or `v1.1-ood` — the
freeze applies to model results, and this produces none. That is why it needs
no ledger gate and carries no seed.

Two measures, because either alone is misleading:

  out-of-support  fraction of live values outside the training [min, max].
                  Catches a hard extrapolation demand but ignores shape.
  overlap         histogram-intersection of the two densities on a shared
                  grid; 1.0 identical, 0.0 disjoint. Catches a distribution
                  that has moved while still nominally in range.

The rescaling arm is the diagnostic half. If a single scalar restores overlap,
the cause is a units or geographic-definition mismatch in the serving path --
something we did -- rather than a market that changed. Those two carry
completely different consequences for the thesis, and the difference is not
visible without this comparison.

Usage:  ./.venv/Scripts/python.exe scripts/run_feature_support.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader  # noqa: E402

SERIES = ["price", "exog_1", "exog_2"]
OUT = REPO_ROOT / "reports" / "tables" / "feature_support.csv"


def overlap(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
    """Histogram-intersection of two densities over a shared grid."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    grid = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=grid, density=True)
    hb, _ = np.histogram(b, bins=grid, density=True)
    return float(np.sum(np.minimum(ha, hb) * np.diff(grid)))


def out_of_support(train: np.ndarray, live: np.ndarray) -> float:
    return float(((live < train.min()) | (live > train.max())).mean())


def load_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    train, _ = BenchmarkLoader(cfg).load()
    live = pd.read_csv(REPO_ROOT / "data" / "raw" / "live_ood_de.csv",
                       index_col=0, parse_dates=True)
    # The live cache is tz-aware UTC; the benchmark index is naive. Compare on
    # the same footing rather than silently aligning on mismatched types.
    live.index = live.index.tz_convert(None)
    return train, live


def main() -> None:
    train, live = load_windows()
    print(f"TRAIN  {train.index.min()} .. {train.index.max()}  n={len(train):,}")
    print(f"LIVE   {live.index.min()} .. {live.index.max()}  n={len(live):,}\n")

    rows = []
    for col in SERIES:
        a = train[col].dropna().to_numpy()
        b = live[col].dropna().to_numpy()
        ratio = a.mean() / b.mean()
        row = {
            "series": col,
            "train_mean": a.mean(), "train_min": a.min(), "train_max": a.max(),
            "live_mean": b.mean(), "live_min": b.min(), "live_max": b.max(),
            "live_over_train_mean": b.mean() / a.mean(),
            "out_of_support_raw": out_of_support(a, b),
            "overlap_raw": overlap(a, b),
            "out_of_support_rescaled": out_of_support(a, b * ratio),
            "overlap_rescaled": overlap(a, b * ratio),
        }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("series")
    pd.set_option("display.width", 200)
    print(table.T.to_string(float_format=lambda v: f"{v:,.4f}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT)
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")

    exog1 = table.loc["exog_1"]
    print(
        "\nexog_1 overlap {:.4f} raw -> {:.4f} rescaled by {:.3f}x: a single "
        "scalar restores support,\nwhich is a definition mismatch (Amprion "
        "zonal vs national aggregate), not market drift."
        .format(exog1["overlap_raw"], exog1["overlap_rescaled"],
                exog1["live_over_train_mean"])
    )


if __name__ == "__main__":
    main()
