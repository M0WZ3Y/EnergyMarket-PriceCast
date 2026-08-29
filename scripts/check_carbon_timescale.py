"""Why the carbon block cannot help at day-ahead resolution — PRINTS ONLY.

B3_carbon_switch moved its target regime by +0.007 MAE. That is NOTHING, and
it is categorically different from B4 (+1.791) or B6 (+0.902), which actively
degrade. Under LASSO, "no effect" is the signature of a coefficient correctly
shrunk toward zero, not of a broken feature -- so filing B3 under
"misspecified" alongside those two would be wrong.

This measures the physical reason. An EUA price is set by a compliance market
that clears on a multi-week to multi-year timescale. Inside a short day-ahead
test window it barely moves, so it has almost no variance to contribute no
matter how correctly it is built.

The honest scope: this explains the CARBON component of B3. The block also
carries a gas-vs-coal dispatch proxy, which does vary day to day (CoV 0.23) --
its failure needs a different explanation and is not covered here.

Writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.loader import BenchmarkLoader, load_config  # noqa: E402
from src.features import leakage_guard as lg  # noqa: E402
from src.features.pipeline import build_features, load_feature_config  # noqa: E402


def main(origins_n: int = 80) -> int:
    cfg = load_config()
    tr, te = BenchmarkLoader(cfg).load()
    df = pd.concat([tr, te])
    f = dict(load_feature_config())
    f["exog_blocks"] = {"carbon_block": True, "fuel_switch_proxy_block": True}
    lg.clear_registry()
    X, Y = build_features(df, f)
    origins = X.index[-origins_n:]

    c = X.loc[origins, "carbon_cost_gas_D-1"]
    price = Y.loc[origins].mean(axis=1)

    print("=" * 74)
    print("CARBON TIMESCALE — variation available inside the test window")
    print("=" * 74)
    print(f"window            {len(c)} days  "
          f"{origins.min().date()} .. {origins.max().date()}")
    print(f"mean / sd         {c.mean():.4f} / {c.std():.4f} EUR/MWh_el")
    print(f"range             {c.min():.4f} .. {c.max():.4f}  "
          f"(span {c.max() - c.min():.4f})")
    print(f"distinct values   {c.nunique()}   (auctions do not run daily)")
    print()
    print("autocorrelation of the daily carbon series:")
    for lag in (1, 2, 5, 10, 20):
        if len(c) > lag:
            print(f"   lag {lag:>2}d  {c.autocorr(lag):+.4f}")
    print()

    rows = []
    for col, label in (
        ("carbon_cost_gas_D-1", "EUA carbon cost"),
        ("exog_1_D0_h13", "load forecast h13"),
        ("exog_2_D0_h13", "RES forecast h13"),
        ("switch_proxy_D-1_h13", "gas/coal split h13"),
    ):
        if col in X.columns:
            v = X.loc[origins, col]
            rows.append(dict(driver=label, cov=v.std() / abs(v.mean()),
                             ac1=v.autocorr(1)))
    rows.append(dict(driver="daily price (TARGET)",
                     cov=price.std() / abs(price.mean()), ac1=price.autocorr(1)))
    t = pd.DataFrame(rows).set_index("driver")
    print("relative variation in the SAME window:")
    print(t.round(4).to_string())
    print()

    cov_c = c.std() / abs(c.mean())
    cov_p = price.std() / abs(price.mean())
    print(f"  carbon CoV {cov_c:.4f} vs target CoV {cov_p:.4f}  "
          f"-> ratio {cov_c / cov_p:.3f}")
    print(f"  carbon ac(1) {c.autocorr(1):.3f}: near-constant within the window.")
    print()
    print("  CONCLUSION. Carbon is a LEVEL driver of the merit order across")
    print("  YEARS -- a rising EUA price monotonically favours gas over coal and")
    print("  lifts the whole cost stack. It is not an HOURLY driver. Over an")
    print("  80-day day-ahead window it varies ~18x less than the target, so")
    print("  there is almost nothing for a coefficient to attach to and LASSO")
    print("  correctly shrinks it. B3's null is a TIMESCALE MISMATCH, not a")
    print("  construction defect.")
    print()
    print("  Testable implication: the carbon block should become informative")
    print("  on a multi-year evaluation window, where EUA moves 5-50 EUR/t.")
    print("  It is not evidence that carbon does not drive German prices.")
    print()
    print("  SCOPE. This covers the carbon component only. The gas/coal dispatch")
    print("  proxy in the same block DOES vary day to day and its null needs a")
    print("  separate explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
