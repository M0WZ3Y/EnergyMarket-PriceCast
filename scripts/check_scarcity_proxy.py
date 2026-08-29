"""Does the capacity-based scarcity proxy add anything over load? — PRINTS ONLY.

Tests one narrow claim, and is careful about what it does NOT claim.

WHAT IS BEING TESTED. `reserve_margin_block` builds forecast load divided by
installed dispatchable capacity. Installed capacity is published yearly and is
a step function: it changes once a year and is constant in between. So the
proxy is suspected of being little more than a per-year rescaling of load, in
which case it carries no scarcity information the model does not already have
from `exog_1`.

METHOD. Residualise the proxy on load, then ask whether what is left over
explains price — overall, and in the upper price tail where scarcity is
supposed to act.

THE CONTROL THAT MATTERS, and two wrong answers on the way to it. Capacity is
constant WITHIN a year, so within any year proxy = load / c is an EXACT linear
function of load. The residual from a within-year regression is therefore zero
by construction, and the test must be run that way.

  Attempt 1 pooled all years into one regression and reported "carries
  measurable signal in spike hours" (mean |r| 0.117). Wrong: one pooled slope
  cannot fit several years that each need their own, so the residual retained
  a LOAD component, and load does correlate with tail price.

  Attempt 2 added a year dummy to remove the level difference. Still wrong,
  and barely moved the number (0.126), because the problem was never the
  per-year INTERCEPT — it was the per-year SLOPE.

  Attempt 3, done here, regresses within each year. Residual standard
  deviation comes out at ~3e-15 of the proxy's own SD: exactly zero to machine
  precision.

This is why the script gates on residual MAGNITUDE before reporting any
correlation. Once the residual is numerical noise, its correlation with price
is also numerical noise and will happily print values like 0.32 that mean
nothing at all.

SCOPE OF THE CONCLUSION. The null here says the CAPACITY-BASED proxy is
an exact function of load and year. It does NOT say scarcity is unimportant.
Real scarcity is driven by short-term AVAILABILITY, which varies daily;
installed capacity does not. That gap is exactly the argument for the ENTSO-E
outage feed, and this script quantifies the gap rather than dismissing the
mechanism.

Writes nothing. Not ledger-gated: it produces no thesis artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader, load_config  # noqa: E402
from src.features import leakage_guard as lg  # noqa: E402
from src.features.pipeline import build_features, load_feature_config  # noqa: E402

SPIKE_Q = 0.95
#: |r| below which a residual is treated as carrying no usable signal.
NULL_R = 0.10


def _corr(x, y) -> float:
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    cfg = load_config()
    tr, te = BenchmarkLoader(cfg).load()
    df = pd.concat([tr, te])

    f = dict(load_feature_config())
    f["exog_blocks"] = {"reserve_margin_block": True}
    lg.clear_registry()
    X, Y = build_features(df, f)
    years = pd.Index(X.index.year)

    print("=" * 76)
    print("SCARCITY PROXY CHECK — does load/capacity add anything over load?")
    print("=" * 76)
    print(f"rows {len(X)}   {X.index.min().date()} .. {X.index.max().date()}")
    print()

    rows = []
    for h in range(24):
        prox = X[f"resmargin_nooutage_D0_h{h:02d}"].to_numpy(float)
        load = X[f"exog_1_D0_h{h:02d}"].to_numpy(float)
        price = Y[f"y_h{h:02d}"].to_numpy(float)
        ok = np.isfinite(prox) & np.isfinite(load) & np.isfinite(price)
        prox, load, price, yr = prox[ok], load[ok], price[ok], years[ok]
        if len(prox) < 50:
            continue

        # Pooled regression -- kept only to show what it gets wrong.
        b, a = np.polyfit(load, prox, 1)
        resid_pooled = prox - (a + b * load)

        # WITHIN-YEAR regression: the correct control. Capacity is constant
        # inside a year, so each year needs its own slope, not a shared one.
        resid = np.full_like(prox, np.nan)
        for y in np.unique(yr):
            m = yr == y
            if m.sum() < 10:
                continue
            by, ay = np.polyfit(load[m], prox[m], 1)
            resid[m] = prox[m] - (ay + by * load[m])

        cut = np.quantile(price, SPIKE_Q)
        tail = price >= cut

        # Residual size relative to the proxy's own spread. When this is at
        # machine-epsilon the residual is noise and any correlation computed
        # from it is meaningless -- reported, but flagged as such.
        rel = float(np.nanstd(resid) / np.std(prox)) if np.std(prox) else np.nan

        rows.append(
            dict(
                hour=h,
                r_proxy_load=_corr(prox, load),
                resid_sd_rel=rel,
                r_pooled_spike=_corr(resid_pooled[tail], price[tail]),
                r_withinyear_spike=_corr(resid[tail], price[tail]),
                n_spike=int(tail.sum()),
            )
        )

    res = pd.DataFrame(rows).set_index("hour")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string())
    print()

    print("-" * 76)
    print("SUMMARY")
    print("-" * 76)
    print(f"  mean |r| proxy vs load                      : "
          f"{res['r_proxy_load'].abs().mean():.4f}")
    print(f"  mean residual SD / proxy SD (within-year)   : "
          f"{res['resid_sd_rel'].mean():.3e}")
    print(f"  mean |r| residual vs spike price (POOLED)   : "
          f"{res['r_pooled_spike'].abs().mean():.4f}   <- misleading")
    print(f"  mean |r| residual vs spike price (within-yr): "
          f"{res['r_withinyear_spike'].abs().mean():.4f}   <- noise, see below")
    print()

    rel = res["resid_sd_rel"].mean()
    if rel < 1e-10:
        print(f"  The within-year residual is ZERO to machine precision "
              f"({rel:.1e} of the")
        print("  proxy's own SD). Capacity is constant within a year, so")
        print("  proxy = load / capacity_year is an EXACT linear function of load.")
        print("  The correlations printed for it are correlations of floating-point")
        print("  noise and carry no meaning -- which is exactly why residual")
        print("  MAGNITUDE has to be checked before any correlation is believed.")
        print()
        verdict = ("NULL, EXACTLY — the capacity-based proxy contains no "
                   "information beyond load and the calendar year. Not "
                   "'little'; none.")
    else:
        m = res["r_withinyear_spike"].abs().mean()
        verdict = (f"residual is non-degenerate ({rel:.2e}); mean |r| {m:.3f} "
                   "against spike price")
    print(f"  VERDICT: {verdict}")
    print()
    print("  SCOPE: this is a statement about THIS PROXY's construction, not about")
    print("  scarcity in DE-LU. Installed capacity is slow-moving; real scarcity is")
    print("  driven by short-term availability, which varies daily. Quantifying that")
    print("  gap is the argument for the ENTSO-E outage feed, not against the")
    print("  mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
