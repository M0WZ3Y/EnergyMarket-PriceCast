"""Diebold-Mariano tests over the ablation variants — PRINTS ONLY.

The ablation reports MAE deltas, several of which are sub-1% of a baseline MAE
of 6.26 over an 80-day window. At that size "does nothing", "does something
small" and "sampling noise" are indistinguishable by point estimate alone, so
every delta the analysis leans on has to be tested.

Reads the PERSISTED predictions written by run_full_physical_ablation.py, so it
refits nothing and cannot disagree with the numbers already reported.

WHAT IS TESTED
  baseline vs each block          does the block change accuracy at all
  B1 vs ALL                       the headline claim -- is the full set really
                                  worse than the ramp block alone, or are
                                  -6.1% and -7.7% not separable
  per-REGIME, not only pooled     B4's off-target gains (spike -12.3%,
                                  high_hydro -7.2%) are among the largest
                                  single-regime effects measured and nothing
                                  should be built on them untested

TEST. Multivariate DM in the Lago et al. sense: losses are aggregated per DAY
across the 24 hours before differencing, because a day's 24 prices come from
one auction and are not independent draws. Treating each hour as its own
observation would inflate the effective sample ~24x and manufacture
significance. epftoolbox's own DM is used for the pooled case; the per-regime
case cannot use it (a regime is a subset of hours, not whole days) and uses a
HAC-corrected t-statistic on the hourly loss differential instead, with the
bandwidth reported.

One-sided by construction: DM(a, b) asks whether b is MORE accurate than a, so
p < 0.05 means the second model wins. Both directions are printed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader, load_config  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.metrics import diebold_mariano_hac  # noqa: E402
from src.features.pipeline import _pivot_to_daily_wide  # noqa: E402

CACHE_ROOT = REPO_ROOT / "data" / "ablation_cache"
ALPHA = 0.05


def load_variants(cache_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob(str(cache_dir / "*.csv"))):
        name = os.path.basename(f).split("__")[0]
        out[name] = pd.read_csv(f, parse_dates=["origin"])
    return out


def _aligned(a: pd.DataFrame, b: pd.DataFrame):
    """Common (origin, hour) rows of two prediction frames, in one order."""
    ka = a.set_index(["origin", "hour"]).sort_index()
    kb = b.set_index(["origin", "hour"]).sort_index()
    idx = ka.index.intersection(kb.index)
    return ka.loc[idx], kb.loc[idx], idx


def dm_pooled(a: pd.DataFrame, b: pd.DataFrame, norm: int = 1) -> float:
    """Multivariate DM p-value: is `b` more accurate than `a`?

    Losses are summed over the 24 hours of a day BEFORE differencing -- the
    multivariate variant, because one auction sets all 24 prices and treating
    hours as independent would inflate the sample and the significance with it.
    """
    ka, kb, idx = _aligned(a, b)
    days = idx.get_level_values(0)
    la = np.abs(ka["y_true"] - ka["y_pred"]) ** norm
    lb = np.abs(kb["y_true"] - kb["y_pred"]) ** norm
    da = la.groupby(days).mean().to_numpy()
    db = lb.groupby(days).mean().to_numpy()
    d = da - db
    if np.allclose(d, 0):
        return float("nan")
    from scipy import stats

    mean_d = d.mean()
    var_d = d.var(ddof=1) / len(d)
    if var_d <= 0:
        return float("nan")
    stat = mean_d / np.sqrt(var_d)
    return float(1 - stats.norm.cdf(stat))


def dm_regime(a: pd.DataFrame, b: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
    """HAC-corrected DM on a REGIME subset.

    A regime is a set of hours, not a set of whole days, so the multivariate
    daily aggregation does not apply. The hourly loss differential is
    autocorrelated, hence the HAC correction; the bandwidth is returned so the
    reader can see how much correction was applied.
    """
    ka, kb, idx = _aligned(a, b)
    m = mask.reindex(idx).fillna(False).to_numpy()
    if m.sum() < 30:
        return float("nan"), 0
    la = np.abs(ka["y_true"].to_numpy() - ka["y_pred"].to_numpy())[m]
    lb = np.abs(kb["y_true"].to_numpy() - kb["y_pred"].to_numpy())[m]
    d = la - lb
    if np.allclose(d, 0):
        return float("nan"), 0
    n = len(d)
    bw = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dm = d.mean()
    g0 = np.var(d, ddof=1)
    s = g0
    for lag in range(1, bw + 1):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        s += 2 * (1 - lag / (bw + 1)) * cov
    if s <= 0:
        return float("nan"), bw
    from scipy import stats

    stat = dm / np.sqrt(s / n)
    return float(1 - stats.norm.cdf(stat)), bw


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, matching run_combination_ladder.py.

    The per-regime grid runs 7 variants x 8 regimes = 56 tests. At alpha 0.05
    that expects ~3 false positives from noise alone, so reporting raw
    per-regime stars would manufacture findings. NaNs (degenerate tests) are
    passed through and do not count toward the family.
    """
    idx = [i for i, v in enumerate(pvals) if not np.isnan(v)]
    m = len(idx)
    adj = [float("nan")] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def stars(p: float) -> str:
    if np.isnan(p):
        return " n/a "
    if p < 0.01:
        return "***  "
    if p < 0.05:
        return "**   "
    if p < 0.10:
        return "*    "
    return "     "


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="o80_c990_aic")
    args = ap.parse_args(argv)

    cache_dir = CACHE_ROOT / args.cache
    res = load_variants(cache_dir)
    if "baseline" not in res:
        raise SystemExit(f"no baseline predictions in {cache_dir}")

    print("=" * 78)
    print("DIEBOLD-MARIANO TESTS OVER THE ABLATION")
    print("=" * 78)
    print(f"cache      {cache_dir.name}")
    print(f"variants   {', '.join(sorted(res))}")
    n_days = res['baseline']['origin'].nunique()
    print(f"window     {n_days} days x 24 h = {len(res['baseline'])} hourly obs")
    print()
    print("Multivariate DM (Lago et al.): losses aggregated per DAY before")
    print("differencing, because one auction sets all 24 prices. Treating hours")
    print("as independent would inflate the effective sample ~24x.")
    print("One-sided: p < 0.05 means the SECOND model is significantly better.")
    print("*** p<0.01  ** p<0.05  * p<0.10")
    print()

    # ---------------- pooled: baseline vs each ----------------
    print("-" * 78)
    print("POOLED — baseline vs each variant")
    print("-" * 78)
    print(f"  {'variant':<20} {'MAE':>8} {'delta':>9} "
          f"{'p(var better)':>14} {'p(base better)':>15}")
    base = res["baseline"]
    bmae = np.abs(base["y_true"] - base["y_pred"]).mean()
    for name in sorted(res):
        if name == "baseline":
            continue
        v = res[name]
        vmae = np.abs(v["y_true"] - v["y_pred"]).mean()
        p_better = dm_pooled(base, v)
        p_worse = dm_pooled(v, base)
        print(f"  {name:<20} {vmae:8.4f} {vmae - bmae:+9.4f} "
              f"{p_better:14.4f}{stars(p_better)}{p_worse:10.4f}{stars(p_worse)}")
    print()

    # ---------------- the headline comparison ----------------
    if "B1_ramp" in res and "ALL" in res:
        print("-" * 78)
        print("HEADLINE — is ALL actually worse than B1 alone?")
        print("-" * 78)
        b1, al = res["B1_ramp"], res["ALL"]
        m1 = np.abs(b1["y_true"] - b1["y_pred"]).mean()
        ma = np.abs(al["y_true"] - al["y_pred"]).mean()
        p_all_better = dm_pooled(b1, al)
        p_b1_better = dm_pooled(al, b1)
        print(f"  B1_ramp MAE {m1:.4f}   ALL MAE {ma:.4f}   diff {ma - m1:+.4f}")
        print(f"  p(ALL better than B1) = {p_all_better:.4f} {stars(p_all_better)}")
        print(f"  p(B1 better than ALL) = {p_b1_better:.4f} {stars(p_b1_better)}")
        if not np.isnan(p_b1_better) and p_b1_better < ALPHA:
            print("  => B1 is SIGNIFICANTLY better than ALL. The headline stands.")
        elif not np.isnan(p_all_better) and p_all_better < ALPHA:
            print("  => ALL is SIGNIFICANTLY better than B1. The headline REVERSES.")
        else:
            print("  => NOT SEPARABLE at the 5% level. The headline must be")
            print("     restated: ALL and B1 are statistically indistinguishable")
            print("     on this window, so 'full coverage lost to one block' is")
            print("     not supported -- only 'full coverage did not beat it'.")
        print()

    # ---------------- per regime ----------------
    cfg = load_config()
    tr, te = BenchmarkLoader(cfg).load()
    df = pd.concat([tr, te])
    wide = _pivot_to_daily_wide(df)
    context = regimes.physical_context(wide)
    mech = regimes._stack_hourly(regimes.mechanism_context(df.index))
    if not mech.empty:
        context = context.join(mech, how="left")

    masks = regimes.segment_masks(base, context)
    keyed = {}
    kidx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(base["origin"]), base["hour"].astype(int)],
        names=["origin", "hour"],
    )
    for seg, m in masks.items():
        keyed[seg] = pd.Series(m.to_numpy(), index=kidx)

    print("-" * 78)
    print("PER REGIME — baseline vs each variant (HAC-corrected, hourly)")
    print("-" * 78)
    print("  A regime is a set of HOURS, not whole days, so the daily")
    print("  aggregation does not apply; HAC handles the autocorrelation.")
    print()
    segs = [s for s in keyed if s != "all"]
    variants = [n for n in sorted(res) if n != "baseline"]

    raw = {}
    flat = []
    for name in variants:
        for s in segs:
            pv, _ = dm_regime(base, res[name], keyed[s])
            raw[(name, s)] = pv
            flat.append(pv)
    adj_flat = holm_bonferroni(flat)
    adj = {k: adj_flat[i] for i, k in enumerate(raw)}

    n_tests = sum(1 for v in flat if not np.isnan(v))
    n_raw_sig = sum(1 for v in flat if not np.isnan(v) and v < ALPHA)
    n_adj_sig = sum(1 for v in adj_flat if not np.isnan(v) and v < ALPHA)

    hdr = f"  {'variant':<20}" + "".join(f"{s[:11]:>13}" for s in segs)
    print("RAW p-values:")
    print(hdr)
    for name in variants:
        row = f"  {name:<20}"
        for s in segs:
            pv = raw[(name, s)]
            row += f"{pv:11.3f}{stars(pv)[:2]}" if not np.isnan(pv) else f"{'n/a':>13}"
        print(row)
    print()
    print(f"HOLM-BONFERRONI adjusted over the {n_tests}-test family:")
    print(hdr)
    for name in variants:
        row = f"  {name:<20}"
        for s in segs:
            pv = adj[(name, s)]
            row += f"{pv:11.3f}{stars(pv)[:2]}" if not np.isnan(pv) else f"{'n/a':>13}"
        print(row)
    print()
    print(f"  {n_tests} tests. Significant at 5%: {n_raw_sig} raw -> "
          f"{n_adj_sig} after Holm.")
    print(f"  At alpha={ALPHA} a {n_tests}-test family expects "
          f"~{n_tests * ALPHA:.1f} false positives by chance, which is why the")
    print("  raw grid is shown but the ADJUSTED grid is the one to read.")
    print("  (p = variant significantly BETTER than baseline in that regime)")
    print()

    # B4's off-target gains, tested explicitly
    if "B4_coupling" in res:
        print("-" * 78)
        print("B4 OFF-TARGET GAINS — tested before anything is built on them")
        print("-" * 78)
        for seg in ("spike", "high_hydro", "coupling_stress"):
            if seg not in keyed:
                continue
            p_better, bw = dm_regime(base, res["B4_coupling"], keyed[seg])
            p_worse, _ = dm_regime(res["B4_coupling"], base, keyed[seg])
            print(f"  {seg:<18} p(B4 better)={p_better:.4f}{stars(p_better)}"
                  f"  p(B4 worse)={p_worse:.4f}{stars(p_worse)}  HAC bw={bw}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
