"""Collinearity / VIF diagnostics — src/features/collinearity.py

The redundancy rule for this feature set is maximal mechanism coverage with
MINIMAL redundancy: one clean feature per physical driver. That rule needs a
measurement, because the physical blocks are built from overlapping inputs
and collinearity here is not hypothetical -- residual load is exactly
`exog_1 - exog_2`, and dispatch shares sum to one by construction.

Why it matters beyond tidiness: collinear inputs do not usually hurt point
forecasts much, but they make coefficient and SHAP attributions unstable, and
SHAP is a deliverable of this thesis, not a diagnostic. Two collinear columns
split one mechanism's importance arbitrarily between them, and the split
moves from fold to fold. So this is reported per block, and the report is
printed rather than acted on automatically -- dropping a column is a
modelling decision, not something a diagnostic should do silently.

Both diagnostics are computed on the ACTUAL feature matrix, never on a
sample or a proxy for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: |r| above which a pair is reported. 0.95 is deliberately high: the point
#: is to surface near-duplicates, not every correlated pair, and hourly
#: columns of the same series are correlated by nature.
CORR_THRESHOLD = 0.95

#: VIF above which a column is reported. 10 is the conventional cutoff.
VIF_THRESHOLD = 10.0


def correlated_pairs(
    X: pd.DataFrame, threshold: float = CORR_THRESHOLD, max_pairs: int = 40
) -> pd.DataFrame:
    """Feature pairs whose absolute correlation exceeds `threshold`.

    Only the upper triangle is scanned, so each pair is reported once. Zero-
    variance columns are excluded first: their correlation is undefined, and
    including them yields NaN comparisons that silently drop out of the
    result rather than being reported as the degenerate columns they are.
    """
    num = X.select_dtypes(include=[np.number])
    nunique = num.nunique()
    degenerate = list(nunique[nunique <= 1].index)
    num = num.drop(columns=degenerate)

    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "r"])

    corr = num.corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    stacked = corr.where(mask).stack()
    hits = stacked[stacked >= threshold].sort_values(ascending=False)

    out = pd.DataFrame(
        {
            "feature_a": [i[0] for i in hits.index[:max_pairs]],
            "feature_b": [i[1] for i in hits.index[:max_pairs]],
            "r": hits.values[:max_pairs],
        }
    )
    out.attrs["n_total_pairs"] = int(len(hits))
    out.attrs["degenerate_columns"] = degenerate
    return out


def variance_inflation(X: pd.DataFrame, columns=None, sample: int | None = None) -> pd.Series:
    """VIF per column, computed by regressing each column on the others.

    VIF_j = 1 / (1 - R2_j). Computed from the correlation matrix inverse
    rather than by fitting p separate regressions, which is the same
    quantity at a fraction of the cost.

    A singular correlation matrix means at least one column is an EXACT
    linear combination of others -- infinite VIF, and precisely the case
    worth knowing about here (residual load is exactly exog_1 - exog_2). The
    pseudo-inverse is used so those columns report a huge-but-finite VIF and
    appear in the ranking instead of raising and hiding every other result.
    """
    num = X.select_dtypes(include=[np.number])
    if columns is not None:
        num = num[[c for c in columns if c in num.columns]]

    nunique = num.nunique()
    num = num.drop(columns=list(nunique[nunique <= 1].index))
    num = num.dropna()
    if sample and len(num) > sample:
        num = num.sample(sample, random_state=42)
    if num.shape[1] < 2:
        return pd.Series(dtype=float)

    corr = num.corr().to_numpy()
    inv = np.linalg.pinv(corr)
    vif = pd.Series(np.diag(inv), index=num.columns)
    return vif.sort_values(ascending=False)


def block_of(column: str) -> str:
    """Best-effort mapping from a column name to its feature block.

    Used only to group the printed report; an unrecognised prefix is
    reported under its own name rather than silently bucketed as 'other',
    so a new block cannot hide inside an existing group.
    """
    if column.startswith("price_D-"):
        return "benchmark:price_lags"
    if column.startswith("exog_"):
        return "benchmark:exog"
    if column.startswith("dow_"):
        return "benchmark:calendar"
    for prefix, name in (
        ("resload_grad", "physical:ramp"),
        ("resload_ramp", "physical:ramp"),
        ("resload_", "physical:residual_load"),
        ("merit_pos", "physical:merit_order"),
        ("res_share", "physical:merit_order"),
        ("tightness_proxy", "physical:scarcity_proxy"),
        ("nbprice", "exog:coupling"),
        ("nbspread", "exog:coupling"),
        ("netimport", "exog:flows"),
        ("absflow", "exog:flows"),
        ("gasshare", "exog:merit_explicit"),
        ("coalshare", "exog:merit_explicit"),
        ("ligshare", "exog:merit_explicit"),
        ("nucshare", "exog:merit_explicit"),
        ("resgenshare", "exog:merit_explicit"),
        ("thermalshare", "exog:merit_explicit"),
        ("switch_proxy", "exog:fuel_switch"),
        ("carbon_", "exog:carbon"),
        ("eua_", "exog:carbon"),
        ("disp_capacity", "exog:capacity"),
        ("resmargin", "exog:reserve_margin"),
        ("pump", "exog:storage"),
    ):
        if column.startswith(prefix):
            return name
    return f"unmapped:{column.split('_')[0]}"


def report(
    X: pd.DataFrame,
    corr_threshold: float = CORR_THRESHOLD,
    vif_threshold: float = VIF_THRESHOLD,
    vif_sample: int | None = 2000,
    max_print: int = 25,
) -> str:
    """The printed collinearity report for one feature matrix."""
    lines = []
    lines.append(f"features: {X.shape[1]} columns x {X.shape[0]} rows")

    blocks = pd.Series({c: block_of(c) for c in X.columns}).value_counts()
    lines.append("")
    lines.append("columns by block:")
    for b, n in blocks.items():
        lines.append(f"  {b:<28} {n}")

    pairs = correlated_pairs(X, threshold=corr_threshold)
    n_total = pairs.attrs.get("n_total_pairs", len(pairs))
    degenerate = pairs.attrs.get("degenerate_columns", [])
    lines.append("")
    lines.append(f"collinear pairs |r| >= {corr_threshold}: {n_total}")
    if degenerate:
        lines.append(
            f"  zero-variance columns excluded ({len(degenerate)}): "
            f"{degenerate[:5]}"
        )
    if len(pairs):
        shown = pairs.head(max_print)
        for _, r in shown.iterrows():
            a_b, b_b = block_of(r.feature_a), block_of(r.feature_b)
            flag = "  <-- CROSS-BLOCK" if a_b != b_b else ""
            lines.append(f"  {r.r:.4f}  {r.feature_a:<30} {r.feature_b:<30}{flag}")
        if n_total > max_print:
            lines.append(f"  ... and {n_total - max_print} more")

    vif = variance_inflation(X, sample=vif_sample)
    high = vif[vif >= vif_threshold]
    lines.append("")
    lines.append(f"VIF >= {vif_threshold}: {len(high)} of {len(vif)} columns")
    if len(high):
        for c, v in high.head(max_print).items():
            lines.append(f"  {v:12.1f}  {c:<34} [{block_of(c)}]")
        if len(high) > max_print:
            lines.append(f"  ... and {len(high) - max_print} more")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Diagnosis of an underperforming block
# --------------------------------------------------------------------------

#: |r| against the baseline above which a block's columns are judged to be
#: already covered by existing features.
REDUNDANT_R = 0.90

#: Median |r| below which a block is judged independent of the baseline.
INDEPENDENT_R = 0.50

#: Coefficient of variation, relative to the TARGET's own CoV, below which a
#: block's columns are treated as too static to inform a forecast on this
#: window. A driver that barely moves while the target swings cannot have a
#: coefficient fitted to it, however correctly it is built -- that is a
#: mismatch of timescale, not a defect of construction, and conflating the two
#: would file a correct feature as broken. Measured example: EUA carbon has
#: CoV 0.031 against a target CoV of 0.553 over 80 days (ratio 0.056), because
#: a compliance market clears on a multi-week horizon.
STATIC_COV_RATIO = 0.25


def diagnose_block(
    X_variant: pd.DataFrame,
    X_baseline: pd.DataFrame,
    improved_target_regime: bool,
    aggregate_gain: float,
    target: pd.Series | None = None,
    window_index=None,
) -> dict:
    """Classify why a block underperformed: REDUNDANT / MISSPECIFIED / WEAK.

    The three are different problems with different fixes, and an ablation
    delta alone cannot tell them apart -- a block that does nothing because
    the mechanism is already covered looks identical to one that does nothing
    because it was built wrong.

    The discriminator is how much of each NEW column is already explained by
    the columns that were there before:

      REDUNDANT     new columns are highly correlated with existing ones. The
                    mechanism is already represented; adding it again buys
                    nothing and costs degrees of freedom.
      TIMESCALE_MISMATCH
                    new columns are independent of the baseline but barely
                    MOVE inside the evaluation window, relative to how much
                    the target moves. Correctly built, simply too static to
                    inform a forecast at this resolution over this horizon.
                    Distinct from misspecification and much more optimistic:
                    the same feature may be informative on a longer window.
      MISSPECIFIED  new columns are INDEPENDENT of the baseline (so they do
                    carry new information) yet fail to help the regime they
                    physically target. Independent-but-useless points at the
                    construction, not the mechanism.
      GENUINELY_WEAK new columns are independent and DID move their target
                    regime, but the effect is small. Correct, distinct, and
                    simply low-signal in this market.

    `aggregate_gain` is signed as MAE delta: negative means the variant
    improved. A block that improves aggregate MAE while failing its target
    regime is flagged separately -- it is picking up a correlate rather than
    the mechanism it claims, and that is the case worth investigating.
    """
    new_cols = [c for c in X_variant.columns if c not in set(X_baseline.columns)]
    if not new_cols:
        return {"verdict": "NO_NEW_COLUMNS", "n_new": 0}

    common_rows = X_variant.index.intersection(X_baseline.index)
    V = X_variant.loc[common_rows, new_cols].select_dtypes(include=[np.number])
    B = X_baseline.loc[common_rows].select_dtypes(include=[np.number])

    V = V.loc[:, V.nunique() > 1]
    B = B.loc[:, B.nunique() > 1]
    if V.empty or B.empty:
        return {"verdict": "DEGENERATE", "n_new": len(new_cols)}

    joint = pd.concat([V, B], axis=1).dropna()
    if len(joint) < 30:
        return {"verdict": "INSUFFICIENT_OVERLAP", "n_new": len(new_cols)}

    corr = joint.corr().abs()
    cross = corr.loc[V.columns, B.columns]
    # For each new column, how much of it is already in the baseline.
    best = cross.max(axis=1)

    frac_redundant = float((best >= REDUNDANT_R).mean())
    median_r = float(best.median())

    # Relative variability of the new columns inside the scored window,
    # against the target's own. Computed before the verdict so a static block
    # is never filed as misspecified.
    cov_ratio = float("nan")
    if target is not None:
        rows = joint.index if window_index is None else joint.index.intersection(window_index)
        sub = V.loc[V.index.intersection(rows)]
        tgt = target.loc[target.index.intersection(rows)]
        if len(sub) > 10 and len(tgt) > 10 and tgt.std() > 0 and abs(tgt.mean()) > 0:
            m = sub.mean().abs()
            covs = (sub.std() / m.where(m > 0)).replace([np.inf, -np.inf], np.nan)
            tgt_cov = tgt.std() / abs(tgt.mean())
            if covs.notna().any() and tgt_cov > 0:
                cov_ratio = float(covs.median() / tgt_cov)

    static = (not np.isnan(cov_ratio)) and cov_ratio < STATIC_COV_RATIO

    if frac_redundant >= 0.5:
        verdict = "REDUNDANT"
    elif static and not improved_target_regime:
        verdict = "TIMESCALE_MISMATCH"
    elif median_r < INDEPENDENT_R and not improved_target_regime:
        verdict = "MISSPECIFIED"
    elif not improved_target_regime:
        verdict = "PARTIALLY_REDUNDANT"
    else:
        verdict = "GENUINELY_WEAK" if abs(aggregate_gain) < 0.05 else "EFFECTIVE"

    worst = best.sort_values(ascending=False)
    top = [
        (c, cross.loc[c].idxmax(), float(best[c])) for c in worst.index[:5]
    ]
    return {
        "verdict": verdict,
        "n_new": len(new_cols),
        "median_max_r_vs_baseline": median_r,
        "frac_cols_r_ge_0.90": frac_redundant,
        "most_redundant": top,
        "improved_target_regime": improved_target_regime,
        "aggregate_mae_delta": aggregate_gain,
        "cov_ratio_vs_target": cov_ratio,
    }


def format_diagnosis(name: str, d: dict) -> str:
    """One block's diagnosis, printed."""
    if d.get("n_new", 0) == 0:
        return f"  {name:<20} {d['verdict']}"
    lines = [
        f"  {name:<20} {d['verdict']:<20} "
        f"{d['n_new']:>4} new cols, median max|r| vs baseline "
        f"{d['median_max_r_vs_baseline']:.3f}, "
        f"{100 * d['frac_cols_r_ge_0.90']:.0f}% at |r|>=0.90"
        + (f", CoV ratio vs target {d['cov_ratio_vs_target']:.3f}"
           if not np.isnan(d.get("cov_ratio_vs_target", float("nan"))) else "")
    ]
    for col, against, r in d.get("most_redundant", [])[:3]:
        lines.append(f"       {r:.4f}  {col:<32} ~ {against}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Scaler compatibility
# --------------------------------------------------------------------------

#: Number of trailing weekday-dummy columns that LEAR excludes from scaling.
#: This is why LEARLassoModel._assert_dow_columns_last exists: epftoolbox's
#: LEAR standardises all but the last 7 columns, so the dummies -- which have
#: MAD 0 by construction, being 6/7 zeros -- never reach the scaler.
N_DOW_COLUMNS = 7


def zero_mad_columns(X: pd.DataFrame, exclude_trailing: int = N_DOW_COLUMNS) -> list[str]:
    """Columns LEAR's median/MAD scaler cannot standardise.

    epftoolbox's LEAR scales with median and MEDIAN ABSOLUTE DEVIATION, not
    mean and standard deviation. A column can therefore have healthy variance
    and still be unscalable: if more than half its values equal its median,
    MAD is exactly 0 and the transform divides by zero, producing NaN that
    surfaces much later as an opaque "Input X contains NaN" from sklearn.

    This is not a hypothetical. A neighbour-price SPREAD is zero-inflated by
    construction: market coupling means DE and its neighbour clear at exactly
    the same price whenever the interconnector is not binding, which for
    DE-NL is over half of all hours at 06 and 18. The feature is physically
    meaningful and would be usable by a tree or a neural model; it is this
    particular scaler it defeats.

    The trailing weekday dummies are excluded because LEAR does not scale
    them, so their MAD of 0 is harmless.
    """
    if exclude_trailing:
        candidate = X.iloc[:, :-exclude_trailing]
    else:
        candidate = X
    num = candidate.select_dtypes(include=[np.number])
    if num.empty:
        return []
    med = num.median()
    mad = (num - med).abs().median()
    return list(mad[mad == 0].index)


def drop_unscalable(
    X: pd.DataFrame, exclude_trailing: int = N_DOW_COLUMNS
) -> tuple[pd.DataFrame, list[str]]:
    """Drop zero-MAD columns, preserving the dummies-last ordering.

    Returns (X_kept, dropped). Dropping is done here rather than inside the
    model wrapper deliberately: the limitation belongs to LEAR's scaler, not
    to the feature, and the existing model code is frozen. Callers MUST report
    what was dropped -- a feature silently removed from a variant makes that
    variant a different experiment from the one its name claims.
    """
    dropped = zero_mad_columns(X, exclude_trailing=exclude_trailing)
    if not dropped:
        return X, []
    return X.drop(columns=dropped), dropped


def zero_mad_in_any_window(
    X: pd.DataFrame,
    origins,
    window: int,
    exclude_trailing: int = N_DOW_COLUMNS,
) -> list[str]:
    """Columns with zero MAD in ANY training window the run will use.

    A global MAD check is not sufficient, and assuming otherwise costs a
    whole run: LEAR refits on a trailing window at every origin and scales
    against THAT window's median and MAD. A column can have healthy MAD over
    the full series and still be exactly constant-at-its-median inside one
    899-day window, which divides by zero for that origin alone and fails the
    run partway through.

    So the check has to be the union over every window actually used, which
    is also what makes the resulting feature set identical across origins --
    a per-window drop would silently change the model's inputs from day to
    day, making the ablation a comparison of moving targets.
    """
    if exclude_trailing:
        candidate = X.iloc[:, :-exclude_trailing]
    else:
        candidate = X
    num = candidate.select_dtypes(include=[np.number])
    if num.empty:
        return []

    bad: set[str] = set()
    idx = X.index
    for o in origins:
        pos = idx.get_loc(o)
        start = max(0, pos - window)
        chunk = num.iloc[start:pos]
        if len(chunk) < 2:
            continue
        med = chunk.median()
        mad = (chunk - med).abs().median()
        bad.update(mad[mad == 0].index)
    return sorted(bad)


def drop_unscalable_over_windows(
    X: pd.DataFrame,
    origins,
    window: int,
    exclude_trailing: int = N_DOW_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns unscalable in any training window, dummies kept last."""
    dropped = zero_mad_in_any_window(
        X, origins, window, exclude_trailing=exclude_trailing
    )
    if not dropped:
        return X, []
    return X.drop(columns=dropped), dropped
