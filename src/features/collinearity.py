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
