"""SHAP interpretability run for thesis section 4-6.

Two stages, mirroring scripts/run_ood_stress.py:

    run_shap.py --fit     fit the interpretation-only models, save + metadata
    run_shap.py           (default) compute SHAP over the test days, cache the
                          summaries, export figures 10-15 and the 4-6 table

Why a separate fit at all: the models in models/frozen/ were trained on the
trailing 1092 days ending 2017-12-31, which contains the entire test period,
so explaining them over test days would be in-sample. `--fit` trains the same
wrappers on the trailing 1092 days ending strictly BEFORE the first test day,
so every explained day is unseen. See src/interpretability/shap_analysis.py.

FREEZE DISCIPLINE. This script only ever creates new paths --
models/interpretation/, data/processed/shap/, reports/figures/1[0-5]_*,
reports/tables/shap_importance.* -- and never opens an existing v1.0-results
artifact. `_assert_writable` enforces that at runtime rather than trusting it:
the PreToolUse freeze hook guards Edit/Write tool calls, not scripts.

Determinism: TreeSHAP with no background dataset is exact and has no random
component, so repeated runs are bit-identical. Seed 42 is still forced through
the model wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs unattended, never in a GUI session

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader, load_config
from src.evaluation.walk_forward import load_evaluation_config
from src.features.pipeline import build_features, daily_target
from src.interpretability.shap_analysis import (
    FEATURE_GROUPS,
    fit_interpretation_models,
    group_importance,
    interpretation_train_days,
    regime_split,
    shap_values_daily,
    shap_values_hourly,
    split_boundary,
)
from src.models import load_models_config
from src.models.daily import DailyLightGBMModel
from src.models.lgbm import LightGBMModel

INTERP_DIR = REPO_ROOT / "models" / "interpretation"
CACHE_DIR = REPO_ROOT / "data" / "processed" / "shap"
# A truncated smoke run (--limit) writes here instead, so it can never replace
# the committed artifacts behind figures 10-15. The captions interpolate their
# own statistics, so a smoke run's output stays truthful -- but it would still
# silently become the thing the thesis figures were drawn from.
SMOKE_DIR = CACHE_DIR / "_smoke"
FIG_DIR = REPO_ROOT / "reports" / "figures"
TAB_DIR = REPO_ROOT / "reports" / "tables"

# Every path this script is permitted to create. Anything else is a bug, and
# an unguarded one would silently overwrite a frozen result.
WRITABLE_PREFIXES = (INTERP_DIR, CACHE_DIR)
WRITABLE_FILES = tuple(
    FIG_DIR / name
    for name in (
        "10_shap_global_importance_hourly.png",
        "11_shap_beeswarm_hourly.png",
        "12_shap_hour_profile.png",
        "13_shap_calm_vs_stressed.png",
        "14_shap_hourly_vs_daily.png",
        "15_shap_waterfall_case_study.png",
    )
) + tuple(TAB_DIR / f"shap_importance.{ext}" for ext in ("tex", "csv"))

# The hour whose beeswarm goes in the thesis: the evening peak, where the
# price-formation story is most pronounced.
BEESWARM_HOUR = 18

ATTRIBUTION = "Data: Lago et al. (2021) open EPEX-DE benchmark via epftoolbox."


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _assert_writable(path: Path) -> Path:
    """Refuse to write anywhere outside this script's own new namespace.

    The freeze hook intercepts Edit/Write TOOL calls, not scripts, so a script
    is exactly the thing that can still clobber a frozen artifact. This makes
    that impossible here rather than merely unlikely.
    """
    path = Path(path).resolve()
    for prefix in WRITABLE_PREFIXES:
        if path == prefix.resolve() or prefix.resolve() in path.parents:
            return path
    if path in {p.resolve() for p in WRITABLE_FILES}:
        return path
    raise RuntimeError(
        f"run_shap: refusing to write {_rel(path)} -- outside the interpretability "
        "namespace. v1.0-results artifacts are frozen; this script only creates "
        "models/interpretation/, data/processed/shap/, figures 10-15 and "
        "reports/tables/shap_importance.*"
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    _assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    print(f"  wrote {_rel(path)}")


def _savefig(fig, path: Path) -> None:
    _assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_rel(path)}")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    boundary = split_boundary(df_test)
    X, Y = build_features(pd.concat([df_train, df_test]))
    return X, Y, boundary


# --------------------------------------------------------------------------
# stage 1: fit the interpretation-only models
# --------------------------------------------------------------------------
def fit_interpretation() -> None:
    X, Y, boundary = load_features()
    calib = int(load_evaluation_config()["walk_forward"]["calibration_window_days"])
    train_days = interpretation_train_days(X.index, boundary, calib)

    print(
        f"interpretation fit on {len(train_days)} days: "
        f"{train_days.min().date()} -> {train_days.max().date()} "
        f"(test opens {boundary.date()})"
    )

    hourly, daily = fit_interpretation_models(X, Y, train_days, load_models_config())

    INTERP_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in (("lightgbm", hourly), ("daily_lightgbm", daily)):
        path = _assert_writable(INTERP_DIR / name)
        model.save(path)
        print(f"  saved {name} -> {_rel(path)}")

    meta = {
        "purpose": "SHAP interpretation only (thesis 4-6); NOT a results artifact",
        "train_start": str(train_days.min().date()),
        "train_end": str(train_days.max().date()),
        "n_train_days": int(len(train_days)),
        "calibration_window_days": calib,
        "test_boundary": str(boundary.date()),
        "n_features": int(X.shape[1]),
    }
    meta_path = _assert_writable(INTERP_DIR / "metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {_rel(meta_path)}")


def load_interpretation_models() -> tuple[LightGBMModel, DailyLightGBMModel, dict]:
    meta_path = INTERP_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{_rel(meta_path)} not found -- run `run_shap.py --fit` first. The "
            "interpretation models are deliberately not committed (models/ is "
            "gitignored); the fit is deterministic and takes under a minute."
        )
    meta = json.loads(meta_path.read_text())
    models_cfg = load_models_config()
    hourly = LightGBMModel(models_cfg["lightgbm"]).load(INTERP_DIR / "lightgbm")
    daily = DailyLightGBMModel(models_cfg["daily_lightgbm"]).load(
        INTERP_DIR / "daily_lightgbm"
    )
    return hourly, daily, meta


def assert_explained_days_unseen(train_end: pd.Timestamp, first_explained: pd.Timestamp) -> None:
    """Runtime restatement of the guarantee the unit tests pin.

    Cheap, and it is the single property that makes every figure in 4-6
    defensible. It lives here as well as in the tests because the tests
    check the windowing FUNCTION, while this checks the artifacts actually
    on disk -- a stale models/interpretation/ from an older config could
    satisfy the former and violate the latter.
    """
    train_end, first_explained = pd.Timestamp(train_end), pd.Timestamp(first_explained)
    if train_end >= first_explained:
        raise RuntimeError(
            f"run_shap: interpretation model trained through {train_end.date()} but "
            f"the first explained day is {first_explained.date()} -- the explained "
            "days must be strictly unseen. Refusing to produce in-sample figures. "
            "Re-run `run_shap.py --fit` if the split moved."
        )


def assert_windows_disjoint(train_days: pd.Index, explained_days: pd.Index) -> None:
    """No explained day may appear in the training window, as a SET.

    Comparing only `train_end` against the first explained day is sufficient
    only while both windows are contiguous and adjacent. That is true today
    and silently stops being true if the feature index ever gains a gap, so
    the property is checked directly rather than inferred from two scalars.
    """
    overlap = pd.DatetimeIndex(train_days).intersection(pd.DatetimeIndex(explained_days))
    if len(overlap):
        raise RuntimeError(
            f"run_shap: {len(overlap)} day(s) appear in BOTH the interpretation "
            f"training window and the explained set (first: {overlap[0].date()}) "
            "-- these attributions would be in-sample. Refusing."
        )


# --------------------------------------------------------------------------
# stage 2: compute + cache
# --------------------------------------------------------------------------
def compute(x_limit: int | None = None) -> dict:
    """Compute SHAP over the test days and cache every summary a figure needs.

    Only summaries are cached, never the full (24, n_days, 247) tensor: the
    tensor is ~35 MB of float64 that nothing downstream reads twice, and
    data/processed is a partially-tracked tree where a large binary would be
    easy to commit by accident.
    """
    hourly, daily, meta = load_interpretation_models()
    X, Y, boundary = load_features()

    train_end = pd.Timestamp(meta["train_end"])
    train_start = pd.Timestamp(meta["train_start"])
    test_days = X.index[X.index >= boundary]
    if x_limit:
        test_days = test_days[:x_limit]

    assert_explained_days_unseen(train_end, test_days.min())
    assert_windows_disjoint(
        X.index[(X.index >= train_start) & (X.index <= train_end)], test_days
    )

    X_test = X.loc[test_days]
    print(
        f"explaining {len(test_days)} test days "
        f"({test_days.min().date()} -> {test_days.max().date()}), "
        f"{X_test.shape[1]} features"
    )

    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    regimes = regime_split(Y, threshold).loc[test_days]
    calm = regimes == "calm"
    stressed = regimes == "stressed"
    print(
        f"  regimes at {threshold} EUR/MWh: {int(calm.sum())} calm, "
        f"{int(stressed.sum())} stressed"
    )

    hourly_res = shap_values_hourly(hourly, X_test)
    daily_res = shap_values_daily(daily, X_test)
    columns = list(X_test.columns)

    # group x hour: the heatmap, and the source of the hourly column of every
    # other figure (averaged over hours).
    by_hour = pd.DataFrame(
        {h: group_importance(res.values, columns) for h, res in hourly_res.items()}
    )
    by_hour.columns.name = "hour"
    by_hour.index.name = "feature_group"

    def _grouped(res_values: np.ndarray, mask: pd.Series) -> pd.Series:
        return group_importance(res_values[mask.to_numpy()], columns)

    splits = pd.DataFrame(
        {
            "hourly_all": by_hour.mean(axis=1),
            "hourly_calm": pd.concat(
                [_grouped(r.values, calm) for r in hourly_res.values()], axis=1
            ).mean(axis=1),
            "hourly_stressed": pd.concat(
                [_grouped(r.values, stressed) for r in hourly_res.values()], axis=1
            ).mean(axis=1),
            "daily_all": group_importance(daily_res.values, columns),
            "daily_calm": _grouped(daily_res.values, calm),
            "daily_stressed": _grouped(daily_res.values, stressed),
        }
    )
    splits.index.name = "feature_group"

    # Beeswarm source: top-20 raw features at the evening peak, with the
    # feature VALUES alongside (the beeswarm colours by feature value).
    peak = hourly_res[BEESWARM_HOUR]
    top20 = peak.mean_abs().sort_values(ascending=False).head(20).index.tolist()
    idx = [columns.index(c) for c in top20]
    beeswarm = pd.DataFrame(peak.values[:, idx], index=test_days, columns=top20)
    beeswarm_data = X_test[top20]

    # Waterfall source: the day the MODEL itself called highest, explained by
    # the daily arm.
    #
    # Deliberately not the highest realized baseload. Selecting the argmax of
    # the OUTCOME and then captioning the figure "the model under-forecasts
    # the extreme" is circular: regression to the mean alone guarantees that
    # any well-calibrated forecast sits below the outcome-argmax, so such a
    # figure cannot fail to show under-forecasting whatever the model does,
    # and is therefore no evidence for the claim. Selecting by the model's own
    # prediction asks the question a waterfall actually answers -- what drove
    # this forecast -- and uses no test-period outcome at all.
    daily_prediction = pd.Series(
        daily_res.values.sum(axis=1) + daily_res.expected_value, index=test_days
    )
    baseload = daily_target(Y.loc[test_days])
    case_day = daily_prediction.idxmax()
    w_pos = list(test_days).index(case_day)

    # The under-forecasting claim is instead backed by an aggregate over the
    # top-decile baseload days, which is not selection-fragile the way a
    # single argmax day is.
    top_decile = baseload >= baseload.quantile(0.9)
    top_decile_bias = float(
        (daily_prediction[top_decile] - baseload[top_decile]).mean()
    )
    waterfall = pd.DataFrame(
        {
            "shap_value": daily_res.values[w_pos],
            "feature_value": X_test.iloc[w_pos].to_numpy(),
        },
        index=columns,
    )
    waterfall.index.name = "feature"

    out_dir = SMOKE_DIR if x_limit else CACHE_DIR
    if x_limit:
        print(f"  --limit is set: writing to {_rel(out_dir)}, NOT the thesis cache")

    _write_csv(by_hour, out_dir / "group_by_hour.csv")
    _write_csv(splits, out_dir / "group_importance_splits.csv")
    _write_csv(beeswarm, out_dir / f"beeswarm_h{BEESWARM_HOUR:02d}_shap.csv")
    _write_csv(beeswarm_data, out_dir / f"beeswarm_h{BEESWARM_HOUR:02d}_data.csv")
    _write_csv(waterfall, out_dir / "waterfall_case_study.csv")

    facts = {
        "n_test_days": int(len(test_days)),
        "test_start": str(test_days.min().date()),
        "test_end": str(test_days.max().date()),
        "n_calm": int(calm.sum()),
        "n_stressed": int(stressed.sum()),
        "threshold": threshold,
        "train_start": meta["train_start"],
        "train_end": meta["train_end"],
        "n_train_days": meta["n_train_days"],
        "beeswarm_hour": BEESWARM_HOUR,
        "waterfall_day": str(pd.Timestamp(case_day).date()),
        "waterfall_selection": "highest predicted daily baseload (no outcome used)",
        "waterfall_baseload": float(baseload.loc[case_day]),
        "waterfall_prediction": float(daily_prediction.loc[case_day]),
        "waterfall_regime": str(regimes.loc[case_day]),
        "top_decile_bias": top_decile_bias,
        "n_top_decile": int(top_decile.sum()),
        "daily_expected_value": float(daily_res.expected_value),
    }
    facts_path = _assert_writable(out_dir / "facts.json")
    facts_path.write_text(json.dumps(facts, indent=2))
    print(f"  wrote {_rel(facts_path)}")
    return facts


def load_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    def _read(name: str) -> pd.DataFrame:
        path = CACHE_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{_rel(path)} not found -- run `run_shap.py` without --figures-only "
                "to compute the SHAP cache first."
            )
        return pd.read_csv(path, index_col=0)

    by_hour = _read("group_by_hour.csv")
    by_hour.columns = [int(c) for c in by_hour.columns]
    splits = _read("group_importance_splits.csv")
    beeswarm = _read(f"beeswarm_h{BEESWARM_HOUR:02d}_shap.csv")
    beeswarm_data = _read(f"beeswarm_h{BEESWARM_HOUR:02d}_data.csv")
    waterfall = _read("waterfall_case_study.csv")
    facts = json.loads((CACHE_DIR / "facts.json").read_text())
    return by_hour, splits, beeswarm, beeswarm_data, waterfall, facts


# --------------------------------------------------------------------------
# stage 3: figures
# --------------------------------------------------------------------------
def _order(series: pd.Series) -> pd.Series:
    return series.sort_values(ascending=True)


def figure_global_importance(splits: pd.DataFrame, facts: dict) -> None:
    imp = _order(splits["hourly_all"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.to_numpy(), color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP| (EUR/MWh), averaged over the 24 hourly models")
    ax.set_ylabel("Feature family")
    ax.set_title(
        f"Global feature importance, hourly LightGBM (static fit)\n"
        f"{facts['n_test_days']} unseen test days "
        f"({facts['test_start']} to {facts['test_end']})",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, FIG_DIR / "10_shap_global_importance_hourly.png")


def figure_beeswarm(beeswarm: pd.DataFrame, beeswarm_data: pd.DataFrame, facts: dict) -> None:
    import shap

    expl = shap.Explanation(
        values=beeswarm.to_numpy(),
        data=beeswarm_data.to_numpy(),
        feature_names=list(beeswarm.columns),
    )
    shap.plots.beeswarm(expl, max_display=20, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.5, 6)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (EUR/MWh)")
    ax.set_title(
        f"Feature attributions at hour {facts['beeswarm_hour']:02d} (evening peak)\n"
        f"top 20 individual features, {facts['n_test_days']} unseen test days"
    )
    _savefig(fig, FIG_DIR / "11_shap_beeswarm_hourly.png")


def figure_hour_profile(by_hour: pd.DataFrame, facts: dict) -> None:
    ordered = by_hour.loc[[g for g in FEATURE_GROUPS if g in by_hour.index]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(ordered.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ordered.columns)))
    ax.set_xticklabels([f"{h:02d}" for h in ordered.columns], fontsize=8)
    ax.set_yticks(range(len(ordered.index)))
    ax.set_yticklabels(ordered.index, fontsize=9)
    ax.set_xlabel("Target hour of day")
    ax.set_ylabel("Feature family")
    ax.set_title(
        "Which features matter at which hour — hourly LightGBM\n"
        f"{facts['n_test_days']} unseen test days"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP| (EUR/MWh)")
    _savefig(fig, FIG_DIR / "12_shap_hour_profile.png")


def figure_calm_vs_stressed(splits: pd.DataFrame, facts: dict) -> None:
    order = _order(splits["hourly_all"]).index
    calm = splits.loc[order, "hourly_calm"]
    stressed = splits.loc[order, "hourly_stressed"]

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(y - 0.2, calm.to_numpy(), height=0.4, label=f"Calm ({facts['n_calm']} days)", color="#2b6cb0")
    ax.barh(y + 0.2, stressed.to_numpy(), height=0.4, label=f"Stressed ({facts['n_stressed']} days)", color="#c05621")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean |SHAP| (EUR/MWh), averaged over the 24 hourly models")
    ax.set_ylabel("Feature family")
    ax.set_title(
        "Feature importance by market regime, hourly LightGBM\n"
        f"regime set by the previous day's peak against "
        f"{facts['threshold']:.2f} EUR/MWh"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, FIG_DIR / "13_shap_calm_vs_stressed.png")


def figure_hourly_vs_daily(splits: pd.DataFrame, facts: dict) -> None:
    order = _order(splits["hourly_all"]).index
    hourly = splits.loc[order, "hourly_all"]
    daily = splits.loc[order, "daily_all"]

    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(y - 0.2, hourly.to_numpy(), height=0.4, label="Hourly (24 per-hour models)", color="#2b6cb0")
    ax.barh(y + 0.2, daily.to_numpy(), height=0.4, label="Daily direct (baseload)", color="#276749")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean |SHAP| (EUR/MWh)")
    ax.set_ylabel("Feature family")
    ax.set_title(
        "Hourly versus direct-daily attribution, LightGBM\n"
        f"{facts['n_test_days']} unseen test days (RQ4)"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    _savefig(fig, FIG_DIR / "14_shap_hourly_vs_daily.png")


def figure_waterfall(waterfall: pd.DataFrame, facts: dict) -> None:
    import shap

    expl = shap.Explanation(
        values=waterfall["shap_value"].to_numpy(),
        base_values=facts["daily_expected_value"],
        data=waterfall["feature_value"].to_numpy(),
        feature_names=list(waterfall.index),
    )
    shap.plots.waterfall(expl, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    # The title states the miss, it does not hide it. On the most extreme day
    # of the test period the direct-daily model reaches only a fraction of the
    # realized baseload, and a figure that quoted the actual value while the
    # plot showed f(x) would read as a contradiction rather than as the
    # finding it is: attribution explains what the model DID, not what was
    # right, and every family pushes the same way yet still falls short.
    plt.title(
        f"Direct-daily forecast for {facts['waterfall_day']} "
        f"({facts['waterfall_regime']} regime) — the day the model itself "
        f"called highest\n"
        f"predicted {facts['waterfall_prediction']:.1f} versus "
        f"{facts['waterfall_baseload']:.1f} EUR/MWh realized. Across the "
        f"{facts['n_top_decile']} highest-baseload days the mean signed error "
        f"is {facts['top_decile_bias']:.1f} EUR/MWh.",
        fontsize=9,
        pad=18,
    )
    _savefig(fig, FIG_DIR / "15_shap_waterfall_case_study.png")


# --------------------------------------------------------------------------
# stage 4: table
# --------------------------------------------------------------------------
def export_table(splits: pd.DataFrame, facts: dict) -> None:
    shown = splits[["hourly_calm", "hourly_stressed", "daily_calm", "daily_stressed"]].copy()
    shown.columns = ["Hourly calm", "Hourly stressed", "Daily calm", "Daily stressed"]
    shown = shown.loc[[g for g in FEATURE_GROUPS if g in shown.index]]
    shown.index.name = "Feature family"

    csv_path = _assert_writable(TAB_DIR / "shap_importance.csv")
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    shown.to_csv(csv_path, float_format="%.4f")
    print(f"  wrote {_rel(csv_path)}")

    # Every number in the caption is interpolated, never hand-typed -- the
    # ood_stress.tex caption carried a hand-typed rounding error for exactly
    # this reason (logs/decisions.md 2026-08-05).
    caption = (
        f"Mean |SHAP| attribution per feature family (EUR/MWh), LightGBM, over "
        f"{facts['n_test_days']} unseen test days from {facts['test_start']} to "
        f"{facts['test_end']}. The explained models are interpretation-only fits "
        f"trained on {facts['n_train_days']} days ending {facts['train_end']}, "
        f"strictly before the test period, so no explained day was seen in "
        f"training. Hourly columns average the 24 per-hour models; daily columns "
        f"are the direct baseload model. Regimes follow the ensemble definition "
        f"(previous day's peak above {facts['threshold']:.2f} EUR/MWh): "
        f"{facts['n_calm']} calm and {facts['n_stressed']} stressed days. "
        f"Note that the explained fit is STATIC, held fixed across the whole "
        f"test period, whereas the models behind the chapter's accuracy "
        f"results are recalibrated at every origin; these attributions "
        f"therefore describe a first-origin fit and cannot show drift in "
        f"feature reliance across 2016--2017. {ATTRIBUTION}"
    )
    tex_path = _assert_writable(TAB_DIR / "shap_importance.tex")
    shown.to_latex(
        tex_path,
        float_format="%.2f",
        caption=caption,
        label="tab:shap-importance",
        position="htbp",
        escape=True,
    )
    print(f"  wrote {_rel(tex_path)}")


def make_figures() -> None:
    by_hour, splits, beeswarm, beeswarm_data, waterfall, facts = load_cache()
    figure_global_importance(splits, facts)
    figure_beeswarm(beeswarm, beeswarm_data, facts)
    figure_hour_profile(by_hour, facts)
    figure_calm_vs_stressed(splits, facts)
    figure_hourly_vs_daily(splits, facts)
    figure_waterfall(waterfall, facts)
    export_table(splits, facts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit", action="store_true", help="fit and save the interpretation-only models"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="redraw figures from the existing cache without recomputing SHAP",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="explain only the first N test days (smoke test)"
    )
    args = parser.parse_args(argv)

    if args.fit:
        fit_interpretation()
        return

    if not args.figures_only:
        compute(x_limit=args.limit)
    make_figures()
    print("\nSHAP run complete (thesis 4-6).")


if __name__ == "__main__":
    main()
