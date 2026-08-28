"""Block-by-block ablation over the full physical feature set — PRINTS ONLY.

Extends scripts/run_physical_ablation.py to blocks 2-6 (the snapshot-derived
ones) and adds the leakage-guard status and collinearity report the task
brief asks for.

Writes nothing. No data/processed artifact, no report file, no frozen
v1.0-results or v1.1-ood output is touched, which is why this is correctly
absent from tests/test_ledger_gate.py's GATED list.

THREE THINGS THAT WOULD RIG THE COMPARISON IF LEFT UNHANDLED, all enforced:

1. COMMON DAY POOL. Energy-Charts serves nothing before 2015-01-01 while the
   benchmark starts 2012-01-09, and merit_order/scarcity need a 365-day
   trailing reference. Different variants therefore have different usable
   day sets. Scoring a variant on a later, easier set of days than the
   baseline manufactures an improvement out of nothing, so every variant is
   scored on the INTERSECTION of all variants' origins.

2. LEAKAGE GUARD PER VARIANT. Each variant's columns are checked against the
   pre-gate-closure information set before it is allowed to run. A variant
   that fails is reported and SKIPPED rather than scored -- a leaking
   variant's good number is worse than no number.

3. PROTOCOL HONESTY. Reduced evaluation window and LEAR-LASSO as the vehicle,
   both printed in the header. LEAR is used because its LASSO self-selects
   features, so a block gets a fair hearing without retuning; comparing an
   untuned LightGBM across feature sets would measure stale hyperparameters.
   A known consequence, visible in the earlier run: a block that is an exact
   LINEAR COMBINATION of existing columns (residual load = exog_1 - exog_2)
   can carry no new information for a linear model, and is systematically
   understated here relative to what a tree or an LSTM could extract.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import BenchmarkLoader, load_config  # noqa: E402
from src.data.sources import provenance as prov  # noqa: E402
from src.data.sources import snapshot  # noqa: E402
from src.evaluation import regimes  # noqa: E402
from src.evaluation.run_baselines import run_model  # noqa: E402
from src.features import collinearity as col  # noqa: E402
from src.features import leakage_guard as lg  # noqa: E402
from src.features.audit import audit_features, prioritize_gaps  # noqa: E402
from src.features.pipeline import build_features, load_feature_config  # noqa: E402
from src.models import LEARLassoModel, load_models_config  # noqa: E402

#: variant name -> (physical_blocks, exog_blocks)
VARIANTS: dict[str, tuple[dict, dict]] = {
    "baseline": ({}, {}),
    # Block 1 (keyless) — best performer from the earlier screen.
    "B1_ramp": ({"residual_load_gradient_block": True}, {}),
    # Block 2 — explicit merit order / marginal technology.
    "B2_merit_explicit": ({}, {"merit_order_explicit_block": True,
                               "capacity_structure_block": True}),
    # Block 3 — carbon (reduced) + fuel-switch proxy.
    "B3_carbon_switch": ({}, {"carbon_block": True,
                              "fuel_switch_proxy_block": True}),
    # Block 4 — market coupling.
    "B4_coupling": ({}, {"coupling_block": True,
                         "cross_border_flow_block": True}),
    # Block 5 — scarcity / reserve margin.
    "B5_reserve_margin": ({}, {"reserve_margin_block": True}),
    # Block 6 — storage / hydro.
    "B6_storage": ({}, {"storage_block": True}),
    # Everything buildable at once.
    "ALL": (
        {
            "residual_load_block": True,
            "residual_load_gradient_block": True,
            "merit_order_block": True,
            "scarcity_block": True,
        },
        {
            # REDUNDANCY RULE applied: one lag and the two largest
            # interconnections, not every zone at every lag. The full-fat
            # version is ALL_FULL below and is normally unestimable.
            "merit_order_explicit_block": {"lags": (1,)},
            "capacity_structure_block": True,
            "carbon_block": True,
            "fuel_switch_proxy_block": True,
            "coupling_block": {"zones": ("FR", "NL"), "lags": (1,)},
            "cross_border_flow_block": True,
            "reserve_margin_block": True,
            "storage_block": True,
        },
    ),
    # Every block at full width. Retained deliberately so the p > n limit is
    # measured and reported rather than assumed: with the common day pool
    # capped at ~1079 days by the 2015 Energy-Charts floor, this variant has
    # more features than there are days to fit them, and LEAR cannot estimate
    # it at all. It is skipped with that reason printed, not silently dropped.
    "ALL_FULL": (
        {
            "residual_load_block": True,
            "residual_load_gradient_block": True,
            "merit_order_block": True,
            "scarcity_block": True,
        },
        {
            "merit_order_explicit_block": True,
            "capacity_structure_block": True,
            "carbon_block": True,
            "fuel_switch_proxy_block": True,
            "coupling_block": True,
            "cross_border_flow_block": True,
            "reserve_margin_block": True,
            "storage_block": True,
        },
    ),
}

#: Which regime each block is physically supposed to help. A block that does
#: not move its own target regime is flagged: that is the signature of a
#: specification bug, not of an unhelpful feature.
TARGET_REGIME = {
    "B1_ramp": "steep_ramp",
    "B2_merit_explicit": "high_res",
    "B3_carbon_switch": "spike",
    "B4_coupling": "spike",
    "B5_reserve_margin": "spike",
    "B6_storage": "steep_ramp",
}


def build_variant(df, physical: dict, exog: dict):
    """Build (X, Y) for one variant without touching the config file."""
    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = physical
    cfg["exog_blocks"] = exog
    lg.clear_registry()
    X, Y = build_features(df, cfg)
    declared = lg.registry()
    return X, Y, declared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", type=int, default=100)
    ap.add_argument("--calibration", type=int, default=728)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--collinearity-only", action="store_true",
                    help="print the audit, guard and collinearity report, then stop")
    args = ap.parse_args(argv)

    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    df = pd.concat([df_train, df_test])

    from src.features.pipeline import _pivot_to_daily_wide

    wide = _pivot_to_daily_wide(df)

    variants = VARIANTS if args.only is None else {
        k: v for k, v in VARIANTS.items() if k in set(args.only) | {"baseline"}
    }

    print("=" * 78)
    print("FULL PHYSICAL ABLATION — SCREEN, NOT A THESIS RESULT")
    print("=" * 78)
    print("model           LEAR-LASSO (linear; LASSO self-selects features)")
    print(f"calibration     {args.calibration} days trailing, daily recalibration")
    print("writes          NOTHING — stdout only; no frozen artifact touched")
    print()

    print("-" * 78)
    print("DATA PROVENANCE")
    print("-" * 78)
    print(prov.format_provenance_table())
    missing = [k for k in ("entsoe_outages", "entsoe_ntc", "entsoe_reservoir")
               if not snapshot.has(k)]
    if missing:
        print()
        print(f"Tier B NOT pinned ({', '.join(missing)}) — ENTSOE_API_TOKEN unset.")
        print("  True outage-adjusted reserve margin, NTC and reservoir levels are")
        print("  therefore absent. Dependent features skip; nothing is backfilled.")
    print()

    # --- build every variant, check each for leakage -----------------------
    built: dict[str, tuple] = {}
    rejected: dict[str, str] = {}
    for name, (physical, exog) in variants.items():
        X, Y, declared = build_variant(df, physical, exog)
        try:
            lg.assert_no_leakage(strict=False)
        except lg.LeakageError as exc:
            rejected[name] = str(exc)
            continue
        built[name] = (X, Y, declared)

    print("-" * 78)
    print("LEAKAGE GUARD")
    print("-" * 78)
    for name, (X, _, declared) in built.items():
        n_ext = sum(1 for d in declared.values()
                    if d.source not in ("price", "exog_1", "exog_2", "calendar"))
        print(f"  PASS  {name:<20} {X.shape[1]:>5} cols, "
              f"{len(declared):>4} declared ({n_ext} external)")
    for name, why in rejected.items():
        print(f"  FAIL  {name:<20} SKIPPED — {why.splitlines()[1].strip()}")
    if not rejected:
        print("  no variant reads data published after the forecast origin")
    print()

    # --- Stage 2/3 audit ---------------------------------------------------
    print("-" * 78)
    print("STAGE 2 — FEATURE AUDIT (ALL blocks enabled)")
    print("-" * 78)
    ref = built.get("ALL", built.get("baseline"))
    a = audit_features(ref[0])
    print(a[["status", "driver_strength", "data_availability", "n_columns"]].to_string())
    print()
    print("STAGE 3 — GAP PRIORITIZATION")
    print(prioritize_gaps(a)[["rank", "status", "score", "actionable", "blocked_by"]]
          .to_string())
    print()

    # --- collinearity ------------------------------------------------------
    print("-" * 78)
    print("COLLINEARITY / VIF  (redundancy rule)")
    print("-" * 78)
    print("BASELINE, for context — high VIF is inherent to the 24h-wide design:")
    bX = built["baseline"][0]
    bv = col.variance_inflation(bX, sample=2000)
    bp = col.correlated_pairs(bX)
    print(f"  {int((bv >= 10).sum())} of {len(bv)} columns at VIF>=10; "
          f"max {bv.max():.3e}; {bp.attrs['n_total_pairs']} pairs |r|>=0.95")
    print()
    if "ALL" in built:
        print(col.report(built["ALL"][0], max_print=12))
    print()

    if args.collinearity_only:
        return 0

    # --- fair common day pool ---------------------------------------------
    common = None
    for X, _, _ in built.values():
        common = X.index if common is None else common.intersection(X.index)
    origins = common[-args.origins:]
    print("-" * 78)
    print("EVALUATION WINDOW")
    print("-" * 78)
    print(f"common day pool  {len(common)} days shared by all variants "
          f"({common.min().date()} .. {common.max().date()})")
    print(f"scored origins   {len(origins)} "
          f"({origins.min().date()} .. {origins.max().date()})")
    # LassoLarsIC cannot estimate a model with more features than samples, and
    # the ceiling here is structural: the common day pool is bounded by the
    # 2015 Energy-Charts floor, so a wide variant is not merely expensive but
    # UNESTIMABLE. Report which variants that excludes and carry on with the
    # rest -- aborting the whole run would lose the seven measurable blocks
    # because of the one that cannot be measured.
    max_calibration = len(common) - len(origins)
    calibration = min(args.calibration, max_calibration)
    if calibration != args.calibration:
        print(f"calibration      {args.calibration} -> {calibration} "
              f"(capped by the {len(common)}-day common pool)")

    unestimable = {n: X.shape[1] for n, (X, _, _) in built.items()
                   if X.shape[1] >= calibration}
    for n, p_cols in unestimable.items():
        print(f"  SKIP {n:<20} {p_cols} features >= {calibration} calibration days "
              f"— p > n, not estimable by LEAR on the available history")
        built.pop(n)
    if "baseline" not in built:
        raise SystemExit("baseline itself is unestimable; nothing to compare against")
    print()

    eval_cfg = {"walk_forward": {"calibration_window_days": calibration,
                                 "step_days": 1}}
    models_cfg = load_models_config()
    context = regimes.physical_context(wide)

    results: dict[str, pd.DataFrame] = {}
    for name, (X, Y, _) in built.items():
        Xa, Ya = X.loc[common], Y.loc[common]
        print(f"[run] {name:<20} X={Xa.shape} ...", end=" ", flush=True)
        t0 = time.time()
        results[name] = run_model(
            name, LEARLassoModel(models_cfg["lear_lasso"]), Xa, Ya,
            eval_cfg=eval_cfg, first_origin=origins.min(),
        )
        print(f"{len(results[name]) // 24} origins in {time.time() - t0:.0f}s")

    # --- pooled ------------------------------------------------------------
    print()
    print("=" * 78)
    print("POOLED METRICS")
    print("=" * 78)
    pooled = pd.DataFrame(
        {n: regimes.segmented_metrics(f).loc["all"] for n, f in results.items()}
    ).T
    base = pooled.loc["baseline", "mae"]
    pooled["mae_delta"] = pooled["mae"] - base
    pooled["mae_pct"] = 100.0 * pooled["mae_delta"] / base
    pooled["n_cols"] = [built[n][0].shape[1] for n in pooled.index]
    print(pooled[["n_cols", "n", "mae", "rmse", "mae_delta", "mae_pct"]].to_string())

    # --- per regime --------------------------------------------------------
    print()
    print("=" * 78)
    print("REGIME-SEGMENTED METRICS")
    print("=" * 78)
    for name, reason in regimes.UNAVAILABLE_SEGMENTS.items():
        print(f"segment '{name}' UNAVAILABLE: {reason}")
    print()
    print("baseline by segment:")
    print(regimes.segmented_metrics(results["baseline"], context).to_string())
    print()

    verdicts = []
    for name, frame in results.items():
        if name == "baseline":
            continue
        print(f"--- {name} vs baseline ---")
        cmp = regimes.compare_segmented(results["baseline"], frame, context)
        print(cmp.to_string())
        tgt = TARGET_REGIME.get(name)
        if tgt and tgt in cmp.index:
            d = cmp.loc[tgt, "mae_delta"]
            ok = d < 0
            print(f"  target regime '{tgt}': MAE delta {d:+.4f}  "
                  f"{'IMPROVED' if ok else 'DID NOT IMPROVE  <-- check specification'}")
            verdicts.append((name, tgt, d, ok))
        print()

    print("=" * 78)
    print("TARGET-REGIME VERDICTS")
    print("=" * 78)
    for name, tgt, d, ok in verdicts:
        print(f"  {name:<20} {tgt:<14} {d:+9.4f}  "
              f"{'OK' if ok else 'FLAGGED'}")

    # --- target sanity checks the brief asks for ---------------------------
    print()
    print("=" * 78)
    print("TARGET SANITY")
    print("=" * 78)
    yt = results["baseline"]["y_true"]
    print(f"  negative prices present : {bool((yt < 0).any())} "
          f"({int((yt < 0).sum())} of {len(yt)} hours)")
    print(f"  min / max observed      : {yt.min():.2f} / {yt.max():.2f} EUR/MWh")
    print(f"  within [-500, 4000]     : "
          f"{bool(yt.min() >= -500 and yt.max() <= 4000)}")
    print("  target transform        : none (raw EUR/MWh) — no log, so negative "
          "prices survive intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
