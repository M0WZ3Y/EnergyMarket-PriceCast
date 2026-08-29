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
import hashlib
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

#: Where per-variant predictions are cached.
#:
#: Deliberately OUTSIDE data/processed. That directory is listed in
#: .claude/hooks/frozen_results_guard.py's FROZEN_DIRS and the v1.0-results
#: tag exists, so it is frozen -- and while a brand-new subdirectory there
#: would not overwrite any existing result, putting ablation output inside
#: the frozen tree invites exactly the ambiguity the freeze rule exists to
#: prevent. This path is new, unfrozen, and regenerable.
CACHE_DIR = REPO_ROOT / "data" / "ablation_cache"

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
    # ------------------------------------------------------------------
    # PAIRWISE variants, measured against B1 rather than against ALL.
    #
    # The single-vs-ALL design cannot separate "this feature is weak" from
    # "this feature is drowned in a 6-block set": in ALL every block competes
    # for the same degrees of freedom at p/n ~ 0.8, so a modest real effect
    # disappears. B1 is the only block demonstrated significant (DM p<0.001),
    # so pairing against it asks the question that matters -- does this block
    # add anything ON TOP of what already works.
    # ------------------------------------------------------------------
    "B1+coupling_state": (
        {"residual_load_gradient_block": True},
        {"coupling_state_block": True},
    ),
    "B1+coupling_split": (
        {"residual_load_gradient_block": True},
        {"coupling_split_block": True},
    ),
    "B1+headroom": (
        {"residual_load_gradient_block": True},
        {"dispatchable_headroom_block": True},
    ),
    "B1+B2": (
        {"residual_load_gradient_block": True},
        {"merit_order_explicit_block": {"lags": (1,)}, "capacity_structure_block": True},
    ),
    "B1+B3": (
        {"residual_load_gradient_block": True},
        {"carbon_block": True, "fuel_switch_proxy_block": True},
    ),
    "B1+B4": (
        {"residual_load_gradient_block": True},
        {"coupling_block": {"zones": ("FR", "NL"), "lags": (1,)},
         "cross_border_flow_block": True},
    ),
    "B1+B5": (
        {"residual_load_gradient_block": True},
        {"reserve_margin_block": True},
    ),
    "B1+B6": (
        {"residual_load_gradient_block": True},
        {"storage_block": True},
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
    # Block 1: ramps, and the high-RES / low-residual-load hours the residual
    # load construct is about.
    "B1_ramp": ("steep_ramp", "high_res", "low_residual"),
    # Block 2: which technology is setting the price -> gas-marginal hours.
    "B2_merit_explicit": ("gas_marginal",),
    # Block 3: carbon shifts the coal/gas switching point -> gas-marginal.
    "B3_carbon_switch": ("gas_marginal",),
    # Block 4: coupling -> hours where DE and its neighbours diverge.
    "B4_coupling": ("coupling_stress",),
    # Block 5: scarcity -> the price tail. NOTE this is a PROXY target: the
    # spike segment selects hours where price WAS high, not hours where
    # capacity WAS tight (see regimes.UNAVAILABLE_SEGMENTS['outage_scarcity']).
    "B5_reserve_margin": ("spike",),
    # Block 6: storage arbitrages the intraday shape -> low-residual-load and
    # high pumped-storage-activity hours.
    "B6_storage": ("low_residual", "high_hydro"),
    "ALL": ("steep_ramp", "gas_marginal", "coupling_stress", "spike"),
    "B1+coupling_state": ("coupling_stress",),
    "B1+coupling_split": ("coupling_stress",),
    "B1+headroom": ("spike",),
    "B1+B2": ("gas_marginal",),
    "B1+B3": ("gas_marginal",),
    "B1+B4": ("coupling_stress",),
    "B1+B5": ("spike",),
    "B1+B6": ("high_hydro",),
}

#: Blocks running on keyless PROXIES rather than the real feed, with what the
#: real feed would add. Muted or negative deltas here are an expected and
#: legitimate result, not a failure -- and must never be reported as
#: "this mechanism does not matter in DE-LU".
PROXY_BLOCKS = {
    "B5_reserve_margin": (
        "installed dispatchable capacity, NOT net of outages. The real "
        "ENTSO-E outage feed would make the denominator AVAILABLE capacity, "
        "which is what actually tightens before a scarcity event; the proxy "
        "denominator is near-constant within a year and so carries almost no "
        "event-time information."
    ),
    "B6_storage": (
        "observed pumped-storage dispatch, NOT reservoir filling rates. "
        "Dispatch is the RESPONSE to the price shape; the filling rate is the "
        "state variable that drives it. The proxy can only echo what the "
        "market already did, one day late."
    ),
    "B3_carbon_switch": (
        "carbon leg of SRMC only, plus an observed gas-vs-coal dispatch "
        "split. The fuel legs (TTF gas, API2 coal) are Montel-licensed, so a "
        "true clean spark/dark spread and a real switch indicator cannot be "
        "built at all."
    ),
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
    ap.add_argument("--criterion", choices=["aic", "bic"], default="aic",
                    help="LASSO selection criterion. 'aic' reproduces epftoolbox's "
                         "LEAR exactly. 'bic' is a HARNESS DIAGNOSTIC (see "
                         "src/models/_lear_bic_diagnostic.py): AIC under-penalises "
                         "at p/n~0.8, so a wide variant keeps too many coefficients "
                         "and its degradation is partly an artifact of the criterion.")
    ap.add_argument("--no-cache", action="store_true",
                    help="refit every variant even if a cached prediction exists")
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
    if args.criterion == "bic":
        from src.models._lear_bic_diagnostic import LEARLassoBICModel as _Model
        print("MODEL           LEAR-LASSO with BIC selection (HARNESS DIAGNOSTIC,")
        print("                not a thesis model; see _lear_bic_diagnostic.py)")
    else:
        _Model = LEARLassoModel
    # Segmentation context = physical state (residual load, ramp, RES
    # share) PLUS the mechanism-specific state (gas share, coupling
    # spread, pumped-storage activity) that the physics check needs.
    # Merged into one frame so every segment is defined on the same index.
    context = regimes.physical_context(wide)
    mech = regimes._stack_hourly(regimes.mechanism_context(df.index))
    if not mech.empty:
        context = context.join(mech, how="left")
    missing_segments = [c for c in ("gas_share", "coupling_spread",
                                    "pumped_activity")
                        if c not in context.columns]
    if missing_segments:
        print(f"  NOTE: mechanism context missing {missing_segments}; the")
        print("  segments depending on them are OMITTED, not reported empty.")

    # LEAR standardises with median/MAD, so a column whose values are more
    # than half identical to its median has MAD 0 and divides by zero --
    # surfacing much later as an opaque "Input X contains NaN" from sklearn.
    # Neighbour-price SPREADS are zero-inflated exactly this way, because
    # market coupling clears DE and its neighbour at the same price whenever
    # the interconnector is not binding. Drop them for THIS model and say so:
    # a column silently removed would make the variant a different experiment
    # from the one its name claims.
    # PREDICTIONS ARE PERSISTED PER VARIANT, and reloaded on a later run.
    #
    # An earlier version held every variant's predictions in memory and
    # computed all tables only after the last one finished. A kill during the
    # final variant therefore discarded seven completed fits -- about two
    # hours of compute -- with nothing recoverable, because success was only
    # ever recorded at the very end. Each variant now writes its own file as
    # soon as it finishes, so an interruption costs one variant instead of
    # all of them, and a re-run resumes rather than restarts.
    #
    # This writes to a NEW directory and touches no frozen artifact.
    cache_dir = CACHE_DIR / f"o{args.origins}_c{calibration}_{args.criterion}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"prediction cache  {cache_dir}")
    print()

    unscalable: dict[str, list[str]] = {}
    results: dict[str, pd.DataFrame] = {}
    for name, (X, Y, _) in built.items():
        Xa, Ya = X.loc[common], Y.loc[common]
        Xa, dropped = col.drop_unscalable_over_windows(Xa, origins, calibration)
        if dropped:
            unscalable[name] = dropped

        # The cache key includes the feature-set fingerprint, so a variant
        # whose columns changed since it was cached is refit rather than
        # silently reused -- a stale prediction file would otherwise be
        # scored as if it came from the current feature set.
        fingerprint = hashlib.md5(
            ("|".join(Xa.columns) + f"|{len(Xa)}").encode()
        ).hexdigest()[:12]
        cache_file = cache_dir / f"{name}__{fingerprint}.csv"

        if cache_file.exists() and not args.no_cache:
            results[name] = pd.read_csv(cache_file, parse_dates=["origin"])
            print(f"[cached] {name:<20} X={Xa.shape}  "
                  f"{len(results[name]) // 24} origins from {cache_file.name}")
            continue

        print(f"[run] {name:<20} X={Xa.shape} ...", end=" ", flush=True)
        t0 = time.time()
        frame = run_model(
            name, _Model(models_cfg["lear_lasso"]), Xa, Ya,
            eval_cfg=eval_cfg, first_origin=origins.min(),
        )
        results[name] = frame
        # Write immediately -- the whole point is that this survives a kill.
        frame.to_csv(cache_file, index=False, lineterminator="\n")
        print(f"{len(frame) // 24} origins in {time.time() - t0:.0f}s "
              f"-> {cache_file.name}")

    # --- pooled ------------------------------------------------------------
    print()
    if unscalable:
        print("-" * 78)
        print("COLUMNS DROPPED FOR LEAR (zero MAD — unscalable by its "
              "median/MAD scaler)")
        print("-" * 78)
        for n, cols in unscalable.items():
            print(f"  {n:<20} {len(cols)} dropped: {cols}")
        print("  Checked over EVERY training window the run uses, not just the")
        print("  full series: LEAR rescales per window, so a column can be fine")
        print("  globally and degenerate inside one window.")
        print("  These are physically meaningful and would be usable by a tree")
        print("  or neural model; a spread is zero-inflated because coupled")
        print("  markets clear at the SAME price whenever the border is not")
        print("  binding, so over half its values equal its median.")
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

    # ---------------------------------------------------------------- (a)
    print("(a) PER-BLOCK x PER-REGIME  -  MAE delta vs baseline (negative = better)")
    print("-" * 78)
    seg_tables = {}
    for name, frame in results.items():
        if name == "baseline":
            continue
        seg_tables[name] = regimes.compare_segmented(
            results["baseline"], frame, context
        )
    base_seg = regimes.segmented_metrics(results["baseline"], context)
    segs = list(base_seg.index)

    print("  baseline by segment:")
    print("    " + base_seg.to_string().replace("\n", "\n    "))
    print()
    hdr = "  " + "block".ljust(20) + "".join(x[:12].rjust(13) for x in segs)
    print(hdr)
    for name, cmp in seg_tables.items():
        row = "  " + name.ljust(20)
        for x in segs:
            row += (f"{cmp.loc[x, 'mae_delta']:13.3f}" if x in cmp.index
                    else "-".rjust(13))
        print(row)
    print()
    print("  full per-block tables (MAE, RMSE, n):")
    for name, cmp in seg_tables.items():
        print(f"  --- {name} ---")
        print("    " + cmp.to_string().replace("\n", "\n    "))
        print()

    # ---------------------------------------------------------------- (b)
    print("=" * 78)
    print("(b) PHYSICS CHECK - did each block improve the regime it TARGETS?")
    print("=" * 78)
    physics = {}
    flagged = []
    for name, cmp in seg_tables.items():
        targets = TARGET_REGIME.get(name, ())
        hits = []
        for t in targets:
            hits.append((t, float(cmp.loc[t, "mae_delta"]) if t in cmp.index else None))
        measured = [(t, d) for t, d in hits if d is not None]
        improved = bool(measured) and all(d < 0 for _, d in measured)
        any_improved = any(d < 0 for _, d in measured)
        agg = float(pooled.loc[name, "mae_delta"])
        physics[name] = dict(targets=hits, improved=improved,
                             any_improved=any_improved, aggregate=agg)
        verdict = "YES" if improved else ("PARTIAL" if any_improved else "NO")
        print(f"  {name:<20} target improved: {verdict:<8} "
              f"aggregate MAE delta {agg:+.4f}")
        for t, d in hits:
            if d is None:
                print(f"       {t:<18} NOT MEASURED (segment unavailable)")
            else:
                print(f"       {t:<18} {d:+9.4f}  "
                      f"{'improved' if d < 0 else 'WORSE'}")
        if agg < 0 and not any_improved:
            flagged.append(name)
            print("       *** FLAG: improved AGGREGATE MAE but NOT its own "
                  "target regime.")
            print("           A feature that helps everywhere except where its "
                  "mechanism operates")
            print("           is picking up a correlate, not the mechanism it "
                  "claims. Diagnosed below.")
        print()

    # ---------------------------------------------------------------- (c)
    print("=" * 78)
    print("(c) DIAGNOSIS OF UNDERPERFORMING / FLAGGED BLOCKS")
    print("=" * 78)
    print("  REDUNDANT      new columns already explained by baseline features")
    print("                 (the mechanism is already covered)")
    print("  MISSPECIFIED   independent of baseline yet fails its target regime")
    print("                 (independent-but-useless points at construction)")
    print("  GENUINELY_WEAK independent, moved its regime, but small effect")
    print()
    if flagged:
        print(f"  FLAGGED (aggregate gain without target-regime gain): "
              f"{', '.join(flagged)}")
        print()
    Xb = built["baseline"][0]
    for name in seg_tables:
        ph = physics[name]
        d = col.diagnose_block(
            built[name][0], Xb,
            improved_target_regime=ph["improved"],
            aggregate_gain=ph["aggregate"],
        )
        print(col.format_diagnosis(name, d))
        if name in PROXY_BLOCKS:
            print(f"       PROXY BLOCK - {PROXY_BLOCKS[name]}")
        print()

    # ---------------------------------------------------------------- (d)
    print("=" * 78)
    print("(d) COLLINEARITY PAIRS ABOVE THRESHOLD")
    print("=" * 78)
    target = built.get("ALL", built["baseline"])[0]
    pairs = col.correlated_pairs(target, threshold=col.CORR_THRESHOLD)
    print(f"  |r| >= {col.CORR_THRESHOLD}: {pairs.attrs['n_total_pairs']} pairs "
          f"in the {target.shape[1]}-column ALL variant")
    for _, r in pairs.head(15).iterrows():
        ba, bb = col.block_of(r.feature_a), col.block_of(r.feature_b)
        flag = "  <-- CROSS-BLOCK" if ba != bb else ""
        print(f"    {r.r:.4f}  {r.feature_a:<30} {r.feature_b:<30}{flag}")
    print()

    # ---------------------------------------------------------------- (e)
    print("=" * 78)
    print("(e) LEAKAGE GUARD STATUS")
    print("=" * 78)
    print(f"  variants checked : {len(built)}   rejected: {len(rejected)}")
    print(f"  {lg.summary()}")
    print("  negative tests in tests/test_leakage_guard.py - the guard must")
    print("  FAIL the build in each of these, and does:")
    for t in (
        "test_end_to_end_same_day_neighbour_price_fails_the_build",
        "test_end_to_end_target_day_realized_flow_fails_the_build",
        "test_end_to_end_same_day_realized_generation_fails_the_build",
        "test_the_unpatched_versions_of_those_blocks_pass   [control]",
    ):
        print(f"    {t}")
    print()

    # ---------------------------------------------------------------- (f)
    print("=" * 78)
    print("(f) SOURCE STATUS - real data / proxy / unavailable stub")
    print("=" * 78)
    rows = [
        ("REAL", "residual load, ramps", "benchmark exog_1/exog_2 forecasts"),
        ("REAL", "merit-order position", "derived from the same forecasts"),
        ("REAL", "dispatch shares", "Energy-Charts /public_power (lagged)"),
        ("REAL", "installed capacity", "Energy-Charts /installed_power"),
        ("REAL", "EUA carbon", "EEX auction archive 2012-2025"),
        ("REAL", "neighbour prices", "Energy-Charts /price (lagged)"),
        ("REAL", "cross-border flows", "Energy-Charts /cbpf (realized, lagged)"),
        ("REAL", "pumped storage", "Energy-Charts /public_power (lagged)"),
        ("PROXY", "fuel switching", "dispatch split; TTF/API2 are licensed"),
        ("PROXY", "reserve margin", "installed capacity, NOT net of outages"),
        ("PROXY", "scarcity (keyless)", "trailing-max residual load"),
        ("STUB", "generation outages", "ENTSO-E token required"),
        ("STUB", "NTC", "ENTSO-E token required"),
        ("STUB", "hydro reservoir levels", "ENTSO-E token required"),
        ("STUB", "clean spark spread", "TTF gas price is Montel-licensed"),
        ("STUB", "clean dark spread", "API2 coal price is Montel-licensed"),
    ]
    for kind, what, why in rows:
        print(f"  {kind:<6} {what:<24} {why}")
    print()
    print("  UNBUILDABLE SUB-FEATURES, explicitly: clean spark spread, clean")
    print("  dark spread, and a true coal-to-gas switch indicator. All three")
    print("  need gas and coal prices. Ember publishes only series DERIVED from")
    print("  Montel-licensed inputs, which makes Ember a citation, not a source.")
    print()

    # --- target sanity -----------------------------------------------------
    print("=" * 78)
    print("TARGET SANITY")
    print("=" * 78)
    yt = results["baseline"]["y_true"]
    print(f"  negative prices present : {bool((yt < 0).any())} "
          f"({int((yt < 0).sum())} of {len(yt)} hours)")
    print(f"  min / max observed      : {yt.min():.2f} / {yt.max():.2f} EUR/MWh")
    print(f"  within [-500, 4000]     : "
          f"{bool(yt.min() >= -500 and yt.max() <= 4000)}")
    print("  target transform        : none (raw EUR/MWh); no log, so negative")
    print("                            prices survive intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
