"""Ablation harness for the physical feature blocks — PRINTS ONLY.

Runs baseline -> baseline+block for each buildable physical block from
src/features/physical.py, and reports pooled and regime-segmented metric
deltas.

This script is deliberately NOT gated by src/ledger_gate (see
tests/test_ledger_gate.py's GATED/UNGATED lists): it writes nothing. No file
under data/processed, reports/ or models/ is created or modified, and no
frozen v1.0-results / v1.1-ood artifact is touched. Results go to stdout.

TWO PROTOCOL DEVIATIONS, both deliberate and both stated in the printed
header so no number from here can be mistaken for a thesis result:

  1. Reduced evaluation window. The thesis protocol recalibrates daily over
     the full 728-day test period; at ~5 s per origin per variant that is
     ~8 h for one model across five variants. This is a SCREEN over the most
     recent `--origins` days, meant to decide whether a block earns a full
     run, not to replace one.

  2. LEAR-LASSO as the vehicle. It is linear and its LASSO performs its own
     feature selection, so an added block gets a fair hearing without
     retuning. Giving LightGBM the same fairness would mean 50 Optuna trials
     per variant, and comparing an untuned LightGBM across feature sets
     would measure the stale hyperparameters, not the features.

FAIR-COMPARISON RULE (the thing most easily got wrong here): merit_order and
scarcity need a 365-day trailing reference, so enabling them drops ~358
early rows from X. Scoring a variant on a different, later, easier set of
days than the baseline would manufacture an improvement out of nothing.
Every variant is therefore evaluated on the INTERSECTION of all variants'
available origins, printed in the header.
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
from src.evaluation import regimes  # noqa: E402
from src.evaluation.run_baselines import run_model  # noqa: E402
from src.features.audit import audit_features, prioritize_gaps  # noqa: E402
from src.features.pipeline import (  # noqa: E402
    _pivot_to_daily_wide,
    build_features,
    load_feature_config,
)
from src.models import LEARLassoModel, load_models_config  # noqa: E402

#: One variant per buildable block, plus the baseline. Ordered by the
#: Stage-3 gap ranking so the highest-value block is measured first.
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "+residual_load": {"residual_load_block": True},
    "+merit_order": {"merit_order_block": True},
    "+ramp_gradient": {"residual_load_gradient_block": True},
    "+scarcity_proxy": {"scarcity_block": True},
    "+all_physical": {
        "residual_load_block": True,
        "merit_order_block": True,
        "residual_load_gradient_block": True,
        "scarcity_block": True,
    },
}


def build_variant(df: pd.DataFrame, blocks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (X, Y) with `blocks` enabled, without touching the config file.

    The on-disk configs/features.yaml keeps every flag false. This mutates
    only an in-memory copy, so a crashed run can never leave the repo in a
    state where the default pipeline silently includes physical features.
    """
    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = blocks
    return build_features(df, cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origins", type=int, default=100,
                    help="number of most-recent origin days to score (default 100)")
    ap.add_argument("--calibration", type=int, default=728,
                    help="trailing calibration window in days (default 728)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these variant names")
    args = ap.parse_args(argv)

    data_cfg = load_config()
    df_train, df_test = BenchmarkLoader(data_cfg).load()
    df = pd.concat([df_train, df_test])
    wide = _pivot_to_daily_wide(df)

    variants = VARIANTS if args.only is None else {
        k: v for k, v in VARIANTS.items() if k in set(args.only) | {"baseline"}
    }

    built = {name: build_variant(df, blocks) for name, blocks in variants.items()}

    # Fair-comparison rule: score every variant on the same days.
    common = None
    for X, _ in built.values():
        common = X.index if common is None else common.intersection(X.index)
    origins = common[-args.origins:]

    eval_cfg = {
        "walk_forward": {"calibration_window_days": args.calibration, "step_days": 1},
    }

    # LassoLarsIC cannot estimate noise variance when n_samples < n_features,
    # and dies deep inside sklearn with a message that says nothing about the
    # calibration window. The widest variant sets the requirement, so check it
    # up front against every variant rather than after minutes of fitting.
    widest = max((X.shape[1], name) for name, (X, _) in built.items())
    if args.calibration <= widest[0]:
        raise SystemExit(
            f"--calibration {args.calibration} is too small: variant "
            f"'{widest[1]}' has {widest[0]} features, and LEAR's LassoLarsIC "
            f"needs more calibration days than features. Use at least "
            f"{widest[0] + 1}."
        )

    print("=" * 78)
    print("PHYSICAL FEATURE ABLATION — SCREEN, NOT A THESIS RESULT")
    print("=" * 78)
    print(f"model             LEAR-LASSO (linear; LASSO self-selects features)")
    print(f"calibration       {args.calibration} days trailing, daily recalibration")
    print(f"scored origins    {len(origins)}  ({origins.min().date()} .. {origins.max().date()})")
    print(f"common day pool   {len(common)} days shared by all variants")
    print(f"variants          {', '.join(variants)}")
    print("writes            NOTHING — stdout only; no frozen artifact touched")
    print()

    print("--- STAGE 2/3: audit of the baseline feature matrix ---")
    a = audit_features(built["baseline"][0])
    print(a[["status", "driver_strength", "data_availability", "n_columns"]].to_string())
    print()
    print(prioritize_gaps(a)[["rank", "status", "score", "actionable", "blocked_by"]].to_string())
    print()

    models_cfg = load_models_config()
    context = regimes.physical_context(wide)

    results: dict[str, pd.DataFrame] = {}
    for name, (X, Y) in built.items():
        Xa, Ya = X.loc[common], Y.loc[common]
        print(f"[run] {name:<16} X={Xa.shape} ...", end=" ", flush=True)
        t0 = time.time()
        frame = run_model(
            name,
            LEARLassoModel(models_cfg["lear_lasso"]),
            Xa,
            Ya,
            eval_cfg=eval_cfg,
            first_origin=origins.min(),
        )
        results[name] = frame
        print(f"{len(frame)//24} origins in {time.time() - t0:.0f}s")

    print()
    print("=" * 78)
    print("POOLED METRICS")
    print("=" * 78)
    pooled = pd.DataFrame(
        {name: regimes.segmented_metrics(f).loc["all"] for name, f in results.items()}
    ).T
    base_mae = pooled.loc["baseline", "mae"]
    pooled["mae_delta_vs_base"] = pooled["mae"] - base_mae
    pooled["mae_pct"] = 100.0 * pooled["mae_delta_vs_base"] / base_mae
    print(pooled[["n", "mae", "rmse", "mae_delta_vs_base", "mae_pct"]].to_string())

    print()
    print("=" * 78)
    print("REGIME-SEGMENTED METRICS  (stage 5)")
    print("=" * 78)
    print(f"segments unavailable: {', '.join(regimes.UNAVAILABLE_SEGMENTS)}")
    for name, reason in regimes.UNAVAILABLE_SEGMENTS.items():
        print(f"  {name}: {reason}")
    print()

    base_seg = regimes.segmented_metrics(results["baseline"], context)
    print("baseline by segment:")
    print(base_seg.to_string())
    print()

    #: Which segment each block is supposed to help. A block that does not
    #: move its own target regime is flagged: that is the signature of a
    #: specification bug, not of an unhelpful feature.
    targets = {
        "+residual_load": "low_residual",
        "+merit_order": "high_res",
        "+ramp_gradient": "steep_ramp",
        "+scarcity_proxy": "spike",
    }

    for name, frame in results.items():
        if name == "baseline":
            continue
        print(f"--- {name} vs baseline ---")
        cmp = regimes.compare_segmented(results["baseline"], frame, context)
        print(cmp.to_string())
        tgt = targets.get(name)
        if tgt and tgt in cmp.index:
            d = cmp.loc[tgt, "mae_delta"]
            verdict = "IMPROVED" if d < 0 else "DID NOT IMPROVE  <-- check specification"
            print(f"  target regime '{tgt}': MAE delta {d:+.4f}  {verdict}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
