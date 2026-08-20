"""Combination-design ablation ladder — scripts/run_combination_ladder.py

Governed by the pre-registered improvement gate in logs/decisions.md
(2026-08-20), plus its amendment the same day. Read both before changing
anything here: the criterion, the 0.02 EUR/MWh floor, the six-test
Holm-Bonferroni family and the stopping rule were all fixed in advance and
must not be revised to suit an outcome.

WHAT THIS ANSWERS. The test-fitted oracle over the four members scores MAE
3.558 while the legitimate validation-fitted regime-aware ensemble already
scores 3.5569 -- so GLOBAL convex reweighting has no headroom left. That
bound was computed for one scalar weight per model applied identically
across all 24 hours. Everything here sits outside that family and is
therefore unmeasured, not known-futile:

  rung 0  global convex weights                    (baseline, not DM-tested)
  rung 1  per-hour convex weights                  (24 simplex vectors)
  rung 2  per-hour unconstrained linear stacking   (negative weights allowed)
  rung 3  regime-gated per-hour convex weights     (2 x 24 vectors)

NO INTERCEPT ANYWHERE, including rung 2. An intercept is a bias correction,
a different mechanism with its own literature and its own arm of this
project (the OOD recalibration experiment). Allowing one here would confound
"negative weights help" with "a constant shift helps", and the whole point of
a ladder is that each rung isolates ONE change.

LEAKAGE. Weights are fitted on the validation window only, never on test.
The gate's amendment splits that window temporally -- fit on the first 273
days, measure the adoption criterion on the held-out last 91 -- because the
weights are fitted on validation, so validation MAE would otherwise be
in-sample and would reward parameters for their own sake. An adopted rung is
refitted on the full validation window and scored ONCE on test.

Two member sets run side by side (user decision 2026-08-20):
  frozen  the frozen seed-42 members; isolates the combiner
  seedens the seed-ensembled LSTM from the supplementary arm

Writes nothing under reports/ by default -- use --export once results are
final. Touches nothing frozen.

Usage:
    python scripts/run_combination_ladder.py
    python scripts/run_combination_ladder.py --export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_seed_ensemble import seed_ensemble_frame
from src.evaluation.ensemble import _aligned_pivots, regime_labels
from src.evaluation.metrics import diebold_mariano_hac, mae, rmae
from src.evaluation.walk_forward import load_evaluation_config

VAL_DIR = REPO_ROOT / "data" / "processed" / "validation_preds"
TEST_DIR = REPO_ROOT / "data" / "processed" / "baselines"
SEED_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble"
SEED_VAL_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble_val"
OUT_DIR = REPO_ROOT / "data" / "processed" / "combination_ladder"
TABLES_DIR = REPO_ROOT / "reports" / "tables"

MEMBERS = ("SARIMAX", "LEAR-LASSO", "LightGBM", "LSTM")
FILES = {"SARIMAX": "sarimax.csv", "LEAR-LASSO": "lear_lasso.csv",
         "LightGBM": "lightgbm.csv", "LSTM": "lstm.csv"}

# Pre-registered, logs/decisions.md 2026-08-20. Not tunable from the CLI on
# purpose: a gate you can pass a flag to is not a gate.
FLOOR_EUR_MWH = 0.02
INNER_SELECT_DAYS = 91
FAMILY_SIZE = 6


def _long(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin"])


def _fit_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MAE-minimizing convex weights. Mirrors ensemble.fit_weights.

    Kept local rather than imported because fit_weights takes long frames and
    enforces the validation-before-test contract at frame level; here the
    caller has already sliced down to one hour (or one regime-hour) and the
    ordering check has already run on the full frames.
    """
    from scipy.optimize import minimize

    n = P.shape[0]
    if not np.isfinite(P).all() or not np.isfinite(y).all():
        raise ValueError("non-finite values reached the weight fit")
    res = minimize(
        lambda w: float(np.mean(np.abs(y - w @ P))),
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
    )
    # SLSQP returns its starting point on failure, which would silently
    # degrade the rung to an equal-weight average while still being labelled
    # "fitted". Fail loudly instead.
    if not res.success:
        raise RuntimeError(f"simplex weight fit did not converge: {res.message}")
    return res.x / res.x.sum()


def _fit_unconstrained(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MAE-minimizing linear weights, no simplex constraint, NO intercept."""
    from scipy.optimize import minimize

    n = P.shape[0]
    res = minimize(
        lambda w: float(np.mean(np.abs(y - w @ P))),
        x0=np.full(n, 1.0 / n),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"unconstrained weight fit did not converge: {res.message}")
    return res.x


def load_member_set(which: str) -> tuple[dict, dict]:
    """Return (validation frames, test frames) for one member set."""
    val = {m: _long(VAL_DIR / FILES[m]) for m in MEMBERS}
    test = {m: _long(TEST_DIR / FILES[m]) for m in MEMBERS}
    if which == "seedens":
        val["LSTM"], _ = seed_ensemble_frame(SEED_VAL_DIR)
        test["LSTM"], _ = seed_ensemble_frame(SEED_DIR, extra=TEST_DIR / "lstm.csv")
    elif which != "frozen":
        raise ValueError(f"unknown member set: {which}")
    return val, test


def _split_validation(truth: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Temporal inner split — never random, and never reordered.

    The last INNER_SELECT_DAYS origins are held out to measure the adoption
    criterion; everything before them fits the weights.
    """
    days = truth.index.sort_values()
    if len(days) <= INNER_SELECT_DAYS:
        raise SystemExit(f"validation window too short: {len(days)} days")
    return days[:-INNER_SELECT_DAYS], days[-INNER_SELECT_DAYS:]


def _predict(preds: dict, weights, days: pd.DatetimeIndex, mode: str,
             labels: dict | None = None) -> np.ndarray:
    """Apply a rung's weights to produce a [days x 24] forecast matrix."""
    names = list(preds)
    stack = np.stack([preds[m].loc[days].to_numpy() for m in names])  # [M, D, 24]
    out = np.zeros(stack.shape[1:])
    if mode == "global":
        out = np.tensordot(weights, stack, axes=(0, 0))
    elif mode in ("per_hour", "per_hour_unconstrained"):
        for h in range(stack.shape[2]):
            out[:, h] = weights[h] @ stack[:, :, h]
    elif mode == "regime_per_hour":
        regimes = np.array([labels.get(d, "calm") for d in days])
        for r in ("calm", "stressed"):
            sel = regimes == r
            if not sel.any():
                continue
            for h in range(stack.shape[2]):
                out[sel, h] = weights[r][h] @ stack[:, sel, h]
    else:
        raise ValueError(mode)
    return out


def _fit_rung(preds: dict, truth: pd.DataFrame, days: pd.DatetimeIndex,
              mode: str, labels: dict | None = None):
    names = list(preds)
    stack = np.stack([preds[m].loc[days].to_numpy() for m in names])
    y = truth.loc[days].to_numpy()

    if mode == "global":
        return _fit_simplex(stack.reshape(len(names), -1), y.ravel())
    if mode == "per_hour":
        return [_fit_simplex(stack[:, :, h], y[:, h]) for h in range(y.shape[1])]
    if mode == "per_hour_unconstrained":
        return [_fit_unconstrained(stack[:, :, h], y[:, h]) for h in range(y.shape[1])]
    if mode == "regime_per_hour":
        regimes = np.array([labels.get(d, "calm") for d in days])
        out = {}
        for r in ("calm", "stressed"):
            sel = regimes == r
            n_days = int(sel.sum())
            if n_days < 2:
                raise SystemExit(f"regime '{r}' has {n_days} fit days — too few")
            out[r] = [_fit_simplex(stack[:, sel, h], y[sel, h]) for h in range(y.shape[1])]
            print(f"    regime {r}: fitted on {n_days} days")
        return out
    raise ValueError(mode)


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).mean())


def _rmae_from_matrix(days: pd.DatetimeIndex, y: np.ndarray, p: np.ndarray) -> float:
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)])
    real = pd.Series(y.ravel(), index=idx).sort_index().to_frame("price")
    pred = pd.Series(p.ravel(), index=idx).sort_index().to_frame("price")
    return float(rmae(real, pred, m="W"))


RUNGS = [
    ("rung 0: global convex", "global"),
    ("rung 1: per-hour convex", "per_hour"),
    ("rung 2: per-hour unconstrained", "per_hour_unconstrained"),
    ("rung 3: regime-gated per-hour", "regime_per_hour"),
]


def run_ladder(which: str, threshold: float) -> pd.DataFrame:
    val_frames, test_frames = load_member_set(which)
    val_truth, val_preds = _aligned_pivots(val_frames)
    test_truth, test_preds = _aligned_pivots(test_frames)

    fit_days, sel_days = _split_validation(val_truth)
    val_labels = regime_labels(val_frames["LightGBM"], threshold=threshold)
    test_labels = regime_labels(test_frames["LightGBM"], threshold=threshold)
    n_stressed = sum(1 for d in fit_days if val_labels.get(d) == "stressed")
    print(f"\n=== member set: {which} ===")
    print(f"validation inner-fit {len(fit_days)} days ({n_stressed} stressed), "
          f"inner-select {len(sel_days)} days")
    print(f"test {len(test_truth.index)} origins")

    rows, prev = [], None
    consecutive_failures = 0

    for label, mode in RUNGS:
        print(f"  {label}")
        try:
            w_inner = _fit_rung(val_preds, val_truth, fit_days, mode, val_labels)
        except (SystemExit, RuntimeError) as exc:
            print(f"    SKIPPED — {exc}")
            rows.append({"member set": which, "rung": label, "status": f"not fitted: {exc}"})
            continue

        # Criterion: OUT-OF-SAMPLE on the held-out inner-select days.
        sel_pred = _predict(val_preds, w_inner, sel_days, mode, val_labels)
        sel_mae = _mae(val_truth.loc[sel_days].to_numpy(), sel_pred)

        # Refit on the FULL validation window, then score test exactly once.
        w_full = _fit_rung(val_preds, val_truth, val_truth.index, mode, val_labels)
        test_pred = _predict(test_preds, w_full, test_truth.index, mode, test_labels)
        y_test = test_truth.to_numpy()
        row = {
            "member set": which,
            "rung": label,
            "val MAE (held-out)": sel_mae,
            "test MAE": _mae(y_test, test_pred),
            "test rMAE": _rmae_from_matrix(test_truth.index, y_test, test_pred),
        }

        if prev is not None:
            row["val delta"] = prev["sel_mae"] - sel_mae
            row["clears 0.02 floor"] = bool(row["val delta"] >= FLOOR_EUR_MWH)
            # One-sided DM: does THIS rung beat the previous one on test?
            row["DM p vs prev (raw)"] = float(
                diebold_mariano_hac(p_real=y_test, p_pred_1=prev["test_pred"],
                                    p_pred_2=test_pred)
            )
            passed = row["clears 0.02 floor"]
            consecutive_failures = 0 if passed else consecutive_failures + 1
        rows.append(row)
        prev = {"sel_mae": sel_mae, "test_pred": test_pred}

        if consecutive_failures >= 2:
            print("    stopping rule (§6.5): two consecutive rungs failed the floor")
            break

    return pd.DataFrame(rows)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values over the pre-registered family."""
    m = FAMILY_SIZE
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main(export: bool = False) -> None:
    threshold = float(load_evaluation_config()["regime"]["stress_threshold_eur_mwh"])
    table = pd.concat([run_ladder(w, threshold) for w in ("frozen", "seedens")],
                      ignore_index=True)

    mask = table["DM p vs prev (raw)"].notna() if "DM p vs prev (raw)" in table else None
    if mask is not None and mask.any():
        raw = table.loc[mask, "DM p vs prev (raw)"].tolist()
        if len(raw) != FAMILY_SIZE:
            print(f"\nNOTE: {len(raw)} DM tests ran, family was pre-registered as "
                  f"{FAMILY_SIZE}. Holm correction still divides by "
                  f"{FAMILY_SIZE} — the family is fixed in advance, so a rung "
                  "skipped by the stopping rule does not shrink it.")
        table.loc[mask, "DM p vs prev (Holm)"] = holm_bonferroni(raw)

    print("\n" + "=" * 70)
    print(table.to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "ladder.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'ladder.csv'}")
    if export:
        print("(--export: table export not wired until results are final)")


if __name__ == "__main__":
    from src.ledger_gate import require_ledger_progress

    require_ledger_progress(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    main(export=args.export)
