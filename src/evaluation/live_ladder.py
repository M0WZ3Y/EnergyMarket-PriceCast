"""v2 live ladder helpers — src/evaluation/live_ladder.py

Rungs (spec §6): R0 as served, R1a timezone fixed, R1b ENTSO-E definitions,
R2 + causal bias correction, R3 rolling recalibration, R4 re-tuned.

THE SELECTION CONTRACT. Every choice made here -- bias window, ensemble
weights, regime threshold, served model -- receives validation-window frames
only, and the functions that make those choices assert it against
TEST_START. Test frames are scored afterwards and never feed a choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VAL_START = pd.Timestamp("2025-07-01")
VAL_END = pd.Timestamp("2025-12-31")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-09-30")
HISTORY_END = pd.Timestamp("2025-06-30")

MEMBERS = ["SARIMAX", "LEAR-LASSO", "LightGBM", "LSTM"]
ALL_MODELS = ["naive", *MEMBERS]
ENSEMBLES = ["Ensemble (mean)", "Ensemble (static)", "Ensemble (regime-aware)"]
BIAS_WINDOWS = (3, 7, 14, 30, 60)
REGIME_KS = (3.0, 2.5, 2.0, 1.5)
MIN_DAYS_PER_REGIME = 20
REGIME_REFERENCE_DAYS = 364  # mean/std reference for the stress threshold (decision 2026-10-16 a)
LEAR_WINDOWS = (364, 728, 1092, 1456)

# Ex-ante information set (decision 2026-10-15 e): D+1 wind+solar is published
# about six hours after the auction, so the ex-ante variant drops it. Everything
# else in the feature set is known by 11:00 on D-1.
# ALLOW-list, not a block-list (leakage review 2026-10-16, F3): the only
# target-day (D0) input known before the auction is the D+1 load forecast. Every
# other D0 column -- exog_2_D0, resload_D0 and every physical block derived from
# D+1 wind/solar (resload_grad/ramp, merit_pos, res_share, tightness_proxy) -- is
# dropped, so enabling a new block can never smuggle D+1 renewables in.
EXANTE_SAFE_D0 = ("exog_1_D0_",)
EXANTE_SARIMAX_EXOG = ("exog_1_D0", "exog_2_D-1")


def exante(X: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix restricted to what is known before the D-1 noon auction."""
    return X.drop(columns=[c for c in X.columns if "_D0" in c and not c.startswith(EXANTE_SAFE_D0)])


def origins(frame: pd.DataFrame) -> pd.Series:
    """Parse the origin column. Resumed walk-forward files mix "YYYY-MM-DD" and
    "YYYY-MM-DD 00:00:00" (pandas formats each appended chunk independently),
    which a single-format parse rejects; both mean the same midnight."""
    return pd.to_datetime(frame["origin"], format="mixed")


def read_frame(path) -> pd.DataFrame:
    """Read a long forecast frame with a normalised datetime origin."""
    f = pd.read_csv(path)
    f["origin"] = origins(f)
    return f


def slug(name: str) -> str:
    """Same file naming as scripts/run_ood_stress.py."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def long_frame(name: str, Y: pd.DataFrame, preds: pd.DataFrame, days) -> pd.DataFrame:
    days = list(days)
    return pd.DataFrame(
        {
            "origin": np.repeat(pd.DatetimeIndex(days), 24),
            "hour": np.tile(np.arange(24), len(days)),
            "y_true": Y.loc[days].to_numpy().ravel(),
            "y_pred": preds.loc[days].to_numpy().ravel(),
            "model": name,
        }
    )


def window_slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    o = origins(frame)
    return frame[(o >= start) & (o <= end)].reset_index(drop=True)


def validation(frame: pd.DataFrame) -> pd.DataFrame:
    return window_slice(frame, VAL_START, VAL_END)


def test_window(frame: pd.DataFrame) -> pd.DataFrame:
    return window_slice(frame, TEST_START, TEST_END)


def _assert_validation_only(frame: pd.DataFrame, what: str) -> None:
    if len(frame) and origins(frame).max() >= TEST_START:
        raise AssertionError(f"{what}: received origins on/after {TEST_START.date()} -- selection must use validation only")


def mae_of(frame: pd.DataFrame) -> float:
    return float((frame["y_true"] - frame["y_pred"]).abs().mean())


# -- R2 ---------------------------------------------------------------------------
def select_bias_window(val_frame: pd.DataFrame, windows=BIAS_WINDOWS) -> tuple[int, dict[int, float]]:
    """Window minimising validation MAE of the causal correction (cold start 'expanding')."""
    from scripts.run_ood_recalibration import recalibrate

    _assert_validation_only(val_frame, "select_bias_window")
    scores = {w: mae_of(recalibrate(val_frame, w, cold_start="expanding")) for w in windows}
    return min(scores, key=scores.get), scores


def apply_bias_correction(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal correction over a continuous val+test frame (history flows into test)."""
    from scripts.run_ood_recalibration import recalibrate

    out = recalibrate(frame, window, cold_start="expanding")
    return out.drop(columns=["correction"])


# -- regime threshold (R3/R4) ----------------------------------------------------
def regime_threshold(history_prices: pd.Series, val_frame: pd.DataFrame,
                     ks=REGIME_KS, min_days: int = MIN_DAYS_PER_REGIME) -> tuple[float, float]:
    """v1's validation-only k-rule on the v2 history: largest k with >= min_days per regime.

    history_prices must end strictly before the validation window (it is the
    'train mean + k*std' statistic).
    """
    from src.evaluation.ensemble import regime_labels

    _assert_validation_only(val_frame, "regime_threshold")
    if history_prices.index.max() >= VAL_START:
        raise AssertionError("regime_threshold: history must end before the validation window")
    mu, sd = float(history_prices.mean()), float(history_prices.std())
    for k in sorted(ks, reverse=True):
        thr = mu + k * sd
        labels = regime_labels(val_frame, threshold=thr)
        n_st = sum(v == "stressed" for v in labels.values())
        if n_st >= min_days and len(labels) - n_st >= min_days:
            return k, thr
    raise ValueError(f"no k in {ks} gives >= {min_days} days per regime on validation")


def fit_ensembles(val_members: dict[str, pd.DataFrame], threshold: float) -> dict:
    """Static and regime-aware weights from validation frames only."""
    from src.evaluation.ensemble import fit_weights, regime_labels

    for m, f in val_members.items():
        _assert_validation_only(f, f"fit_ensembles[{m}]")
    static = fit_weights(val_members, test_days=pd.DatetimeIndex([TEST_START]))
    labels = regime_labels(next(iter(val_members.values())), threshold=threshold)
    regime = {}
    for reg in ("calm", "stressed"):
        days = [o for o, lab in labels.items() if lab == reg]
        regime[reg] = fit_weights({m: f[f["origin"].isin(days)] for m, f in val_members.items()},
                                  test_days=pd.DatetimeIndex([TEST_START]))
    return {"static": static, "regime": regime, "threshold": threshold}


def apply_ensembles(members: dict[str, pd.DataFrame], spec: dict) -> dict[str, pd.DataFrame]:
    from src.evaluation.ensemble import combine_forecasts, combine_regime_aware

    equal = {m: 1.0 for m in members}  # zero fitted parameters: cannot overfit validation
    return {
        "Ensemble (mean)": combine_forecasts(members, equal, name="Ensemble (mean)"),
        "Ensemble (static)": combine_forecasts(members, spec["static"], name="Ensemble (static)"),
        "Ensemble (regime-aware)": combine_regime_aware(
            members, spec["regime"], threshold=spec["threshold"], name="Ensemble (regime-aware)"
        ),
    }


# -- cross-fitted selection scores (leakage review 2026-10-16, F1) -----------------
# A candidate whose parameters are FITTED on validation (ensemble weights, R2's
# bias window) must not be ranked by its in-sample validation MAE against single
# models whose validation forecasts are out-of-sample. Its selection score is a
# 2-fold cross-fit: fit on one half of validation, score on the other, and swap.
# The SERVED parameters are still fitted on the whole validation window.
CV_SPLIT = pd.Timestamp("2025-10-01")  # halves: Jul-Sep | Oct-Dec


def _halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = origins(frame)
    return frame[o < CV_SPLIT], frame[o >= CV_SPLIT]


def _fit_ensembles_half(val_members: dict[str, pd.DataFrame], threshold: float) -> dict:
    """fit_ensembles on a half; a regime with < MIN_DAYS_PER_REGIME days in that
    half falls back to the half's static weights (too few days to fit its own)."""
    from src.evaluation.ensemble import fit_weights, regime_labels

    for m, f in val_members.items():
        _assert_validation_only(f, f"crossfit[{m}]")
    static = fit_weights(val_members, test_days=pd.DatetimeIndex([TEST_START]))
    labels = regime_labels(next(iter(val_members.values())), threshold=threshold)
    regime = {}
    for reg in ("calm", "stressed"):
        days = [o for o, lab in labels.items() if lab == reg]
        regime[reg] = (fit_weights({m: f[f["origin"].isin(days)] for m, f in val_members.items()},
                                   test_days=pd.DatetimeIndex([TEST_START]))
                       if len(days) >= MIN_DAYS_PER_REGIME else static)
    return {"static": static, "regime": regime, "threshold": threshold}


def crossfit_ensembles(val_members: dict[str, pd.DataFrame], threshold: float) -> dict[str, pd.DataFrame]:
    """Out-of-sample validation frames for the FITTED ensembles (static, regime-aware)."""
    halves = [{m: _halves(f)[i] for m, f in val_members.items()} for i in (0, 1)]
    parts: dict[str, list] = {"Ensemble (static)": [], "Ensemble (regime-aware)": []}
    for fit_i, score_i in ((0, 1), (1, 0)):
        ens = apply_ensembles(halves[score_i], _fit_ensembles_half(halves[fit_i], threshold))
        for n in parts:
            parts[n].append(ens[n])
    return {n: pd.concat(p).sort_values(["origin", "hour"]).reset_index(drop=True) for n, p in parts.items()}


def crossfit_bias(val_frame: pd.DataFrame) -> pd.DataFrame:
    """R2: bias window chosen on one validation half, its (causal) correction scored on the other."""
    _assert_validation_only(val_frame, "crossfit_bias")
    a, b = _halves(val_frame)
    wa, _ = select_bias_window(a)
    wb, _ = select_bias_window(b)
    return pd.concat([_halves(apply_bias_correction(val_frame, wb))[0],
                      _halves(apply_bias_correction(val_frame, wa))[1]]).sort_values(["origin", "hour"]).reset_index(drop=True)


# -- selection -----------------------------------------------------------------------
def select_served(val_scores: pd.DataFrame) -> dict:
    """Lowest validation MAE over (rung, model). val_scores: columns rung, model, val_mae, n_days."""
    full = val_scores[val_scores["n_days"] == val_scores["n_days"].max()]
    best = full.sort_values(["val_mae", "rung", "model"]).iloc[0]
    return {"rung": str(best["rung"]), "model": str(best["model"]), "val_mae": float(best["val_mae"]),
            "n_val_days": int(best["n_days"])}


def common_days(frames: list[pd.DataFrame]) -> pd.DatetimeIndex:
    sets = [set(origins(f)) for f in frames]
    return pd.DatetimeIndex(sorted(set.intersection(*sets)))


def restrict(frame: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    return frame[origins(frame).isin(days)].sort_values(["origin", "hour"]).reset_index(drop=True)
