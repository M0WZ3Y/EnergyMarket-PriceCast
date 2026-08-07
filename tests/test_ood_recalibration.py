"""Tests for scripts/run_ood_recalibration.py — post-hoc OOD bias correction.

This analysis is supplementary: it sits on top of v1.1-ood and must not be
able to change it. So the tests pin two different kinds of property.

The STATISTICAL properties (causality, cold-start handling, metric
agreement) are what make the number defensible. The rolling correction is
the whole experiment; if a single future day leaks into a day's correction
the result is not a bias correction at all, it is hindsight, and it would
produce exactly the flattering "recovery" the hypothesis hopes to see.
That is the failure mode this file exists to prevent.

The INTEGRITY property is that running the whole thing leaves the frozen
artifacts byte-identical. The freeze hook only intercepts Edit/Write tool
calls, not scripts writing the same paths (NEXT_SESSION.md), so a script
is precisely the thing that could silently overwrite a tagged number.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import mae, rmae  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_ood_recalibration", REPO_ROOT / "scripts" / "run_ood_recalibration.py"
)
recal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recal)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(y_true_by_day, y_pred_by_day, start="2026-01-08", model="M") -> pd.DataFrame:
    """Build a long frame [origin, hour, y_true, y_pred, model].

    Each day is flat across its 24 hours, so a day's mean signed error is
    exactly y_true - y_pred for that day and the expected correction can be
    written down by hand.
    """
    days = pd.date_range(start, periods=len(y_true_by_day), freq="D", tz="UTC")
    parts = []
    for day, t, p in zip(days, y_true_by_day, y_pred_by_day):
        parts.append(
            pd.DataFrame(
                dict(origin=[day] * 24, hour=range(24), y_true=float(t),
                     y_pred=float(p), model=model)
            )
        )
    return pd.concat(parts, ignore_index=True)


def _hash_tree(paths) -> dict[str, str]:
    out = {}
    for path in sorted(paths):
        if path.is_file():
            out[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------
# causality — the property the whole experiment rests on
# --------------------------------------------------------------------------
def test_correction_uses_only_strictly_past_days():
    """A step change in the future must not move today's correction.

    Two series identical up to day 9 and wildly different afterwards must
    produce identical corrections for every day up to and including day 9.
    If any future day leaked in, the later divergence would show up early.
    """
    n = 20
    true_a = [100.0] * n
    pred_a = [90.0] * n                      # constant -10 signed error
    true_b = list(true_a)
    pred_b = [90.0] * 10 + [-500.0] * 10     # identical, then a huge step

    ca = recal.rolling_signed_correction(_frame(true_a, pred_a), window=5)
    cb = recal.rolling_signed_correction(_frame(true_b, pred_b), window=5)

    days = ca.index[:10]
    pd.testing.assert_series_equal(ca.loc[days], cb.loc[days])


def test_correction_equals_mean_signed_error_of_the_previous_window_days():
    """Hand-computable case: the correction is the mean of the prior window's
    signed errors, and nothing else."""
    true = [100.0] * 10
    pred = [90.0, 80.0, 95.0, 70.0, 100.0, 60.0, 85.0, 90.0, 75.0, 88.0]
    signed = [t - p for t, p in zip(true, pred)]

    corr = recal.rolling_signed_correction(_frame(true, pred), window=3)

    # day 3 (0-indexed) is the first with 3 prior days
    assert corr.iloc[3] == pytest.approx(np.mean(signed[0:3]))
    assert corr.iloc[5] == pytest.approx(np.mean(signed[2:5]))
    assert corr.iloc[9] == pytest.approx(np.mean(signed[6:9]))
    # cold-start days carry no correction at all, rather than a filled zero
    assert corr.iloc[:3].isna().all()


def test_a_constant_bias_is_fully_removed():
    """The hypothesis in miniature: a pure level shift should recalibrate to
    (near) zero error once the window is populated."""
    true = [100.0] * 30
    pred = [70.0] * 30                       # constant -30 underforecast
    out = recal.recalibrate(_frame(true, pred), window=7, cold_start="exclude")
    assert np.abs(out["y_true"] - out["y_pred"]).max() == pytest.approx(0.0, abs=1e-9)


def test_correction_is_applied_additively_to_every_hour_of_the_day():
    true = [100.0] * 10
    pred = [90.0] * 10
    raw = _frame(true, pred)
    out = recal.recalibrate(raw, window=3, cold_start="exclude")
    day = out["origin"].iloc[0]
    assert out[out["origin"] == day]["y_pred"].nunique() == 1
    assert out[out["origin"] == day]["y_pred"].iloc[0] == pytest.approx(100.0)


# --------------------------------------------------------------------------
# cold start — documented behaviour, tested explicitly
# --------------------------------------------------------------------------
def test_exclude_drops_exactly_the_first_window_days():
    out = recal.recalibrate(_frame([100.0] * 30, [70.0] * 30), window=7, cold_start="exclude")
    assert out["origin"].nunique() == 30 - 7


def test_expanding_drops_only_the_very_first_day():
    """Expanding uses every prior day available. Day 0 still has no prior
    day, so it can never be corrected — it is dropped, not filled."""
    out = recal.recalibrate(_frame([100.0] * 30, [70.0] * 30), window=7, cold_start="expanding")
    assert out["origin"].nunique() == 29


def test_expanding_uses_partial_history_during_the_cold_start_only():
    """'expanding' is a cold-start policy, not a different estimator.

    Early days average whatever history exists; once `window` days are
    available it must behave exactly like 'exclude'. Expanding over the
    whole series instead would ignore --window altogether and make every
    window in a sweep return identical numbers — which is how this was
    caught.
    """
    true = [100.0] * 10
    pred = [50.0] + [90.0] * 9   # a large day-0 error a 3-day window must forget
    signed = [t - p for t, p in zip(true, pred)]
    corr = recal.rolling_signed_correction(_frame(true, pred), window=3, cold_start="expanding")

    # cold start: day 1 sees only day 0, day 2 sees days 0-1
    assert corr.iloc[1] == pytest.approx(signed[0])
    assert corr.iloc[2] == pytest.approx(np.mean(signed[0:2]))
    # past the cold start: a strict 3-day window that has forgotten day 0
    assert corr.iloc[6] == pytest.approx(np.mean(signed[3:6]))


def test_expanding_and_exclude_agree_once_the_window_is_full():
    true = [100.0, 120.0, 90.0, 105.0, 95.0, 130.0, 85.0, 110.0, 100.0, 115.0]
    pred = [90.0, 80.0, 95.0, 70.0, 100.0, 60.0, 85.0, 90.0, 75.0, 88.0]
    a = recal.rolling_signed_correction(_frame(true, pred), window=3, cold_start="exclude")
    b = recal.rolling_signed_correction(_frame(true, pred), window=3, cold_start="expanding")
    pd.testing.assert_series_equal(a.iloc[3:], b.iloc[3:])


def test_cold_start_days_are_never_silently_filled():
    """The one behaviour explicitly forbidden by the task: a cold-start day
    must not appear in the output with a zero (or any) correction."""
    raw = _frame([100.0] * 30, [70.0] * 30)
    out = recal.recalibrate(raw, window=7, cold_start="exclude")
    dropped = set(raw["origin"].unique()) - set(out["origin"].unique())
    assert len(dropped) == 7
    assert dropped == set(sorted(raw["origin"].unique())[:7])


def test_unknown_cold_start_mode_is_refused():
    with pytest.raises(ValueError, match="cold_start"):
        recal.recalibrate(_frame([100.0] * 10, [90.0] * 10), window=3, cold_start="fill")


# --------------------------------------------------------------------------
# metrics — must be the repo's own implementation, not a reimplementation
# --------------------------------------------------------------------------
def test_metrics_match_src_evaluation_on_a_known_case():
    rng = np.random.default_rng(42)
    n_days = 40
    true = rng.normal(100, 20, n_days)
    pred = true + rng.normal(-5, 3, n_days)
    frame = _frame(true, pred)

    got = recal.metrics(frame)

    ts = pd.DatetimeIndex(frame["origin"] + pd.to_timedelta(frame["hour"], unit="h"))
    real = pd.Series(frame["y_true"].values, index=ts).sort_index().to_frame("price")
    prd = pd.Series(frame["y_pred"].values, index=ts).sort_index().to_frame("price")

    assert got["MAE"] == pytest.approx(mae(real.values, prd.values))
    assert got["rMAE"] == pytest.approx(rmae(real, prd, m="W"))


def test_raw_metrics_reproduce_the_frozen_ood_summary():
    """Regression guard against the published v1.1-ood numbers.

    If this pipeline's raw (uncorrected) metrics over all 173 days do not
    reproduce ood_summary.csv, then the loading or metric path differs from
    the one that produced the tagged result, and any recalibrated number
    built on it would be measuring something else.
    """
    summary = pd.read_csv(recal.OOD_DIR / "ood_summary.csv", index_col=0)
    for model, filename in recal.OOD_FILES.items():
        frame = recal.load_ood_frame(recal.OOD_DIR / filename)
        got = recal.metrics(frame)
        assert got["MAE"] == pytest.approx(float(summary.loc[model, "MAE"]), rel=1e-9)
        assert got["rMAE"] == pytest.approx(float(summary.loc[model, "rMAE"]), rel=1e-9)


def test_raw_and_recalibrated_are_scored_on_the_identical_day_subset():
    """The comparison is only meaningful if both arms cover the same days.

    Scoring raw over 173 days and recalibrated over 159 would let the
    cold-start exclusion itself move the headline number — an easy way to
    manufacture an improvement that is really a change of sample.
    """
    row = recal.compare_one(
        recal.load_ood_frame(recal.OOD_DIR / recal.OOD_FILES["LEAR-LASSO"]),
        window=14,
        cold_start="exclude",
    )
    assert row["n days"] == 173 - 14
    assert row["n_days_raw"] == row["n days"]


def test_no_column_name_contains_a_latex_breaking_underscore():
    """A bare '_' in a column name is a LaTeX error in text mode — the same
    trap export_tables.py guards against in its 'days' column."""
    table, _ = recal.sweep([7], cold_start="exclude")
    exported = table.drop(columns=["n_days_raw"])
    assert not [c for c in exported.columns if "_" in str(c)]
    assert not [n for n in exported.index.names if "_" in str(n)]


def test_naive_is_excluded_from_the_crossing_count():
    """rMAE is normalised BY naive, so counting naive among rows that 'beat
    naive' would be circular — and here also false, since correcting naive
    makes it worse."""
    table, _ = recal.sweep([7, 14], cold_start="exclude")
    assert "naive" not in recal.crossings(table).index.get_level_values("model")
    naive = table.loc["naive"]
    assert (naive["rMAE recal"] > naive["rMAE raw"]).all()


# --------------------------------------------------------------------------
# integrity — the frozen tags must be untouched by a full run
# --------------------------------------------------------------------------
def test_full_run_leaves_frozen_artifacts_byte_identical(tmp_path, monkeypatch):
    frozen_paths = list((REPO_ROOT / "data" / "processed" / "ood").glob("*.csv"))
    frozen_paths += list((REPO_ROOT / "reports" / "tables").glob("*"))
    frozen_paths += list((REPO_ROOT / "models" / "frozen").glob("*"))
    before = _hash_tree(frozen_paths)
    assert before, "expected frozen artifacts to hash — check the paths"

    monkeypatch.setattr(recal, "OUT_DIR", tmp_path / "ood_recalibrated")
    monkeypatch.setattr(recal, "TABLES_DIR", tmp_path / "tables")
    recal.main(windows=[7], cold_start="exclude")

    assert _hash_tree(frozen_paths) == before


def test_output_goes_to_the_new_namespace_only(tmp_path, monkeypatch):
    monkeypatch.setattr(recal, "OUT_DIR", tmp_path / "ood_recalibrated")
    monkeypatch.setattr(recal, "TABLES_DIR", tmp_path / "tables")
    recal.main(windows=[7], cold_start="exclude")

    assert (tmp_path / "tables" / "ood_recalibration.csv").exists()
    assert (tmp_path / "tables" / "ood_recalibration.tex").exists()
    # the frozen table names must never be produced by this script
    assert not (tmp_path / "tables" / "ood_stress.csv").exists()
    assert not (tmp_path / "tables" / "results_canonical.csv").exists()
