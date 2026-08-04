"""Guard tests for the regime threshold's provenance and config contract.

The stress threshold is the single scalar the whole regime mechanism rests
on, and its train-only property was previously asserted only in a YAML
comment. After the v1.0-results freeze the number cannot be changed, so the
claim is checked here instead of trusted (leakage review, 2026-08-04).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DE = REPO_ROOT / "data" / "raw" / "DE.csv"
EVAL_CFG = REPO_ROOT / "configs" / "evaluation.yaml"

# Strictly before BOTH the Optuna tuning window (opens 2015-01-05) and the
# ensemble weight-fitting window (opens 2015-01-12). See the config comment.
TRAIN_CUTOFF = "2015-01-04"
K = 1.5


def _cfg() -> dict:
    with open(EVAL_CFG) as f:
        return yaml.safe_load(f)


def _require_raw_data() -> None:
    """Skip when the benchmark CSV is absent, but FAIL when the environment
    declares that full data should be present.

    data/raw/ is gitignored, so on a fresh clone these tests would skip —
    and the one test that actually checks the leakage property would report
    green while checking nothing. Setting THESIS_FULL_DATA=1 (CI, or before
    tagging v1.0-results) turns that silence into a failure.
    """
    if RAW_DE.exists():
        return
    if os.environ.get("THESIS_FULL_DATA") == "1":
        pytest.fail(
            f"{RAW_DE} missing while THESIS_FULL_DATA=1: the regime-threshold "
            "provenance check cannot run, and must not pass silently"
        )
    pytest.skip("benchmark CSV not cached locally (set THESIS_FULL_DATA=1 to require it)")


def test_stress_threshold_matches_train_only_statistics():
    """The configured threshold must equal train mean + 1.5*std computed on
    data strictly before every window that could contaminate it.

    A drifting threshold, or one silently recomputed over a window that
    reaches into validation/test, is a leak that no other test would catch.
    """
    _require_raw_data()

    prices = pd.read_csv(RAW_DE, index_col=0, parse_dates=True).iloc[:, 0]
    train = prices.loc[:TRAIN_CUTOFF]
    assert not train.empty, "train slice is empty -- wrong column or index"

    expected = train.mean() + K * train.std()
    actual = float(_cfg()["regime"]["stress_threshold_eur_mwh"])
    assert np.isclose(actual, expected, atol=0.01), (
        f"configured threshold {actual} != train mean + {K}*std = {expected:.4f} "
        f"on data <= {TRAIN_CUTOFF}. Either the config drifted or the slice "
        "changed; both break the train-only provenance claim."
    )


def test_train_cutoff_precedes_the_derived_tuning_window():
    """The cutoff must precede the tuning window as the CONFIG derives it.

    Comparing two hardcoded dates would be tautological: it could never
    fail, yet the property it claims depends on
    walk_forward.calibration_window_days and the data start. Shortening the
    calibration window would open the tuning window before the cutoff while
    a hardcoded test stayed green. So derive the window start here.
    """
    _require_raw_data()

    prices = pd.read_csv(RAW_DE, index_col=0, parse_dates=True).iloc[:, 0]
    data_start = prices.index.min().normalize()
    calib_days = int(_cfg()["walk_forward"]["calibration_window_days"])

    # First origin that has a full trailing calibration window behind it.
    first_tuning_origin = data_start + pd.Timedelta(days=calib_days)
    assert pd.Timestamp(TRAIN_CUTOFF) < first_tuning_origin, (
        f"threshold cutoff {TRAIN_CUTOFF} does not precede the first tuning "
        f"origin {first_tuning_origin.date()} implied by "
        f"calibration_window_days={calib_days} and data start "
        f"{data_start.date()} -- the train-only claim would be false"
    )


def test_train_cutoff_precedes_the_weight_fitting_window():
    """The cutoff must precede the earliest ensemble weight-fitting origin,
    read from the committed validation predictions rather than hardcoded."""
    val_dir = REPO_ROOT / "data" / "processed" / "validation_preds"
    frames = sorted(val_dir.glob("*.csv"))
    if not frames:
        pytest.skip("validation predictions not present")

    earliest = min(
        pd.read_csv(f, usecols=["origin"], parse_dates=["origin"])["origin"].min()
        for f in frames
    )
    assert pd.Timestamp(TRAIN_CUTOFF) < earliest, (
        f"threshold cutoff {TRAIN_CUTOFF} does not precede the first "
        f"weight-fitting origin {earliest.date()}"
    )


def test_config_has_no_legacy_spike_key():
    """The pre-2026-08-04 key must be gone, not merely shadowed.

    combine_regime_aware() rejects a legacy 'spike' weights dict, but nothing
    rejected a legacy config key -- a half-migrated config would fail at
    runtime in a long walk-forward rather than here in seconds.
    """
    regime = _cfg()["regime"]
    assert "stress_threshold_eur_mwh" in regime
    assert "spike_threshold_eur_mwh" not in regime, (
        "configs/evaluation.yaml still carries the legacy spike key; "
        "the rename to stress_threshold_eur_mwh is incomplete"
    )
