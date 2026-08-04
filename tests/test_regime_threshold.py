"""Guard tests for the regime threshold's provenance and config contract.

The stress threshold is the single scalar the whole regime mechanism rests
on, and its train-only property was previously asserted only in a YAML
comment. After the v1.0-results freeze the number cannot be changed, so the
claim is checked here instead of trusted (leakage review, 2026-08-04).
"""

from __future__ import annotations

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


def test_stress_threshold_matches_train_only_statistics():
    """The configured threshold must equal train mean + 1.5*std computed on
    data strictly before every window that could contaminate it.

    A drifting threshold, or one silently recomputed over a window that
    reaches into validation/test, is a leak that no other test would catch.
    """
    if not RAW_DE.exists():
        pytest.skip("benchmark CSV not cached locally")

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


def test_train_cutoff_precedes_every_downstream_window():
    """Pin the ordering the threshold's provenance depends on."""
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    optuna_window_start = pd.Timestamp("2015-01-05")
    weight_fitting_start = pd.Timestamp("2015-01-12")
    assert cutoff < optuna_window_start
    assert cutoff < weight_fitting_start


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
