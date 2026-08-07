"""Tests for scripts/run_seed_ensemble.py — the LSTM seed ensemble.

The claim this work supports is fragile (p = 0.080 against a 0.05
threshold), so the tests pin the things that would make it fragile in the
wrong way: the seed policy, the member set actually being averaged, and the
fact that weights are fitted on validation rather than on test.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_seed_ensemble", REPO_ROOT / "scripts" / "run_seed_ensemble.py"
)
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

from src.models import LSTMModel, load_models_config  # noqa: E402


# --------------------------------------------------------------------------
# seed policy: 42 stays the default
# --------------------------------------------------------------------------
def test_lstm_default_seed_is_still_42():
    """The project rule is seed 42 everywhere. Making the seed configurable
    for ensembling must not have changed what a normal run does."""
    assert LSTMModel(load_models_config()["lstm"]).seed == 42


def test_lstm_seed_is_overridable():
    assert LSTMModel({**load_models_config()["lstm"], "seed": 43}).seed == 43


def test_baseline_seed_constant_matches_the_project_rule():
    assert se.BASELINE_SEED == 42


# --------------------------------------------------------------------------
# member selection: exactly the seed runs, nothing else
# --------------------------------------------------------------------------
def _seed_file(directory: Path, seed: int, value: float, n_days: int = 3) -> Path:
    days = pd.date_range("2016-01-04", periods=n_days, freq="D")
    rows = pd.DataFrame(
        {
            "origin": np.repeat(days, 24),
            "hour": list(range(24)) * n_days,
            "y_true": 100.0,
            "y_pred": value,
            "model": f"LSTM-s{seed}",
        }
    )
    path = directory / f"lstm_s{seed}.csv"
    rows.to_csv(path, index=False)
    return path


def test_seed_ensemble_averages_exactly_the_seed_runs(tmp_path):
    _seed_file(tmp_path, 43, 10.0)
    _seed_file(tmp_path, 44, 20.0)
    _seed_file(tmp_path, 45, 30.0)
    frame, n = se.seed_ensemble_frame(tmp_path)
    assert n == 3
    assert frame["y_pred"].unique() == pytest.approx([20.0])


def test_the_averaged_output_file_is_not_folded_back_in_as_a_member(tmp_path):
    """Regression: a bare lstm_s*.csv glob also matches the module's own
    lstm_seed_ensemble.csv output, which sits in the same directory. Folding
    the average back in as a member over-weights it. It happened to cancel
    when the file was exactly the mean of the others, but a STALE file (from
    a different seed count) would have silently skewed the result."""
    _seed_file(tmp_path, 43, 10.0)
    _seed_file(tmp_path, 44, 20.0)
    # a deliberately stale "average" that is not the mean of the two above
    stale = _seed_file(tmp_path, 99, 999.0)
    stale.rename(tmp_path / "lstm_seed_ensemble.csv")

    frame, n = se.seed_ensemble_frame(tmp_path)
    assert n == 2, "lstm_seed_ensemble.csv must not be counted as a seed run"
    assert frame["y_pred"].unique() == pytest.approx([15.0])


def test_misaligned_seed_runs_are_refused(tmp_path):
    _seed_file(tmp_path, 43, 10.0, n_days=3)
    _seed_file(tmp_path, 44, 20.0, n_days=2)
    with pytest.raises(SystemExit, match="identical"):
        se.seed_ensemble_frame(tmp_path)


def test_a_single_run_is_not_an_ensemble(tmp_path):
    _seed_file(tmp_path, 43, 10.0)
    with pytest.raises(SystemExit, match="need >=2"):
        se.seed_ensemble_frame(tmp_path)


# --------------------------------------------------------------------------
# the produced artifacts
# --------------------------------------------------------------------------
VAL_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble_val"
TEST_DIR = REPO_ROOT / "data" / "processed" / "seed_ensemble"

pytestmark_needs_runs = pytest.mark.skipif(
    not (VAL_DIR.exists() and TEST_DIR.exists()),
    reason="seed-ensemble runs not present (regenerable via run_seed_ensemble.py)",
)


@pytestmark_needs_runs
def test_test_window_ensemble_has_four_members_covering_728_origins():
    frame, n = se.seed_ensemble_frame(TEST_DIR, extra=se.FROZEN_LSTM)
    assert n == 4, "expected seeds 43/44/45 plus the frozen seed-42 run"
    counts = frame.groupby("origin").size()
    assert len(counts) == 728
    assert (counts == 24).all()


@pytestmark_needs_runs
def test_validation_window_ensemble_covers_the_weight_fitting_window():
    """Weights must be fitted on validation. That requires the seed ensemble
    to exist there, over the same 357 origins the frozen weight fit used."""
    frame, n = se.seed_ensemble_frame(VAL_DIR)
    assert n == 4
    origins = pd.DatetimeIndex(sorted(frame["origin"].unique()))
    assert len(origins) == 357
    assert origins.min() == pd.Timestamp("2015-01-12")
    assert origins.max() == pd.Timestamp("2016-01-03")


@pytestmark_needs_runs
def test_validation_and_test_windows_do_not_overlap():
    """The whole legitimacy of the result rests on this."""
    val, _ = se.seed_ensemble_frame(VAL_DIR)
    test, _ = se.seed_ensemble_frame(TEST_DIR, extra=se.FROZEN_LSTM)
    assert val["origin"].max() < test["origin"].min()


@pytestmark_needs_runs
def test_seed_ensemble_beats_every_individual_seed():
    """Variance reduction is the entire mechanism; if the average were not
    better than its members, there would be nothing here."""
    frame, _ = se.seed_ensemble_frame(TEST_DIR, extra=se.FROZEN_LSTM)
    truth = frame.pivot(index="origin", columns="hour", values="y_true").sort_index().to_numpy()
    ens = frame.pivot(index="origin", columns="hour", values="y_pred").sort_index().to_numpy()
    ens_mae = float(np.abs(truth - ens).mean())

    # Same digit-only filter the module uses. Globbing "lstm_s*.csv" here
    # would pick up lstm_seed_ensemble.csv and compare the ensemble against
    # itself -- this test failed that way first time, which is the bug class
    # it exists to guard, reappearing in the guard.
    paths = [
        p for p in sorted(TEST_DIR.glob("lstm_s*.csv")) if re.fullmatch(r"lstm_s\d+", p.stem)
    ] + [se.FROZEN_LSTM]
    assert len(paths) == 4
    for path in paths:
        m = se._long(path)
        p = m.pivot(index="origin", columns="hour", values="y_pred").sort_index().to_numpy()
        assert ens_mae < float(np.abs(truth - p).mean()), f"{path.name} beats the ensemble"
