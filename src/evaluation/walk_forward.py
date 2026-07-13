"""Rolling-origin (walk-forward) validation harness — src/evaluation/

Implements the daily-recalibration protocol from Lago et al. (2021) /
epftoolbox (epftoolbox.models._lear.LEAR.recalibrate_and_forecast_next_day):
each forecast origin trains on a fixed-size trailing calibration window,
forecasts exactly one day ahead, then the origin advances by step_days.
Only rolling-origin splits are produced here — random/shuffled splits are
never valid for this project (CLAUDE.md).

Operates on the daily calendar produced by src/features/pipeline.py's
build_features() (X, Y indexed by target_day), not on raw hourly data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation.yaml"


def load_evaluation_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class WalkForwardSplit(NamedTuple):
    origin: pd.Timestamp
    train_days: pd.DatetimeIndex
    test_days: pd.DatetimeIndex


def walk_forward_splits(
    days: pd.DatetimeIndex,
    cfg: dict | None = None,
    first_origin: pd.Timestamp | None = None,
) -> Iterator[WalkForwardSplit]:
    """Yield rolling-origin splits over a sorted daily calendar.

    Each split's train_days is exactly the `calibration_window_days` days
    immediately preceding the origin (never the origin itself or any later
    day); test_days is the origin day alone. The origin then advances by
    `step_days`. `first_origin` restricts which days are used as origins
    (e.g. the nominal test period) while training history is still drawn
    from everything before each origin, matching epftoolbox's LEAR
    recalibration loop, which recalibrates on trailing history regardless
    of the nominal train/test label of that history.
    """
    if cfg is None:
        cfg = load_evaluation_config()
    wf_cfg = cfg["walk_forward"]
    window = wf_cfg["calibration_window_days"]
    step = wf_cfg["step_days"]

    days = pd.DatetimeIndex(days).sort_values()
    n = len(days)

    start_pos = window
    if first_origin is not None:
        origin_positions = [i for i in range(n) if days[i] >= first_origin]
        if not origin_positions:
            return
        start_pos = max(start_pos, origin_positions[0])

    for pos in range(start_pos, n, step):
        origin = days[pos]
        train_days = days[pos - window : pos]
        test_days = days[pos : pos + 1]
        assert train_days.max() < origin, "train window must end strictly before the origin"
        yield WalkForwardSplit(origin=origin, train_days=train_days, test_days=test_days)


def assert_validation_before_test(
    validation_days: pd.DatetimeIndex, test_days: pd.DatetimeIndex
) -> None:
    """Enforce the project's non-negotiable ordering rule: the validation
    window (used for Optuna tuning) must sit strictly before the test
    window (CLAUDE.md: "validation window strictly before test window").
    Raises AssertionError if the ordering is violated, or ValueError if
    either window is empty — an empty window can't be verified as
    correctly ordered, so silently passing would defeat the whole point
    of this check (it exists to catch upstream slicing bugs).
    """
    validation_days = pd.DatetimeIndex(validation_days)
    test_days = pd.DatetimeIndex(test_days)
    if len(validation_days) == 0 or len(test_days) == 0:
        raise ValueError("validation_days and test_days must both be non-empty")
    assert validation_days.max() < test_days.min(), (
        "validation window must end strictly before the test window starts: "
        f"validation ends {validation_days.max()}, test starts {test_days.min()}"
    )


def carve_validation_from_train(
    train_days: pd.DatetimeIndex, cfg: dict | None = None
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a train calendar into (fit_days, validation_days).

    validation_days is the trailing `validation.validation_days` days of
    train, used for Optuna hyperparameter search. Callers must additionally
    verify the validation window ends before the real test period via
    assert_validation_before_test(validation_days, test_days) — this
    function only guarantees validation sits at the end of train, not that
    train/test themselves are correctly ordered.
    """
    if cfg is None:
        cfg = load_evaluation_config()
    val_n = cfg["validation"]["validation_days"]
    train_days = pd.DatetimeIndex(train_days).sort_values()
    if val_n <= 0:
        # train_days[:-0] is train_days[:0] (empty) and train_days[-0:] is
        # all of train_days -- the classic Python negative-zero slicing
        # trap would silently invert fit/validation instead of raising.
        raise ValueError("validation_days must be a positive integer")
    if val_n >= len(train_days):
        raise ValueError("validation_days must be smaller than the train period")
    fit_days = train_days[:-val_n]
    validation_days = train_days[-val_n:]
    return fit_days, validation_days
