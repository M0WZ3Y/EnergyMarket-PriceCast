"""Tests for scripts/run_ood_stress.py — the OOD stress test harness.

The OOD result itself needs live data and a network, so it cannot be
asserted here. What IS tested is the machinery that decides whether an OOD
run is VALID — the guards that stop a meaningless run from being reported
as a result, which is the part that would quietly corrupt chapter 4.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_ood_stress", REPO_ROOT / "scripts" / "run_ood_stress.py"
)
ood = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ood)


def _write_cache(path: Path, start: str, days: int = 30) -> Path:
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    pd.DataFrame(
        {"price": 50.0, "exog_1": 40000.0, "exog_2": 15000.0}, index=idx
    ).rename_axis("timestamp").to_csv(path)
    return path


def _write_frozen_meta(tmp_path: Path, frozen_on: str) -> Path:
    frozen = tmp_path / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "metadata.json").write_text(
        json.dumps(
            {
                "frozen_on": frozen_on,
                "train_start": "2015-01-05",
                "calibration_window_days": 1092,
                "n_train_days": 1092,
                "benchmark_price_mean": 34.69,
                "benchmark_price_std": 16.7,
            }
        )
    )
    return frozen


def test_replay_refuses_cache_overlapping_the_training_window(tmp_path, monkeypatch):
    """A window overlapping the freeze date scores models on their own
    training data and would report flattering numbers under an OOD heading.

    This is the guard that matters most: such a run fails silently — it
    produces plausible metrics — so nothing downstream would catch it.
    """
    monkeypatch.setattr(ood, "FROZEN_DIR", _write_frozen_meta(tmp_path, "2017-12-31"))
    cache = _write_cache(tmp_path / "live.csv", "2017-06-01")

    with pytest.raises(SystemExit, match="overlaps the training window|on or before"):
        ood.replay(cache=cache)


def test_replay_accepts_cache_strictly_after_the_freeze(tmp_path, monkeypatch):
    """The mirror case: a window after the freeze must get past the overlap
    guard. It fails later for an unrelated reason (no frozen model files),
    which is fine — this pins the guard boundary, not the whole run.

    Asserted POSITIVELY on where the run got to, not on the absence of two
    magic substrings: a wording change in the guard would silently turn a
    negative-substring assertion into a test of nothing (while its sibling
    above, which uses the same strings in `match=`, breaks loudly). The
    positive identification is the exception TYPE plus the cause chain —
    replay() re-raises the missing-artifact FileNotFoundError as SystemExit
    with `from exc`, and that only happens downstream of the overlap guard,
    of the metadata check, and of feature construction succeeding.
    """
    monkeypatch.setattr(ood, "FROZEN_DIR", _write_frozen_meta(tmp_path, "2017-12-31"))
    cache = _write_cache(tmp_path / "live.csv", "2026-01-01")

    with pytest.raises(SystemExit) as exc:
        ood.replay(cache=cache)

    # the run reached the model-loading stage: it died on a real missing
    # artifact, chained from the underlying filesystem error
    assert isinstance(exc.value.__cause__, (FileNotFoundError, OSError))
    message = str(exc.value)
    assert "Re-run with --fit to rebuild the frozen models." in message
    # ...and specifically on the first frozen model the loop reaches
    assert "frozen naive missing or unreadable" in message
    # the two guards that must NOT have fired (kept from the original test)
    assert "overlaps the training window" not in message
    assert "on or before" not in message
    assert "no complete days" not in message


def test_replay_requires_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ood, "FROZEN_DIR", _write_frozen_meta(tmp_path, "2017-12-31"))
    with pytest.raises(SystemExit, match="--fetch"):
        ood.replay(cache=tmp_path / "does_not_exist.csv")


def test_replay_requires_frozen_models(tmp_path, monkeypatch):
    monkeypatch.setattr(ood, "FROZEN_DIR", tmp_path / "empty")
    cache = _write_cache(tmp_path / "live.csv", "2026-01-01")
    with pytest.raises(SystemExit, match="--fit"):
        ood.replay(cache=cache)


def test_replay_rejects_a_window_too_short_for_lag_features(tmp_path, monkeypatch):
    """Fewer days than the longest lag leaves no complete rows. Without this
    guard the run would report metrics over zero days."""
    monkeypatch.setattr(ood, "FROZEN_DIR", _write_frozen_meta(tmp_path, "2017-12-31"))
    cache = _write_cache(tmp_path / "live.csv", "2026-01-01", days=3)
    with pytest.raises(SystemExit, match="no complete days|longer range"):
        ood.replay(cache=cache)


@pytest.mark.network
def test_fetch_live_hits_the_api(tmp_path, monkeypatch):
    """Network-marked: excluded from the offline suite by design."""
    monkeypatch.setattr(ood, "LIVE_CACHE", tmp_path / "live.csv")
    ood.fetch_live("2026-06-01", "2026-06-08")
    df = pd.read_csv(tmp_path / "live.csv", index_col=0, parse_dates=True)
    assert {"price", "exog_1", "exog_2"} <= set(df.columns)
    assert len(df) > 0
