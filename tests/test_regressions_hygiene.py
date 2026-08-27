"""Wave F regression reproductions — script hygiene defects.

Lower severity than waves A-E: none of these produce a wrong number. They
waste a run, mislead an operator, or duplicate a source of truth that
configs/ already owns. They are in scope because the whole point of this
sweep is that "low severity" defects in an untested script layer are how
the 2026-08-02 CSV corruption happened in the first place.

Nothing here writes outside tmp_path.
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


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


task_monitor = _load_script("task_monitor")
run_full_baselines = _load_script("run_full_baselines")
run_daily_direct = _load_script("run_daily_direct")
run_dm_ensembles = _load_script("run_dm_ensembles")
run_ood_stress = _load_script("run_ood_stress")


# ==========================================================================
# scripts/task_monitor.py — resume advice that does not work
# ==========================================================================
def test_resume_commands_cover_every_walk_forward_model():
    """RESUME_COMMANDS has no 'lstm' entry.

    The LSTM walk-forward is the longest-running job in the project and the
    one most likely to be interrupted, so it is precisely the one that
    needs a resume path. Without an entry it falls through to
    "no resume command known -- skipped" and silently does not restart.
    """
    assert "lstm" in task_monitor.RESUME_COMMANDS


def test_suggested_resume_commands_are_accepted_by_the_runner():
    """The printed resume hint derives the model name from the CSV stem,
    yielding lowercase 'sarimax'/'lear_lasso'/'lightgbm'/'lstm'.

    run_full_baselines.py matches against 'SARIMAX', 'LEAR-LASSO',
    'LightGBM', 'LSTM', so the suggested command exits with
    "no matching models" and produces nothing. RESUME_COMMANDS already
    stores correctly-cased names -- the two resume paths disagree with
    each other.
    """
    accepted = set(run_full_baselines.MODEL_KEYS) if hasattr(run_full_baselines, "MODEL_KEYS") else None
    if accepted is None:  # fall back to the resume table's own values
        accepted = {
            args[1]
            for args in task_monitor.RESUME_COMMANDS.values()
            if len(args) > 1 and args[0].endswith("run_full_baselines.py")
        }

    for stem in ("sarimax", "lear_lasso", "lightgbm", "lstm"):
        suggestion = task_monitor.resume_command_for(stem)
        assert suggestion is not None, f"no resume command for {stem!r}"
        model_arg = suggestion[-1]
        assert model_arg in accepted, (
            f"resume hint suggests {model_arg!r}, which run_full_baselines.py rejects"
        )


def test_expected_origins_comes_from_config_not_a_literal():
    """EXPECTED_ORIGINS = 728 hardcodes what configs/data.yaml's test-year
    span determines. Change market or years_test and every progress bar and
    every DONE/STALLED verdict is silently wrong.
    """
    assert task_monitor.EXPECTED_ORIGINS == task_monitor.expected_origins_from_config()


# ==========================================================================
# scripts/run_dm_ensembles.py — crashes on a small regime subset
# ==========================================================================
def test_report_survives_a_regime_subset_too_small_to_bootstrap():
    """blocks = [b for b in (3,4,5,7,9,10) if b < n] is EMPTY for n <= 3, so
    min(sweep.values()) raises "min() arg is an empty sequence".

    Not hypothetical: configs/evaluation.yaml documents that at k=3.0 the
    validation window held just 3 stressed days, and the threshold is the
    documented tuning lever. Raising it enough to leave <= 3 stressed test
    days kills the script mid-report, after printing the MAE lines and
    before writing anything.
    """
    days = pd.date_range("2020-01-01", periods=3, freq="D")
    rng = np.random.default_rng(42)
    truth = pd.DataFrame(rng.normal(50, 5, size=(3, 24)), index=days, columns=range(24))
    piv = {
        "Ensemble (static)": truth + rng.normal(0, 1, size=(3, 24)),
        "Ensemble (regime-aware)": truth + rng.normal(0, 1, size=(3, 24)),
    }

    # Must not raise ValueError("min() arg is an empty sequence"); either it
    # reports with the corrections that ARE computable, or it declines with
    # an explanatory message.
    try:
        run_dm_ensembles._report(piv, truth, days=days, name="tiny subset")
    except ValueError as exc:
        assert "empty sequence" not in str(exc), (
            f"crashed on a small regime subset instead of handling it: {exc}"
        )


# ==========================================================================
# scripts/run_full_baselines.py + run_daily_direct.py — silent empty runs
# ==========================================================================
@pytest.mark.parametrize("module", [run_full_baselines, run_daily_direct])
def test_inverted_origin_range_is_rejected_not_silently_empty(module):
    """--last-origin before --first-origin filters every split away, the
    loop body never executes, and the script prints "0 of 0 origins" and
    exits 0 -- having produced nothing while reporting success.

    An aggregate exit code of 0 meaning "nothing failed" rather than
    "everything ran" already cost this project a deliverable once.
    """
    with pytest.raises((ValueError, SystemExit), match="(?i)(origin|range|before|after)"):
        module.validate_origin_range(
            first_origin=pd.Timestamp("2017-01-01"),
            last_origin=pd.Timestamp("2016-01-01"),
        )


# ==========================================================================
# scripts/run_ood_stress.py — flags accepted and ignored
# ==========================================================================
def test_inverted_fetch_range_does_not_silently_rewrite_the_cache(tmp_path, monkeypatch):
    """pd.date_range(start, end) with end < start is empty, so no fetch
    happens, the existing cache is re-read and re-written, and the script
    prints "cached N hourly rows" as though it had worked. The `if not
    parts` guard never fires because the cached frame was appended.
    """
    cache = tmp_path / "live.csv"
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    pd.DataFrame({"price": 50.0, "exog_1": 1.0, "exog_2": 2.0}, index=idx).rename_axis(
        "timestamp"
    ).to_csv(cache)
    monkeypatch.setattr(run_ood_stress, "LIVE_CACHE", cache)

    # `loader` is a local inside fetch_live(), so the class is the seam.
    called = []

    class _FakeLoader:
        attribution = "test fixture"

        def __init__(self, *a, **k):
            pass

        def fetch_exog(self, *a, **k):
            called.append((a, k))
            return pd.DataFrame()

    monkeypatch.setattr(run_ood_stress, "EnergyChartsLoader", _FakeLoader)

    with pytest.raises((ValueError, SystemExit), match="(?i)(range|before|after|start|end)"):
        run_ood_stress.fetch_live("2026-07-01", "2026-06-01")

    assert called == [], "attempted a fetch on an inverted range"


def test_fetch_live_honours_an_explicit_cache_override(tmp_path, monkeypatch):
    """--cache is documented as "override the cached live-data CSV (for
    pipeline validation)" but fetch_live() writes to the module-level
    LIVE_CACHE and main() never passes args.cache through.

    So `--fetch --cache other.csv` overwrites the committed
    data/raw/live_ood_de.csv while the user believes they redirected it --
    a non-reproducible artifact destroyed by a flag meant to protect it.
    """
    import inspect

    sig = inspect.signature(run_ood_stress.fetch_live)
    assert "cache" in sig.parameters, (
        "fetch_live() takes no cache argument, so --cache cannot be honoured"
    )


# T3: configs/evaluation.yaml is the single source for the regime threshold.
# Matches the canonical value, the superseded 2026-08-04 one, and its stale
# rounding -- any of them appearing as a literal in code is the defect.
_THRESHOLD_LITERAL = re.compile(r"62\.6989|62\.6522|62\.65(?![0-9])")


def test_the_stress_threshold_is_never_hardcoded_outside_configs():
    """The value drifted across the repo once already (CLAUDE.md said 62.65
    while configs said 62.6989). Prose can be corrected by reading; code
    cannot, so code must read configs. Figure captions that round it to
    62.70 are correct and are not scanned -- only .py files are.
    """
    here = Path(__file__).resolve()
    offenders = []
    for folder in ("src", "scripts", "tests", "app"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if path.resolve() == here:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _THRESHOLD_LITERAL.search(line):
                    offenders.append(f"  {path.relative_to(REPO_ROOT)}:{n}: {line.strip()}")
    assert not offenders, (
        "regime threshold hardcoded outside configs/evaluation.yaml:\n"
        + "\n".join(offenders)
    )
