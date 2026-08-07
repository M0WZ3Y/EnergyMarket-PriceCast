"""Tests for src/ledger_gate.py — the writing-progress gate.

The gate's whole value is that it is hard to ignore, so the tests pin the
two things that would quietly destroy that: that a blocking condition
really exits non-zero (not warns), and that the bypass really leaves a
trace (not just a flag nobody sees).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from src import ledger_gate


def _ledger(tmp_path: Path, rows: list[tuple[str, float, str]]) -> Path:
    path = tmp_path / "page_ledger.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "pages_banked", "note"])
        for row in rows:
            w.writerow(row)
    return path


NOW = datetime(2026, 8, 7, 12, 0)


# --------------------------------------------------------------------------
# blocking conditions
# --------------------------------------------------------------------------
def test_blocks_when_the_latest_entry_is_stale(tmp_path):
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "3-5"), ("2026-08-04", 8.0, "3-6")])
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert blocked
    assert "hours old" in why


def test_blocks_when_the_latest_entry_did_not_move_forward(tmp_path):
    path = _ledger(tmp_path, [("2026-08-06", 8.0, "3-6"), ("2026-08-07", 8.0, "no progress")])
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert blocked
    assert "no forward movement" in why


def test_blocks_on_an_empty_or_missing_ledger(tmp_path):
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(tmp_path / "nope.csv"), now=NOW)
    assert blocked
    assert "empty or missing" in why


def test_the_repo_ledger_today_is_blocking():
    """The state that motivated the gate must actually trip it.

    A gate that would have let 2026-08-07's ledger through would not have
    solved the problem it was built for.
    """
    blocked, why = ledger_gate.evaluate(
        ledger_gate.read_ledger(ledger_gate.DEFAULT_LEDGER), now=NOW
    )
    assert blocked, f"expected the repo ledger to block on {NOW.date()}, got: {why}"


# --------------------------------------------------------------------------
# allowing conditions
# --------------------------------------------------------------------------
def test_allows_a_fresh_progressing_ledger(tmp_path):
    path = _ledger(tmp_path, [("2026-08-06", 8.0, "3-6"), ("2026-08-07", 11.5, "3-7-1")])
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked, why


def test_a_lone_fresh_entry_is_allowed(tmp_path):
    """Documented choice: with one entry there is no previous count to
    compare, so only staleness applies. Treating it as 'no progress' would
    block every new ledger forever."""
    path = _ledger(tmp_path, [("2026-08-07", 3.0, "first")])
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked, why


def test_exactly_48_hours_is_still_allowed(tmp_path):
    """Boundary: the rule is 'more than 48 hours', so exactly 48 passes."""
    path = _ledger(tmp_path, [("2026-08-04", 1.0, "a"), ("2026-08-05", 2.0, "b")])
    blocked, _ = ledger_gate.evaluate(
        ledger_gate.read_ledger(path), now=datetime(2026, 8, 7, 0, 0)
    )
    assert not blocked


def test_a_page_count_that_goes_down_counts_as_movement(tmp_path):
    """A cut section is real forward work on the thesis, and the ledger is
    explicitly allowed to decrease (page_quota.pages_banked takes the latest
    row, not the maximum). Only an UNCHANGED count means nothing happened."""
    path = _ledger(tmp_path, [("2026-08-06", 8.0, "3-6"), ("2026-08-07", 6.0, "cut 3-5")])
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked, why


# --------------------------------------------------------------------------
# the enforcement itself: exit code, not a warning
# --------------------------------------------------------------------------
def test_require_exits_nonzero_when_blocked(tmp_path):
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "old")])
    with pytest.raises(SystemExit) as exc:
        ledger_gate.require_ledger_progress(
            "scripts/run_something.py", ledger_path=path, now=NOW, env={},
            decisions_path=tmp_path / "decisions.md",
        )
    assert exc.value.code == 1


def test_require_returns_quietly_when_allowed(tmp_path):
    path = _ledger(tmp_path, [("2026-08-06", 8.0, "a"), ("2026-08-07", 11.0, "b")])
    ledger_gate.require_ledger_progress(
        "scripts/run_something.py", ledger_path=path, now=NOW, env={},
        decisions_path=tmp_path / "decisions.md",
    )  # must not raise


def test_blocked_message_names_the_script_and_the_escape_hatch(tmp_path, capsys):
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "old")])
    with pytest.raises(SystemExit):
        ledger_gate.require_ledger_progress(
            "scripts/run_ood_recalibration.py", ledger_path=path, now=NOW, env={},
            decisions_path=tmp_path / "decisions.md",
        )
    err = capsys.readouterr().err
    assert "run_ood_recalibration.py" in err
    assert ledger_gate.BYPASS_ENV in err
    assert "page_quota.py --add" in err


# --------------------------------------------------------------------------
# bypass: works, and is never free
# --------------------------------------------------------------------------
def test_bypass_allows_a_blocked_run_and_writes_a_trace(tmp_path):
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "old")])
    decisions = tmp_path / "decisions.md"
    decisions.write_text("# existing content\n", encoding="utf-8")

    ledger_gate.require_ledger_progress(
        "scripts/run_ood_recalibration.py",
        ledger_path=path,
        now=NOW,
        env={ledger_gate.BYPASS_ENV: "1"},
        decisions_path=decisions,
    )  # must not raise

    text = decisions.read_text(encoding="utf-8")
    assert "# existing content" in text, "the trace must append, never overwrite"
    assert "LEDGER GATE BYPASSED" in text
    assert "run_ood_recalibration.py" in text
    assert "(no reason given)" in text
    assert "2026-08-01" in text, "the trace should record the ledger state at bypass"


def test_bypass_value_is_recorded_as_the_reason(tmp_path):
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "old")])
    decisions = tmp_path / "decisions.md"

    ledger_gate.require_ledger_progress(
        "scripts/run_shap.py",
        ledger_path=path,
        now=NOW,
        env={ledger_gate.BYPASS_ENV: "supervisor asked for the SHAP figure today"},
        decisions_path=decisions,
    )

    text = decisions.read_text(encoding="utf-8")
    assert "supervisor asked for the SHAP figure today" in text
    assert "(no reason given)" not in text


def test_empty_bypass_value_does_not_bypass(tmp_path):
    """An unset-but-present variable must not silently disable the gate."""
    path = _ledger(tmp_path, [("2026-08-01", 4.0, "old")])
    with pytest.raises(SystemExit):
        ledger_gate.require_ledger_progress(
            "scripts/run_something.py",
            ledger_path=path,
            now=NOW,
            env={ledger_gate.BYPASS_ENV: "   "},
            decisions_path=tmp_path / "decisions.md",
        )


def test_bypass_on_an_already_passing_ledger_still_logs(tmp_path):
    """Setting the variable when it was not needed still leaves a trace —
    otherwise it could be left permanently exported with no record."""
    path = _ledger(tmp_path, [("2026-08-06", 8.0, "a"), ("2026-08-07", 11.0, "b")])
    decisions = tmp_path / "decisions.md"
    ledger_gate.require_ledger_progress(
        "scripts/run_ensemble.py", ledger_path=path, now=NOW,
        env={ledger_gate.BYPASS_ENV: "1"}, decisions_path=decisions,
    )
    assert "LEDGER GATE BYPASSED" in decisions.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# wiring: the gate must be on the right scripts, and only at __main__
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
GATED = [
    "run_full_baselines.py", "run_daily_direct.py", "run_ensemble.py",
    "run_dm_ensembles.py", "run_shap.py", "run_ood_stress.py",
    "run_ood_recalibration.py", "tune_lightgbm.py", "tune_lstm.py",
    "tune_daily.py", "export_tables.py", "week5_checkpoint.py",
]
UNGATED = ["page_quota.py", "task_monitor.py", "verify_dataset.py",
           "smoke_test_energycharts.py"]


@pytest.mark.parametrize("name", GATED)
def test_output_producing_scripts_are_gated(name):
    text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "require_ledger_progress" in text, f"{name} produces output but is not gated"


@pytest.mark.parametrize("name", UNGATED)
def test_readonly_and_reporting_scripts_are_not_gated(name):
    text = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "require_ledger_progress" not in text, f"{name} is read-only and must not be gated"


@pytest.mark.parametrize("name", GATED)
def test_the_gate_is_called_only_under_main(name):
    """Critical: a module-level call would fire on IMPORT, and the test suite
    imports several of these modules (tests/test_ood_stress.py loads
    run_ood_stress.py via importlib). That would gate pytest itself."""
    lines = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8").splitlines()
    main_guard = next(
        (i for i, ln in enumerate(lines) if ln.startswith('if __name__ == "__main__"')), None
    )
    assert main_guard is not None, f"{name} has no __main__ guard"
    calls = [i for i, ln in enumerate(lines) if "require_ledger_progress(" in ln]
    assert calls, f"{name} never calls the gate"
    assert all(i > main_guard for i in calls), (
        f"{name} calls require_ledger_progress outside its __main__ guard — "
        "that would gate every import, including pytest's"
    )
