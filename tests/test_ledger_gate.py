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

# Design B fixtures. Quota 3 pages/week accruing from the FIRST ledger row;
# blocks at 2 weeks of debt (6 pages), bypass disabled at 3 weeks (9 pages).
ON_QUOTA = [("2026-08-01", 4.0, "3-5"), ("2026-08-04", 8.0, "3-6")]  # debt -8
NEARLY = [("2026-07-27", 0.0, "opened")]                             # debt  3
BLOCKED = [("2026-07-20", 0.0, "opened")]                            # debt  6
HARD_CAPPED = [("2026-07-06", 0.0, "opened")]                        # debt 12


# --------------------------------------------------------------------------
# blocking conditions
# --------------------------------------------------------------------------
def test_staleness_alone_does_not_block(tmp_path):
    """The whole point of design B. Under the old gate this ledger blocked
    for being three days old; a stale ledger is the normal state of a
    legitimate technical session, and blocking on it is what trained the
    bypass reflex."""
    path = _ledger(tmp_path, ON_QUOTA)
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked, why


def test_blocks_once_two_weeks_of_quota_have_accrued_unbanked(tmp_path):
    path = _ledger(tmp_path, BLOCKED)
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert blocked
    assert "behind quota" in why


def test_one_week_of_debt_is_not_yet_blocking(tmp_path):
    """Boundary: the rule is two weeks, so one week of debt still runs."""
    path = _ledger(tmp_path, NEARLY)
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked, why


def test_blocks_on_an_empty_or_missing_ledger(tmp_path):
    blocked, why = ledger_gate.evaluate(ledger_gate.read_ledger(tmp_path / "nope.csv"), now=NOW)
    assert blocked
    assert "empty or missing" in why


def test_debt_accrues_from_the_first_row_not_the_last(tmp_path):
    """Adding a row that banks nothing must not reset the clock -- that
    would let an empty entry buy another two weeks, indefinitely."""
    path = _ledger(tmp_path, BLOCKED + [("2026-08-07", 0.0, "still nothing")])
    assert ledger_gate.page_debt(ledger_gate.read_ledger(path), NOW) == 6.0


def test_writing_ahead_buys_slack(tmp_path):
    """Negative debt is returned as-is, so pages banked early genuinely
    count later. Under the old rule this ledger blocked for not moving."""
    path = _ledger(tmp_path, [("2026-07-06", 40.0, "way ahead")])
    debt = ledger_gate.page_debt(ledger_gate.read_ledger(path), NOW)
    assert debt < 0
    blocked, _ = ledger_gate.evaluate(ledger_gate.read_ledger(path), now=NOW)
    assert not blocked


def test_the_repo_ledger_is_blocking_today():
    """The state that motivated the gate must actually trip it.

    Dated 2026-08-27, not 2026-08-07: the ledger opened 2026-08-05, so on
    08-07 it was two days old and genuinely owed nothing yet. Design B
    blocks on falling behind, not on being new.
    """
    blocked, why = ledger_gate.evaluate(
        ledger_gate.read_ledger(ledger_gate.DEFAULT_LEDGER),
        now=datetime(2026, 8, 27, 12, 0),
    )
    assert blocked, f"expected the repo ledger to block, got: {why}"


# --------------------------------------------------------------------------
# the enforcement itself: exit code, not a warning
# --------------------------------------------------------------------------
def test_require_exits_nonzero_when_blocked(tmp_path):
    path = _ledger(tmp_path, BLOCKED)
    with pytest.raises(SystemExit) as exc:
        ledger_gate.require_ledger_progress(
            "scripts/run_something.py", ledger_path=path, now=NOW, env={},
            decisions_path=tmp_path / "decisions.md",
        )
    assert exc.value.code == 1


def test_require_returns_quietly_when_allowed(tmp_path):
    path = _ledger(tmp_path, ON_QUOTA)
    ledger_gate.require_ledger_progress(
        "scripts/run_something.py", ledger_path=path, now=NOW, env={},
        decisions_path=tmp_path / "decisions.md",
    )  # must not raise


def test_blocked_message_names_the_script_and_the_escape_hatch(tmp_path, capsys):
    path = _ledger(tmp_path, BLOCKED)
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
# bypass: works, is never free, and stops working past the hard cap
# --------------------------------------------------------------------------
def test_bypass_allows_a_blocked_run_and_writes_a_trace(tmp_path):
    path = _ledger(tmp_path, BLOCKED)
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


def test_the_bypass_says_it_does_not_reduce_the_debt(tmp_path, capsys):
    path = _ledger(tmp_path, BLOCKED)
    ledger_gate.require_ledger_progress(
        "scripts/run_shap.py", ledger_path=path, now=NOW,
        env={ledger_gate.BYPASS_ENV: "1"}, decisions_path=tmp_path / "decisions.md",
    )
    assert "does not reduce it" in capsys.readouterr().err


def test_bypass_is_refused_past_the_hard_cap(tmp_path, capsys):
    """Constraint (a): the gate must not be satisfiable by bypassing. Debt
    rolls forward, and this is where the escape hatch closes."""
    path = _ledger(tmp_path, HARD_CAPPED)
    decisions = tmp_path / "decisions.md"
    with pytest.raises(SystemExit) as exc:
        ledger_gate.require_ledger_progress(
            "scripts/run_shap.py", ledger_path=path, now=NOW,
            env={ledger_gate.BYPASS_ENV: "urgent"}, decisions_path=decisions,
        )
    assert exc.value.code == 1
    assert "DISABLED" in capsys.readouterr().err
    assert not decisions.exists(), "a refused bypass must not log itself as taken"


def test_bypass_value_is_recorded_as_the_reason(tmp_path):
    path = _ledger(tmp_path, BLOCKED)
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
    path = _ledger(tmp_path, BLOCKED)
    with pytest.raises(SystemExit):
        ledger_gate.require_ledger_progress(
            "scripts/run_something.py",
            ledger_path=path,
            now=NOW,
            env={ledger_gate.BYPASS_ENV: "   "},
            decisions_path=tmp_path / "decisions.md",
        )


def test_bypass_on_an_already_passing_ledger_still_logs(tmp_path):
    """Setting the variable when it was not needed still leaves a trace --
    otherwise it could be left permanently exported with no record."""
    path = _ledger(tmp_path, ON_QUOTA)
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
