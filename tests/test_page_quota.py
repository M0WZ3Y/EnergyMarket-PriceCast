"""Page-quota tracker — scripts/page_quota.py.

A tracker that flatters you is worse than no tracker. The tests here are
mostly about the arithmetic refusing to look better than reality:

* a passed deadline must not produce a negative or "achievable" rate,
* a deadline that is today must demand the whole remainder today, not divide
  by zero,
* pages banked must come from the LATEST ledger row, not the maximum — using
  the max would silently hide a cut chapter and keep reporting the high-water
  mark as progress.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "page_quota_under_test", REPO_ROOT / "scripts" / "page_quota.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pq = _load()

TODAY = pq.date(2026, 8, 5)


# ---------------------------------------------------------------------------
# Required-rate arithmetic
# ---------------------------------------------------------------------------


def test_required_rate_is_remaining_pages_over_remaining_days():
    rate = pq.required_rate(banked=7, target=100, today=TODAY, deadline=pq.date(2026, 9, 7))
    assert rate == pytest.approx(93 / 33)


def test_a_deadline_today_demands_the_whole_remainder_today():
    """Not a division by zero, and not 'infinite' — a finite, honest number."""
    rate = pq.required_rate(banked=90, target=100, today=TODAY, deadline=TODAY)
    assert rate == pytest.approx(10.0)


def test_a_passed_deadline_is_reported_as_missed_not_as_a_negative_rate():
    with pytest.raises(pq.DeadlinePassed):
        pq.required_rate(banked=10, target=100, today=TODAY, deadline=pq.date(2026, 8, 1))


def test_meeting_the_target_early_gives_a_rate_of_zero_not_a_negative_one():
    rate = pq.required_rate(banked=120, target=100, today=TODAY, deadline=pq.date(2026, 9, 7))
    assert rate == 0.0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def _write_ledger(path: Path, rows: list[tuple[str, float, str]]) -> Path:
    lines = ["date,pages_banked,note"]
    lines += [f"{d},{p},{n}" for d, p, n in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_pages_banked_is_the_latest_row_not_the_maximum(tmp_path):
    """A cut chapter must show as a drop. Taking the max would keep reporting
    the high-water mark and quietly overstate progress for the rest of the
    project."""
    ledger = _write_ledger(
        tmp_path / "l.csv",
        [("2026-08-05", 12, "ch3 start"), ("2026-08-06", 18, "ch3"), ("2026-08-07", 15, "cut 3-4")],
    )
    assert pq.pages_banked(pq.load_ledger(ledger)) == 15


def test_ledger_rows_are_read_in_date_order_not_file_order(tmp_path):
    ledger = _write_ledger(
        tmp_path / "l.csv",
        [("2026-08-07", 15, "later"), ("2026-08-05", 12, "earlier")],
    )
    assert pq.pages_banked(pq.load_ledger(ledger)) == 15


def test_an_empty_ledger_means_zero_banked_not_a_crash(tmp_path):
    ledger = _write_ledger(tmp_path / "l.csv", [])
    assert pq.pages_banked(pq.load_ledger(ledger)) == 0


def test_a_missing_ledger_is_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="page_ledger"):
        pq.load_ledger(tmp_path / "page_ledger.csv")


def test_ledger_rejects_a_negative_page_count(tmp_path):
    ledger = _write_ledger(tmp_path / "l.csv", [("2026-08-05", -3, "typo")])
    with pytest.raises(ValueError, match="negative"):
        pq.load_ledger(ledger)


# ---------------------------------------------------------------------------
# The shipped config must be internally consistent
# ---------------------------------------------------------------------------


def test_shipped_schedule_config_is_loadable_and_ordered():
    cfg = pq.load_schedule()
    assert cfg["page_budget"] == 100
    dates = [m["date"] for m in cfg["milestones"]]
    assert dates == sorted(dates), "milestones must be in chronological order"


def test_no_milestone_targets_more_than_the_page_budget():
    cfg = pq.load_schedule()
    for m in cfg["milestones"]:
        assert m["pages"] <= cfg["page_budget"], f"{m['name']} exceeds the budget"


def test_milestones_fall_inside_the_schedule_window():
    cfg = pq.load_schedule()
    for m in cfg["milestones"]:
        assert cfg["start_date"] <= m["date"] <= cfg["end_date"], m["name"]


def test_the_booked_review_dates_match_the_decision_log():
    """These two dates are commitments to another person. If someone edits
    the config, this test is what notices."""
    cfg = pq.load_schedule()
    by_name = {m["name"]: m["date"] for m in cfg["milestones"]}
    assert by_name["Week-9 partial review"] == pq.date(2026, 8, 31)
    assert by_name["Week-10 full-draft review"] == pq.date(2026, 9, 7)


# ---------------------------------------------------------------------------
# Draft estimation (a signal, explicitly not the ledger)
# ---------------------------------------------------------------------------


def test_draft_estimate_scales_with_words(tmp_path):
    (tmp_path / "a.md").write_text(" ".join(["واژه"] * 500), encoding="utf-8")
    assert pq.estimate_draft_pages(tmp_path, words_per_page=250) == pytest.approx(2.0)


def test_draft_estimate_ignores_non_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text(" ".join(["واژه"] * 250), encoding="utf-8")
    (tmp_path / "b.txt").write_text(" ".join(["واژه"] * 5000), encoding="utf-8")
    assert pq.estimate_draft_pages(tmp_path, words_per_page=250) == pytest.approx(1.0)


def test_draft_estimate_is_zero_for_a_missing_directory(tmp_path):
    assert pq.estimate_draft_pages(tmp_path / "nope", words_per_page=250) == 0.0
