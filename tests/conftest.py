"""Shared test configuration: data-root resolution and the missing-data policy.

Two problems this file exists to fix (2026-08-05 suite sweep):

1. The project's central non-leakage guard (the regime-threshold provenance
   check) reads `data/raw/DE.csv`, which is gitignored. On a clean checkout
   that test SKIPPED and the suite still reported green -- the guard
   CLAUDE.md calls non-negotiable was silently absent exactly where an
   examiner would run it. The old `THESIS_FULL_DATA=1` escape hatch was set
   by nothing: no conftest, no CI, no addopts. An honour-system flag nobody
   sets is not a safeguard.

   So the policy is INVERTED here: strict is the DEFAULT. Missing benchmark
   data FAILS. Someone genuinely working from a bare clone opts out with
   `THESIS_ALLOW_MISSING_DATA=1`, and when they do, a prominent
   "MISSING DATA" banner is printed in the terminal summary -- a skip buried
   in a "148 passed" line is indistinguishable from a pass.

   `THESIS_FULL_DATA=1` remains accepted as a no-op alias so the existing
   instructions in NEXT_SESSION.md stay correct.

2. `THESIS_DATA_DIR` lets a test point the data root at an empty temp
   directory to simulate a clean checkout, so the meta-tests in
   tests/test_suite_integrity.py can verify the policy without ever moving,
   renaming or deleting the real frozen data files.

This module also centralises the `sys.path.insert(0, repo_root)` that seven
test modules each repeat; those lines remain in place (harmless) but are now
redundant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Centralised repo-root import path (previously duplicated per test module).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Populated by require_thesis_data() whenever a test is skipped because its
# input data is absent; drained by pytest_terminal_summary below.
_MISSING_DATA: list[str] = []


def thesis_data_dir() -> Path:
    """Root of the thesis data tree.

    Defaults to `<repo>/data`; overridable with THESIS_DATA_DIR so tests can
    simulate a clean checkout without touching real files.
    """
    override = os.environ.get("THESIS_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data"


def thesis_data_path(*parts: str) -> Path:
    """A path beneath the (possibly redirected) thesis data root."""
    return thesis_data_dir().joinpath(*parts)


def allow_missing_data() -> bool:
    """True only if the operator explicitly opted out of the strict default."""
    return os.environ.get("THESIS_ALLOW_MISSING_DATA") == "1"


def require_thesis_data(path: Path, description: str, *, non_empty_dir: bool = False) -> Path:
    """Assert a data input is present, or apply the missing-data policy.

    Default (strict): FAIL. With THESIS_ALLOW_MISSING_DATA=1: skip, and
    register the omission so it is shouted about in the terminal summary.
    """
    if non_empty_dir:
        present = path.is_dir() and any(path.glob("*.csv"))
    else:
        present = path.exists()

    if present:
        return path

    detail = f"{description} not found at {path}"
    if allow_missing_data():
        _MISSING_DATA.append(detail)
        pytest.skip(f"MISSING DATA (opt-out active): {detail}")

    pytest.fail(
        f"{detail}. This check cannot run and must not pass silently.\n"
        "Restore the data (see NEXT_SESSION.md), or, if you are deliberately "
        "running on a bare clone, set THESIS_ALLOW_MISSING_DATA=1 to downgrade "
        "these checks to loudly-reported skips."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    """Make permitted data-dependent skips impossible to miss."""
    if not _MISSING_DATA:
        return

    write = terminalreporter.write_line
    terminalreporter.section("MISSING DATA", sep="!", red=True, bold=True)
    write(
        "!!! MISSING DATA: data-dependent guards were SKIPPED, not run. "
        "This run does NOT verify them. !!!",
        red=True,
        bold=True,
    )
    for detail in dict.fromkeys(_MISSING_DATA):
        write(f"  - {detail}", red=True)
    write(
        "THESIS_ALLOW_MISSING_DATA=1 is set. Unset it (and restore data/) "
        "before trusting a green run.",
        red=True,
        bold=True,
    )
