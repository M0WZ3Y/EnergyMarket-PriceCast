"""Meta-tests: the suite must not silently stop checking things.

The 2026-08-05 sweep found that the project's central non-leakage guard --
the check that the regime stress threshold was computed on train-only data
-- depends on `data/raw/DE.csv`, which is gitignored. On a clean checkout
that test SKIPPED and the suite still reported green, so the guard CLAUDE.md
calls non-negotiable was silently absent exactly where an examiner or
reviewer would run it.

The `THESIS_FULL_DATA=1` escape hatch worked mechanically but was set by
nothing: no conftest, no CI, no pytest.ini addopts. An honour-system flag
that nobody sets is not a safeguard.

These tests fail if that regression ever returns.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _run_pytest(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.update(env_extra)
    env.pop("THESIS_FULL_DATA", None) if "THESIS_FULL_DATA" not in env_extra else None
    return subprocess.run(
        [str(PY) if PY.exists() else sys.executable, "-m", "pytest",
         *args, "-q", "-p", "no:cacheprovider", "--tb=no"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_missing_benchmark_data_fails_the_suite_instead_of_skipping(tmp_path):
    """A clean checkout must FAIL, not quietly skip.

    Simulated by pointing the data directory at an empty temp dir rather
    than by moving real files, so this test can never damage the working
    copy. The data root must therefore be resolvable via THESIS_DATA_DIR.
    """
    result = _run_pytest(
        ["tests/test_regime_threshold.py"],
        {"THESIS_DATA_DIR": str(tmp_path)},
    )

    assert result.returncode != 0, (
        "the leakage-provenance guard skipped silently with no benchmark data "
        "present -- on a clean checkout the suite reports green while the "
        "check CLAUDE.md calls non-negotiable is not running.\n"
        f"stdout:\n{result.stdout[-2000:]}"
    )


def test_data_dependent_skips_are_reported_not_swallowed(tmp_path):
    """Even when skipping is permitted, it must be VISIBLE.

    A skip that scrolls past in a '122 passed' summary is indistinguishable
    from a pass to anyone not reading -rs output.
    """
    result = _run_pytest(
        ["tests/test_regime_threshold.py"],
        {"THESIS_DATA_DIR": str(tmp_path), "THESIS_ALLOW_MISSING_DATA": "1"},
    )

    combined = result.stdout + result.stderr
    assert "MISSING DATA" in combined.upper(), (
        "data-dependent tests were skipped without a prominent warning in the "
        f"terminal summary.\nstdout:\n{combined[-2000:]}"
    )


def test_pytest_ini_enables_strict_markers():
    """Without --strict-markers a typo'd @pytest.mark.netwrok silently runs
    in the offline set instead of erroring -- and with 775 warnings already
    scrolling past, nobody would notice.
    """
    ini = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "--strict-markers" in ini, "pytest.ini does not enable --strict-markers"


def test_an_unknown_marker_is_an_error_not_a_warning():
    """Behavioural counterpart to the pytest.ini check above."""
    probe = REPO_ROOT / "tests" / "_marker_probe.py"
    probe.write_text(
        "import pytest\n\n\n@pytest.mark.deliberately_unregistered\ndef test_probe():\n    pass\n",
        encoding="utf-8",
    )
    try:
        result = _run_pytest(["tests/_marker_probe.py"], {})
        assert result.returncode != 0, (
            "an unregistered marker was accepted; a typo'd marker name would "
            "silently move a test between the offline and network sets"
        )
    finally:
        probe.unlink(missing_ok=True)
