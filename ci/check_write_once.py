"""Write-once guard for issued forecasts.

Issued forecasts (app/forecast_site/data/forecasts/*.json) may only be ADDED.
A run that modifies or deletes a committed forecast fails the job.

    python ci/check_write_once.py        # from anywhere; exits 1 on violation

Uses git with check=True: if git itself fails (not a repo, no HEAD), the error
propagates and the job fails. A git error is never reported as "clean".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = "app/forecast_site/data/forecasts/"


def check(repo: Path = REPO_ROOT) -> list[str]:
    """Return repo-relative POSIX paths of forecast files modified or deleted vs HEAD."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=MDT",
            "HEAD",
            "--",
            FORECAST_DIR,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    violations = check(REPO_ROOT)
    for path in violations:
        print(f"{path}: issued forecast modified or deleted (forecasts are write-once)")
    if violations:
        print(f"write-once check FAILED: {len(violations)} issued forecast file(s) changed")
        return 1
    print("write-once check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
