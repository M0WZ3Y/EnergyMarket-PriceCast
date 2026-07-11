"""PostToolUse hook: run the offline contract tests after edits under src/ or tests/.

Runs pytest with the network-marked tests deselected (fast, offline-only).
Exit 2 feeds the failure output back to Claude so it fixes the breakage
immediately — this keeps the loader-schema and leakage assertions green.
"""
import json
import subprocess
import sys
from pathlib import Path

WATCHED_DIRS = ("src", "tests")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    raw_path = tool_input.get("file_path") or ""
    if not raw_path:
        return 0

    try:
        rel = Path(raw_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return 0

    if not any(rel == d or rel.startswith(d + "/") for d in WATCHED_DIRS):
        return 0
    if not rel.endswith(".py"):
        return 0

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not network", "-q"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        tail = (result.stdout or "")[-3000:] + (result.stderr or "")[-1000:]
        print(f"Offline contract tests failed after editing {rel}:\n{tail}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
