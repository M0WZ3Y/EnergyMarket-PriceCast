"""PreToolUse hook: block edits to results artifacts once v1.0-results exists.

Per CLAUDE.md: after the v1.0-results tag (end of week 7), model results are
frozen — the thesis text depends on those numbers. Until the tag exists this
hook is a no-op.
"""
import json
import subprocess
import sys
from pathlib import Path

FROZEN_DIRS = ("reports/figures", "reports/tables", "models", "data/processed")
TAG = "v1.0-results"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    raw_path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not raw_path:
        return 0

    try:
        rel = Path(raw_path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return 0  # outside the repo — not ours to police

    if not any(rel == d or rel.startswith(d + "/") for d in FROZEN_DIRS):
        return 0

    try:
        tag = subprocess.run(
            ["git", "tag", "-l", TAG], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return 0  # if git is unavailable, fail open rather than break all edits

    if not tag:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"'{rel}' is frozen: the {TAG} tag exists, and per CLAUDE.md model "
                "results must never be rerun or modified after freezing (thesis "
                "chapters depend on these numbers). If a change is genuinely "
                "required, log a dated entry in logs/decisions.md and deliberately "
                f"delete/re-create the {TAG} tag first."
            ),
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
