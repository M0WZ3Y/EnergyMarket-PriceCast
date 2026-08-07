"""Ledger-progress gate — src/ledger_gate.py

Refuses to run a NEW technical script while the writing ledger shows no
recent forward movement.

Why this exists: on 2026-08-07 `thesis/page_ledger.csv` held a single row
dated 2026-08-05 reading 0.0 pages, while `thesis/drafts/` had grown to four
files of Farsi prose. Technical work kept being picked up ahead of the
conversion that actually banks pages, and nothing in the repo made that
visible, let alone costly. A printed warning would not have changed it —
warnings scroll past. This exits non-zero.

Lives in src/ rather than scripts/ because scripts in this repo deliberately
never import from each other (see the duplicated `_rel` helpers in
run_ood_stress.py and export_tables.py); src/ is the shared layer they all
already import.

The ledger reader here is a deliberately minimal local copy rather than an
import of scripts/page_quota.py's richer `load_ledger`. Importing a script
from src/ would invert the dependency direction the rest of the repo
follows. The cost is a few lines of duplication; the gate needs only dates
and page counts.

Usage, in a gated script's __main__ block ONLY (never at import time — the
test suite imports these modules and must not be gated):

    if __name__ == "__main__":
        require_ledger_progress(__file__)
        main()
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "thesis" / "page_ledger.csv"
DEFAULT_DECISIONS = REPO_ROOT / "logs" / "decisions.md"

BYPASS_ENV = "THESIS_SKIP_LEDGER_GATE"
STALE_HOURS = 48

# Values that mean "on" but carry no explanation. Anything else set in the
# variable is treated as the reason itself, so a bypass can be self-documenting.
_BARE_TRUTHY = {"1", "true", "yes", "on"}


class LedgerGateError(RuntimeError):
    """The ledger shows no recent forward progress."""


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


def read_ledger(path: str | Path = DEFAULT_LEDGER) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if not (raw.get("date") or "").strip():
                continue
            rows.append(
                {
                    "date": _as_date(raw["date"]),
                    "pages_banked": float(raw["pages_banked"]),
                    "note": (raw.get("note") or "").strip(),
                }
            )
    return sorted(rows, key=lambda r: r["date"])


def evaluate(ledger: list[dict], now: datetime | None = None) -> tuple[bool, str]:
    """(blocked, reason). Reason is '' when not blocked.

    Two independent block conditions, per the rule this implements:

      1. The most recent entry is more than STALE_HOURS old.
      2. The most recent entry banked the same page count as the one before
         it — logged, but no forward movement.

    Ledger dates have day granularity, so an entry is treated as made at
    00:00 of its date. That is the conservative reading: it can only make an
    entry look older, never fresher.

    A ledger with exactly ONE entry cannot be evaluated on condition 2 —
    there is nothing to compare against — so only staleness applies. The
    alternative (treating a lone entry as "no progress") would block every
    fresh ledger permanently, including a brand-new project's, which would
    make the gate something people rip out rather than use.
    """
    now = now or datetime.now()

    if not ledger:
        return True, (
            "the page ledger is empty or missing, so there is no evidence of "
            "any writing progress"
        )

    last = ledger[-1]
    age = now - datetime.combine(last["date"], datetime.min.time())
    if age > timedelta(hours=STALE_HOURS):
        hours = age.total_seconds() / 3600
        return True, (
            f"the most recent ledger entry is dated {last['date']}, "
            f"{hours:.0f} hours old (limit {STALE_HOURS})"
        )

    if len(ledger) >= 2:
        prev = ledger[-2]
        if last["pages_banked"] == prev["pages_banked"]:
            return True, (
                f"the most recent ledger entry ({last['date']}) banks "
                f"{last['pages_banked']:g} pages, the same as the entry before it "
                f"({prev['date']}) — logged, but no forward movement"
            )

    return False, ""


def _bypass_reason(env: dict | None = None) -> str | None:
    """The reason string if the bypass is set, else None."""
    env = os.environ if env is None else env
    raw = (env.get(BYPASS_ENV) or "").strip()
    if not raw:
        return None
    if raw.lower() in _BARE_TRUTHY:
        return "(no reason given)"
    return raw


def log_bypass(
    script: str,
    reason: str,
    ledger: list[dict],
    now: datetime | None = None,
    decisions_path: str | Path = DEFAULT_DECISIONS,
) -> None:
    """Append a trace of the bypass to logs/decisions.md.

    Appended at the END of the file rather than inserted before its running
    footer. decisions.md is ~1600 lines of multi-entry prose, and a
    programmatic insert into the middle of it is exactly the mutation class
    that has destroyed content in this project before. Appending cannot
    damage an existing entry.
    """
    now = now or datetime.now()
    decisions_path = Path(decisions_path)
    if ledger:
        last = ledger[-1]
        state = f"last entry {last['date']}, pages_banked {last['pages_banked']:g}"
    else:
        state = "ledger empty or missing"

    entry = (
        f"\n\n### {now.date()} — LEDGER GATE BYPASSED ({Path(script).name})\n\n"
        f"`{BYPASS_ENV}` was set, so the ledger-progress gate did not run before "
        f"`{Path(script).name}`. Reason given: {reason}. Ledger state at bypass: "
        f"{state}.\n\n"
        "Recorded automatically by `src/ledger_gate.py`. The bypass exists so an "
        "urgent technical task is never hard-blocked by writing admin — but it "
        "leaves this trace, so choosing it is visible rather than free.\n"
    )
    with open(decisions_path, "a", encoding="utf-8") as f:
        f.write(entry)


def require_ledger_progress(
    script: str,
    ledger_path: str | Path = DEFAULT_LEDGER,
    now: datetime | None = None,
    env: dict | None = None,
    decisions_path: str | Path = DEFAULT_DECISIONS,
) -> None:
    """Gate entry point. Exits non-zero when the ledger shows no progress."""
    ledger = read_ledger(ledger_path)

    reason = _bypass_reason(env)
    if reason is not None:
        log_bypass(script, reason, ledger, now=now, decisions_path=decisions_path)
        print(
            f"[ledger-gate] BYPASSED via {BYPASS_ENV} ({reason}). "
            f"Logged to {Path(decisions_path).name}.",
            file=sys.stderr,
        )
        return

    blocked, why = evaluate(ledger, now=now)
    if not blocked:
        return

    print(
        f"\n[ledger-gate] refusing to run {Path(script).name}\n\n"
        f"  {why}.\n\n"
        "  Writing is the binding constraint on this project, and new technical\n"
        "  output does not move it. Before starting another technical task:\n\n"
        "    1. Convert a finished draft into the Amirkabir template\n"
        "       (see thesis/CONVERSION_QUEUE.md)\n"
        "    2. Record the real page count:\n"
        "       ./.venv/Scripts/python.exe scripts/page_quota.py --add <pages> "
        '--note "<section>"\n\n'
        f"  If this task genuinely cannot wait, set {BYPASS_ENV}=<reason> — it will\n"
        f"  run, and the bypass will be logged to logs/decisions.md with that reason.\n",
        file=sys.stderr,
    )
    raise SystemExit(1)
