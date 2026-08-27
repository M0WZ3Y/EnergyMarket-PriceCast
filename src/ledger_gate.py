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
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "thesis" / "page_ledger.csv"
DEFAULT_DECISIONS = REPO_ROOT / "logs" / "decisions.md"

BYPASS_ENV = "THESIS_SKIP_LEDGER_GATE"

# Weekly settlement (T4, design B, chosen 2026-08-26). The gate no longer
# blocks on staleness: a stale ledger is the NORMAL state during a
# legitimate technical session, and blocking on it is what trained the
# bypass reflex -- every block this gate ever produced was a staleness
# block, because the "no forward movement" rule needs two ledger rows and
# the ledger has only ever had one.
#
# What is measured instead is banked template pages against an accruing
# weekly quota. Debt rolls forward, so a bypass does not clear it, and past
# DEBT_HARD_CAP_WEEKS the bypass stops working altogether -- the cost that
# makes the gate not satisfiable by bypassing.
WEEKLY_PAGE_QUOTA = 3.0
DEBT_BLOCK_WEEKS = 2
DEBT_HARD_CAP_WEEKS = 3

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


def page_debt(ledger: list[dict], now: datetime | None = None) -> float:
    """Template pages owed: accrued quota since the ledger opened, minus banked.

    Weeks accrue from the FIRST ledger row, so the clock starts when the
    project started counting, not at whatever the latest row happens to say.
    Debt is therefore unaffected by how recently anything was logged — which
    is the point: staleness is not a violation, falling behind is.

    Negative debt means ahead of quota and is returned as-is, so writing
    ahead genuinely buys slack later.
    """
    now = now or datetime.now()
    if not ledger:
        return WEEKLY_PAGE_QUOTA * DEBT_BLOCK_WEEKS
    weeks = max(0, (now.date() - ledger[0]["date"]).days // 7)
    return WEEKLY_PAGE_QUOTA * weeks - ledger[-1]["pages_banked"]


def evaluate(ledger: list[dict], now: datetime | None = None) -> tuple[bool, str]:
    """(blocked, reason). Reason is '' when not blocked.

    ONE block condition: the accrued page debt has reached
    DEBT_BLOCK_WEEKS worth of quota. Deliberately NOT staleness — a ledger
    untouched for days is the normal state of a legitimate technical
    session, and blocking on it is what made the old gate something to
    bypass reflexively rather than satisfy.

    A brand-new ledger owes nothing until a week has accrued, so this does
    not block a fresh project, and it does not need two rows to work.
    """
    now = now or datetime.now()

    if not ledger:
        return True, (
            "the page ledger is empty or missing, so there is no evidence of "
            "any writing progress"
        )

    debt = page_debt(ledger, now)
    if debt >= WEEKLY_PAGE_QUOTA * DEBT_BLOCK_WEEKS:
        return True, (
            f"{debt:g} template pages behind quota "
            f"({WEEKLY_PAGE_QUOTA:g}/week accruing since {ledger[0]['date']}; "
            f"{ledger[-1]['pages_banked']:g} banked)"
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
        debt = page_debt(ledger, now or datetime.now())
        if debt >= WEEKLY_PAGE_QUOTA * DEBT_HARD_CAP_WEEKS:
            # The cost that makes this gate not satisfiable by bypassing.
            print(
                f"\n[ledger-gate] refusing to run {Path(script).name}\n\n"
                f"  {debt:g} template pages behind quota — past the "
                f"{WEEKLY_PAGE_QUOTA * DEBT_HARD_CAP_WEEKS:g}-page hard cap.\n\n"
                f"  {BYPASS_ENV} is DISABLED at this level of debt. Bypassing does\n"
                "  not clear debt, and this is the point where it stops working.\n\n"
                "    ./.venv/Scripts/python.exe scripts/page_quota.py --add <pages> "
                '--note "<section>"\n',
                file=sys.stderr,
            )
            raise SystemExit(1)
        log_bypass(script, reason, ledger, now=now, decisions_path=decisions_path)
        print(
            f"[ledger-gate] BYPASSED via {BYPASS_ENV} ({reason}). "
            f"Logged to {Path(decisions_path).name}. "
            f"Page debt is {debt:g} and this bypass does not reduce it.",
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
