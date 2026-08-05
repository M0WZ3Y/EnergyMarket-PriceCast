"""Page-quota tracker for the Farsi thesis body.

    page_quota.py                 status against every upcoming milestone
    page_quota.py --add 12        record 12 pages banked as of today
    page_quota.py --add 12 --note "3-3 done"

Why this exists: the README's standing rule is "45-60 min thesis writing
daily before code (week 2+), page quotas tracked", and it lapsed --- every
weekly footer in logs/decisions.md reads "Pages banked: 0 / quota 0" because
the quota was never set. All dates and targets live in configs/schedule.yaml;
nothing here is hardcoded.

Design rule: this tool must never look better than reality. A passed deadline
raises rather than returning a negative rate, a deadline that is today demands
the whole remainder today rather than dividing by zero, and pages banked come
from the LATEST ledger row rather than the maximum --- taking the max would
keep reporting a high-water mark after a chapter was cut.

The ledger is human-maintained on purpose. The real page count comes from the
docx in the official Amirkabir template, which lives outside this repo, so no
script here can measure it. The report also prints an estimate from
thesis/drafts/*.md word counts, as a rough signal only -- never as the ledger.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCHEDULE = REPO_ROOT / "configs" / "schedule.yaml"
DRAFTS_DIR = REPO_ROOT / "thesis" / "drafts"

LEDGER_HEADER = ["date", "pages_banked", "note"]


class DeadlinePassed(RuntimeError):
    """Raised instead of returning a rate that cannot be achieved."""


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def load_schedule(path: str | Path = DEFAULT_SCHEDULE) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["start_date"] = _as_date(cfg["start_date"])
    cfg["end_date"] = _as_date(cfg["end_date"])
    for milestone in cfg["milestones"]:
        milestone["date"] = _as_date(milestone["date"])
    return cfg


def load_ledger(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No page_ledger at {path}. Create it with a header row "
            f"'{','.join(LEDGER_HEADER)}', or record your first entry with "
            "`page_quota.py --add <pages>`."
        )

    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if not (raw.get("date") or "").strip():
                continue
            pages = float(raw["pages_banked"])
            if pages < 0:
                raise ValueError(
                    f"page_ledger {path}: negative page count {pages} on "
                    f"{raw['date']} -- pages banked cannot be below zero"
                )
            rows.append(
                {
                    "date": _as_date(raw["date"].strip()),
                    "pages_banked": pages,
                    "note": (raw.get("note") or "").strip(),
                }
            )
    return sorted(rows, key=lambda r: r["date"])


def pages_banked(ledger: list[dict]) -> float:
    """The most RECENT entry, not the largest.

    Pages can legitimately go down (a cut section). Reporting the maximum
    would hide that and overstate progress for the rest of the project.
    """
    return ledger[-1]["pages_banked"] if ledger else 0.0


def required_rate(banked: float, target: float, today: date, deadline: date) -> float:
    """Pages per day needed from `today` to hit `target` by `deadline`."""
    if deadline < today:
        raise DeadlinePassed(
            f"{deadline} has passed (today is {today}); there is no achievable rate"
        )
    remaining = target - banked
    if remaining <= 0:
        return 0.0
    days_left = (deadline - today).days
    # A deadline of today means the whole remainder is due today -- a finite,
    # honest number, not a ZeroDivisionError and not "infinity".
    return remaining / days_left if days_left > 0 else float(remaining)


def estimate_draft_pages(drafts_dir: str | Path, words_per_page: int) -> float:
    """Rough page estimate from drafted markdown. A signal, not the ledger."""
    drafts_dir = Path(drafts_dir)
    if not drafts_dir.is_dir():
        return 0.0
    words = sum(
        len(p.read_text(encoding="utf-8").split()) for p in sorted(drafts_dir.glob("*.md"))
    )
    return words / words_per_page


def add_entry(pages: float, note: str, ledger_path: Path, today: date) -> None:
    """Append (or replace) today's row."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_ledger(ledger_path) if ledger_path.exists() else []
    rows = [r for r in rows if r["date"] != today]
    rows.append({"date": today, "pages_banked": pages, "note": note})
    rows.sort(key=lambda r: r["date"])

    with open(ledger_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"date": row["date"], "pages_banked": row["pages_banked"], "note": row["note"]}
            )
    print(f"recorded {pages:g} pages on {today}" + (f" ({note})" if note else ""))


def report(cfg: dict, ledger: list[dict], today: date) -> list[str]:
    banked = pages_banked(ledger)
    budget = cfg["page_budget"]
    lines = [
        "",
        f"Thesis page quota - {today}",
        "=" * 58,
        f"  Pages banked      {banked:g} / {budget}  ({banked / budget:.0%})",
    ]

    if ledger:
        last = ledger[-1]
        lines.append(
            f"  Last entry        {last['date']}"
            + (f" - {last['note']}" if last["note"] else "")
        )
        stale = (today - last["date"]).days
        if stale >= 2:
            lines.append(f"  !! ledger is {stale} days stale - is the daily rule running?")
    else:
        lines.append("  Last entry        (none - ledger is empty)")

    estimate = estimate_draft_pages(DRAFTS_DIR, cfg["words_per_page"])
    if estimate:
        lines.append(
            f"  Drafts estimate   ~{estimate:.1f}pp of markdown in thesis/drafts/ "
            f"(at {cfg['words_per_page']} w/p, uncalibrated)"
        )

    lines += ["", "  Milestone                     due        target   need/day", "  " + "-" * 56]
    for milestone in cfg["milestones"]:
        try:
            rate = required_rate(banked, milestone["pages"], today, milestone["date"])
        except DeadlinePassed:
            short = max(0.0, milestone["pages"] - banked)
            verdict = "MISSED" if short else "met"
            lines.append(
                f"  {milestone['name']:<28}  {milestone['date']}  "
                f"{milestone['pages']:>6}   {verdict}"
                + (f" ({short:g}pp short)" if short else "")
            )
            continue

        days = (milestone["date"] - today).days
        verdict = "on target" if rate == 0 else f"{rate:.2f}pp/day over {days}d"
        lines.append(
            f"  {milestone['name']:<28}  {milestone['date']}  "
            f"{milestone['pages']:>6}   {verdict}"
        )

    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add", type=float, metavar="PAGES", help="record pages banked today")
    parser.add_argument("--note", default="", help="note for the --add entry")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    args = parser.parse_args(argv)

    cfg = load_schedule(args.schedule)
    ledger_path = REPO_ROOT / cfg["ledger"]
    today = date.today()

    if args.add is not None:
        add_entry(args.add, args.note, ledger_path, today)

    ledger = load_ledger(ledger_path) if ledger_path.exists() else []
    print("\n".join(report(cfg, ledger, today)))


if __name__ == "__main__":
    main()
