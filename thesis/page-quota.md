# Page quota — the writing plan

Set 2026-08-05, week 5 of 12. This exists because the README's standing rule
("45–60 min thesis writing daily before code, week 2+, page quotas tracked")
was never operationalised: every weekly footer in `logs/decisions.md` reads
`Pages banked: 0 / quota 0`, because the quota was never set. It is set here.

## Where the numbers live

| What | Where |
|---|---|
| Dates, targets, page budget | `configs/schedule.yaml` (single source of truth) |
| Daily record of pages banked | `thesis/page_ledger.csv` |
| Status report | `./.venv/Scripts/python.exe scripts/page_quota.py` |

Nothing is hardcoded in the script. Change a target by editing the config; the
tests in `tests/test_page_quota.py` will catch a milestone that drifts past the
page budget, falls outside the schedule window, or moves either of the two
supervisor dates that are commitments to another person.

## Daily use

```bash
# Status only
./.venv/Scripts/python.exe scripts/page_quota.py

# Record progress (replaces today's row if you run it twice)
./.venv/Scripts/python.exe scripts/page_quota.py --add 12 --note "3-3 done"
```

Record the **cumulative page count in the official Amirkabir docx**, not words
written and not markdown drafted. The docx lives outside this repo, so no
script can measure it — that number has to come from you. If a section gets
cut, record the lower number; the tracker deliberately reports the latest
entry rather than the high-water mark, so a cut shows up as a drop instead of
being quietly hidden.

## The plan as of 2026-08-05

Banked: **0 of 100**. (`thesis/drafts/` holds Farsi drafts of 3-5 and 3-6,
roughly 4pp, but they are not in the template, so they are not banked.)

| Milestone | Due | Target | Required pace |
|---|---|---|---|
| Week-9 partial review | 2026-08-31 | 60pp | 2.31 pp/day over 26 days |
| Week-10 full-draft review | 2026-09-07 | 100pp | 3.03 pp/day over 33 days |
| Schedule end | 2026-09-27 | 100pp | 1.89 pp/day over 53 days |

The week-9 target is 60pp because chapters 3 (37pp) and 4 (29pp) are the two a
partial review can actually act on — both are fully backed by the frozen
`v1.0-results` numbers and `reports/tables/*.tex`, so they can be written now
without waiting on anything.

Both review dates are booked with the supervisor (`logs/decisions.md`,
2026-07-06 Admin entry). They do not move.

## Calibrate `words_per_page` before trusting the drafts estimate

`configs/schedule.yaml` sets `words_per_page: 250` as a placeholder. Once one
section is in the template, divide its word count by the pages it actually
occupied and set the real value. Until then the "drafts estimate" line in the
report is a rough signal only.

## Why the pace is what it is

The code is roughly three weeks ahead of schedule — the `v1.0-results` freeze
landed 19 days early, SHAP (planned week 8) landed 19 days early. None of that
slack helps unless it is spent on pages: the two review dates are fixed, and
writing is now the entire critical path. Suggested order, since it follows the
frozen numbers rather than waiting on anything:

1. Chapter 3 (methodology, 37pp) — sections 3-5 and 3-6 already have drafts.
2. Chapter 4 (results, 29pp) — every table and figure it needs now exists,
   including SHAP figures 10–15 and `reports/tables/shap_importance.tex`.
3. Chapters 1, 2, 5 (34pp).

Claim discipline for chapter 4 is written down in `NEXT_SESSION.md` — the
regime-aware significance is stressed-days-only, LSTM vs LEAR-LASSO is a tie,
and the SHAP attributions describe a *static* fit where the results models
recalibrate daily. Do not restate any of those loosely.
