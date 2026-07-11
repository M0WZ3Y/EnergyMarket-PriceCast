---
name: log-decision
description: Append a dated decision entry to logs/decisions.md in the project's standard format
disable-model-invocation: true
---

# Log a decision

Append a decision entry to `logs/decisions.md`. The user's input (`$ARGUMENTS`)
is the decision summary; it may be terse — expand it into the file's format
without inventing facts.

## Format rules (match the existing file exactly)

1. Read `logs/decisions.md` first.
2. Entries live under `## Week N` sections. Append to the **latest** week
   section unless the user names a different week. If the current calendar
   week has no section yet, ask the user whether to open a new `## Week N`
   before creating one (week numbering is the 12-week personal schedule, not
   ISO weeks).
3. Entry heading: `### YYYY-MM-DD — Short title` using **today's date** unless
   the user gives another date.
4. Body: markdown bullets. State the decision, the alternative(s) rejected,
   and the reason — this file feeds the methodology chapter, so write it as
   material a reader of chapter 3 could cite.
5. Insert the entry **before** the `---` + weekly footer line at the bottom
   (`Pages banked: ...`), never after it.
6. Do not reflow or edit existing entries.

## When to remind the user

Per CLAUDE.md, any scope change beyond the approved list, any new
model/feature/data source, and the week-7 France stretch decision all require
an entry here. If the summary describes one of those, say so in the entry
title (e.g. "Scope: ...").
