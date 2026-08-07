# Conversion queue — drafts → Amirkabir template

Generated 2026-08-07. **This file is maintained by hand after the first
generation.** The conversion itself is manual: the official Amirkabir `.docx`
template lives outside this repo, so no script here can paste into it or
measure the resulting page count.

Bodies in `thesis/drafts/*.md` are now clean Farsi prose — the `> DRAFT …`
source blocks have been moved to `thesis/drafts/sources/<stem>.sources.md`
and are **not** meant to be pasted. Each body keeps its `#` title line, which
maps onto the template's heading style.

## How to use this

1. Open a draft, select the body, paste into the template under the matching
   heading.
2. Read the page number the template actually reports for that section.
3. Write it into the **Pages (real)** column below and tick the box.
4. Once at least one section has a real number, record the total with
   `./.venv/Scripts/python.exe scripts/page_quota.py --add <pages> --note "3-6"`.

## The queue

Order follows `thesis/outline.md`.

| ✓ | Section | Title (outline) | Draft file | Body words | Pages (real) |
|---|---|---|---|---|---|
| [ ] | 3-2 | فرضیات بنیادی پژوهش [3pp] | **not drafted** | — | |
| [ ] | 3-5 | چارچوب اعتبارسنجی و معیارهای ارزیابی [4pp] | `3-5-evaluation-framework.md` | 487 | |
| [ ] | 3-6 | مدل‌های پایه [4pp] | `3-6-baseline-models.md` | 1047 | |
| [ ] | 3-7-1 | LightGBM [3pp] | `3-7-1-lightgbm.md` | 706 | |
| [ ] | 3-7-2 | LSTM [4pp] | `3-7-2-lstm.md` | 952 | |

**Total drafted body words: 3,192** across 4 files.

## Why the page column is blank

`configs/schedule.yaml` carries `words_per_page: 250`, which is an
**uncalibrated placeholder** — nobody has yet measured how many words of this
Farsi prose fill one page of the official template. Multiplying by it would
produce a number that looks like a measurement and is not one, and the page
ledger is the one place in this project that must never look better than
reality.

So: no estimate is recorded here. The column stays empty until a human pastes
a section and reports the page count the template itself shows. That first
real number is also what calibrates `words_per_page` for everything after it.

## Note on 3-2

Section 3-2 (the six formal assumptions) appears in the outline and in
`CHECKLIST.md`, but **no draft exists in this repo** — searched across the
full git history, all branches and stashes on 2026-08-07. It is listed here so
the queue matches the outline rather than only the files that happen to exist.
If it was drafted in a separate web session, it lives there and has not been
brought back.

Section 3-2 also has a dependency worth noting before it is written: two of
the six assumptions — (4) model generalization and (5) stable market
conditions — are directly challenged by the OOD result and by the
2026-08-07 recalibration finding. That tension belongs in the section, stated
plainly.
