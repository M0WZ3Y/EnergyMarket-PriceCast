# Section approval tracker

The author approves **one subsection at a time**. Nothing proceeds to the next
unit until the current one is approved — see `thesis/WRITING_HANDOFF.md` §1.5.

This file is the durable record of that. It has to be a file, not a
conversation: chapters 1–2 are drafted in Claude chat and 3–5 in Claude Code,
neither of which can see the other's history. **Update it in the same turn a
unit is approved**, before starting the next.

## Status values

| Value | Meaning |
|---|---|
| `—` | not started |
| `draft-md` | Farsi draft exists in `thesis/drafts/`, not yet in LaTeX |
| `in-tex` | converted into `thesis/latex/`, compiles |
| `review` | presented to the author, awaiting verdict |
| `changes` | author asked for changes; not approved |
| `APPROVED` | signed off. Only then may the next unit start. |

Record pages only after the unit is `in-tex` and the real count is read from
the compiled PDF — never from a word-count estimate.

## Chapter 1 — مقدمه (7pp) · Claude chat

| § | Title | pp | Status | Pages banked | Note |
|---|---|---|---|---|---|
| 1-1 | انگیزه و اهمیت موضوع | 2 | — | | |
| 1-2 | بیان مسئله | 1 | — | | |
| 1-3 | سؤالات پژوهش | 1 | **BLOCKED** | | needs RQ1–RQ3 verbatim |
| 1-4 | نوآوری‌ها و دستاوردها | 1 | — | | |
| 1-5 | ساختار پایان‌نامه | 1 | — | | |
| 1-6 | محدوده و مفروضات | 1 | — | | |

## Chapter 2 — مروری بر پیشینه (17pp) · Claude chat

Each unit needs its **source map approved first** (handoff §15.5), then the
prose. Two gates per subsection, not one.

| § | Theme | pp | Map | Prose | Note |
|---|---|---|---|---|---|
| 2-1 | reviews / bibliometric | 2 | — | — | |
| 2-2 | classical / statistical | 2 | — | — | |
| 2-3 | classical ML | 2 | — | — | |
| 2-4 | deep learning | 2 | — | — | |
| 2-5 | hybrid / attention | 2 | — | — | |
| 2-6 | explainability | 2 | — | — | |
| 2-7 | applied / market-specific (incl. Iran) | 2 | — | — | |
| 2-8 | long-term / probabilistic | 2 | — | — | |
| 2-9 | شکاف پژوهشی و جایگاه کار | 2 | — | — | positioning |

## Chapter 3 — روش تحقیق (37pp) · Claude Code

| § | Title | pp | Status | Pages banked | Note |
|---|---|---|---|---|---|
| 3-1 | مقدمه | 1 | — | | |
| 3-2 | فرضیات بنیادی | 3 | — | | no RQs needed — best starting point |
| 3-3-1 | بنچمارک EPEX-DE | 3 | — | | |
| 3-3-2 | داده زنده Energy-Charts | 2 | — | | CC BY 4.0 attribution required |
| 3-3-3 | تحلیل اکتشافی داده | 3 | — | | figures 01–09 |
| 3-4 | مهندسی ویژگی | 7 | — | | must justify `exog_*_D0` legality |
| 3-5 | چارچوب اعتبارسنجی و معیارها | 4 | `draft-md` | | `3-5-evaluation-framework.md` |
| 3-6 | مدل‌های پایه | 4 | `draft-md` | | `3-6-baseline-models.md` |
| 3-7-1 | LightGBM | 3 | `draft-md` | | `3-7-1-lightgbm.md` |
| 3-7-2 | LSTM | 4 | `draft-md` | | `3-7-2-lstm.md` |
| 3-8 | مدل ترکیبی | 3 | — | | threshold `62.6989`, label **stressed** |

**~15pp already drafted.** Converting 3-5, 3-6, 3-7-1, 3-7-2 is the fastest
route off a zero ledger and should be approved before any new drafting.

## Chapter 4 — نتایج و تحلیل (29pp) · Claude Code

| § | Title | pp | Status | Pages banked | Note |
|---|---|---|---|---|---|
| 4-1 | مقدمه | 1 | — | | |
| 4-2 | نتایج ساعتی | 6 | — | | `results_canonical` |
| 4-3 | نتایج روزانه | 5 | — | | |
| 4-4 | مستقیم در برابر تجمیعی | 2 | — | | RQ4 |
| 4-5-1 | DM — مدل‌های ما | ~1.5 | — | | HAC only |
| 4-5-2 | DM — در برابر Lago | ~1.5 | — | | seed ensemble = SUPPLEMENTARY |
| 4-6 | تفسیرپذیری SHAP | 8 | — | | largest section; figures 10–15 |
| 4-7 | پاسخ به سؤالات پژوهش | 4 | **BLOCKED** | | needs RQ1–RQ3 |

## Chapter 5 — جمع‌بندی (10pp) · Claude Code

| § | Title | pp | Status | Pages banked | Note |
|---|---|---|---|---|---|
| 5-1 | جمع‌بندی نتایج | 2 | — | | no new data |
| 5-2 | محدودیت‌ها | 2 | — | | OOD result + data limits + assumptions |
| 5-3 | ابزار PriceCast | 2 | — | | figure 16 |
| 5-4 | پیشنهادات | 2 | — | | grounded in real limitations |
| 5-5 | نتیجه‌گیری نهایی | 2 | — | | |

---

## Log

Append one dated line per verdict. Terse is fine; the table above carries the
state, this carries the history.

```
2026-08-30  tracker created; nothing approved yet
```
