---
name: thesis-fa-latex
description: Write or edit the Farsi thesis body in the AUTthesis (Amirkabir) XeLaTeX class. Use for any .tex under thesis/latex/, any Farsi thesis prose, and before quoting any number into the thesis. Covers the mandatory \lr{} numeral rule, the template fixes this machine needs, and where the frozen numbers live.
---

# Writing the Farsi thesis in LaTeX

The thesis body is Farsi, RTL, and compiled with **XeLaTeX** against the
official `AUTthesis` class. Chapters 3–5 are written in this repo; chapters 1–2
are drafted elsewhere and pasted in.

Everything below was established empirically on 2026-08-30 by compiling and
visually inspecting the output. None of it is inferred.

## RULE 1 — Every numeral goes inside `\lr{}`. No exceptions.

This is the single most dangerous thing in the project. It fails **silently**:
it compiles clean, emits no warning, and produces plausible-looking Persian
digits that are simply the wrong number.

```latex
% WRONG — renders as 90.3
خطای MAE برابر 3.90 است.

% RIGHT — renders as 3.90
خطای \lr{MAE} برابر \lr{3.90} است.
```

Proven, by compiling both forms side by side in the real class:

| Source | Bare in Farsi | Inside `\lr{}` |
|---|---|---|
| `62.6989` | `۶۹۸۹.۶۲` → 6989.62 ✗ | `62.6989` ✓ |
| `0.404` | `۴۰۴.۰` → 404.0 ✗ | `0.404` ✓ |
| `3.90` | `۹۰.۳` → 90.3 ✗ | `3.90` ✓ |

**Why:** the decimal point is a bidi-neutral character. The renderer reads it
as a sentence-ending full stop, splits `3.90` into the runs `3` and `90`, and
reorders them right-to-left into `90.3`.

Applies to: every metric, p-value, threshold, count, year, date, and price.
Latin identifiers (`MAE`, `LEAR-LASSO`, `Diebold-Mariano`, `p`) also take
`\lr{}` — they render correctly without it, but wrapping keeps them
typographically consistent and immune to the same class of bug.

Integers with no decimal point survive unwrapped. **Wrap them anyway** — the
rule is only enforceable if it has no exceptions to remember.

Deliberate exception: Persian digits written as literal Persian characters
(`۱۳۹۹`) are fine bare. Use those only for Persian calendar years and page
numbers, never for a measured quantity.

**Headings too — and there it fails differently.** Latin text in a
`\section{}`/`\subsection{}` is not reordered, it is *missing*: the heading
font has no Latin glyphs, so the table of contents prints tofu boxes.

```latex
\subsection{بنچمارک EPEX-DE}          % ToC: بنچمارک □□-□□□□
\subsection{بنچمارک \lr{EPEX-DE}}     % ToC: بنچمارک EPEX-DE
```

Compiles clean with zero warnings either way. The only detection is to render
the contents page and look at it.

## RULE 2 — Numbers are read from disk, never recalled

Every number in chapters 3–5 exists in a frozen file. Read it, then wrap it.

| Section | Source file |
|---|---|
| 4-2, 4-3, 4-4 | `reports/tables/results_canonical.csv` |
| 4-5-1 | `reports/tables/dm_tests.csv`, `dm_regime_split.csv` |
| 4-5-1 (robustness) | `reports/tables/dm_bootstrap_sensitivity.csv` |
| 4-5-2 | `reports/tables/lago_comparison.csv`, `dm_vs_lago.csv`, `seed_ensemble.csv` |
| 4-6 | `reports/tables/shap_importance.csv`, figures 10–15 |
| 5-2 (OOD) | `reports/tables/ood_stress.csv`, `ood_recalibration.csv` |
| 3-8 threshold | `configs/evaluation.yaml` → `regime.stress_threshold_eur_mwh` |

The `.tex` twin of each table is ready to `\input` — do not retype a table by
hand. The `.csv` twin is for quoting individual values in prose.

The frozen tables were produced under `v1.0-results` and must never be
regenerated. If a number in a draft disagrees with the file, the file is right.

## RULE 3 — Terms that must never appear

These are decided, logged, and wrong to reintroduce:

- **`84.04`** and the word **spike / جهش** as a regime label. Superseded
  2026-08-04; `84.04` left only 3 stressed validation days. The threshold is
  `62.6989` and the label is **stressed / پرتنش**. `84.04` remains valid only
  as a descriptive EDA statistic, never as a switch.
- **MAPE**. Excluded by design — negative prices exist in this market. Report
  MAE, RMSE, sMAPE, rMAE only. Lago et al. print MAPE; say why it is not
  comparable rather than repeating it.
- **Uncorrected `epftoolbox` p-values**, and any claim of **1% significance**.
  DM tests are HAC-corrected; quote those.

## Template setup

`thesis/latex/` holds the official class. Two fixes are required or it will
not build on this machine — both are already applied and commented in place:

1. `\setdigitfont{PGaramond}` → **`{Persian Modern}`**. PGaramond is not
   installed; XeTeX silently falls back to METAFONT and the build dies with
   `Missing argument to beginchar`. Install the font with
   `mpm --install=persian-modern` if a fresh machine lacks it.
2. **`booktabs` and `multirow` added to `commands.tex`** — every exported table
   needs `\toprule`/`\midrule`/`\multirow`. They must be loaded **before**
   `xepersian`, which must stay the last package: it patches everything before
   it.

Fonts the class expects, all present: `B Nazanin` (text),
`Times New Roman` (Latin), `Persian Modern` (digits).

## Building

```bash
"C:/Program Files/MiKTeX/miktex/bin/x64/xelatex.exe" -interaction=nonstopmode AUTthesis.tex
```

Run twice for cross-references, plus `biber`/`bibtex` for the bibliography.
Never trust exit code alone — a clean exit with mangled digits is the exact
failure this skill exists to prevent. **Render and look at the page.**

## Tables and figures

- `\input` the frozen `.tex`; never retype.
- The table body is LTR. Wrap it in `\begin{latin}...\end{latin}`, but keep
  the **caption outside** that environment, or it prints `Table 3-1` in English
  instead of `جدول ۳-۱`.
- Let the class number captions. Do not hardcode numbers.
- Figures live in `reports/figures/` and are exported once, in final captioned
  form — see the `export-results` skill.

## Citations

The bibliography is `thesis/latex/Latex template 2018/references.bib` —
**41 entries**, derived from the Zotero export in `Thesis References/` with
three corrections documented in its header. A fresh Zotero export undoes them.

- **Style is `unsrt-fa`** (changed from the template's `plain-fa`): the
  reference list is ordered by **first appearance in the text**, not
  alphabetically. LaTeX renumbers on every build — never hand-number a
  reference, never reorder the `.bib`. Entry order in the `.bib` is irrelevant.
- Cite with `\cite{exact_key}`. A typo'd key silently prints `[?]`.
- **Cite only from those 41 keys.** Never invent a reference, DOI, year or
  author. Never state what a paper found unless that text is in front of you —
  summarising a paper from its title is fabrication.
- If a claim needs a source that is not in the library, write
  `[NEEDS CITATION]` and stop. Do not attribute it to the nearest plausible
  paper; a real citation on a claim the source does not make is worse than no
  citation.
- **Translation is not paraphrase.** The sources are English and the thesis is
  Farsi; a translated sentence still belongs to its author and still needs its
  citation. If a sentence tracks the original phrase-for-phrase, rewrite it.
- Cite at sentence/clause level, not once per paragraph. Your own results take
  no citation — point at your own table or figure.

`persian-bib` supplies `unsrt-fa.bst` and is installed. Build in four steps:

```
xelatex AUTthesis && bibtex AUTthesis && xelatex AUTthesis && xelatex AUTthesis
```

Skipping `bibtex` reuses the old `.bbl`. The template ships one full of
unrelated Finsler-geometry papers, so the document **compiles fine while
showing someone else's bibliography**. Always check the `.blg`.

**BibTeX does not honour `%` comments.** In a `.bib`, an `@` always starts an
entry — even inside a comment, even inside `@comment{...}`. A header note that
spelled out entry types with their `@` prefix made BibTeX skip real records
with only `I'm skipping whatever remains of this entry` as a clue. The header
of `references.bib` therefore contains no `@` characters at all; keep it that
way.

## Farsi typography

- Use ZWNJ (نیم‌فاصله, U+200C) for prefixes and plurals: `می‌شود`, `داده‌ها`,
  `پیش‌بینی` — not `می شود` and not `میشود`.
- Latin technical terms stay Latin inside `\lr{}`; give the Persian gloss on
  first use, then use one form consistently.
- Section numbering is the class's job. Write `\section{...}`, not `3-5`.

## Section map

`thesis/outline.md` is the authority for which section a result belongs to and
its page budget. Read it before writing; never guess a section number.

## One subsection at a time — stop for approval

**The author approves every part before the next begins.** Hard gate.

- The unit is a subsection (`3-2`, `3-3-1`, `4-5-1`), never a whole chapter.
- Draft one unit, present it, **stop**. Do not draft ahead or start the next.
- Approval on one unit never extends to the next.
- Record the verdict in `thesis/APPROVALS.md` in the same turn it is given —
  that file is the only state shared with the Claude chat sessions writing
  chapters 1–2.

A wrong number or a mis-attributed source is a paragraph to fix inside one
reviewed subsection and a rewrite after eight unreviewed ones. `unsrt-fa` also
numbers references by first appearance, so inserting an approved-late section
renumbers everything after it — sequential approval keeps that stable.

## Before claiming a section is done

1. Compile.
2. Render the changed pages to PNG and **look at them** (`pymupdf` is in
   `.venv`).
3. Check every numeral against its source file.
4. Bank the pages:
   `./.venv/Scripts/python.exe scripts/page_quota.py --add N --note "3-5"`
   — or write a dated deferral line in `logs/decisions.md`. Never neither.
