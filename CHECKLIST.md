# What's left — prioritized checklist

Generated 2026-08-05, refreshed **2026-08-30** (week 8 of 12). The single
actionable list; `NEXT_SESSION.md` is the context handoff and
`thesis/page-quota.md` is the pace tracker.

**The whole picture in one line:** all code is done, frozen and now fully
green (481 tests), 0 of 100 pages are written, and two supervisor dates do
not move.

| | Date | Days from 2026-08-30 | Needs |
|---|---|---|---|
| Week-9 partial review | **2026-08-31** | **1** | ~60pp |
| Week-10 full-draft review | **2026-09-07** | 8 | 100pp |
| Schedule end | 2026-09-27 | 28 | revisions, article, defense |

> ⛔ **Read that table again.** The partial review is *tomorrow* and the ledger
> holds **0 banked pages** against a ~60pp expectation. The 2026-08-05 plan
> targeted 66pp by Aug 28; the intervening three weeks went into code
> (power-engineering features, the ablation study and its controls, the
> benchmark pin, the MVP dashboard) — all of it real and all of it now closed,
> but none of it pages. Nothing in this file can fix that; only writing can.
> The technical side is done, so from here the only P1 is prose.

---

## P0 — Blocking. Do before writing the sections they gate.

- [ ] **Reconcile RQ1–RQ3 wording with the approved proposal.** No longer
      blocking: a defensible working set is adopted in
      `thesis/WRITING_HANDOFF.md` §7, each bounded to the frozen evidence
      with an explicit "not a claim" line. Sections **1-3** and **4-7** can
      be drafted now; check the wording against the proposal at review and
      rule then on whether the OOD result becomes RQ5.
- [ ] **Apply the §16A examiner-pass findings when drafting.** One is
      mandatory: LSTM vs LightGBM (raw p = 0.0405) must NOT be called
      significant — Holm-corrected 0.1215 over the 21-test family. Declare
      the confirmatory/exploratory family split in 3-5.
- [ ] **Calibrate `words_per_page`** in `configs/schedule.yaml` (currently a
      placeholder 250). Put one section into the Amirkabir docx, divide its
      word count by the pages it actually occupied, set the real value.
      Until then the drafts estimate is noise.

## P1 — The writing. 93% of the remaining work.

Order follows dependency: chapters 3 and 4 wait on nothing because every
number and figure they need is frozen.

### Chapter 3 — روش تحقیق (37pp) · target Aug 5–17

- [ ] 3-1 مقدمه — 1pp
- [ ] 3-2 فرضیات بنیادی — 3pp · the six formal assumptions. Address the honest
      tension: the OOD result challenges assumptions (4) generalization and
      (5) stable market conditions.
- [ ] 3-3-1 بنچمارک EPEX-DE — 3pp · train/test spans, price statistics,
      negative-price counts
- [ ] 3-3-2 داده زنده Energy-Charts — 2pp · CC BY 4.0 attribution required
- [ ] 3-3-3 تحلیل اکتشافی داده — 3pp · figures 01–09 already exported
- [ ] 3-4 مهندسی ویژگی — 7pp · the 247 columns; why `exog_*_D0` is legal
- [ ] 3-5 چارچوب اعتبارسنجی و معیارها — 4pp · **draft exists**:
      `thesis/drafts/3-5-evaluation-framework.md`. Includes why not MAPE.
- [ ] 3-6 مدل‌های پایه — 4pp · **draft exists**:
      `thesis/drafts/3-6-baseline-models.md`
- [ ] 3-7-1 LightGBM — 3pp
- [ ] 3-7-2 LSTM — 4pp
- [ ] 3-8 مدل ترکیبی — 3pp · threshold is **62.6989**, label is **stressed**.
      Disclose that ensemble weights were fitted on the members' tuning window.

### Chapter 4 — نتایج و تحلیل (29pp) · target Aug 18–28

- [ ] 4-1 مقدمه — 1pp
- [ ] 4-2 نتایج ساعتی — 6pp
- [ ] 4-3 نتایج روزانه — 5pp
- [ ] 4-4 مستقیم در برابر تجمیعی — 2pp · RQ4: aggregation wins for LEAR-LASSO,
      LightGBM, LSTM; SARIMAX is the exception
- [ ] 4-5 آزمون دیبولد-ماریانو — 3pp · HAC-corrected only. LSTM vs LEAR-LASSO
      is a **tie** (p=0.404).
- [ ] 4-6 تفسیرپذیری SHAP — 8pp · largest section. Figures 10–15 exported.
      Lead finding: `price_D-1` +77% under stress.
- [ ] 4-7 پاسخ به سؤالات پژوهش — 4pp · answers in HANDOFF §6;
      follow the §16A wording rules (nulls as findings, Holm where required)

**→ 66pp by Aug 28, clearing the 60pp partial review on Aug 31 with slack.**

### Chapter 5 — جمع‌بندی (10pp) · target Aug 29 – Sep 1

- [ ] 5-1 جمع‌بندی نتایج — 2pp
- [ ] 5-2 محدودیت‌ها — 2pp · the OOD stress test lives here
- [ ] 5-3 ابزار PriceCast — 2pp · needs the screenshot in P2
- [ ] 5-4 پیشنهادها — 2pp
- [ ] 5-5 نتیجه‌گیری نهایی — 2pp

### Chapters 1 and 2 (24pp) · target Sep 2–7

- [ ] 1-1 انگیزه و اهمیت — 2pp
- [ ] 1-2 بیان مسئله — 1pp
- [ ] 1-3 سؤالات پژوهش — 1pp · RQs adopted, HANDOFF §7
- [ ] 1-4 نوآوری‌ها — 1pp
- [ ] 1-5 ساختار پایان‌نامه — 1pp
- [ ] 1-6 محدوده و مفروضات — 1pp
- [ ] 2-1 … 2-8 — ~2pp each across the 8 Zotero subcollections
- [ ] 2-9 شکاف پژوهشی و جایگاه کار — 2pp

> ⚠ **The one scheduling risk worth acting on.** Chapter 2 is 17pp and is the
> *only* chapter that depends on nothing frozen — it needs literature, not
> results. Leaving it until Sep 2–7 concentrates 24pp into the final 6 days at
> ~4pp/day, above the average pace, right before the full-draft review.
> **Recommendation:** start the chapter-2 reading and note-taking now, in
> parallel, on days when the results chapters stall. It is the only work that
> can be moved earlier without waiting on anything.

### Ongoing

- [ ] Log pages daily: `./.venv/Scripts/python.exe scripts/page_quota.py --add N --note "3-3"`
      The tracker warns if the ledger goes 2 days stale.

## P2 — Small and closable. An hour each, at most.

- [x] **Thesis 5-3 screenshot** → `reports/figures/16_pricecast_screenshot.png`.
      Done 2026-08-05 (commit `9d14fed`): captured with both the accuracy
      warning and the forecast-vs-actual chart in frame, plus the sidebar
      showing the cached-demo source and 173 forecastable days.
- [x] **Decide the week-7 pre-freeze reproducibility check.** Decided: run it
      late. Done 2026-08-07 — fresh venv, fresh benchmark download, naive exact
      match and LEAR-LASSO matching within 1e-12 over all 728 origins. Logged in
      `logs/decisions.md` as a sanity check explicitly *not* a pre-freeze gate,
      since it ran after the 08-04 freeze.
- [x] **Fix the stale data-source test table** in `logs/decisions.md`. Done
      2026-08-06. Note the scope was wider than this item stated: **four** rows
      carried `Scheduled`, not one. Rows 4, 8/11 and 11 were all already
      satisfied (2026-07-28, 2026-08-04 `v1.1-ood`, 2026-08-05 `4243571`) and
      now cite their evidence; row 7 is resolved by the reproducibility check
      below.

- [x] **Promote the seed-ensemble result** (added 2026-08-20). Was reachable
      only from a commit message, with its prediction frames gitignored on one
      disk. Now: frames committed, `reports/tables/seed_ensemble.{csv,tex}`
      exported for new thesis section 4-5-2, numbers pinned by
      `tests/test_seed_ensemble_table.py`, branch merged to `main`. Kept
      SUPPLEMENTARY — no re-freeze, no frozen table touched.
      See `logs/decisions.md` 2026-08-20.

## P3 — Unscheduled deliverables. Decide *when*, not *whether*.

- [ ] **English journal article.** An approved deliverable with **no slot
      anywhere in the 12-week plan**. Assign one — realistically after the
      full draft, in the Sep 8–27 window.
- [ ] **`defense/` assets.** Directory is empty, also unscheduled. Same window.
      Confirm defense format/duration and whether a live demo is permitted (the
      cached-demo mode exists so PriceCast cannot fail on the day).

## P4 — Optional. My recommendation is to decline.

- [ ] **France stretch** (`dataset='FR'`). Config change only, but it produces a
      second results set needing its own tables, DM tests and prose, and it
      competes directly with 3pp/day. Logged as third priority from the start,
      "skip unless genuinely ahead". Being ahead *on code* is not the same as
      being ahead — the writing is at zero.
- [ ] **Skill observation backlog** — 22 open, never reviewed. Worth an hour
      someday, not ahead of pages.

---

## Done — for reference, not action

- [x] Data verification, EDA, feature pipeline, evaluation framework
- [x] All five models + static and regime-aware ensembles
- [x] Week-5 checkpoint → Plan B leads
- [x] Walk-forward results, hourly + daily, direct + aggregated
- [x] Diebold–Mariano with HAC correction and bootstrap cross-check
- [x] **`v1.0-results` freeze** (2026-08-04, 19 days early)
- [x] **`v1.1-ood`** OOD stress test on live 2026 data
- [x] Debug sweep: ~30 defects closed test-first; suite 122 → 189
- [x] **SHAP interpretability, section 4-6** (2026-08-05, 19 days early)
- [x] **PriceCast MVP**, week-11 live-path test closed (2026-08-05, ~6 weeks early)
- [x] Page quota configured and tracked
- [x] Writing handoff, numbers verified against the frozen tables
- [x] Everything pushed to GitHub; both freeze tags on the remote

### Added since the 2026-08-05 generation

- [x] **Power-engineering physical features** + `v1.2-physical-features` tag.
      Six-block ablation with significance testing; merged to `main` 2026-08-29.
- [x] **Ablation controls.** B5's spike gain survived three refutation attempts
      (penalty artefact, year interaction, scrambled-year control) — see
      `logs/decisions.md` 2026-08-27..29.
- [x] **Benchmark reproducibility gap closed** (2026-08-29). `data/raw/DE.csv`
      pinned with a sha256 provenance record and `-text` in `.gitattributes`;
      the tags themselves ship no benchmark data, which is now stated in the
      provenance and pinned by `tests/test_benchmark_provenance.py`.
- [x] **MVP results dashboard** (2026-08-30, branch `mvp-dashboard`). Offline,
      self-contained, Persian/RTL; consumer-only over byte-pinned copies of the
      frozen ablation cache. Pinned by `tests/test_dashboard_generator.py`.
- [x] **Full technical checkup** (2026-08-30). 481 tests green, no frozen
      artifact modified since `v1.0-results`, no debt markers in the tree.

Suite: **481 tests** (474 offline + 7 network), 0 skipped. Working tree clean.
Run it with the project interpreter — `./.venv/Scripts/python.exe -m pytest`.
A bare `python` resolves to the Windows Store build, which lacks `lightgbm`
and fails collection on 16 modules.
