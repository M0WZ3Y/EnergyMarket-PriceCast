# Thesis writing handoff — read this before drafting any chapter

**Project:** Day-ahead electricity price forecasting for the German market.
**Deliverable:** 100-page Farsi (Persian) MSc thesis, Amirkabir University of
Technology, compiled with XeLaTeX in the official `AUTthesis` class.
**Generated:** 2026-08-30, after the technical phase was closed.

---

## 0. How to use this file

Read the whole thing once. Then, before each chapter, re-read §1 (rules), §2
(the project), and the per-chapter brief in §10.

**Two audiences, and the difference matters:**

- **Claude chat — chapters 1 and 2.** Chat has **no access to the repository**.
  Every number, name and date it needs is written out inline in this file. If
  a fact is not in here, it is not available in chat — ask for it rather than
  reconstructing it. Paste this file at the start of the conversation.
- **Claude Code — chapters 3, 4, 5.** Reads the repo directly and must quote
  numbers **from the frozen files**, not from this summary. This file is the
  map; the files are the territory. Where the two disagree, the files win and
  this file is wrong and should be fixed.

**The single most important instruction in this document is §1.1.** If you
read nothing else, read that.

---

## 1. Non-negotiable writing rules

### 1.1 Every numeral must be wrapped in `\lr{}`

The thesis is right-to-left. A bare decimal in Farsi text is **silently
corrupted**: it compiles clean, raises no warning, and prints a plausible but
wrong number. This was verified by compiling both forms in the real template
and inspecting the rendered PDF:

| You write | Bare in Farsi text | Wrapped in `\lr{}` |
|---|---|---|
| `62.6989` | `۶۹۸۹.۶۲` → **6989.62** ✗ | `62.6989` ✓ |
| `0.404` | `۴۰۴.۰` → **404.0** ✗ | `0.404` ✓ |
| `3.90` | `۹۰.۳` → **90.3** ✗ | `3.90` ✓ |

The decimal point is bidi-neutral; the renderer treats it as a full stop,
splits the number into two runs, and reorders them RTL.

```latex
% WRONG
خطای MAE مدل LEAR-LASSO برابر 3.90 است.

% RIGHT
خطای \lr{MAE} مدل \lr{LEAR-LASSO} برابر \lr{3.90} است.
```

Wrap **every** numeral — metrics, p-values, counts, years, dates, prices —
even integers that would survive. A rule with exceptions is a rule nobody
applies. Wrap Latin identifiers (`MAE`, `LEAR-LASSO`, `p`) too.

**The rule extends to section headings, and there it fails differently.**
Latin text in a `\section{}`/`\subsection{}` does not merely reorder — the
heading font has no Latin glyphs, so the table of contents prints **tofu
boxes**:

```latex
\subsection{بنچمارک EPEX-DE}          % ToC shows: بنچمارک □□-□□□□
\subsection{بنچمارک \lr{EPEX-DE}}     % ToC shows: بنچمارک EPEX-DE
```

Six headings in the scaffold hit this (`EPEX-DE`, `Energy-Charts`,
`LightGBM`, `LSTM`, `SHAP`, `EnergyMarket-PriceCast`) and are now fixed. It
compiled cleanly with zero warnings in both states — the only way to catch it
is to look at the rendered contents page.

*(If drafting in chat as plain Persian prose rather than LaTeX, write numbers
normally — the wrapping is applied when the text is moved into the `.tex`.
But flag it, because whoever pastes it must do the wrapping.)*

### 1.2 Never invent a number

Every quantity in this thesis exists in a frozen file. Chapter 3–5 drafting in
Claude Code must read it. Chapter 1–2 drafting in chat must use §5–§7 of this
file verbatim. **Do not round, restate from memory, or "approximately"
anything.** A wrong number in a results chapter is the one error that can sink
a defense.

### 1.3 Terms that must never appear

| Forbidden | Use instead | Why |
|---|---|---|
| `84.04` as a regime threshold; the word **spike / جهش** as a regime label | `62.6989`; the label **stressed / پرتنش** | Superseded 2026-08-04. At 3σ only 3 stressed validation days remained — too few to fit a weight set. `84.04` is still valid as a *descriptive* EDA statistic, never as a switch. |
| **MAPE** | MAE, RMSE, sMAPE, rMAE | Negative prices exist in this market; MAPE is undefined/unstable. Lago et al. print it and flag it themselves as unreliable here. |
| Uncorrected `epftoolbox` p-values; any claim of **1% significance** | HAC-corrected Diebold-Mariano p-values | Autocorrelated forecast errors inflate significance without the correction. |
| "our model beats the state of the art" | See §7.3 — we tie two of their variants and lose to their best | Overclaiming against a published benchmark is the fastest route to a hostile defense. |

### 1.4 Persian typography

Use ZWNJ (نیم‌فاصله, U+200C): `می‌شود` not `می شود`/`میشود`; `داده‌ها`,
`پیش‌بینی`, `پیش‌بینی‌ها`. Keep Latin technical terms in Latin script, give the
Persian gloss on first use, then stay consistent.

### 1.5 One part at a time — stop and wait for approval

**The author approves every part before the next one begins.** This is a hard
gate, not a courtesy.

- The unit is a **subsection** (`3-2`, `3-3-1`, `4-5-1`), not a chapter.
- Draft **one** unit, present it, then **stop**. Do not continue to the next
  unit, do not draft ahead, do not offer the next one half-written.
- "Looks good" on 3-2 approves **3-2 only**. Approval never extends forward.
- If a unit is too large to review in one sitting, split it and say so — but
  still stop at the boundary you chose.
- Record the outcome in `thesis/APPROVALS.md` before moving on.

**Why this is worth the friction.** 100 pages of Farsi built on frozen numbers
and 41 citations has two failure modes that only get more expensive with
distance: a wrong number and a mis-attributed source. Both are cheap to fix in
one reviewed subsection and brutal to fix after eight unreviewed ones, because
later sections come to depend on earlier phrasing, claims and citation
numbering. Catching a problem in 3-4 is a paragraph; catching the same problem
after 4-6 is a rewrite.

It also keeps the citation ledger honest: `unsrt-fa` numbers references by
first appearance, so inserting an approved-late section reshuffles every
number after it. Sequential approval keeps that ordering stable and meaningful.

### 1.6 Write the tensions, do not hide them

This project has four genuinely awkward results (§8). A thesis that states
them plainly and explains them is stronger than one that buries them — and
every one of them is discoverable by an examiner in ten minutes. They are
assets, not liabilities: they are what makes this an honest empirical study
rather than a leaderboard entry.

---

## 2. The project in one page

Day-ahead electricity price forecasting for Germany, on two targets:

- **Hourly** — a 24-price vector for day D+1.
- **Daily baseload** — the 24-hour average, produced **two ways**: trained
  directly on the daily series (*direct*), and by averaging the hourly
  forecasts (*aggregated*). Comparing the two **is** research question 4.

**Five models, fixed list, no additions:** naive, SARIMAX, LEAR-LASSO,
LightGBM, LSTM — plus two ensembles (static, and regime-aware). Random
Forest, XGBoost, SVR and GRU were deliberately cut and must not reappear.

**Two contributions beyond a standard benchmark study**, both formally
sanctioned (2026-07-11):

1. **Regime-aware ensemble weighting** — separate weight sets for calm and
   stressed days, switched on a price threshold.
2. **Out-of-distribution stress test** — the frozen 2016–2017-era models
   evaluated on live 2026 data. The result is negative and is the most
   interesting finding in the thesis (§7.5).

Plus a SHAP interpretability analysis (calm vs stressed, hourly vs daily) and
a deployed tool, **EnergyMarket-PriceCast**.

Everything is frozen behind three git tags: `v1.0-results`, `v1.1-ood`,
`v1.2-physical-features`. Results are never regenerated.

---

## 3. Thesis structure, page budget, and status

Total 100pp. **0 pages are currently banked.** The authority on section
numbering is `thesis/outline.md`.

| Ch | Title | pp | Who writes it | Status |
|---|---|---|---|---|
| 1 | مقدمه | 7 | **Claude chat** | Not started; 1-3 uses the RQs in §7 |
| 2 | مروری بر پیشینه پژوهش | 17 | **Claude chat** | Not started |
| 3 | روش تحقیق | 37 | **Claude Code** | ~15pp drafted (3-5, 3-6, 3-7-1, 3-7-2) — see §10 |
| 4 | نتایج و تحلیل | 29 | **Claude Code** | Not started; 4-7 answers §7 under §16A |
| 5 | جمع‌بندی، بحث و پیشنهادات | 10 | **Claude Code** | Not started |

**Chapter 1 (7pp)** — 1-1 انگیزه و اهمیت [2] · 1-2 بیان مسئله [1] ·
1-3 سؤالات پژوهش [1] · 1-4 نوآوری‌ها [1] · 1-5 ساختار پایان‌نامه [1] ·
1-6 محدوده و مفروضات [1]

**Chapter 2 (17pp)** — eight subsections of ~2pp each, mapping to eight Zotero
subcollections: 2-1 reviews/bibliometric · 2-2 classical/statistical ·
2-3 classical ML · 2-4 deep learning · 2-5 hybrid/attention ·
2-6 explainability · 2-7 applied/market-specific (incl. Iran) ·
2-8 long-term/probabilistic · then 2-9 شکاف پژوهشی و جایگاه کار [2pp].

**Chapter 3 (37pp)** — 3-1 مقدمه [1] · 3-2 فرضیات بنیادی [3] ·
3-3 داده‌ها [8: 3-3-1 بنچمارک [3], 3-3-2 داده زنده [2], 3-3-3 تحلیل اکتشافی [3]] ·
3-4 مهندسی ویژگی [7] · 3-5 چارچوب اعتبارسنجی و معیارها [4] ·
3-6 مدل‌های پایه [4] · 3-7 مدل‌های ML/DL [7: LightGBM [3], LSTM [4]] ·
3-8 مدل ترکیبی [3]

**Chapter 4 (29pp)** — 4-1 مقدمه [1] · 4-2 نتایج ساعتی [6] ·
4-3 نتایج روزانه [5] · 4-4 مستقیم در برابر تجمیعی [2] ·
4-5 آزمون دیبولد-ماریانو [3: 4-5-1 ours, 4-5-2 vs Lago + seed ensemble] ·
4-6 تفسیرپذیری SHAP [8] · 4-7 پاسخ به سؤالات پژوهش [4]

**Chapter 5 (10pp)** — 5-1 جمع‌بندی [2] · 5-2 محدودیت‌ها [2, incl. OOD] ·
5-3 ابزار PriceCast [2] · 5-4 پیشنهادات [2] · 5-5 نتیجه‌گیری [2]

---

## 4. Data

### 4.1 Benchmark (the thesis's primary data)

- **Market:** EPEX-DE, via the `epftoolbox` open benchmark dataset (Lago et
  al.). Dataset code `DE`.
- **Span:** 2012-01-09 → 2017-12-31. 52,416 hourly rows.
- **Columns:** `price`, plus two exogenous series `exog_1` and `exog_2`
  (day-ahead load forecast and day-ahead renewable generation forecast — both
  published *before* the forecast origin, which is why using their D0 values
  is legal).
- **Test period:** 2016-01-04 → 2017-12-31 = **728 forecast origins**,
  17,472 hourly predictions per model. Held-out, 2 years, per the Lago
  protocol.
- **Feature matrix:** 2,177 rows × **247 columns**.
- **Provenance:** pinned as an immutable snapshot with a sha256 record
  (`data/raw/DE.csv`, `data/raw/provenance_benchmark.json`). A fresh download
  on 2026-08-29 was byte-identical.

### 4.2 Live data (used only for the OOD test and the tool)

- **Source:** Energy-Charts API (Fraunhofer ISE), keyless, **CC BY 4.0**.
- **Required attribution, must appear in the thesis:**
  *Data: Energy-Charts (Fraunhofer ISE) / Bundesnetzagentur SMARD.de, CC BY 4.0*
- **Bidding zone:** `DE-LU`. Note the zone was `DE-AT-LU` before 2018-10-01 —
  a real discontinuity worth one sentence.
- **OOD window:** 173 complete days of 2026 data, 4,343 hourly rows, no gaps.

### 4.3 Data constraints to state honestly in 3-3 and 5-2

Clean spark spreads and dark spreads are **not buildable** in this thesis: gas
(TTF) and coal (API2) futures are commercially licensed (Montel/ICE), and Ember
republishes only series *derived* from them — which makes Ember a citation, not
a data source. Investing.com and Trading Economics were rejected on terms-of-
service and reproducibility grounds. This is a genuine limitation, and stating
it is better than being asked about it.

---

## 5. Methodology

### 5.1 Validation protocol

**Walk-forward (rolling-origin) with daily recalibration** — never a random
split. This mirrors `epftoolbox`/Lago et al. exactly:

- Trailing calibration window: **1,092 days** (= 364×3, the `epftoolbox` LEAR
  default).
- Step: **1 day** (daily recalibration).
- Validation window: **364 days**, carved from the end of the training split
  and lying **strictly before** the test window — asserted in code, not assumed.
- Tuning window: 2015-01-05 → 2016-01-03.
- Ensemble weight-fitting window: 2015-01-12 → 2016-01-03.

**Hyperparameters:** Optuna, **50 trials per model**. **Seed 42 everywhere.**

**Leakage rule:** no feature may use information after the forecast origin.
Enforced by a declarative information-set guard and an assertion test, not by
convention. Feature code is also forbidden from touching the network — also
enforced by a test.

### 5.2 Metrics

MAE, RMSE, sMAPE, rMAE, and Diebold-Mariano for significance. **No MAPE**
(§1.3). rMAE is MAE divided by the MAE of a weekly-lag naive forecast (same
weekday, previous week).

### 5.3 The regime threshold (section 3-8)

- **Threshold: `62.6989` EUR/MWh** = train mean `37.6876` + `1.5` × std
  `16.6742`, computed on prices **up to and including 2015-01-04**.
- That cutoff sits strictly before *both* the tuning and weight-fitting
  windows, so the threshold cannot be contaminated by either.
- **k = 1.5 was chosen by a validation-only rule**: take the largest k in
  {3.0, 2.5, 2.0, 1.5} for which both regimes hold ≥ 20 validation days. k=2.0
  gives 10 stressed days; k=1.5 gives 37. **The test window was never
  consulted** — choosing k by test behaviour would be in-sample selection.
- Resulting split: validation **37 / 357**; test **77 stressed / 728**.
- A day is labeled stressed when the **previous** day's realized prices
  breached the threshold — enforced in code, so the label itself cannot leak.
- At ~1.5σ this marks an *elevated* day, not a price spike. Hence "stressed".

**Disclose in 3-8:** ensemble weights were fitted on the members' tuning
window.

---

## 6. Results — the frozen numbers

All from `v1.0-results`. **728 origins, EPEX-DE, 2016-01-04 → 2017-12-31.**

### 6.1 Headline accuracy (MAE, EUR/MWh — lower is better)

| Model | Hourly | Daily-direct | Daily-aggregated |
|---|---|---|---|
| naive | 7.75 | 6.36 | 6.36 |
| SARIMAX | 4.35 | **3.20** | 3.27 |
| LEAR-LASSO | 3.90 | 2.90 | 2.84 |
| LightGBM | 3.97 | 3.30 | 2.99 |
| LSTM | 3.87 | 3.18 | 2.78 |
| Ensemble (static) | 3.57 | — | 2.68 |
| **Ensemble (regime-aware)** | **3.56** | — | **2.65** |

Ensembles have no daily-direct arm — their members are hourly.

Full precision, plus RMSE / sMAPE / rMAE, in
`reports/tables/results_canonical.csv`.

Selected rMAE (hourly): naive 0.849, SARIMAX 0.477, LEAR-LASSO 0.427,
LightGBM 0.435, LSTM 0.424, static 0.392, regime-aware 0.390.

### 6.2 RQ4 — direct vs aggregated daily baseload

**Aggregation wins for three of four models; SARIMAX is the exception.**

| Model | Direct | Aggregated | Winner |
|---|---|---|---|
| SARIMAX | 3.20 | 3.27 | **direct** |
| LEAR-LASSO | 2.90 | 2.84 | aggregated |
| LightGBM | 3.30 | 2.99 | aggregated |
| LSTM | 3.18 | 2.78 | aggregated |

The naive model is identical either way. The SARIMAX exception is a real
result, not noise — explain it rather than averaging it away.

### 6.3 Diebold-Mariano, our own models (section 4-5-1)

HAC-corrected, one-sided. p is for "row model is better than column model".

- **LSTM vs LEAR-LASSO: p = 0.404 → a tie.** The best statistical model and
  the best deep model are statistically indistinguishable. This is a headline
  finding, not a disappointment.
- LightGBM vs LEAR-LASSO: p = 0.734 → no difference.
- **LSTM vs LightGBM: raw p = 0.0405, but Holm-corrected p = 0.1215 over
  the 21-test matrix — do NOT call this significant.** Write it as a lower
  point estimate that does not survive correction for multiplicity. See
  §16A/A1.
- All models beat naive at p ≈ 0 (survives correction).
- Both ensembles beat every member at Holm p ≤ 3.4e-05 (survives).
- Both ensembles beat every individual member (p < 1e-5).
- **Regime-aware vs static ensemble: p = 0.0226** → significant at 5%.

### 6.4 Where the regime-aware gain actually comes from

| Subset | Days | MAE regime-aware | MAE static | DM p (HAC) |
|---|---|---|---|---|
| all | 728 | 3.5569 | 3.5742 | **0.0226** |
| stressed | 77 | 5.5132 | 5.6830 | **0.0063** |
| calm | 651 | 3.3255 | 3.3248 | 0.847 |

**The entire gain is on the 77 stressed days.** On calm days the regime-aware
ensemble is very slightly *worse* and utterly insignificant. This is exactly
what the method was designed to do, and saying so is the strongest version of
the claim.

**Robustness — state this honestly.** A block-bootstrap cross-check gives, over
block lengths 3–10 days:

- all 728 days: p ranges **0.0129 → 0.0571** — *it crosses 0.05 at the longest
  blocks.*
- stressed 77 days: p ranges **0.0081 → 0.0439** — significant throughout.
- calm: p ≈ 0.84–0.85 throughout.

So the pooled result is borderline under the most conservative dependence
assumption, while the stressed-subset result holds. Write it that way.

### 6.5 Comparison against Lago et al. (2021) — section 4-5-2

Their forecasts were re-scored with **our** metric code, so the comparison is
like-for-like.

| Model | Source | MAE | rMAE |
|---|---|---|---|
| DNN Ensemble | Lago et al. | **3.413** | **0.374** |
| Ensemble (regime-aware) | **this thesis** | 3.557 | 0.390 |
| Ensemble (static) | **this thesis** | 3.574 | 0.392 |
| DNN 4 | Lago et al. | 3.592 | 0.394 |
| LEAR Ensemble | Lago et al. | 3.609 | 0.395 |
| LSTM | this thesis | 3.873 | 0.424 |
| LEAR 1092 | Lago et al. | 3.930 | 0.431 |

DM verdicts for our regime-aware ensemble:

- vs **DNN Ensemble**: **theirs better, p = 0.0127** (significant).
- vs **DNN 4**: no significant difference (p = 0.322).
- vs **LEAR Ensemble**: no significant difference (p = 0.127).
- vs **LEAR 1092**: **ours better, p = 1.0e-07**.

**The honest headline: we do not beat their best. We are statistically
indistinguishable from two of their five published variants, and we beat their
LEAR-1092 baseline.** Say precisely this.

**A genuine methodological finding worth its own paragraph:** for the LEAR
variants, Lago et al.'s **published table and their shipped forecast files
disagree**. Their LEAR Ensemble prints MAE 3.955 in the paper but scores 3.609
from the shipped forecasts; LEAR 1092 prints 4.108 but scores 3.930. The DNN
rows reproduce exactly. We used the shipped forecasts (the conservative
choice — it makes their models look *better*, not worse). This is a
reproducibility observation about the benchmark literature and belongs in the
thesis.

### 6.6 Seed ensemble — SUPPLEMENTARY, not frozen (section 4-5-2)

Averaging four LSTM seeds (42–45) instead of the single frozen seed-42 model:

| Configuration | MAE | rMAE |
|---|---|---|
| LSTM seed 42 (frozen) | 3.873 | 0.424 |
| LSTM 4-seed ensemble | 3.646 | 0.399 |
| Ensemble (static), frozen LSTM | 3.574 | 0.392 |
| Ensemble (static), seed-ensembled | 3.526 | 0.386 |
| Ensemble (regime-aware), frozen LSTM | 3.557 | 0.390 |
| Ensemble (regime-aware), seed-ensembled | **3.499** | **0.383** |

With seed ensembling, the gap to Lago's DNN Ensemble stops being significant
(p = 0.0803 vs p = 0.0127 frozen).

**This must be labelled SUPPLEMENTARY and kept visibly separate.** The
headline numbers in 4-2/4-3, the SHAP analysis in 4-6, and the OOD addendum
all remain on the **frozen seed-42 LSTM**. Say so explicitly so the two sets
of numbers are never read as one.

Oracle (best achievable static weights, for context): 3.558 with the frozen
LSTM, 3.502 with the seed-ensembled LSTM.

### 6.7 SHAP interpretability (section 4-6, 8pp — the largest section)

Mean |SHAP| per feature family, EUR/MWh:

| Feature family | Hourly calm | Hourly stressed | Daily calm | Daily stressed |
|---|---|---|---|---|
| `price_D-1` | 3.4833 | **6.1471** | 3.1109 | 5.1335 |
| `price_D-2` | 0.8610 | 1.2550 | 0.3778 | 0.5177 |
| `price_D-3` | 0.8279 | 1.0302 | 0.3328 | 0.3916 |
| `price_D-7` | 1.7454 | 2.1473 | 1.0802 | 1.4784 |
| `exog_1_D-1` | 0.6591 | 0.6735 | 0.2479 | 0.2611 |
| `exog_1_D-7` | 0.7131 | 0.7487 | 0.3699 | 0.3991 |
| `exog_1_D0` | 3.5197 | 3.3680 | 3.4393 | 3.0130 |
| `exog_2_D-1` | 0.9947 | 0.9221 | 0.4077 | 0.3986 |
| `exog_2_D-7` | 0.6740 | 0.7761 | 0.2230 | 0.2754 |
| `exog_2_D0` | **5.8422** | **5.9761** | 4.4929 | 5.0366 |
| `dow` | 0.1464 | 0.1358 | 0.0395 | 0.0379 |

**Lead findings:**

1. `price_D-1` importance rises **+76.5%** under stress (3.4833 → 6.1471). The
   model leans much harder on yesterday's price when the market is stressed.
2. `exog_2_D0` (same-day renewables forecast) is the single most important
   feature overall, and is stable across regimes.
3. `exog_1_D0` (same-day load forecast) *falls* slightly under stress — the
   mirror image of finding 1.
4. Day-of-week is near-irrelevant once lags are present.

### 6.8 OOD stress test (section 5-2) — the most important negative result

Frozen benchmark-era models, evaluated on 173 days of live 2026 data:

| Model | MAE | rMAE | MAE vs benchmark |
|---|---|---|---|
| **naive** | **29.18** | **0.808** | +3.77 |
| LEAR-LASSO | 39.27 | 1.087 | +10.07 |
| SARIMAX | 41.39 | 1.145 | +9.51 |
| Ensemble (regime-aware) | 42.17 | 1.167 | +11.86 |
| Ensemble (static) | 44.43 | 1.230 | +12.43 |
| LSTM | 54.93 | 1.520 | +14.18 |
| LightGBM | 66.06 | 1.828 | +16.65 |

**Every trained model falls below the naive benchmark.** rMAE > 1 means worse
than a weekly-lag naive forecast. The ordering also **inverts**: LightGBM and
LSTM, among the best on the benchmark, are the worst out of distribution.

**Post-hoc recalibration partially rescues it.** A rolling bias correction
with a **7-day window**, rMAE before → after, all figures on the same 166-day
subset that window supports:

| Model | rMAE raw | rMAE recalibrated |
|---|---|---|
| naive | 0.8035 | 0.8678 *(hurt)* |
| SARIMAX | 1.1547 | 1.1295 |
| LEAR-LASSO | 1.0919 | 1.1323 *(hurt)* |
| LightGBM | 1.8258 | **1.0784** |
| LSTM | 1.5354 | **1.0135** |
| Ensemble (static) | 1.2390 | **0.9064** |
| Ensemble (regime-aware) | 1.1756 | **0.8946** |

Recalibration helps most exactly where the failure was worst (LightGBM, LSTM)
and *hurts* naive and LEAR-LASSO — the two models that were least biased to
begin with.

Interpretation: the failure is dominated by a **level shift**, not a collapse
of learned structure — the models still rank hours correctly, they are simply
biased. That is a substantive, defensible finding and it directly challenges
formal assumptions (4) and (5) in §9.

---

## 7. The four research questions

**Status (2026-08-31): RQ1–RQ3 below are PROVISIONAL — reconstructed backwards
from the work that was actually done, not copied from the approved proposal.**
The proposal was never committed to this repo. Before section **1-3** or **4-7**
is drafted, the author must open the approved proposal and either confirm these
or replace them with the verbatim text. A research question the committee can
line up against the proposal has to match it word for word; a plausible
paraphrase that drifts is worse than a blank. RQ4 is not provisional — it is
inferable from the code and is what the direct/aggregated split was built for.

### What makes an RQ defensible here

You will be held to every word of these in the defense. Each one below is
therefore written to five rules — apply the same rules to any rewording you
bring back from the proposal:

1. **Bounded.** Name the models, the market, the period and the horizon inside
   the question. "Do deep models outperform statistical ones?" is a claim about
   the whole field and cannot be answered by five models on one market over two
   years. "Do these five, here, over this window?" can.
2. **Pre-committed.** The metric and the significance test are named in the
   question, not chosen after seeing the numbers. Ours were fixed in
   `configs/evaluation.yaml` before the test window was scored.
3. **A null is an answer.** Each question is phrased so "no difference" or "only
   under condition X" resolves it. A question that can only be resolved by a win
   forces you to defend a win you do not have — this is the single most common
   way an RQ becomes indefensible.
4. **Conditions are asked for, not discovered.** RQ2 asks *under which market
   conditions*, so the calm-day null is part of the answer rather than a caveat
   bolted on afterwards.
5. **Answerable from frozen files.** Every answer below traces to a file under
   `reports/tables/`, produced at `v1.0-results` and never regenerated.

Each RQ carries an explicit **Claim / Not a claim** pair. Write the "not a
claim" line into the thesis too — pre-empting the over-reading is far stronger
than being caught in it.

---

### RQ1 — comparative accuracy of the five model families

> در پیش‌بینی روز-پیشِ قیمت برق بازار \lr{EPEX-DE}، پنج مدل منتخب — ساده،
> \lr{SARIMAX}، \lr{LEAR-LASSO}، \lr{LightGBM} و \lr{LSTM} — تحت یک پروتکل
> اعتبارسنجی پیش‌رونده یکسان، بر پایه معیارهای \lr{MAE}، \lr{RMSE}، \lr{sMAPE}
> و \lr{rMAE}، چه عملکردی نسبت به یکدیگر دارند، و کدام تفاوت‌ها بر اساس آزمون
> دیبولد-ماریانو با تصحیح \lr{HAC} از نظر آماری معنادار است؟

*Under a single walk-forward protocol on the EPEX-DE benchmark, how do the five
selected models — naive, SARIMAX, LEAR-LASSO, LightGBM and LSTM — compare on
MAE, RMSE, sMAPE and rMAE, and which of the differences are statistically
significant under a HAC-corrected Diebold-Mariano test?*

**Answered in 4-2, 4-3, 4-5-1** (§6.1, §6.3) — `results_canonical.csv`,
`dm_tests.csv`. 728 origins, 2016-01-04 → 2017-12-31.

- **Claim:** on this market and window, LSTM (3.87), LEAR-LASSO (3.90) and
  LightGBM (3.97) are mutually indistinguishable once multiplicity is
  accounted for — LSTM vs LEAR-LASSO p = 0.404 outright, and the nominal
  LSTM-over-LightGBM edge (p = 0.0405) does not survive Holm correction
  (0.1215, §16A/A1). All four trained models beat naive at p ≈ 0, and both
  ensembles beat every member — both survive correction.
- **Not a claim:** that deep learning is or is not superior to statistical
  methods in general. One architecture per family, one market, one window, one
  tuning budget (50 Optuna trials each).

**Why this survives questioning.** The comparison is *fair by construction* —
identical feature matrix, identical splits, identical tuning budget, one metric
module — and the tie is reported as a finding rather than treated as a failure
to separate the models. The equal-budget design is the answer to "did you just
under-tune the LSTM?", and it is worth stating before you are asked.

### RQ2 — ensembling and regime-aware weighting

> آیا ترکیب وزن‌دار مدل‌های پایه دقت را نسبت به بهترین تک‌مدل بهبود می‌دهد، و
> آیا مشروط‌کردن وزن‌ها به رژیم بازار — که با آستانه‌ای صرفاً بر پایه داده
> اعتبارسنجی به دو حالت آرام و پرتنش تفکیک می‌شود — بهبود بیشتری نسبت به
> وزن‌دهی ایستا ایجاد می‌کند؟ این بهبود، در صورت وجود، تحت کدام شرایط بازار
> پدید می‌آید؟

*Does a weighted ensemble of the base models improve accuracy over the best
single model, and does conditioning the weights on market regime — calm vs
stressed, split by a threshold set from validation data alone — improve further
over static weighting? Under which market conditions does any such gain arise?*

**Answered in 3-8, 4-2, 4-5-1** (§6.2, §6.4) — `dm_tests.csv`,
`dm_regime_split.csv`, `dm_bootstrap_sensitivity.csv`.

- **Claim:** both ensembles beat every member (p < 1e-5). Regime-aware beats
  static at p = 0.0226 pooled, p = 0.0063 on the 77 stressed days, and is null
  on the 651 calm days (p = 0.847). **The gain is entirely stress-conditional.**
- **Not a claim:** that regime-aware weighting improves average accuracy
  generally. Pooled significance is borderline — block bootstrap gives 0.0129 →
  0.0571 across 3–10-day blocks, crossing 0.05 at the longest. Report the range.

**Why this survives questioning.** The second sentence — *under which market
conditions* — is what makes the calm-day null part of the answer instead of an
embarrassment. The method was designed to act only under stress; it does exactly
that, and the question was posed to detect it either way.

Two disclosures belong in 3-8, and volunteering them is what keeps this
defensible:

- k = 1.5 was chosen by a **validation-only** rule (largest k in {3.0, 2.5, 2.0,
  1.5} leaving ≥ 20 days in both regimes). The test window was never consulted.
  This is the answer to "did you tune the threshold on the test set?".
- Ensemble weights were fitted on the members' **tuning** window. Say it.

### RQ3 — interpretability

> بر پایه مقادیر \lr{SHAP} محاسبه‌شده برای مدل \lr{LightGBM}، کدام خانواده‌های
> ویژگی بیشترین سهم را در پیش‌بینی دارند، و آیا این سهم میان روزهای آرام و
> پرتنش، و میان افق ساعتی و روزانه، تغییر می‌کند؟

*Based on SHAP values computed for the LightGBM model, which feature families
contribute most to the forecasts, and does that contribution differ between calm
and stressed days, and between the hourly and daily horizons?*

**Answered in 4-6** (§6.7) — `shap_importance.csv`, figures 10–15.

- **Claim:** for the gradient-boosted model, `exog_2_D0` (same-day renewables
  forecast) is the largest single contributor and is regime-stable;
  `price_D-1` rises **+76.5%** under stress (3.4833 → 6.1471) while `exog_1_D0`
  (load) falls slightly; `dow` is near-irrelevant once lags are present.
- **Not a claim:** that these are "the" drivers of German electricity prices, or
  that LEAR-LASSO and the LSTM use the same information. SHAP is model-specific.
  It is an attribution of one model's output, not a causal statement about the
  market.

**Why this survives questioning — and the one thing you must disclose.** The
question names LightGBM explicitly because `shap_analysis.py` uses
`TreeExplainer` and covers only the tree arms. Equally important: the explained
model is **refit on the trailing 1092 days ending 2017-12-31**, because the
frozen walk-forward models saw those days and explaining them would be in-sample.
So the SHAP model is a faithful *twin* of the reported model, not the identical
object. State this in 4-6 in one sentence. An examiner who finds it themselves
reads it as a gap; volunteered, it reads as care — and the module refuses to fit
on a shorter window precisely so the twin cannot silently drift.

### RQ4 — direct vs aggregated daily baseload  *(not provisional)*

> برای پیش‌بینی میانگین بار پایه روزانه، کدام مسیر دقت بالاتری دارد: مدل‌سازی
> مستقیمِ هدف روزانه، یا میانگین‌گیری از ۲۴ پیش‌بینی ساعتی؟ و آیا این نتیجه
> برای هر پنج مدل یکسان است؟

*For forecasting the daily baseload average, which route is more accurate:
modelling the daily target directly, or averaging 24 hourly forecasts? And does
the answer hold for all five models?*

**Answered in 4-4** (§6.2) — `results_canonical.csv`, both arms.

- **Claim:** aggregation wins for LEAR-LASSO (2.84 vs 2.90), LightGBM (2.99 vs
  3.30) and LSTM (2.78 vs 3.18); SARIMAX is the exception (3.20 direct vs 3.27
  aggregated). Naive is identical either way.
- **Not a claim:** a universal rule about temporal aggregation in forecasting.
  Four models, one market, one window.

**Why this survives questioning.** The second sentence — *does it hold for all
five* — is deliberate: it makes the SARIMAX exception a reported result rather
than an inconvenience to average away. Both arms use the same feature matrix,
the same origins and the same five models (`run_daily_direct.py`, and the tuning
budget was matched in `tune_daily.py` for exactly this reason), so the
difference is attributable to the route and not to unequal effort. That equal-
effort design *is* the defense — without it the comparison would be confounded,
and the code comments say so.

### The one judgment call in this reconstruction

**The out-of-distribution failure (§6.8) is not covered by any RQ above.** That
is deliberate: the OOD stress test was added by the 2026-07-11 gameplan, after
the proposal, so it is unlikely to appear as a proposal RQ. It is written up in
**5-2** as a limitation and as the closing of formal assumptions (4) and (5).

If the author prefers it to be a research question — it is arguably the
project's most substantive finding — the clean move is to **append it as RQ5**
rather than displace RQ3, because RQ1–RQ4 each already own a section and
renumbering would break 4-7's structure:

> آیا مدل‌های تثبیت‌شده بر دوره بنچمارک (\lr{2016}–\lr{2017})، بدون هرگونه
> بازآموزی، عملکرد نسبی خود را بر \lr{173} روز داده زنده \lr{DE-LU} در سال
> \lr{2026} حفظ می‌کنند؟ و اگر نه، آیا افت عملکرد ناشی از انحراف سطح قیمت است
> یا از فروپاشی ساختار آموخته‌شده؟

*Do models frozen on the benchmark period (2016–2017), without any retraining,
retain their relative performance on 173 days of live DE-LU data from 2026? And
if not, is the degradation a shift in price level or a collapse of the learned
structure?*

**Answered in 5-2** (§6.8) — `ood_stress.csv`, `ood_recalibration.csv`.

- **Claim:** no. Every trained model falls below naive (rMAE > 1) and the
  ranking inverts — LightGBM and LSTM, near the best on the benchmark, are the
  worst here. A 7-day rolling bias correction recovers most of it (LSTM 1.535 →
  1.014; regime-aware ensemble 1.176 → 0.895) while *hurting* naive and
  LEAR-LASSO, the two least-biased models. The failure is therefore dominated by
  a **level shift**, not a loss of learned structure.
- **Not a claim:** a general law about model decay, or a quantified shelf life.
  One market, one 173-day window, one freeze date, no retraining.

**Why the second sentence matters.** Without it the answer is just "the models
broke", which invites "so what was the point?". With it, the finding is a
*diagnosis* — the models still rank hours correctly and are merely biased — and
it closes formal assumptions (4) *model generalization* and (5) *stable market
conditions* from your own proposal with evidence instead of a disclaimer. That
is the most defensible negative result in the thesis; do not soften it.

---

### Wording the answers in 4-7 so they stay defensible

Section 4-7 is where an over-claim would do the most damage, because it is the
one place all four answers are stated in a row. Rules:

- **Answer the question that was asked**, in the same scope it was asked in. If
  the RQ names five models on EPEX-DE, the answer says so too.
- **State nulls as findings**, in their own sentence — the LSTM/LEAR tie and the
  calm-day null are results, not missing wins.
- **Quote HAC-corrected p-values only**, and give the bootstrap range wherever
  the pooled regime result is mentioned. Never quote uncorrected `epftoolbox`
  p-values or claim 1% significance.
- **Never write that the thesis beats the state of the art.** It ties two of
  Lago et al.'s five variants and loses to their best DNN ensemble (p = 0.0127).
- **Every numeral inside `\lr{}`** (§1.1), read from the frozen file (§1.2).

**These are adopted as the working set — 1-3 and 4-7 are drafted from them
now.** The only outstanding item is a wording reconciliation against the
approved proposal, which happens at review of those two units: if the
proposal words a question differently, the proposal wins and the answer
text is unaffected, because every answer is scoped to the frozen evidence
rather than to the phrasing. Rule on RQ5 at the same review.

---

## 8. The four honest tensions — write these, do not hide them

1. **We do not beat the published benchmark.** We tie two of Lago et al.'s
   five variants and lose to their best DNN ensemble (p = 0.0127). The
   project's defense is innovation-led (regime-aware weighting, OOD analysis,
   interpretability), not leaderboard-led. This was a deliberate, logged
   strategic choice made at the week-5 checkpoint ("Plan B").
2. **The regime-aware gain is borderline when pooled.** Significant at
   p = 0.0226 under HAC, but the block bootstrap reaches p = 0.0571 at
   10-day blocks, and it would not survive Holm correction had it been
   drawn from the 21-test exploratory family (0.0903) — it is defended as a
   pre-specified confirmatory test, logged 2026-07-11 (§16A/A1). It is solid
   on the stressed subset (p = 0.0063) and null on calm days.
3. **The models fail out of distribution**, losing to naive on 2026 data.
4. **The best statistical model and the best deep model tie** (p = 0.404) —
   which is itself a finding about how much deep learning buys here.

---

## 9. The six formal assumptions (must appear in section 3-2)

From the approved proposal: (1) stationarity, (2) data availability, (3) data
quality, (4) model generalization, (5) stable market conditions,
(6) model interpretability.

**Address the tension head-on:** the OOD result (§6.8) directly challenges
(4) and (5), and arguably (1). Section 3-2 should state the assumptions as the
proposal framed them, and section 5-2 should return to say which ones the
evidence did not support. That round trip is one of the strongest structural
moves available in this thesis.

---

## 10. Per-chapter briefs

### Chapter 1 — مقدمه (7pp, Claude chat)
Motivation: price volatility, renewables penetration, the economic value of
day-ahead accuracy. State the problem, then the four RQs (§7; reconcile wording with
RQ1–3 verbatim**). Innovations to claim in 1-4: named benchmark tied to
published literature (Lago protocol), hourly *and* daily targets with the
direct-vs-aggregated comparison, regime-aware ensembling, OOD stress test,
SHAP as a deliverable, and the PriceCast tool. Keep 1-6 (scope and
assumptions) consistent with §9 and §4.3.

### Chapter 2 — پیشینه (17pp, Claude chat)
Eight ~2pp subsections mapping to the eight Zotero subcollections listed in
§3, then 2-9 positioning. **Chapter 2 depends on nothing frozen** — it is the
only chapter that can be written entirely without the repo, which is exactly
why it was assigned to chat. The gap argument in 2-9 should land on: published
EPF benchmarks evaluate in-distribution only, and rarely test regime-conditional
ensembling or out-of-distribution robustness — which is what this thesis adds.

### Chapter 3 — روش تحقیق (37pp, Claude Code)
**Four Farsi drafts already exist** in `thesis/drafts/` — more than the
tracking files claim:

| Draft | Section | Budget |
|---|---|---|
| `3-5-evaluation-framework.md` | 3-5 چارچوب اعتبارسنجی | 4pp |
| `3-6-baseline-models.md` | 3-6 مدل‌های پایه | 4pp |
| `3-7-1-lightgbm.md` | 3-7-1 LightGBM | 3pp |
| `3-7-2-lstm.md` | 3-7-2 LSTM | 4pp |

That is **~15pp of chapter 3 already drafted** and awaiting conversion.
`CHECKLIST.md` lists only 3-5 and 3-6, and `thesis/CONVERSION_QUEUE.md` lists
3-2/3-5/3-6 — both are stale. Converting these four is the single fastest way
to move the ledger off zero, and it should happen before any new drafting.

Bodies are clean Farsi prose; the `> DRAFT …` source blocks live separately in
`thesis/drafts/sources/` and are **not** meant to be pasted.

3-2 is the natural next section to write — it needs no RQs. 3-4 (7pp) must explain
the 247 columns and, critically, **why `exog_*_D0` is legal**: those series
are day-ahead forecasts published before the origin. 3-3-3 uses figures 01–09,
already exported.

### Chapter 4 — نتایج (29pp, Claude Code)
Every number is frozen and in §6. 4-6 is the largest single section at 8pp and
has the richest material. 4-7 answers the §7 RQs. Keep the supplementary
seed-ensemble numbers (§6.6) visibly separate from the frozen headline numbers.

### Chapter 5 — جمع‌بندی (10pp, Claude Code)
5-2 carries the OOD result and the data limitations of §4.3, and closes the
loop on the assumptions of §9. 5-3 covers PriceCast, with figure 16 already
captured.

---

## 11. Open items — for the author, none of them blocking

1. **RQ1–RQ3 — reconcile wording with the approved proposal.** They appear
   nowhere in the repository, so §7 adopts a working set reconstructed from
   the work actually done, each bounded to the frozen evidence. **This does
   not block drafting.** At review of 1-3 and 4-7, check the phrasing against
   the proposal — if it differs, the proposal wins and only the question text
   changes, never the answers. Rule on RQ5 at the same review.
2. **`words_per_page` in `configs/schedule.yaml` is still the placeholder
   `250`.** *(Note: moving to LaTeX largely retires this — the compiled PDF
   gives a true page count. Update the schedule config or the page-counting
   script accordingly.)*

---

## 12. Assets already produced

**Figures** (`reports/figures/`, final captioned form, do not regenerate):
01 price distribution · 02 hourly seasonality · 03 weekly seasonality ·
04 annual seasonality · 05 volatility clustering · 06 ACF/PACF ·
07 exogenous correlation · 07b rolling correlation · 08 full-series structural
breaks · 09 daily baseload · 10 SHAP global importance · 11 SHAP beeswarm ·
12 SHAP hour profile · 13 SHAP calm vs stressed · 14 SHAP hourly vs daily ·
15 SHAP waterfall case study · 16 PriceCast screenshot

**Tables** (`reports/tables/`, `.csv` + `.tex` twins; `\input` the `.tex`):
`results_canonical` · `dm_tests` · `dm_regime_split` ·
`dm_bootstrap_sensitivity` · `lago_comparison` · `dm_vs_lago` ·
`seed_ensemble` · `shap_importance` · `ood_stress` · `ood_recalibration` ·
`oracle_bound`

---

## 13. Skills

**Claude Code (chapters 3–5):**

| Skill | When |
|---|---|
| `thesis-fa-latex` | **Any** Farsi thesis prose or `.tex` edit, and before quoting any number. Carries the `\lr{}` rule and the template fixes. |
| `export-results` | Any new figure/table (should be none — everything is frozen) |
| `shap` | Section 4-6 |
| `superpowers:verification-before-completion` | Before claiming any section is done |

**Claude chat (chapters 1–2):** no repo-backed skills apply. If a
literature-review skill is wanted, chapter 2 would have to move into Claude
Code, where `bytedance/deer-flow@systematic-literature-review` or
`bahayonghang/academic-writing-skills@bib-search-citation` could be installed.

**For the English journal article later:** `latex-paper-en` is already
installed; `paper-audit` from the same family is worth adding.

---

## 14. LaTeX environment

- **Engine:** XeLaTeX (MiKTeX 26.5 / XeTeX 4.18). Not pdfLaTeX.
- **Class:** `AUTthesis`, `\documentclass[oneside,msc,12pt]{AUTthesis}`.
- **Fonts:** `B Nazanin` (text), `Times New Roman` (Latin), `Persian Modern`
  (digits).
- **Two required template fixes**, already applied:
  1. `\setdigitfont{PGaramond}` → `{Persian Modern}`. PGaramond is not
     installed and the stock template **does not compile** without this — it
     falls back to METAFONT and dies.
  2. `booktabs` and `multirow` added to `commands.tex`, **before**
     `xepersian` (which must remain the last package loaded).
- **Build:** run XeLaTeX twice, plus `biber`/`bibtex`. Then **render the pages
  and look at them** — a clean exit with corrupted digits is the exact failure
  mode §1.1 describes.
- Captions go **outside** `\begin{latin}`, or they print `Table 3-1` instead
  of `جدول ۳-۱`.

---

## 15. Citations, references, and academic integrity

**This is the highest-risk part of the thesis.** A wrong number is an error; an
uncited borrowed sentence is misconduct. Treat everything below as binding.

### 15.1 The library

`Thesis References/` holds **41 usable sources** — a Zotero export
(`Thesis References.bib`) plus the PDFs in `Thesis References/files/`.

The **working bibliography** is
`thesis/latex/Latex template 2018/references.bib`. It is the Zotero export
with three corrections applied, documented in its own header:

1. **Removed** `noauthor_notitle_nodate` — a completely empty entry (no
   author, title, year or file). A Zotero export artifact.
2. **Fixed** `nogales_forecasting_nodate` → `nogales_forecasting_2002`, with
   journal, volume, number and pages read off the PDF's first page: *IEEE
   Trans. Power Systems*, vol. 17, no. 2, May 2002, pp. 342–348.
3. **Retyped** `trebbien_explainable_nodate` → `@mastersthesis`
   `trebbien_explainable_2023`. It was typed `@article`, but the PDF's title
   page reads *Master Thesis, University of Cologne, Institute for Theoretical
   Physics, March 15, 2023*. **Citing a master's thesis as a peer-reviewed
   journal article is a citation-integrity error, not a formatting nit** — an
   examiner who checks one reference may well check that one.

Every correction was read from the cited PDF, never recalled.

⚠ **A fresh Zotero export will silently undo all three.** If you re-export,
re-apply them — better, fix them in Zotero itself.

### 15.2 Reference order — already handled, do not do it by hand

The list must be ordered by **first appearance in the text**. That is a
one-line setting, now applied:

```latex
\bibliographystyle{unsrt-fa}   % was plain-fa (alphabetical)
```

`unsrt` = "unsorted" = citation order. `[1]` is the first work cited anywhere
in the thesis; the printed list follows that same order; LaTeX renumbers
automatically on every build as citations are inserted or moved.

**Never hand-number a reference, and never reorder the `.bib` file.** Entry
order in `.bib` is irrelevant — only `\cite{}` order in the text matters.

### 15.3 How to cite — the mechanics

```latex
پژوهش‌های پیشین نشان داده‌اند که ... \cite{lago_forecasting_2021}.
دو مطالعه به نتیجه مشابهی رسیده‌اند \cite{lago_forecasting_2018, ziel_probabilistic_2018}.
```

- Cite with the **exact key** from `references.bib`. A typo'd key prints `[?]`
  and is easy to miss — see the check in §15.7.
- Place `\cite{}` **before** the sentence-ending period.
- Numbers rendered by the class are automatic; nothing needs `\lr{}`.

### 15.4 What must be cited

Cite at the **sentence or clause** level, not once per paragraph:

| Situation | Requirement |
|---|---|
| A direct quotation | Quotation marks (`«…»`) + citation. Use sparingly; in a technical thesis, almost never. |
| A paraphrase of someone's idea, however reworded | Citation. **Rewording does not remove the obligation.** |
| A number, dataset, or result from another work | Citation, and name the source in the sentence. |
| A method you used that someone else devised (LEAR, DM test, SHAP, LSTM) | Citation on first substantive use in each chapter. |
| A claim of fact not common knowledge in the field | Citation, or delete the claim. |
| Your own result | **No citation** — point at your own table/figure instead. |

**The line that gets theses failed:** taking a source's sentence, swapping a
few words and the clause order, and citing it — or worse, not citing it. That
is close paraphrase, and plagiarism checkers detect it. Read the passage,
close it, write the idea in your own words in Farsi, then cite. If your
sentence still tracks the original phrase-for-phrase, it is too close.

The thesis is in Farsi and the sources are in English. **Translation is not
paraphrase.** A translated sentence is still that author's sentence and needs
a citation exactly as an English quotation would.

### 15.5 The source map — mandatory for chapter 2

Chapter 2 is 17pp built almost entirely from other people's work. Before
drafting each subsection, write the mapping down; keep it beside the draft:

```
2-4 deep learning
  claim: LSTM captures long-range dependence in price series  -> ugurlu_electricity_2018
  claim: CNN-LSTM hybrids outperform plain LSTM on DE data    -> zhang_deep_2020
  claim: [NEEDS CITATION] transformers are now standard        -> NOT IN LIBRARY
```

Any claim that reaches `[NEEDS CITATION]` gets **cut or softened**, never
attributed to the nearest plausible paper. Attaching a real citation to a
claim that source does not actually make is its own integrity failure, and it
is the single most likely way an AI-assisted literature review goes wrong.

### 15.6 Rules for any model drafting prose

1. Cite **only** from the 41 keys in `references.bib`. Never invent a
   reference, a DOI, a year, or an author.
2. Never cite a paper whose relevant passage was not supplied. If a claim
   needs a source you do not have, output `[NEEDS CITATION]` and stop.
3. Never state what a paper "found" unless that finding is in the text given
   to you. Summarising an unread paper from its title is fabrication.
4. Chapters 1 and 2 are drafted in Claude chat, which **cannot read the
   PDFs**. It must be given the passages. Anything it writes about a source it
   was not shown is unverified and must be checked against the PDF before it
   goes in the thesis.

### 15.7 Verification before submission

- **Every key resolves.** After a build, grep the `.log`/`.blg` for
  `Warning--I didn't find a database entry` and `Citation ... undefined`.
  Either means a `[?]` is printed somewhere.
- **No orphan entries.** `unsrt-fa` prints only cited works, so an
  uncited entry vanishes silently — a source you meant to use and forgot
  leaves no trace. Diff the cited keys against the 41.
- **Spot-check five references against their PDFs** — author, year, venue,
  page range. That is how the Nogales and Trebbien errors above were found,
  and there were two of them in 42 entries.
- **Run the finished text through the university's similarity checker before
  the supervisor does.**

### 15.8 Verified working — end to end

`persian-bib` is **installed** and the full chain was tested against the real
41-entry library on 2026-08-30. It also supplies `ieeetr-fa.bst` if IEEE
formatting is ever preferred (also citation-ordered).

**The test:** three papers cited in deliberately reverse-alphabetical order —
Ziel → Aggarwal → Lago. The reference list printed **[1] Ziel, [2] Aggarwal,
[3] Lago**, i.e. citation order, not alphabetical. `bibtex` ran with zero
errors and zero warnings.

**Build order (all four steps, in this order):**

```
xelatex AUTthesis      # writes the .aux with \citation{} records
bibtex  AUTthesis      # reads .aux + references.bib -> writes .bbl
xelatex AUTthesis      # pulls in the .bbl
xelatex AUTthesis      # resolves the reference numbers
```

Skipping `bibtex` leaves the previous `.bbl` in place. The template ships one
full of unrelated Finsler-geometry papers, so **the document compiles happily
while displaying someone else's bibliography** — that is the failure to watch
for, not a crash.

### 15.9 BibTeX does not understand `%` comments — a real trap

`%` is a LaTeX comment character, **not** a BibTeX one. In a `.bib` file an
`@` *always* starts an entry, even inside what looks like a comment and even
inside an `@comment{...}` block. A header note that mentioned entry types with
their `@` prefix made BibTeX try to parse the note and **silently skip real
records**, reporting only `I'm skipping whatever remains of this entry`.

This was hit and fixed while setting the file up. The header of
`references.bib` is now an `@comment{...}` block containing **no `@`
characters at all**. If you edit it, keep it that way.

### 15.10 Two cosmetic points to settle with your supervisor

Both visible in the verified output; neither is a bug, both are choices:

1. **In-text markers print Persian numerals `[۱] [۲] [۳]`, while the Latin
   reference list prints `[1] [2] [3]`.** The class does this deliberately —
   in-text numbers sit in Farsi text, list entries sit in LTR blocks. Confirm
   the department accepts the mismatch.
2. **Citations render in red**, from `commands.tex`:
   `\usepackage[colorlinks,linkcolor=blue,citecolor=red]{hyperref}`. The
   template's own comment says to disable `hyperref` for the final submitted
   version. Do that before printing.

---

## 16. Per-chapter prompts

Generic academic-writing prompts, adapted to this project. **Use the adapted
version**, not the generic one — four things differ in ways that matter here
(see §16.8). The generic originals are kept at the end for reference.

Every prompt below assumes this whole file has already been pasted or read.

### 16.1 Chapter 1 — Introduction & problem statement

> You are helping write chapter 1 of a Farsi MSc thesis on day-ahead
> electricity price forecasting for the German market (Amirkabir University).
> **Write in Farsi.** Context is §2, §4 and §8 of the handoff you have been
> given.
>
> Draft the background and statement of the problem in this order: the ideal
> situation, where current practice falls short, the consequences, and the
> explicit research gap. Then draft 1-4 (innovations) from the list in §10.
>
> Constraints: do not fabricate facts, statistics or citations. Every number
> must come from §4–§6 of the handoff; if a number you need is not there, say
> so instead of estimating it. Section 1-3 (the research questions) is
> the RQs in §7, quoted as written there — never a fresh invention.
> Budget: 1-1 [2pp], 1-2 [1pp], 1-4 [1pp], 1-5 [1pp], 1-6 [1pp].

### 16.2 Chapter 2 — Literature review

> You are helping write chapter 2 of a Farsi MSc thesis on electricity price
> forecasting. **Write in Farsi.**
>
> Here are the papers for subcollection [2-N: theme]. For each I give the
> BibTeX key and the passages I actually read: [paste key + passages].
>
> **Step 1 — build the source map first, before any prose.** For every claim
> you intend to make, output one line: `claim -> bibtex_key`. Show me this map
> and stop. Any claim you cannot map to a supplied passage gets
> `[NEEDS CITATION]`, not the nearest plausible key.
>
> **Step 2 — after I approve the map**, write the ~2-page subsection in Farsi:
> group by theme, state where studies agree, and be explicit where they
> disagree or where evidence is thin. A subsection that reads as a list of
> summaries has failed; synthesize.
>
> Citation rules, absolute:
> - Cite as `\cite{exact_bibtex_key}` using **only** the keys I supplied.
> - Never invent a reference, DOI, year, author or finding.
> - Never state what a paper "found" unless that finding is in the passage I
>   gave you. Summarising a paper from its title is fabrication.
> - These sources are in English and you are writing Farsi. **Translation is
>   not paraphrase** — a translated sentence still belongs to its author and
>   still needs its citation.
> - If your sentence tracks the original phrase-for-phrase, rewrite it.
>
> Run this once per subcollection: 2-1 reviews/bibliometric · 2-2
> classical/statistical · 2-3 classical ML · 2-4 deep learning · 2-5
> hybrid/attention · 2-6 explainability · 2-7 applied/market-specific
> (incl. Iran) · 2-8 long-term/probabilistic. Then 2-9 [2pp] positions this
> thesis in the gap — see §10 for the intended gap argument.

### 16.3 Chapter 3 — Methodology

> You are helping write chapter 3 of a Farsi MSc thesis. **Write in Farsi.**
>
> The methodology is **already fixed and implemented** — it is documented in
> §5 of the handoff and in the repository's configs and code. Your job is to
> *describe and justify* it, not to propose or improve it. Do not suggest
> alternative designs, extra models, or different validation schemes; the
> model list and protocol are locked.
>
> For section [3-N], write the prose that explains what was done and **why**,
> at the level of rigour a thesis examiner expects: the choice, the
> alternatives considered, and the reason for the decision. Where a design
> limitation exists, state it plainly (§8, §4.3).
>
> Every numeral must be wrapped in `\lr{}` (§1.1). Every quantity must be
> read from its source file, never recalled (§1.2).

### 16.4 Chapter 4 — Results

> You are helping write chapter 4 of a Farsi MSc thesis. **Write in Farsi.**
>
> Note: in this thesis **results and discussion are split**. Chapter 4
> presents and analyses the findings; the broader discussion, limitations and
> implications live in chapter 5. Do not write chapter 5's material here.
>
> Here are the exact frozen numbers for section [4-N]: [paste from §6, or read
> the source file]. Write the results description: what was measured, what the
> numbers show, and what follows from them. Compare to Lago et al. (2021)
> where §6.5 applies, using the stated DM verdicts — do not soften them.
>
> Constraints: never round or restate a number from memory. Keep the
> supplementary seed-ensemble results (§6.6) visibly separate from the frozen
> headline numbers. Do not claim we beat the benchmark (§1.3). Every numeral
> in `\lr{}`.

### 16.5 Chapter 5 — Conclusion & recommendations

> You are acting as a thesis committee reviewer helping write chapter 5 of a
> Farsi MSc thesis. **Write in Farsi.**
>
> Here are the key findings from chapters 3 and 4: [paste]. Draft 5-1
> (summary) restating the primary aim and the study's significance
> **without introducing any new data or numbers not already reported**.
>
> Then 5-2 (limitations): carry the OOD result (§6.8) and the data
> constraints (§4.3), and close the loop on the six formal assumptions (§9) —
> state explicitly which ones the evidence did not support.
>
> Then 5-4: two to four realistic recommendations for future research, each
> grounded in a specific limitation this study actually hit — not generic
> "more data, more models" filler.

### 16.6 Draft → `.tex` conversion (the step that actually banks pages)

Four Farsi drafts (~15pp) already exist in Markdown and are worth more banked
than any new drafting. This is where `\lr{}` errors will concentrate, so it
gets its own prompt.

> Convert `thesis/drafts/[name].md` into a `.tex` chapter body for the
> `AUTthesis` class. This is a **conversion, not a rewrite** — do not improve,
> expand, condense or reorder the Farsi prose. If a sentence seems wrong,
> flag it separately; do not silently fix it.
>
> Apply exactly these transformations:
> 1. Wrap **every** numeral in `\lr{}` — metrics, p-values, counts, years,
>    dates, prices, integers included (§1.1).
> 2. Wrap Latin identifiers (`MAE`, `LEAR-LASSO`, `Diebold-Mariano`) in
>    `\lr{}`.
> 3. Markdown headings → `\section{}` / `\subsection{}`. Never hardcode a
>    section number; the class numbers them.
> 4. Replace any hand-written table with `\input{}` of the frozen `.tex`, with
>    the caption **outside** `\begin{latin}` (§14).
> 5. Convert citations to `\cite{exact_key}`; verify each key exists in
>    `references.bib`.
> 6. Check ZWNJ (نیم‌فاصله) on `می‌`, `‌ها`, `پیش‌بینی` (§1.4).
>
> Then verify, and report the result rather than asserting success: compile,
> render the changed pages to PNG, and read every numeral off the rendered
> page against the source draft. A clean exit proves nothing here.

### 16.7 A reusable guard clause

Append to any prompt in a fresh session:

> Draft **one subsection only**, then stop and wait for my approval before
> going further — do not draft ahead. Do not fabricate facts, statistics,
> citations or file contents. Cite only from the keys I have given you; if you
> need a number or a source you have not been given, output `[NEEDS CITATION]`
> or say so and stop — never approximate and never attribute to the nearest
> plausible paper. Write in Farsi. Wrap every numeral in `\lr{}`.

Every prompt in §16.1–16.6 assumes this gate. Whichever unit is being written,
the turn ends with the draft presented and nothing else started — and
`thesis/APPROVALS.md` updated once a verdict comes back.

### 16.8 What was changed from the generic versions, and why

1. **Language.** None of the generic prompts said Farsi; all of them would
   have produced English. The body is Farsi.
2. **Chapter 3's framing was inverted.** The generic prompt says "act as a
   methodology critic… suggest improvements" and assumes
   survey/interview/lab work. This methodology is finished, frozen behind
   three tags, and implemented in code — inviting improvements would generate
   suggestions that cannot be acted on and would read as scope creep.
   The adapted version asks for description and justification instead.
3. **Chapter 4 was split.** The generic prompt merges "Results & Discussion".
   This thesis puts results in chapter 4 (29pp) and discussion, limitations
   and implications in chapter 5 (10pp). Merging them would produce material
   in the wrong chapter and blow both page budgets.
4. **The numeral and frozen-number rules were added.** The generic prompts
   have no way to know about the `\lr{}` corruption (§1.1) or that every
   result is pinned to a file (§1.2) — the two ways this thesis is most
   likely to end up quietly wrong.

Two smaller notes: chapter 2's generic prompt asks for 5 papers and 500 words;
the real shape here is 8 subsections of ~2pp each, so the adapted version runs
once per subcollection. And `[NEEDS CITATION]` was added because a
literature-review model with no sources will otherwise fill the gap with a
plausible-looking reference.

<details>
<summary>Generic originals, as supplied</summary>

- **Ch1:** "Act as an academic research advisor. I am writing a Master's
  thesis in [discipline] on [topic]. Based on this broad context [paste
  notes/stats], help me draft a background and a statement of the problem.
  Follow this structure: describe the ideal situation, explain where current
  reality falls short, outline the consequences, and explicitly highlight the
  research gap. Do not fabricate facts or statistics."
- **Ch2:** "Act as a literature review assistant. Here are summaries of 5 key
  research papers related to [my research objective] [paste text/summaries].
  Synthesize these findings, compare their agreements and disagreements, and
  write a cohesive 500-word subsection. Cite them strictly as [Author, Year]
  using the provided text. Do not invent outside references."
- **Ch3:** "Act as a methodology critic. I used a [quantitative/qualitative/
  mixed] approach with [survey/interview/lab procedure] to study [topic].
  Review my rough notes below [paste notes] and suggest improvements to the
  academic clarity, rigor, and justification of these choices. Highlight any
  missing design limitations I should address."
- **Ch4:** "Act as a data analyst and academic editor. Here is my raw data/
  statistical output: [paste exact numbers or themes]. Format this into a
  clear results description. Then, in a separate discussion section, help me
  interpret what these findings mean, how they compare to [previous study/
  theory], and what the practical limitations are."
- **Ch5:** "Act as a thesis committee reviewer. Here are the key findings from
  my chapters: [paste chapter conclusions]. Draft a final conclusion section
  that summarizes the primary aim, restates the study's significance without
  introducing new data, and provides two realistic recommendations for future
  research."

</details>

---

## 16A. The examiner pass - loose ends, and how each one closes

Run 2026-08-31 against the frozen tables, adversarially: the goal was to break
the answers, not to confirm them. Nine attacks were tried. **Seven close
cleanly, one forces a change in wording, one is a disclosed limitation.**
Every item below belongs in the thesis text - an examiner who finds one of
these unaided reads it as a hole; pre-empted, each reads as control of the
method.

### A1. Multiplicity - THE ONE THAT CHANGES THE WRITING

**Attack:** section 4-5-1 reports a 7x7 DM matrix. That is **21 pairwise
tests**. At uncorrected alpha = 0.05 you expect about one spurious winner, so any
result near 0.05 is unsafe.

**Result of applying Holm-Bonferroni across all 21** (computed from the frozen
p-values in `dm_tests.csv`; no model was rerun):

| Comparison | raw p | Holm p | verdict |
|---|---|---|---|
| all 6 comparisons vs naive | 0 | 0 | survives |
| ensembles vs each member (8 tests) | <= 4.8e-06 | <= 3.4e-05 | survives |
| SARIMAX vs LSTM / LEAR / LightGBM | <= 1.0e-04 | <= 5.1e-04 | survives |
| **Ensemble regime-aware vs static** | **0.0226** | **0.0903** | **fails** |
| **LSTM vs LightGBM** | **0.0405** | **0.1215** | **fails** |
| LEAR-LASSO vs LightGBM | 0.2665 | 0.5329 | fails (already null) |
| LEAR-LASSO vs LSTM | 0.4036 | 0.5329 | fails (already null) |

**17 of 21 survive.** Everything structural - every model beats naive, both
ensembles beat every member - is untouched. But two claims that are currently
written as significant do not survive a correction over that family.

**How this closes, and it does close.** Declare **two families explicitly** in
3-5, before any p-value is quoted:

1. **Confirmatory, one pre-specified hypothesis:** regime-aware vs static
   weighting. Logged in `logs/decisions.md` on **2026-07-11**, when the
   regime-aware ensemble did not yet exist and no ensemble had been scored on
   the test window (`v1.0-results` is dated 2026-08-04). It cannot have been
   selected by looking at outcomes. A single pre-specified hypothesis is tested
   at alpha = 0.05 without correction - this is standard, not a loophole, and the
   dated log entry is what makes it checkable.
2. **Exploratory, the 21-cell matrix:** report raw **and** Holm-corrected
   p-values in the table. Both columns. Reporting only the raw ones is wrong;
   hiding them looks evasive.

**What must change in the writing:**

- **LSTM vs LightGBM (p = 0.0405) may no longer be called significant.** It sits
  in the exploratory family and does not survive (Holm 0.1215). Write it as
  *"LSTM shows a lower point estimate; the difference does not survive
  correction for multiplicity."* This **strengthens** RQ1 rather than weakening
  it: the honest reading becomes LEAR-LASSO, LightGBM and LSTM are mutually
  indistinguishable, and only the ensembles separate from them.
- **Regime-aware vs static stays at p = 0.0226**, defended as the pre-specified
  confirmatory test - but the sentence must carry all three numbers together:
  pooled 0.0226, stressed-subset 0.0063, bootstrap range 0.0129-0.0571, and the
  note that it would not survive correction had it been drawn from the
  exploratory family. State that yourself.

This precedent already exists in the project: the combination-ladder arm
declared a family of six and applied Holm-Bonferroni to it
(`logs/decisions.md` 2026-08-20). 4-5-1 is simply held to the same standard.

### A2. "Your naive model scores rMAE 0.849 - it beats the benchmark it defines?"

**Attack:** an rMAE below 1 for the naive model looks self-contradictory and, if
unexplained, makes every other rMAE look unreliable.

**Closes.** They are two different naives, and both are standard:

- **Our naive model** (`src/models/naive.py`) is the Lago "similar day" rule:
  Monday takes D-3, weekends take D-7, midweek takes D-1.
- **The rMAE denominator** is epftoolbox's own naive, built internally by
  `naive_forecast` from the real series.

So 0.849 means the similar-day rule beats the toolbox's normalizing naive by
about 15%, which is exactly what it should do. **One sentence in 3-5 naming both
naives kills this question.** Leave it unwritten and it looks like an error.

### A3. "You banned MAPE for negative prices, then reported sMAPE. Why is that consistent?"

**Attack:** sMAPE is also a percentage error, and prices go negative.

**Closes, with the project's own evidence.** MAPE divides by `p_real`, which is
sign-dependent and undefined at zero, so a single near-zero hour makes it
explode without bound. sMAPE divides by `(|p_real| + |p_pred|)/2` - absolute
values, so it stays finite unless both are near zero simultaneously.

The empirical proof is already in `lago_comparison.csv`: **on identical
forecasts**, Lago et al.'s paper reports MAPE of 77-137% against sMAPE of
14-17% - roughly a factor of ten, and the paper itself flags MAPE as unreliable
on this market. Cite that.

Then add the decisive sentence: **no conclusion in this thesis rests on sMAPE.**
Every DM test is computed on absolute loss (`loss_differential`, norm=1), and
every headline is MAE or rMAE. sMAPE is reported for comparability with the
published table, nothing more.

### A4. "Is your comparison against Lago et al. really like-for-like?"

**Closes hard, and this is a strength worth showing.**
`scripts/run_lago_comparison.py` calls `verify_alignment()` *before* a single
p-value is computed: it asserts the full DatetimeIndex is identical to theirs
element-by-element, reports the first mismatching position if not, and then
cross-checks their `Real price` column against our `y_true` to a maximum
absolute difference. Same 728 origins, 2016-01-04 to 2017-12-31, same rMAE
denominator, our metric code on both sides.

Say in 4-5-2 that alignment is asserted in code rather than assumed. That is a
stronger answer than most published comparisons can give.

### A5. "The paper-vs-shipped discrepancy is convenient for you"

**Closes, and it is stronger than section 6.5 currently states.** The pattern is
perfectly systematic, not scattered:

- **All 5 DNN variants reproduce the paper exactly** (DNN 1-4 and DNN Ensemble).
- **All 5 LEAR variants disagree** (LEAR 56, 84, 1092, 1456, Ensemble), and in
  every case the shipped forecasts score **better** than the printed table.

A bug on our side would not spare five DNN rows and hit five LEAR rows. The DNN
rows are the positive control that proves our loader and metrics are correct;
the LEAR rows are the finding. And we adopted the reading that makes **their**
models look better - the conservative choice, against our own interest. Write
that sentence explicitly; it is what converts the observation from convenient to
credible.

### A6. "Did you tune the regime threshold on the test set?"

**Closes.** k = 1.5 came from a validation-only rule: the largest k in
{3.0, 2.5, 2.0, 1.5} leaving >= 20 days in both regimes. k = 2.0 left 10 validation
days, k = 1.5 left 37. The test window was never consulted, and the earlier
3-sigma 84.04 value was superseded on 2026-08-04 for the same validation-only
reason. The regime label also uses the **previous** day's realized prices,
enforced in code, so the label cannot leak.

### A7. "Your SHAP model is not the model you report results for"

**Disclosed, not defended away.** Correct, and deliberately so. The frozen
walk-forward models saw the days being explained, so explaining them would be
in-sample. `shap_analysis.py` refits on the trailing 1092 days ending
2017-12-31 and **refuses** to fit on a shorter window. It is a faithful twin,
not the same object. One sentence in 4-6. Also state that SHAP covers the
**LightGBM arms only** - no claim is made about what LEAR-LASSO or the LSTM use.

### A8. "The ensemble weights saw their members' tuning data"

**Disclosed.** Weights were fitted on the members' tuning window. State it in
3-8. The mitigation is that the *comparison* it feeds - regime-aware vs static -
gives both arms the identical advantage, so the contrast between them is
unaffected.

### A9. "One market, one window, one freeze date"

**Disclosed limitation, and the honest boundary of everything above.** EPEX-DE
only (France was scoped and not run), 728 origins, a single 173-day OOD window,
one architecture per model family, 50 Optuna trials each. Every claim in
chapter 4 is scoped to that. The "Not a claim" lines in section 7 exist to hold
this boundary sentence by sentence.

---

## 17. Author's additional rules

<!-- Add your own rules here before committing. -->
