# Writing handoff — paste into a web session

Written 2026-08-05. **Self-contained on purpose:** a web session cannot read
this repo, so every number, name and constraint it needs is inline below.
Nothing here says "see file X" without also giving the value.

Copy everything between the outer fences.

---

```
You are helping me write the Farsi text of my MSc thesis. All the research,
code and results are FINISHED and FROZEN. Nothing is left to compute or
decide — your job is prose. You have no access to my repository, so every
number you need is in this brief. NEVER invent a number that is not here; if
you need one I have not given you, stop and ask me for it.

## The thesis

Day-ahead electricity price forecasting for the German market using machine
learning and deep learning, benchmarked against the published protocol of
Lago et al. (2021). Two targets: hourly (a 24-value vector for day D+1) and
daily baseload (the mean of those 24 hours).

- Body language: FARSI. Formal academic register.
- Template: Amirkabir University official thesis template.
- Length: 100 pages, 5 chapters, budgets fixed (see structure below).
- A separate journal article in ENGLISH is a later, separate deliverable.
  Do not mix the two.
- Figure/table text stays in English (they are already exported); Farsi
  captions are written in the document.

## How I want you to work

- Write one section at a time, at the page budget given. Do not run over.
- Ask me before starting a section if anything about its scope is unclear.
- Use [REF: key] as a citation placeholder — I resolve these in Zotero.
  Do NOT invent citations, authors, years, or DOIs. If a claim needs a
  source I have not given you, mark it [REF: NEEDED — what it must support].
- Persian technical terms: give the accepted Farsi term with the English in
  parentheses on first use, then Farsi alone.
- Do not pad. If a section is genuinely shorter than budget, say so rather
  than inflating it.

## Structure and page budget (fixed, sums to 100)

Ch1 مقدمه [7pp]: 1-1 انگیزه و اهمیت [2] · 1-2 بیان مسئله [1] ·
  1-3 سؤالات پژوهش [1] · 1-4 نوآوری‌ها [1] · 1-5 ساختار [1] ·
  1-6 محدوده و مفروضات [1]

Ch2 مروری بر پیشینه [17pp]: 2-1 مرورها/کتاب‌سنجی · 2-2 آماری کلاسیک ·
  2-3 یادگیری ماشین کلاسیک · 2-4 یادگیری عمیق · 2-5 ترکیبی/توجه ·
  2-6 تفسیرپذیری · 2-7 کاربردی/بازار (incl. Iran) · 2-8 بلندمدت/احتمالاتی
  (~2pp each) · 2-9 شکاف پژوهشی و جایگاه کار [2pp]

Ch3 روش تحقیق [37pp]: 3-1 مقدمه [1] · 3-2 فرضیات بنیادی [3] ·
  3-3 داده‌ها و بازار [8: 3-3-1 بنچمارک 3, 3-3-2 داده زنده 2, 3-3-3 EDA 3] ·
  3-4 مهندسی ویژگی [7] · 3-5 چارچوب اعتبارسنجی و معیارها [4] ·
  3-6 مدل‌های پایه [4] · 3-7 ML/DL [7: LightGBM 3, LSTM 4] ·
  3-8 مدل ترکیبی [3]

Ch4 نتایج و تحلیل [29pp]: 4-1 مقدمه [1] · 4-2 نتایج ساعتی [6] ·
  4-3 نتایج روزانه [5] · 4-4 مستقیم در برابر تجمیعی [2] ·
  4-5 آزمون دیبولد-ماریانو [3] · 4-6 تفسیرپذیری SHAP [8] ·
  4-7 پاسخ به سؤالات پژوهش [4]

Ch5 جمع‌بندی [10pp]: 5-1 جمع‌بندی [2] · 5-2 محدودیت‌ها [2] ·
  5-3 ابزار PriceCast [2] · 5-4 پیشنهادها [2] · 5-5 نتیجه‌گیری [2]

WRITE IN THIS ORDER (follows the frozen numbers, waits on nothing):
chapter 3 first, then chapter 4, then 1, 2, 5.

## Data

- Benchmark (all thesis results): EPEX-DE from the Lago et al. (2021) open
  benchmark, via the epftoolbox package. Keyless.
  Train 2012-01-09 → 2016-01-03 (34,944 hourly rows).
  Test  2016-01-04 → 2017-12-31 (17,472 rows; 728 forecast origins).
  Train prices: mean 36.20, std 15.95, min -221.99, max 210.00 EUR/MWh;
    297 negative hours (0.85%).
  Test prices: mean 31.64, std 15.49, min -130.09, max 163.52 EUR/MWh;
    241 negative hours (1.38%).
  Zero missing hours, zero NaNs in both.
  Columns: price, exog_1 (day-ahead load forecast), exog_2 (day-ahead
  renewable generation forecast).
- Live data (tool + OOD test only): Energy-Charts API, Fraunhofer ISE, zone
  DE-LU, keyless. REQUIRED attribution wherever it appears:
  "Data: Energy-Charts (Fraunhofer ISE) / Bundesnetzagentur SMARD.de,
  CC BY 4.0".
- The benchmark era is the DE-AT-LU joint bidding zone; the live feed is
  DE-LU. Mention this when the two are compared.

## Method

- Feature matrix: 247 columns, identical for every model.
  Price lags D-1, D-2, D-3, D-7 (24 columns each = 96);
  for each of exog_1, exog_2: lags D-1, D-7 and the target day's own D0
  (24 each = 144); 7 day-of-week dummies. Target Y = the 24 prices of day D.
  The lag convention was cross-checked line-by-line against epftoolbox's
  LEAR implementation so results stay comparable with the published
  benchmark.
- WHY exog_*_D0 is legal and not leakage: exog_1 and exog_2 are day-AHEAD
  forecasts, published before the forecast origin. No feature ever reads the
  target day's own price. This is asserted by an automated test.
- Validation: rolling-origin walk-forward ONLY, never random splits.
  Calibration window 1092 days, step 1 day. LightGBM and LEAR recalibrate at
  every origin; SARIMAX and LSTM every 7 days for tractability.
- Tuning: 50 Optuna trials per model, on a validation window strictly before
  the test window. Seed 42 everywhere.
- Models (exactly five, plus ensembles): naive (Lago day-of-week rule:
  Monday→D-3, weekend→D-7, Tue–Fri→D-1), SARIMAX(1,1,1)(1,0,1,7),
  LEAR-LASSO (epftoolbox LEAR: asinh-median 'Invariant' scaling +
  LassoLarsIC), LightGBM (24 independent per-hour regressors), LSTM.
- Metrics: MAE, RMSE, sMAPE, rMAE. NEVER plain MAPE — negative prices exist
  in the data and MAPE is undefined/unstable there. Say this explicitly in
  3-5; it is a defensible methodological choice, not an omission.
- Ensembles: convex weights minimising MAE, fitted by SLSQP on a validation
  window strictly before the test period.
- Regime-aware ensemble: two weight sets (calm / stressed), switched by a
  threshold of 62.6989 EUR/MWh = train mean + 1.5*std. A day is "stressed"
  if the PREVIOUS day's realized prices contain at least one hour above the
  threshold — previous-day information only, so no leakage.
  In the test period this gives 651 calm and 77 stressed days.

## Frozen results — hourly target, 728 origins (section 4-2)

model                     MAE     RMSE    sMAPE    rMAE
naive                    7.750   13.257   28.595   0.849
SARIMAX                  4.351    7.117   18.035   0.477
LEAR-LASSO               3.899    6.475   16.657   0.427
LightGBM                 3.968    7.502   15.722   0.435
LSTM                     3.873    7.020   15.599   0.424
Ensemble (static)        3.574    6.610   14.671   0.392
Ensemble (regime-aware)  3.557    6.557   14.644   0.390

## Frozen results — daily baseload (sections 4-3, 4-4)

model                    route        MAE     RMSE    sMAPE   rMAE
naive                    direct      6.357   10.685   23.583  0.795
naive                    aggregated  6.357   10.685   23.583  0.795
SARIMAX                  direct      3.198    5.175   13.523  0.400
SARIMAX                  aggregated  3.269    5.076   13.654  0.409
LEAR-LASSO               direct      2.899    4.379   13.675  0.362
LEAR-LASSO               aggregated  2.839    4.414   12.728  0.355
LightGBM                 direct      3.301    6.075   12.495  0.413
LightGBM                 aggregated  2.993    5.580   11.612  0.374
LSTM                     direct      3.175    5.245   12.734  0.397
LSTM                     aggregated  2.780    5.052   11.394  0.348
Ensemble (static)        aggregated  2.677    4.774   11.140  0.335
Ensemble (regime-aware)  aggregated  2.648    4.713   11.078  0.331

RQ4 ANSWER (section 4-4): aggregating 24 hourly forecasts BEATS direct daily
modelling for LEAR-LASSO, LightGBM and LSTM. SARIMAX is the single exception
and does better direct (3.198 vs 3.269). The ensembles have no direct arm by
construction — they combine hourly member forecasts.

## Diebold–Mariano (section 4-5)

All p-values are HAC-corrected (Newey–West) AND cross-checked with a
circular block bootstrap. Loss differential on absolute errors.

Key one-sided p-values (H1: the first model is more accurate):
- Every model vs naive: p < 1e-15 (all beat naive decisively).
- LEAR-LASSO vs SARIMAX: p = 6.2e-06 (LEAR-LASSO better).
- LSTM vs SARIMAX: p = 3.9e-07 (LSTM better).
- LightGBM vs SARIMAX: p = 1.0e-04 (LightGBM better).
- LSTM vs LEAR-LASSO: p = 0.404 — NOT SIGNIFICANT. A TIE.
- LSTM vs LightGBM: p = 0.040 (LSTM better, marginal).
- Static ensemble vs LEAR-LASSO: p = 4.8e-06 (ensemble better).
- Regime-aware vs static ensemble: p = 0.0226 over all 728 days.

Regime split for the regime-aware vs static comparison:
subset      days   MAE regime-aware   MAE static   DM p (HAC)
all          728        3.557           3.574        0.0226
stressed      77        5.513           5.683        0.0063
calm         651        3.326           3.325        0.8465

## SHAP interpretability (section 4-6, 8pp — the largest results section)

Method: exact TreeSHAP on LightGBM, for both the hourly arm (24 per-hour
models) and the direct-daily arm. Explained over all 728 unseen test days.

CRITICAL METHODOLOGICAL POINT, must appear in the text: the explained model
is NOT the frozen production model. The production models recalibrate at
every origin, and a model fit on the trailing window ending 2017-12-31 would
have seen the whole test period — explaining it would be in-sample. So a
separate interpretation-only model was fitted on the 1092 days ending
2016-01-03, strictly before the test period, and every explained day is
genuinely unseen. State also that this explained fit is STATIC across two
years, so it cannot show drift in feature reliance — figure 10 is not the
importance profile of the model behind the accuracy numbers.

Mean |SHAP| per feature family, EUR/MWh:
family        Hourly calm  Hourly stressed  Daily calm  Daily stressed
price_D-1        3.4833         6.1471        3.1109       5.1335
price_D-2        0.8610         1.2550        0.3778       0.5177
price_D-3        0.8279         1.0302        0.3328       0.3916
price_D-7        1.7454         2.1473        1.0802       1.4784
exog_1_D-1       0.6591         0.6735        0.2479       0.2611
exog_1_D-7       0.7131         0.7487        0.3699       0.3991
exog_1_D0        3.5197         3.3680        3.4393       3.0130
exog_2_D-1       0.9947         0.9221        0.4077       0.3986
exog_2_D-7       0.6740         0.7761        0.2230       0.2754
exog_2_D0        5.8422         5.9761        4.4929       5.0366
dow              0.1464         0.1358        0.0395       0.0379

Findings to build the section around:
1. The day-ahead RENEWABLES forecast (exog_2_D0) is the single largest
   driver overall (5.86 averaged across regimes), ahead of yesterday's
   prices (3.77) and the load forecast (3.50). Day-of-week is negligible.
2. Hour profile is physically coherent: the load forecast dominates the
   07:00 morning ramp, renewables dominate midday and the 18:00 evening
   peak, and yesterday's prices carry the overnight hours.
3. The beeswarm at hour 18 shows the MERIT-ORDER EFFECT directly: high
   renewable forecasts push price down, high previous-day prices push it up.
4. REGIME RESULT, the strongest finding: under stress the model leans much
   harder on persistence. price_D-1 rises from 3.48 to 6.15 (+77%) between
   calm and stressed days, while the fundamentals barely move. This is an
   independent mechanism explaining WHY regime-aware weighting helped in
   section 3-8 — connect the two chapters explicitly.
5. Hourly vs daily: the direct-daily arm relies proportionally less on short
   price lags (price_D-2 falls 0.86 → 0.38) but nearly as much on
   fundamentals (exog_2_D0 5.84 → 4.49). Ties back to RQ4.

Figures available (already exported, English text, you write Farsi captions):
10 global importance · 11 beeswarm at hour 18 · 12 feature-family × hour
heatmap · 13 calm vs stressed · 14 hourly vs daily · 15 waterfall case study.

## OOD stress test (section 5-2, limitations)

Models frozen on the benchmark era (trained through 2017-12-31) applied
WITHOUT recalibration to live 2026 DE-LU data: 173 days, 2026-01-08 to
2026-06-29. Mean price 98.66 EUR/MWh versus 34.69 at training time — roughly
2.8x.

model                     MAE     RMSE    sMAPE   rMAE
naive                    29.183  48.732   45.898  0.808
LEAR-LASSO               39.267  65.314   49.717  1.087
SARIMAX                  41.391  56.504   50.177  1.145
Ensemble (regime-aware)  42.171  53.375   59.404  1.167
Ensemble (static)        44.433  55.519   61.695  1.230
LSTM                     54.929  67.506   79.163  1.520
LightGBM                 66.058  77.349   92.497  1.828

Reading: EVERY trained model exceeds rMAE 1.0 — worse than a naive forecast.
Naive alone stays below it, because it carries no frozen parameters. The
in-era ranking INVERTS: LightGBM and LSTM, the strongest in-sample, degrade
most; SARIMAX and LEAR-LASSO hold up best. This is a genuine and publishable
finding about regime shift, not a failure — write it as such.

## CLAIM DISCIPLINE — non-negotiable, these are honesty constraints

1. Regime-aware vs static ensemble: significant on the 77 stressed days
   (p = 0.0063), and only marginally over all 728 days (p = 0.0226, and the
   bootstrap range across block lengths reaches 0.057). BOTH halves must
   appear together every time this is claimed. Never claim significance at
   the 1% level anywhere in the thesis.
2. On calm days the two ensembles are indistinguishable (p = 0.85). This is
   a sanity check that the mechanism is doing what it should — NOT
   independent corroboration of the gain. Do not present it as support.
3. LSTM vs LEAR-LASSO is a TIE (p = 0.404). Do not write that the deep model
   won. The OOD result shows it is also the more fragile of the two — a
   simple linear model is competitive in-sample and more robust out of it.
   That contrast is one of the thesis's better arguments; make it.
4. Ensemble weights were fitted on the same window the member models were
   tuned on. Disclose this in chapter 3. Phrase the LEAR-LASSO weight
   increase under stress as a shift in FITTED WEIGHTS, not as proof of
   intrinsic superiority.
5. OOD: the regime-aware ensemble still edges the static one (42.17 vs
   44.43), BUT 98.8% of live days fall in one regime — so that is the
   stressed weight set applied throughout, NOT evidence that regime
   switching works out of distribution. Say so.
6. The SHAP case-study day (figure 15) is chosen by the MODEL'S OWN highest
   prediction, deliberately not by the highest realized price. The
   under-forecasting claim rests on an aggregate: mean signed error -9.57
   EUR/MWh across the 73 highest-baseload days. Do not claim
   under-forecasting from the single day alone — that would be circular.
7. The regime threshold is 62.6989 EUR/MWh. An earlier draft used 84.04
   (3-sigma); it was SUPERSEDED because it left only 3 stressed validation
   days. Never write 84.04. Never write "spike" — the label is "stressed".
8. Never quote uncorrected Diebold–Mariano p-values. All reported p-values
   are HAC-corrected.

## The six formal assumptions (must appear in section 3-2, from the
approved proposal)

(1) stationarity, (2) data availability, (3) data quality,
(4) model generalization, (5) stable market conditions,
(6) model interpretability.

Note the honest tension worth writing about: the OOD result directly
challenges assumptions (4) and (5). Address that in 3-2 and return to it in
5-2 rather than leaving the assumptions unexamined.

## Scope notes

Approved additions beyond the original generic proposal, all supervisor-
sanctioned: the named Lago et al. benchmark, hourly forecasting actually
operationalised, the fixed five-model list, the live data feed, significance
testing, SHAP as a real deliverable, the PriceCast tool, regime-aware
ensemble weighting, and the OOD stress test.

The PriceCast tool (section 5-3): a Streamlit application — date picker,
live Energy-Charts fetch, forecast, chart, plus a CSV-upload fallback and an
offline cached-demo mode. It openly displays that its forecasts are worse
than naive on present-day prices, which is the honest framing given the OOD
finding.

## WHAT I MUST GIVE YOU AND HAVE NOT

The exact wording of my four research questions is in the approved proposal,
which is not in this brief. Ask me for it before writing sections 1-3 or
4-7. RQ4 concerns direct versus aggregated daily forecasting; I will supply
RQ1–RQ3 verbatim.

Start by asking me which section to write first, and confirm you have what
you need for it.
```

---

## Notes for me (do not paste)

- The four RQs are the one gap. They live in the approved proposal outside
  this repo — have them to hand before starting 1-3 or 4-7.
- Track pages as they land in the docx:
  `./.venv/Scripts/python.exe scripts/page_quota.py --add N --note "3-3"`
- Existing Farsi drafts to reuse rather than rewrite:
  `thesis/drafts/3-5-evaluation-framework.md` and
  `thesis/drafts/3-6-baseline-models.md` (~1,050 words together).
- Every figure and table referenced above already exists in `reports/`.
  Nothing needs regenerating; the freeze holds.
