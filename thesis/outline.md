# Thesis outline (reference for Claude Code)

Farsi body, 100-page budget, mapped to the official Amirkabir template.
Full skeleton (docx) lives outside this repo; this file is the
machine-readable map so generated tables/figures/results land in the
right section.

When asked to "add this result/figure/table to the thesis" or similar,
use this file to identify the correct section number and expected page
budget — don't guess.

## فصل اول: مقدمه [7pp]
- 1-1 انگیزه و اهمیت موضوع [2pp]
- 1-2 بیان مسئله [1pp]
- 1-3 سؤالات پژوهش [1pp] — the 4 RQs
- 1-4 نوآوری‌ها و دستاوردهای پژوهش [1pp]
- 1-5 ساختار پایان‌نامه [1pp]
- 1-6 محدوده و مفروضات کلی پروژه [1pp]

## فصل دوم: مروری بر پیشینه پژوهش [17pp]
Maps to the 8 Zotero subcollections, ~2pp each:
2-1 reviews/bibliometric · 2-2 classical/statistical · 2-3 classical ML ·
2-4 deep learning · 2-5 hybrid/attention · 2-6 explainability ·
2-7 applied/market-specific (incl. Iran) · 2-8 long-term/probabilistic ·
2-9 gap & positioning [2pp]

## فصل سوم: روش تحقیق [37pp] — code-heavy chapter
- 3-1 مقدمه [1pp]
- 3-2 فرضیات بنیادی پژوهش [3pp] — the 6 formal assumptions
- 3-3 داده‌ها و بازار مورد مطالعه [8pp]
  - 3-3-1 بنچمارک EPEX-DE [3pp] — BenchmarkLoader output, verify_dataset.py report
  - 3-3-2 داده زنده Energy-Charts [2pp] — EnergyChartsLoader, attribution line
  - 3-3-3 تحلیل اکتشافی داده [3pp] — EDA notebook findings (week 2)
- 3-4 مهندسی ویژگی [7pp] — src/features/ pipeline, leakage tests
- 3-5 چارچوب اعتبارسنجی و معیارهای ارزیابی [4pp] — src/evaluation/
- 3-6 مدل‌های پایه [4pp] — naive, SARIMAX, LEAR-LASSO (week 4)
- 3-7 مدل‌های یادگیری ماشین و یادگیری عمیق [7pp]
  - 3-7-1 LightGBM [3pp] (week 5)
  - 3-7-2 LSTM [4pp] (week 6)
- 3-8 مدل ترکیبی [3pp] — ensemble (week 7)

## فصل چهارم: نتایج و تحلیل [29pp] — results-heavy chapter
- 4-1 مقدمه [1pp]
- 4-2 نتایج پیش‌بینی ساعتی [6pp] — canonical results table, hourly target
- 4-3 نتایج پیش‌بینی روزانه [5pp] — daily target
- 4-4 مقایسه روش مستقیم و تجمیعی [2pp] — direct vs. aggregated daily (RQ4)
- 4-5 آزمون دیبولد-ماریانو [3pp] — significance tests
- 4-6 تحلیل تفسیرپذیری SHAP [8pp] — week 8, calm-vs-spike, hourly-vs-daily
- 4-7 پاسخ به سؤالات پژوهش [4pp] — explicit answers to the 4 RQs

## فصل پنجم: جمع‌بندی، بحث و پیشنهادات [10pp]
- 5-1 جمع‌بندی نتایج [2pp]
- 5-2 محدودیت‌ها [2pp] — incl. benchmark-era vs. live-data regime shift
- 5-3 ابزار EnergyMarket-PriceCast [2pp] — week 11 tool, live screenshot
- 5-4 پیشنهادات برای تحقیقات آینده [2pp]
- 5-5 نتیجه‌گیری نهایی [2pp]

Total: 7+17+37+29+10 = 100pp
