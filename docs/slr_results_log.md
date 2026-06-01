# SLR Results Log — First Search Session

**Date:** 21 May 2026  
**Databases:** IEEE Xplore + ScienceDirect  
**Screener:** [Your Name]

---

## Session S01 — IEEE Xplore

**Search string used:**
```
("All Metadata": "electricity price forecast*") AND
("All Metadata": "machine learning" OR "deep learning" OR "LSTM" OR "XGBoost"
 OR "random forest" OR "LightGBM" OR "neural network" OR "ensemble")
```
**Filters:** Years 2015–2026 · Journals + Conferences  
**Raw results:** ___ papers  
**After duplicate removal:** ___ papers  
**After title/abstract screen:** ___ included · ___ excluded

---

## Session S02 — ScienceDirect

**Search string used:**
```
("electricity price" OR "day-ahead price") AND ("forecast" OR "prediction") AND
("machine learning" OR "deep learning" OR "LSTM" OR "XGBoost" OR "random forest"
 OR "gradient boosting" OR "ensemble" OR "hybrid model")
```
**Filters:** Years 2015–2026 · Energy + Computer Science · Research + Review articles  
**Raw results:** ___ papers  
**After duplicate removal:** ___ papers  
**After title/abstract screen:** ___ included · ___ excluded

---

## Abstract Screening Log

> Screen titles and abstracts against inclusion/exclusion criteria.
> Mark each: ✅ Include · ❌ Exclude (+ reason code) · ⏳ Pending full-text

| # | Zotero Key | Authors | Year | Title (shortened) | Database | Decision | Reason if excluded | Hypothesis |
|---|---|---|---|---|---|---|---|---|
| 001 | | | | | IEEE | | | |
| 002 | | | | | IEEE | | | |
| 003 | | | | | IEEE | | | |
| 004 | | | | | IEEE | | | |
| 005 | | | | | IEEE | | | |
| 006 | | | | | IEEE | | | |
| 007 | | | | | IEEE | | | |
| 008 | | | | | IEEE | | | |
| 009 | | | | | IEEE | | | |
| 010 | | | | | IEEE | | | |
| 011 | | | | | SD | | | |
| 012 | | | | | SD | | | |
| 013 | | | | | SD | | | |
| 014 | | | | | SD | | | |
| 015 | | | | | SD | | | |
| 016 | | | | | SD | | | |
| 017 | | | | | SD | | | |
| 018 | | | | | SD | | | |
| 019 | | | | | SD | | | |
| 020 | | | | | SD | | | |
| 021 | | | | | SD | | | |
| 022 | | | | | SD | | | |
| 023 | | | | | SD | | | |
| 024 | | | | | SD | | | |
| 025 | | | | | SD | | | |
| 026 | | | | | SD | | | |
| 027 | | | | | SD | | | |
| 028 | | | | | SD | | | |
| 029 | | | | | SD | | | |
| 030 | | | | | SD | | | |

---

## Exclusion Reason Codes

| Code | Meaning |
|---|---|
| E1 | Forecasts demand/load only, not price |
| E2 | Forecasts renewable generation only |
| E3 | Classical methods only, no ML comparison |
| E4 | No numerical accuracy metrics reported |
| E5 | Published before 2015 |
| E6 | Grey literature |
| E7 | Duplicate |
| E8 | Short paper < 6 pages |

---

## Confirmed Seed Papers (Pre-Identified — Auto-Include)

These high-impact papers are confirmed inclusions regardless of search — they form the backbone of the literature review:

| Zotero Key | Authors | Year | Title | Journal | Relevance |
|---|---|---|---|---|---|
| Weron2014 | Weron | 2014 | Electricity price forecasting: A review of the state-of-the-art | Int. J. Forecasting | All RQs — foundational review |
| Lago2018 | Lago et al. | 2018 | Forecasting spot electricity prices: Deep learning approaches and empirical comparison | Applied Energy | H1, H2 — benchmark comparison |
| Lago2021 | Lago et al. | 2021 | Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms | Applied Energy | All RQs — most comprehensive recent review |
| Uniejewski2019 | Uniejewski et al. | 2019 | Variance stabilizing transformations for electricity spot price forecasting | IEEE Trans. Power Syst. | H1 — preprocessing |
| Marcjasz2020 | Marcjasz et al. | 2020 | Beating the naive — combining LASSO with naive intraday electricity price forecasts | Energies | H1 — LASSO baseline |
| Qian2019 | Qian et al. | 2019 | A review and discussion of decomposition-based hybrid models for wind energy forecasting | Applied Energy | H3 — hybrid models |
| Zhang2022 | Zhang et al. | 2022 | Day-ahead electricity price forecasting with LSTM | Energy Reports | H2 — LSTM |
| Nowotarski2018 | Nowotarski & Weron | 2018 | Recent advances in electricity price forecasting | Renewable & Sustainable Energy Reviews | All RQs — review |

---

## Running Totals

| Stage | Count |
|---|---|
| Total identified (all databases, this session) | |
| Duplicates removed | |
| Title/abstract screened | |
| Excluded at title/abstract | |
| Pending full-text | |
| Confirmed included (seed papers) | 8 |
| **Total included so far** | **8** |

---

## Notes for Next Session (Fri May 22)

- Screen first 30 abstracts from this log against I/E criteria
- Log accepted papers with: method used · dataset · metrics reported
- Identify 3 candidate research gap areas from excluded/accepted papers
- Add all accepted papers to Zotero under `EPF_ML` collection with correct tags
