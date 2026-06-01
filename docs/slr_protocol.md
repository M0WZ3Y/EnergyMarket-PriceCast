# Systematic Literature Review Protocol
## PRISMA-Compliant Search Strategy

**Project:** EnergyMarket-PriceCast — Daily and Hourly Electricity Price Forecasting Using Machine Learning  
**Date:** 21 May 2026  
**Author:** [Your Name]  
**Version:** 1.0 — Final

---

## 1. Review Objectives

This SLR identifies, screens, and synthesises empirical studies on machine learning and deep learning methods for electricity price forecasting. The review directly informs:

- **Ch.2** — Literature review and research gap statement
- **Ch.3** — Justification of methodology choices (models, features, evaluation protocol)
- **Ch.6** — Model architecture decisions (layers, hyperparameters, sequence lengths)
- **H1–H4** — Empirical grounding for all four hypotheses

---

## 2. Research Questions Driving the SLR

| SLR-RQ | Purpose |
|---|---|
| Which ML/DL models have been applied to electricity price forecasting and with what results? | Grounds H1, H2 — benchmark selection |
| Which input features are most commonly used and validated? | Grounds H4 — domain feature justification |
| What evaluation protocols are used? | Grounds walk-forward CV choice |
| What are the documented limitations and research gaps? | Grounds the gap statement (end of Ch.2) |
| Which hybrid/ensemble approaches outperform single models? | Grounds H3 |

---

## 3. Databases Searched

| Database | URL | Rationale |
|---|---|---|
| IEEE Xplore | https://ieeexplore.ieee.org | Primary venue for power systems + ML papers |
| ScienceDirect (Elsevier) | https://www.sciencedirect.com | Applied Energy, Energy, IJEPES — top EPF journals |
| Web of Science | https://www.webofscience.com | Cross-disciplinary coverage, citation metrics |
| Google Scholar | https://scholar.google.com | Catch-all for preprints and conference papers |
| arXiv (cs.LG, eess.SP) | https://arxiv.org | Recent deep learning preprints |

**Primary databases:** IEEE Xplore + ScienceDirect (this session)  
**Secondary databases:** Web of Science + Google Scholar (Week 2–3 sessions)

---

## 4. Search Strings

### 4.1 Core Search String (All Databases)

```
("electricity price" OR "energy price" OR "power price" OR "day-ahead price")
AND
("forecasting" OR "prediction" OR "forecast")
AND
("machine learning" OR "deep learning" OR "neural network" OR "LSTM" OR "GRU"
 OR "random forest" OR "XGBoost" OR "LightGBM" OR "gradient boosting"
 OR "support vector" OR "ensemble" OR "hybrid")
```

### 4.2 IEEE Xplore — Adapted String

```
("All Metadata": "electricity price forecast*") AND
("All Metadata": "machine learning" OR "deep learning" OR "LSTM" OR "XGBoost"
 OR "random forest" OR "LightGBM" OR "neural network" OR "ensemble")
```

**Filters applied:**
- Publication years: 2015–2026
- Content type: Journals + Conference papers
- Topic area: Power, Energy & Industry Applications + Computing & Processing

### 4.3 ScienceDirect — Adapted String

```
Title, abstract, keywords:
("electricity price" OR "day-ahead price") AND ("forecast" OR "prediction") AND
("machine learning" OR "deep learning" OR "LSTM" OR "XGBoost" OR "random forest"
 OR "gradient boosting" OR "ensemble" OR "hybrid model")
```

**Filters applied:**
- Years: 2015–2026
- Subject areas: Energy, Computer Science, Engineering
- Article type: Research articles + Review articles

### 4.4 Topic-Specific Supplementary Strings

Run these in Week 2–3 for targeted paper collection:

| Topic | Search String | Target papers |
|---|---|---|
| LSTM/GRU for EPF | `"electricity price" AND ("LSTM" OR "GRU" OR "recurrent neural")` | 5 papers (Sat May 23) |
| XGBoost/LightGBM for EPF | `"electricity price" AND ("XGBoost" OR "LightGBM" OR "gradient boosting")` | 5 papers (Mon May 25) |
| Walk-forward validation | `"electricity price" AND ("walk-forward" OR "rolling origin" OR "temporal cross-validation")` | 4 papers (Fri May 27) |
| Hybrid/ensemble EPF | `"electricity price" AND ("hybrid" OR "ensemble" OR "stacking" OR "blending") AND "forecast"` | 4 papers (Sat May 30) |
| Price spike forecasting | `"electricity price spike" AND ("forecast" OR "detection" OR "machine learning")` | 4 papers (Mon Jun 1) |

---

## 5. Inclusion Criteria

A paper is **included** if it meets ALL of the following:

| # | Criterion | Rationale |
|---|---|---|
| I1 | Studies electricity price forecasting (day-ahead or hour-ahead) | Direct relevance to thesis RQs |
| I2 | Applies at least one ML, DL, or statistical forecasting method | Methods alignment |
| I3 | Reports at least one quantitative accuracy metric (MAE, RMSE, MAPE, or equivalent) | Enables synthesis matrix comparison |
| I4 | Published in a peer-reviewed journal or top-tier conference (IEEE, Elsevier, Springer) | Quality threshold |
| I5 | Published between 2015 and 2026 | Recency — pre-2015 ML EPF literature is superseded |
| I6 | Full text available in English | Language constraint |

---

## 6. Exclusion Criteria

A paper is **excluded** if it meets ANY of the following:

| # | Criterion | Rationale |
|---|---|---|
| E1 | Forecasts electricity demand/load only (not price) | Different problem domain |
| E2 | Forecasts renewable energy generation only (not price) | Out of scope |
| E3 | Uses only classical statistical methods with no ML comparison | Below methods scope |
| E4 | Does not report numerical accuracy metrics | Cannot be synthesised |
| E5 | Published before 2015 | Predates modern ML/DL methods in EPF |
| E6 | Grey literature (reports, theses, working papers) unless highly cited (>100 citations) | Quality threshold |
| E7 | Duplicate — same study appearing in multiple databases | Deduplication |
| E8 | Short paper < 6 pages (conference abstracts, extended abstracts) | Insufficient methodological detail |

---

## 7. Screening Process (PRISMA Flow)

```
IDENTIFICATION
└── Records from database searches (IEEE Xplore + ScienceDirect + WoS + Scholar)
    └── Duplicates removed
            ↓
SCREENING
└── Title + abstract screening against I1–I6 / E1–E8
    └── Records excluded with reason
            ↓
ELIGIBILITY
└── Full-text assessment
    └── Records excluded with reason
            ↓
INCLUDED
└── Studies included in synthesis
    └── Synthesis matrix populated
```

**Target:** 40–60 papers included in final synthesis

---

## 8. Data Extraction Fields (Synthesis Matrix)

For each included paper, record:

| Field | Description |
|---|---|
| ID | Sequential ID (SLR-001, SLR-002, ...) |
| Author(s) | Last name(s) + year |
| Title | Full title |
| Journal/Conference | Publication venue |
| Year | Publication year |
| Market | Electricity market studied (EPEX, PJM, Nordpool, etc.) |
| Horizon | Forecasting horizon (hour-ahead, day-ahead, week-ahead) |
| Models | All models applied |
| Best model | Top-performing model reported |
| Features | Input features used |
| Domain features | Specific domain-informed features (yes/no + which) |
| Evaluation protocol | Train/test split method (walk-forward, single split, k-fold) |
| Metrics | Accuracy metrics reported |
| Best RMSE | Best RMSE reported (standardise units if possible) |
| Best MAE | Best MAE reported |
| Handles spikes | Whether spike events are analysed separately |
| Key finding | One-sentence summary of main contribution |
| Gap identified | Any research gap stated by authors |
| Relevance to thesis | H1 / H2 / H3 / H4 / RQ1–RQ4 |
| Zotero key | Zotero citation key |

---

## 9. Quality Assessment

Each included paper is rated on:

| Dimension | Score (1–3) | Criteria |
|---|---|---|
| Methodological rigour | 1–3 | Walk-forward CV (3), proper train/test (2), single split (1) |
| Reproducibility | 1–3 | Code + data available (3), data only (2), neither (1) |
| Comparability | 1–3 | Compares ≥3 models (3), 2 models (2), single model (1) |
| Market relevance | 1–3 | European market (3), US market (2), other (1) |

Papers scoring ≤5 total are flagged for lower weight in synthesis.

---

## 10. Zotero Setup Instructions

1. Create Zotero collection: `MSc Thesis > SLR > EPF_ML`
2. Sub-collections:
   - `01_LSTM_GRU`
   - `02_XGBoost_LightGBM_RF`
   - `03_Walk_Forward_Validation`
   - `04_Hybrid_Ensemble`
   - `05_Price_Spikes`
   - `06_Excluded`
3. Use Zotero citation key format: `AuthorYear` (e.g., `Lago2018`)
4. Add tags: `included`, `excluded`, `pending`, `H1`, `H2`, `H3`, `H4`
5. Use Zotero notes field to paste the synthesis matrix row for each paper

---

## 11. Search Session Log

| Session | Date | Database | String used | Raw results | After dedup | After title/abstract screen | Notes |
|---|---|---|---|---|---|---|---|
| S01 | 21 May 2026 | IEEE Xplore | String 4.2 | — | — | — | First search session |
| S02 | 21 May 2026 | ScienceDirect | String 4.3 | — | — | — | First search session |
| S03 | TBD | Web of Science | String 4.1 | — | — | — | Week 2 |
| S04 | TBD | Google Scholar | String 4.1 | — | — | — | Week 2 |

*Fill in raw result counts after running each search.*

---

## 12. References

- Moher, D., et al. (2009). Preferred Reporting Items for Systematic Reviews and Meta-Analyses: The PRISMA Statement. *PLoS Medicine*, 6(7), e1000097.
- Weron, R. (2014). Electricity price forecasting: A review. *International Journal of Forecasting*, 30(4), 1030–1081.
- Lago, J., et al. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.
