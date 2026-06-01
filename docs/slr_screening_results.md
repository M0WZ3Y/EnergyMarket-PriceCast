# SLR Abstract Screening Results — Task 5

**Date:** 30 May 2026  
**Screener:** [Your Name]  
**Source:** Zotero export (My_Library.rdf)  
**Total unique papers screened:** 35  

---

## Screening Summary

| Decision | Count |
|---|---|
| ✅ Included | 24 |
| ❌ Excluded | 6 |
| ⚠️ Borderline (needs full-text check) | 5 |

---

## Group 1 — ML vs Classical Baselines (→ H1)

| ID | Year | Authors | Title | Decision |
|---|---|---|---|---|
| SLR-003 | 2021 | Yousaf et al. | A Novel Machine Learning-Based Price Forecasting for Energy Management Systems | ✅ |
| SLR-005 | 2022 | Nitsch et al. | Applying ML to electricity price forecasting in simulated energy market scenarios | ✅ |
| SLR-007 | 2005 | Conejo et al. | Day-Ahead EPF Using Wavelet Transform and ARIMA | ❌ E5 (pre-2015) |
| SLR-009 | 2020 | Albahli et al. | Electricity Price Forecasting for Cloud Computing | ⚠️ Check EPF metrics |
| SLR-010 | 2020 | Aggarwal et al. | Electricity price forecasting in deregulated markets: A review | ✅ |
| SLR-011 | 2021 | Jedrzejewski et al. | Electricity Price Forecasting: The Dawn of Machine Learning | ✅ |
| SLR-016 | 2020 | Castelli et al. | Forecasting Electricity Prices: A Machine Learning Approach | ✅ |
| SLR-017 | 2003 | Nogales et al. | Forecasting Next-Day Electricity Prices by Time Series Models | ❌ E5 (pre-2015) |
| SLR-021 | 2022 | Zema & Sulich | Models of Electricity Price Forecasting: Bibliometric Research | ✅ |
| SLR-023 | 2020 | Lucas et al. | Price Forecasting for the Balancing Energy Market Using ML | ⚠️ Check day-ahead scope |
| SLR-025 | 2019 | Naz et al. | Short-Term Electric Load and Price Forecasting Using Enhanced ELM | ✅ |
| SLR-027 | 2020 | Mohammadi et al. | A Review of ML Applications in Electricity Market Studies | ✅ |
| SLR-028 | 2018 | Jiang & Hu | A Review on Short-Term EPF Techniques for Energy Markets | ✅ |
| SLR-030 | 2022 | Sun et al. | Day-Ahead EPF Strategy Based on ML and Optimization Algorithm | ✅ |
| SLR-035 | 2015 | Foruzan et al. | A comparative study of ML methods for electricity prices forecasting | ⚠️ At inclusion boundary |

---

## Group 2 — LSTM / GRU Deep Learning (→ H2)

| ID | Year | Authors | Title | Decision |
|---|---|---|---|---|
| SLR-001 | 2024 | Pourdaryaei et al. | Multi-head self-attention and CNN-based EPF framework | ✅ |
| SLR-006 | 2022 | Mubarak et al. | Day-Ahead EPF using CNN-BiLSTM with autoregressive integration | ✅ |
| SLR-012 | 2018 | Ugurlu et al. | Electricity Price Forecasting Using Recurrent Neural Networks | ✅ |
| SLR-014 | 2023 | Trebbien | Explainable AI and Deep Learning for Forecasting of Complex Time Series | ⚠️ Grey lit — check citations |
| SLR-018 | 2018 | Lago et al. | Forecasting spot electricity prices: Deep learning empirical comparison | ✅ SEED |
| SLR-029 | 2022 | Miletic et al. | Day-ahead Electricity Price Forecasting Using LSTM Networks | ✅ |
| SLR-031 | 2018 | Zhang et al. | Forecasting of Electricity Prices Using Deep Learning Networks | ✅ |

---

## Group 3 — Hybrid / Ensemble Models (→ H3)

| ID | Year | Authors | Title | Decision |
|---|---|---|---|---|
| SLR-002 | 2022 | Yang et al. | Novel ML-based EPF with optimal model selection strategy | ✅ |
| SLR-004 | 2018 | Kuo & Huang | EPF Model by Hybrid Structured Deep Neural Networks | ✅ |
| SLR-008 | 2022 | Alkawaz et al. | Day-Ahead EPF Based on Hybrid Regression Model | ✅ |
| SLR-026 | 2020 | Zhang et al. | Deep Learning Based Hybrid Framework for Day-Ahead EPF | ✅ |
| SLR-032 | 2025 | Moradzadeh et al. | Hybrid Deep Learning Model for Accurate Short-Term EPF | ✅ |
| SLR-034 | 2021 | Shah et al. | Short-term EPF using Ensemble Machine Learning Technique | ✅ |

---

## Group 4 — Feature Engineering & Domain Features (→ H4)

| ID | Year | Authors | Title | Decision |
|---|---|---|---|---|
| SLR-022 | 2023 | Jiang et al. | Multivariable short-term EPF using AI and multi-input multi-output | ✅ |

> ⚠️ **GAP ALERT:** Only 1 paper explicitly addresses multi-variable domain features for EPF in the current library. This is thin for H4 justification. Recommendation: add targeted searches for feature engineering papers in Week 3 (Mon Jun 1 task). Search string: `"electricity price" AND ("feature selection" OR "feature importance" OR "renewable penetration" OR "merit order" OR "SHAP")`.

---

## Group 5 — Evaluation Protocol / Walk-forward Validation (→ methodology)

| ID | Year | Authors | Title | Decision |
|---|---|---|---|---|
| SLR-019 | 2022 | Ghelasi & Ziel | From day-ahead to mid and long-term horizons with econometric EPF models | ✅ |

> ⚠️ **GAP ALERT:** Only 1 paper on evaluation protocol. Add walk-forward validation papers in Week 2 (Fri May 27 task).

---

## Excluded Papers

| ID | Year | Authors | Reason |
|---|---|---|---|
| SLR-007 | 2005 | Conejo et al. | E5 — published 2005, before 2015 cutoff |
| SLR-013 | 2022 | Shapi et al. | E1 — forecasts building energy consumption, not electricity price |
| SLR-015 | 2022 | Sen et al. | E1 — forecasts country-level electricity consumption, not price |
| SLR-017 | 2003 | Nogales et al. | E5 — published 2003, before 2015 cutoff |
| SLR-024 | 2022 | Ziel & Steinert | Out of scope — probabilistic mid/long-term only |
| SLR-033 | 2019 | Yousefi et al. | Out of scope — long-term forecasting only |

---

## Borderline Papers — Action Required

| ID | Paper | Action needed |
|---|---|---|
| SLR-009 | Albahli et al. (2020) | Open full text — verify electricity price (not just cloud load) metrics reported |
| SLR-014 | Trebbien (2023) | Check if published in journal or conference; check Google Scholar citation count |
| SLR-020 | Herrera et al. (2022) | Open abstract — confirm electricity price is one of the commodities forecasted |
| SLR-023 | Lucas et al. (2020) | Open full text — balancing market paper; check if day-ahead prices also covered |
| SLR-035 | Foruzan et al. (2015) | Open full text — confirm ≥6 pages and sufficient methodological detail |

---

## 3 Candidate Research Gap Areas

Based on screening of all 35 papers, three research gaps emerge:

### Gap 1 — Lack of domain feature ablation studies (→ directly motivates H4)
No paper in the library explicitly ablates domain-informed features (renewable penetration, merit-order proxy) against purely statistical/lag features using a formal statistical test. Most feature importance analyses use SHAP or permutation importance descriptively rather than hypothesis-testing the incremental value of domain features. **This is the core novelty of H4.**

### Gap 2 — Limited multi-horizon single-framework evaluation (→ motivates RQ4)
Most papers study either hour-ahead OR day-ahead forecasting, rarely both within a unified framework with consistent evaluation. Papers that do cover both horizons (e.g. Lago et al. 2018) do not use walk-forward validation consistently across all models. A single framework benchmarking all 9 models across both horizons with walk-forward CV is underrepresented.

### Gap 3 — Recency gap in hybrid model evaluation on European markets post-2022 (→ motivates H3)
Hybrid and ensemble papers are predominantly pre-2022 or use non-European markets (Indian grid, Ontario). The 2021–2022 gas price crisis created structural breaks in European electricity prices that no hybrid model paper has systematically evaluated. Testing hybrid/ensemble superiority on DE-LU data spanning the gas crisis is a clear contribution.

---

## Next Steps

- [ ] Resolve 5 borderline papers via full-text check
- [ ] Add 4+ feature engineering papers (targeted search — Mon Jun 1)
- [ ] Add 4 walk-forward validation papers (Fri May 27 task)
- [ ] Populate synthesis matrix for all 24 confirmed papers
- [ ] Tag all papers in Zotero: `included` / `excluded` / `H1` / `H2` / `H3` / `H4`
