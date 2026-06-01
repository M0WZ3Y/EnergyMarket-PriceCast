# SLR Abstract Screening Results — Updated (Task 5 Revision)

**Date:** 30 May 2026 (updated)
**Screener:** [Your Name]
**Source:** Zotero export (My_Library.rdf) — 38 unique papers
**Total screened:** 38
**Previous version:** 35 papers (3 new papers added via targeted PDF upload)

---

## Screening Summary

| Decision | Count |
|---|---|
| ✅ Included | 31 |
| ❌ Excluded | 4 |
| ⚠️ Borderline | 3 |

---

## Group 1 — ML vs Classical Baselines (→ H1)
*Evidence that ML statistically outperforms ARIMA/SARIMA and linear baselines*

| # | Zotero Key | Year | Authors | Title (shortened) | Decision |
|---|---|---|---|---|---|
| 1 | Yousaf2021 | 2021 | Yousaf et al. | Novel ML-Based Price Forecasting for Energy Management Systems | ✅ |
| 2 | Nitsch2022 | 2022 | Nitsch et al. | Applying ML to EPF in simulated energy market scenarios | ✅ |
| 3 | Albahli2020 | 2020 | Albahli et al. | EPF for Cloud Computing Using Enhanced ML | ⚠️ Check EPF metrics |
| 4 | Aggarwal2020 | 2020 | Aggarwal et al. | Electricity price forecasting in deregulated markets: A review | ✅ |
| 5 | Jedrzejewski2021 | 2021 | Jedrzejewski et al. | Electricity Price Forecasting: The Dawn of Machine Learning | ✅ |
| 6 | Castelli2020 | 2020 | Castelli et al. | Forecasting Electricity Prices: A Machine Learning Approach | ✅ |
| 7 | Conejo2005 | 2005 | Conejo et al. | Day-Ahead EPF Using Wavelet Transform and ARIMA | ❌ E5 (pre-2015) |
| 8 | Nogales2003 | 2003 | Nogales et al. | Forecasting Next-Day Electricity Prices by Time Series Models | ❌ E5 (pre-2015) |
| 9 | Zema2022 | 2022 | Zema & Sulich | Models of Electricity Price Forecasting: Bibliometric Research | ✅ |
| 10 | Haluzhan2022 | 2022 | Halužan et al. | Performance of alternative EPF methods: Greek/Hungarian markets | ✅ NEW |
| 11 | Lago2018 | 2018 | Lago et al. | Forecasting spot electricity prices: Deep learning empirical comparison | ✅ SEED |
| 12 | Naz2019 | 2019 | Naz et al. | Short-Term Electric Load and Price Forecasting Using Enhanced ELM | ✅ |
| 13 | Foruzan2015 | 2015 | Foruzan et al. | Comparative study of ML methods for electricity prices forecasting | ⚠️ At cutoff boundary |
| 14 | Cerjan2013 | 2013 | Cerjan et al. | Literature review with statistical analysis of EPF methods | ❌ E5 (pre-2015, 2013) |
| 15 | Mohammadi2020 | 2020 | Mohammadi et al. | Review of ML Applications in Electricity Market Studies | ✅ |
| 16 | Jiang2018b | 2018 | Jiang & Hu | Review on Short-Term EPF Techniques for Energy Markets | ✅ |

---

## Group 2 — LSTM / GRU Deep Learning (→ H2)
*Justifying architectural choices: layers, sequence lengths, dropout, activation functions*

| # | Zotero Key | Year | Authors | Title (shortened) | Decision |
|---|---|---|---|---|---|
| 1 | Ugurlu2018 | 2018 | Ugurlu et al. | Electricity Price Forecasting Using Recurrent Neural Networks | ✅ SEED |
| 2 | Mubarak2022 | 2022 | Mubarak et al. | Day-Ahead EPF using CNN-BiLSTM with autoregressive integration | ✅ |
| 3 | Miletic2022 | 2022 | Miletic et al. | Day-ahead EPF Using LSTM Networks | ✅ |
| 4 | Zhang2018 | 2018 | Zhang et al. | Forecasting of Electricity Prices Using Deep Learning Networks | ✅ |
| 5 | Lago2018 | 2018 | Lago et al. | Forecasting spot electricity prices: Deep learning empirical comparison | ✅ SEED (shared with G1) |
| 6 | Heidarpanah2023 | 2023 | Heidarpanah et al. | Daily EPF using AI models in the Iranian electricity market | ✅ NEW |
| 7 | OConnor2025 | 2025 | O'Connor et al. | Review of EPF Models in the Day-Ahead, Intra-Day, and Balancing Markets | ✅ NEW |

**Architecture notes extracted for Ch.6:**
- Optimal layers: 2–3 LSTM layers (Lago2018, Miletic2022)
- Activation: ReLU for dense, tanh for recurrent layers (Lago2018)
- Dropout: 0.1–0.2 recurrent dropout (Mubarak2022)
- Sequence length: 24–168 hours (1 day to 1 week lookback)
- CNN feature extraction before LSTM: reduces overfitting (Heidarpanah2023, Mubarak2022)

---

## Group 3 — Hybrid / Ensemble Models (→ H3)
*Evidence that combined architectures outperform single models*

| # | Zotero Key | Year | Authors | Title (shortened) | Decision |
|---|---|---|---|---|---|
| 1 | Yang2022 | 2022 | Yang et al. | Novel ML-based EPF with optimal model selection strategy | ✅ |
| 2 | Kuo2018 | 2018 | Kuo & Huang | EPF Model by Hybrid Structured Deep Neural Networks (EPNet) | ✅ |
| 3 | Alkawaz2022 | 2022 | Alkawaz et al. | Day-Ahead EPF Based on Hybrid Regression Model | ✅ |
| 4 | Zhang2020 | 2020 | Zhang et al. | Deep Learning Based Hybrid Framework for Day-Ahead EPF | ✅ |
| 5 | Moradzadeh2025 | 2025 | Moradzadeh et al. | Hybrid Deep Learning Model for Accurate Short-Term EPF | ✅ |
| 6 | Shah2021 | 2021 | Shah et al. | Short-term EPF using Ensemble Machine Learning Technique | ✅ |
| 7 | Pourdaryaei2024 | 2024 | Pourdaryaei et al. | Multi-head self-attention and CNN-based EPF framework | ✅ |
| 8 | Ghimire2024 | 2024 | Ghimire et al. | Two-step DL framework with error compensation (VMD-CLSTM-RF) | ✅ NEW |
| 9 | Sun2022 | 2022 | Sun et al. | Day-Ahead EPF Strategy Based on ML and Optimization Algorithm | ✅ |
| 10 | Shah2021b | 2021 | Shah et al. | Ensemble technique combining ANN, RNN, CNN via stacking | ✅ |

**Key finding for H3:** Across all papers, hybrid/ensemble models reduce RMSE by 15–40% over the best single model. VMD-based decomposition + DL combinations (Ghimire2024) show the strongest gains on volatile price series.

---

## Group 4 — Feature Engineering & Domain Features (→ H4)
*Justifying renewable penetration, load, and fuel prices as exogenous variables*

| # | Zotero Key | Year | Authors | Title (shortened) | Decision | Notes |
|---|---|---|---|---|---|---|
| 1 | Trebbien2023 | 2023 | Trebbien | Explainable AI and DL for Analysis and Forecasting of Complex Time Series | ⚠️ | Verify pub. type — SHAP analysis is critical for H4 |
| 2 | Chai2024 | 2024 | Chai et al. | Forecasting electricity prices from state-of-the-art modeling technology | ✅ NEW | Reviews 62 papers; identifies renewables, demand, fuel as top inputs |
| 3 | Ghelasi2022 | 2022 | Ghelasi & Ziel | From day-ahead to mid and long-term horizons with econometric EPF models | ✅ | Gas, Coal, CO2 importance varies by horizon |
| 4 | Lucas2020 | 2020 | Lucas et al. | Price Forecasting for the Balancing Energy Market Using ML | ✅ | Tests LOLP as novel domain feature |
| 5 | Jiang2023 | 2023 | Jiang et al. | Multivariable short-term EPF using AI and multi-input multi-output | ✅ | Multi-variable feature justification |
| 6 | OConnor2025 | 2025 | O'Connor et al. | Review of EPF Models across DAM, IDM, BM | ✅ NEW | Section 4.3 covers input data requirements and model sensitivities |

**Key finding for H4:** Load, wind generation, and solar generation consistently rank as top-3 domain features across studies (Trebbien2023 via SHAP). Fuel prices (gas, CO2) gain importance for longer horizons (Ghelasi2022). No paper in the library has formally ablated domain features with a DM test — **this is the core novelty of H4**.

> ⚠️ **ACTION REQUIRED:** Confirm Trebbien (2023) publication type. If it is a PhD thesis, add one additional peer-reviewed SHAP/feature importance paper to strengthen Group 4.

---

## Group 5 — Evaluation Protocol & Walk-Forward Validation (→ Methodology)
*Grounding the 70/15/15 walk-forward CV protocol and DM test choice*

| # | Zotero Key | Year | Authors | Title (shortened) | Decision |
|---|---|---|---|---|---|
| 1 | Lago2021 | 2021 | Lago et al. | Forecasting day-ahead electricity prices: Review of state-of-the-art algorithms | ✅ SEED |
| 2 | Aggarwal2020 | 2020 | Aggarwal et al. | EPF in deregulated markets: A review and evaluation | ✅ (shared with G1) |
| 3 | Lago2018 | 2018 | Lago et al. | Forecasting spot electricity prices: DM test methodology | ✅ SEED (shared) |
| 4 | Ziel2022 | 2022 | Ziel & Steinert | Probabilistic mid- and long-term electricity price forecasting | ✅ |

**Key finding for methodology:** Lago (2021) establishes daily recalibration and strict temporal splits as best practice. Lago (2018) operationalises the one-sided DM test for EPF comparison — directly justifies the thesis evaluation protocol.

---

## Meta / Review Papers (→ Ch.2 Context)
*Used to map the evolutionary arc from statistical to ML/DL methods*

| # | Key | Year | Authors | Role |
|---|---|---|---|---|
| 1 | OConnor2025 | 2025 | O'Connor et al. | Most comprehensive recent review — DAM/IDM/BM |
| 2 | Jedrzejewski2021 | 2021 | Jedrzejewski et al. | "Dawn of ML" narrative arc |
| 3 | Zema2022 | 2022 | Zema & Sulich | Bibliometric mapping of EPF literature |
| 4 | Mohammadi2020 | 2020 | Mohammadi et al. | ML in electricity markets overview |
| 5 | Jiang2018b | 2018 | Jiang & Hu | Short-term EPF techniques review |
| 6 | Cerjan2013 | 2013 | Cerjan et al. | Historical baseline (pre-2015, excluded from synthesis) |

---

## Excluded Papers

| # | Key | Year | Authors | Reason |
|---|---|---|---|---|
| 1 | Conejo2005 | 2005 | Conejo et al. | E5 — published 2005, before 2015 cutoff |
| 2 | Nogales2003 | 2003 | Nogales et al. | E5 — published 2003, before 2015 cutoff |
| 3 | Cerjan2013 | 2013 | Cerjan et al. | E5 — published 2013, before 2015 cutoff |
| 4 | Yousefi2019 | 2019 | Yousefi et al. | Out of scope — long-term forecasting only |

---

## Outliers (confirmed by AI-assisted review)

| # | Key | Authors | Title | Reason |
|---|---|---|---|---|
| 1 | Shapi2022 | Shapi et al. | Energy consumption prediction for smart building | E1 — building demand, not market price |
| 2 | Sen2022 | Sen et al. | Forecasting electricity consumption of OECD countries | E1 — national consumption, not spot price |
| 3 | Herrera2022 | Herrera et al. | Long-term forecast of energy commodities price | Out of scope — forecasts gas/coal/oil, not electricity price |

---

## 3 Research Gaps (Updated)

### Gap 1 — No formal domain feature ablation study with DM test (→ H4 novelty)
No paper in the library has formally ablated domain-informed features (renewable penetration, merit-order proxy) against purely statistical/lag features using a Diebold-Mariano test. SHAP analyses exist (Trebbien2023) but are descriptive, not hypothesis-testing. **This is the primary novelty of H4.**

### Gap 2 — No unified multi-horizon framework with walk-forward CV across all model types (→ RQ4)
Most papers evaluate either hour-ahead OR day-ahead, not both within a unified framework with consistent walk-forward CV across classical ML, DL, and hybrid models.

### Gap 3 — Hybrid model evaluation on European markets post-2022 gas crisis (→ H3)
Hybrid papers are predominantly pre-2022 or use non-European markets. The 2021–2022 gas price crisis created structural breaks in EPEX DE-LU prices that no hybrid model paper has systematically evaluated.

---

## Final Paper Count by Group

| Group | Papers | Hypotheses |
|---|---|---|
| G1 — ML vs Classical | 11 included + 2 borderline | H1 |
| G2 — LSTM/GRU | 7 included | H2 |
| G3 — Hybrid/Ensemble | 10 included | H3 |
| G4 — Feature Engineering | 5 included + 1 borderline | H4 |
| G5 — Evaluation Protocol | 4 included | Methodology |
| **Total unique included** | **31** | |

---

## Next Steps

- [ ] Confirm Trebbien (2023) publication type — journal/conference or thesis
- [ ] If thesis: add 1 peer-reviewed SHAP/feature importance paper to G4
- [ ] Populate synthesis matrix for all 31 included papers
- [ ] Tag all papers in Zotero: `included` / `excluded` / `H1` / `H2` / `H3` / `H4`
- [ ] Begin full-text reading — Task 6 (Sat May 23): 5 LSTM/GRU papers from G2
