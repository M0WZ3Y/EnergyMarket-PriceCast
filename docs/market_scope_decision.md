# Market Scope Decision — EnergyMarket-PriceCast

**Date:** 20 May 2026  
**Author:** [Your Name]  
**Status:** Final — approved for data collection

---

## Decision: EPEX SPOT Germany/Luxembourg (DE-LU)

After evaluating both candidate markets, **EPEX SPOT Germany/Luxembourg (DE-LU)** is selected as the primary market for this thesis.

---

## Evaluation Criteria

| Criterion | EPEX SPOT DE-LU | Nordpool Nordic |
|---|---|---|
| Market size | Largest electricity market in Europe by volume | Smaller, hydro-dominated |
| Price volatility | High — driven by gas, coal, renewable intermittency | Lower — hydro buffering dampens spikes |
| Data availability | Full hourly day-ahead prices 2015–present via ENTSO-E | Available but requires separate Nordpool API |
| Negative prices | Frequent — adds forecasting challenge and thesis novelty | Rare |
| Academic literature | Extensively studied — 80%+ of benchmark papers use DE | Less represented in ML forecasting literature |
| Generation mix complexity | High (gas, coal, wind, solar, nuclear, hydro) | Simpler (hydro-dominant) |
| Renewable penetration | High and growing — wind + solar create nonlinearity | High hydro share but less intermittent |
| Fuel price sensitivity | High gas/coal exposure — merit order clearly observable | Lower exposure |
| Regulatory context | Single-price zone, well-documented rules | Multi-zone, more complex |

**Verdict:** EPEX DE-LU offers the richest forecasting environment — higher volatility, more complex generation mix, frequent negative prices, and the deepest academic benchmark literature. This maximises both thesis novelty and comparability with prior work.

---

## Market Specification

| Parameter | Value |
|---|---|
| Market | EPEX SPOT Day-Ahead Auction — Germany/Luxembourg (DE-LU) |
| Price type | Day-ahead hourly auction clearing price (€/MWh) |
| Zone | DE-LU bidding zone |
| Study period | 1 January 2018 — 31 December 2025 (8 years) |
| Resolution | Hourly (8,760 observations/year · ~70,080 total) |
| Currency | EUR/MWh |
| Source | ENTSO-E Transparency Platform (dataset: Day-ahead prices, code: A44) |
| ENTSO-E area code | 10Y1001A1001A82H (DE-LU) |

---

## Rationale for Study Period (2018–2025)

- **2018:** Post-nuclear phase-out stabilisation — coherent policy baseline
- **2019:** Pre-COVID reference year — normal demand patterns
- **2020–2021:** COVID demand shock + recovery — tests model robustness to structural breaks
- **2021–2022:** Gas price crisis — extreme price spikes, tests spike forecasting (H4 relevance)
- **2023–2025:** Post-crisis normalisation with high renewable penetration — captures current market dynamics

Eight years ensures sufficient data for robust walk-forward cross-validation while spanning multiple market regimes, directly supporting the regime identification feature (K-means clustering, Week 4).

---

## Covariates — Data Sources by Category

| Category | Variables | Source | Resolution | Access |
|---|---|---|---|---|
| Electricity price | Day-ahead price (€/MWh) | ENTSO-E (A44) | Hourly | Free, registration required |
| Load | Actual total load (MW) | ENTSO-E (A65) | Hourly | Free, registration required |
| Generation mix | Wind onshore, wind offshore, solar, hydro, gas, coal, nuclear (MW) | ENTSO-E (A75) | Hourly | Free, registration required |
| Weather | Temperature (°C), wind speed (m/s), solar irradiance (W/m²) | Open-Meteo API | Hourly | Free, no registration |
| Fuel prices | Natural gas (TTF), coal (API2), crude oil (Brent) | EIA / Investing.com / Quandl | Daily | Free (EIA); Quandl free tier |
| Calendar | Hour-of-day, day-of-week, month, season, public holiday indicator | Python `holidays` library | — | Open source |

---

## Scope Limitations

1. **Single bidding zone:** Results apply to DE-LU and should not be generalised to other European markets without revalidation.
2. **Day-ahead prices only:** Intraday and balancing market prices are excluded from the scope.
3. **Study period:** Findings reflect 2018–2025 market conditions. Structural changes post-2025 are not captured.
4. **Covariate availability:** Fuel price data is daily-resolution, upsampled to hourly via forward-fill — a known limitation noted in Ch.4.

---

## References

- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art. *International Journal of Forecasting*, 30(4), 1030–1081.
- Lago, J., De Ridder, F., & De Schutter, B. (2018). Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms. *Applied Energy*, 221, 386–405.
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu
