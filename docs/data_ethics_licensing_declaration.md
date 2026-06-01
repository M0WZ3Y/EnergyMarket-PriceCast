# Data Ethics and Licensing Declaration

**Project:** EnergyMarket-PriceCast — Daily and Hourly Electricity Price Forecasting Using Machine Learning  
**Author:** [Your Name]  
**Student ID:** [Your Student ID]  
**Programme:** MSc [Programme Name]  
**Institution:** [University Name]  
**Supervisor:** [Supervisor Name]  
**Date:** 20 May 2026  
**Version:** 1.0 — Final

---

## 1. Declaration Statement

I, [Your Name], declare that all data used in this research project has been obtained lawfully, is used in accordance with the terms and conditions of each respective data source, and complies with the ethical guidelines of [University Name] and applicable data protection regulations.

No personal data, sensitive data, or proprietary commercial data is used in this project without authorisation. All data sources are either publicly available under open licences or are accessed under free academic/research tiers with documented terms of use.

---

## 2. Data Sources — Ethics and Licensing Summary

### 2.1 ENTSO-E Transparency Platform

| Field | Detail |
|---|---|
| **Data collected** | Day-ahead electricity prices (A44), actual total load (A65), actual generation per production type (A75) |
| **Market / Zone** | EPEX SPOT DE-LU (Germany/Luxembourg), area code: 10Y1001A1001A82H |
| **Period** | 1 January 2018 — 31 December 2025 |
| **Resolution** | Hourly |
| **Access method** | ENTSO-E Transparency Platform API (registered account) |
| **Licence** | Open Data — free for academic and commercial use under ENTSO-E Terms of Use |
| **Terms URL** | https://transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/terms%20and%20conditions.html |
| **Personal data** | None — aggregated market data only |
| **Registration** | Required — academic email used |
| **Cost** | Free |
| **Citation** | ENTSO-E Transparency Platform (2026). Day-ahead prices, load, and generation data for DE-LU, 2018–2025. Retrieved from https://transparency.entsoe.eu |

**Ethical assessment:** No ethical concerns. Data is publicly mandated under EU Regulation 543/2013 and is designed for transparency and academic access.

---

### 2.2 Open-Meteo API

| Field | Detail |
|---|---|
| **Data collected** | Hourly temperature (°C), wind speed at 10m (m/s), shortwave solar radiation (W/m²) |
| **Location** | Representative grid point for DE-LU zone (approx. 51.5°N, 10.0°E — geographic centre of Germany) |
| **Period** | 1 January 2018 — 31 December 2025 |
| **Resolution** | Hourly |
| **Access method** | REST API — no registration required |
| **Licence** | Attribution 4.0 International (CC BY 4.0) |
| **Terms URL** | https://open-meteo.com/en/terms |
| **Personal data** | None |
| **Cost** | Free for non-commercial and academic use (≤10,000 calls/day) |
| **Citation** | Zippenfenig, P. (2023). Open-Meteo.com Weather API [open source]. Zenodo. https://doi.org/10.5281/zenodo.7970649 |

**Ethical assessment:** No ethical concerns. Openly licensed meteorological reanalysis data. Attribution provided in all outputs.

---

### 2.3 EIA (U.S. Energy Information Administration) — Fuel Prices

| Field | Detail |
|---|---|
| **Data collected** | Henry Hub natural gas spot price ($/MMBtu), Europe Brent crude oil spot price ($/barrel) |
| **Period** | 1 January 2018 — 31 December 2025 |
| **Resolution** | Daily (forward-filled to hourly — documented as limitation in Ch.4) |
| **Access method** | EIA Open Data API — free public access |
| **Licence** | U.S. Government Open Data — public domain (no copyright restriction) |
| **Terms URL** | https://www.eia.gov/about/copyrights_reuse.php |
| **Personal data** | None |
| **Cost** | Free |
| **Citation** | U.S. Energy Information Administration (2026). Natural gas and crude oil spot prices, 2018–2025. Retrieved from https://www.eia.gov/opendata |

**Ethical assessment:** No ethical concerns. U.S. federal government open data, explicitly released into the public domain.

---

### 2.4 Coal Prices — Investing.com / World Bank Commodity Data

| Field | Detail |
|---|---|
| **Data collected** | API2 Rotterdam coal price ($/tonne) or World Bank coal price index |
| **Period** | 1 January 2018 — 31 December 2025 |
| **Resolution** | Daily (forward-filled to hourly) |
| **Access method** | World Bank Commodity Markets data portal (preferred — open licence) |
| **Licence** | Creative Commons Attribution 4.0 (CC BY 4.0) — World Bank |
| **Terms URL** | https://datacatalog.worldbank.org/public-licenses |
| **Personal data** | None |
| **Cost** | Free |
| **Citation** | World Bank (2026). Commodity Markets data — coal prices, 2018–2025. Retrieved from https://www.worldbank.org/en/research/commodity-markets |
| **Note** | If API2 Rotterdam prices are required (more precise for European merit order), source from Quandl/Nasdaq Data Link free tier with academic registration. |

**Ethical assessment:** No ethical concerns. Open licence with attribution.

---

## 3. Personal Data Assessment

| Question | Answer |
|---|---|
| Does the project process any personal data (names, emails, locations, IDs)? | **No** |
| Does the project process any sensitive personal data (health, financial, political)? | **No** |
| Is any data collected directly from human participants? | **No** |
| Is any data scraped from social media or user-generated platforms? | **No** |
| Does the project require informed consent from any individual? | **No** |

**Conclusion:** This project processes exclusively aggregated market, meteorological, and commodity price data. No personal data is involved. GDPR and university data protection requirements are satisfied without further action.

---

## 4. Data Storage and Security

| Aspect | Approach |
|---|---|
| **Storage location** | Local machine + GitHub repository (private during development) |
| **Raw data** | Stored in `data/raw/` — excluded from git via `.gitignore` |
| **Processed data** | Stored in `data/processed/` — excluded from git |
| **API keys / credentials** | Stored in `.env` file — excluded from git via `.gitignore`; never committed |
| **Backup** | Raw data backed up to [University OneDrive / external drive] |
| **Retention** | Data retained for the duration of the thesis and 5 years post-submission per university policy |
| **Deletion** | API credentials revoked upon project completion |

---

## 5. Intellectual Property and Attribution

All data sources are attributed in:
- This declaration (Section 2)
- The thesis Ch.4 Data chapter (full citation with access date)
- The `docs/data_schema.md` file (column-level provenance)
- All figures and tables derived from external data (caption-level attribution)

No proprietary or commercially restricted datasets are used. No data redistribution is intended — raw data files are excluded from the public GitHub repository.

---

## 6. Compliance Checklist

| Requirement | Status |
|---|---|
| All data sources identified and documented | ✅ |
| Licences reviewed and confirmed compatible with academic use | ✅ |
| No personal data processed | ✅ |
| API credentials excluded from version control | ✅ |
| Raw data excluded from public repository | ✅ |
| Attribution provided for all CC-licensed sources | ✅ |
| University data ethics guidelines reviewed | ✅ |
| Supervisor informed of data sources | ✅ (via this declaration) |

---

## 7. Supervisor Sign-Off

| | |
|---|---|
| **Student signature:** | [Your Name] — 20 May 2026 |
| **Supervisor acknowledgement:** | [Supervisor Name] — [Date] |

---

*This declaration forms part of the project documentation and will be included as Appendix C of the final thesis.*
