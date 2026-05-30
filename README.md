# ⚡ EnergyMarket-PriceCast

**Daily and Hourly Electricity Price Forecasting Using Machine Learning Approaches**

MSc Thesis Project · [University Name] · [Programme Name] · 2026

---

## Overview

This project builds a robust and accurate forecasting system for predicting day-ahead and hour-ahead electricity prices in competitive power markets using machine learning and deep learning methods.

Electricity prices are highly volatile and depend on nonlinear, time-varying factors including load demand, generation mix, fuel prices, weather conditions, and renewable energy penetration. Classical methods (ARIMA, SARIMA) fail to capture these relationships. This project applies:

- Random Forest (RF)
- XGBoost
- LightGBM
- Support Vector Regression (SVR)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Hybrid and ensemble models

## Research Questions

| ID | Question |
|---|---|
| RQ1 | Which ML model produces the most accurate electricity price forecasts? |
| RQ2 | Which input features have the greatest impact on electricity prices? |
| RQ3 | How much improvement can ML/DL methods provide over classical statistical models? |
| RQ4 | Can a single framework accurately predict both daily and hourly prices? |

## Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| H1 | ≥1 ML model significantly outperforms ARIMA/SARIMA | Diebold-Mariano, p < 0.05 |
| H2 | ≥1 DL model significantly outperforms best classical ML | Diebold-Mariano, p < 0.05 |
| H3 | Hybrid ensemble achieves lower RMSE than any single model | Diebold-Mariano, p < 0.05 |
| H4 | Domain features improve accuracy over statistical features alone | Ablation study |

## Project Structure

```
EnergyMarket-PriceCast/
├── data/
│   ├── raw/             # Raw data as downloaded — never modified
│   ├── processed/       # Cleaned, merged, feature-engineered data
│   └── external/        # External reference data (fuel prices, weather)
├── notebooks/           # Jupyter notebooks (EDA, experiments, reporting)
├── models/
│   ├── saved/           # Serialised trained models (.pkl, .pt)
│   └── checkpoints/     # Training checkpoints (LSTM/GRU)
├── src/
│   ├── data/            # Data loading and preprocessing modules
│   ├── features/        # Feature engineering pipeline
│   ├── models/          # Model definitions and training scripts
│   ├── evaluation/      # Metrics, DM test, ablation study
│   ├── visualization/   # Plotting and dashboard utilities
│   └── utils/           # Shared utilities (logging, config, paths)
├── tests/               # pytest unit tests
├── docs/                # Documentation, schema, decisions
├── configs/             # YAML configuration files
├── logs/                # Training and experiment logs
├── outputs/             # Generated figures, tables, exports
├── mlruns/              # MLflow experiment tracking
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Deliverables

| # | Deliverable | Target Week |
|---|---|---|
| D1 | Cleaned dataset | W4 |
| D2 | EDA notebook | W6 |
| D3 | Feature pipeline | W6 |
| D4 | Benchmark models | W7 |
| D5 | Best model card | W12 |
| D6 | Visualization dashboard | W12 |
| D7 | SHAP report | W12 |
| D8 | Thesis report | W16 |
| D9 | Journal article | W16 |
| D10 | EnergyMarket-PriceCast app | W16 |

## Setup

```bash
# Clone the repo
git clone https://github.com/[username]/EnergyMarket-PriceCast.git
cd EnergyMarket-PriceCast

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

## Evaluation Protocol

- Walk-forward cross-validation · 70/15/15 train/validation/test split
- Test set opened **once only**, after all model selection and tuning is complete
- Metrics: MAE, RMSE, MAPE, sMAPE
- Hypothesis testing: Diebold-Mariano test (α = 0.05)

## Data Sources

| Source | Data | Resolution |
|---|---|---|
| ENTSO-E Transparency Platform | Electricity prices, load, generation mix | Hourly |
| Open-Meteo API | Temperature, wind speed, solar irradiance | Hourly |
| EIA / Quandl | Gas, coal, oil prices | Daily |

## License

Academic use only · MSc Thesis · [University Name] · 2026
