# Source Code for Electricity Price Forecasting

## Code Architecture

### `data_pipeline/`
Data collection, preprocessing, and feature engineering modules.

**Data Collection**
- PJM API integration
- NOAA weather data fetching
- EIA energy data retrieval
- Automated data updates

**Preprocessing**
- Missing value imputation
- Outlier detection and handling
- Data normalization and scaling
- Time series alignment

**Feature Engineering**
- Lag feature creation
- Calendar features (holidays, seasons)
- Weather feature integration
- Load-demand interactions

### `models/`
Implementation of 11 machine learning models for comparison.

**Baseline Models**
- Linear Regression
- Ridge Regression
- Lasso Regression

**Tree-Based Models**
- Decision Tree
- Random Forest
- Gradient Boosting

**Other ML Models**
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)
- XGBoost

**Deep Learning**
- LSTM for temporal patterns

### `evaluation/`
Model evaluation and analysis frameworks.

**Metrics**
- MAE, RMSE, MAPE calculation
- Custom evaluation metrics
- Performance benchmarking

**Cross Validation**
- Time series cross-validation
- Rolling window validation
- Forward chaining

**Statistical Tests**
- Diebold-Mariano test
- Statistical significance testing
- Model comparison statistics

**Explainability**
- SHAP value computation
- Feature importance analysis
- Model interpretation tools

### `forecasting/`
Forecasting implementations for different horizons.

**Multi-Resolution**
- Hourly (24-hour ahead) forecasting
- Daily aggregated forecasting
- Horizon-specific optimization

**Volatility Analysis**
- High-volatility period detection
- Volatility-specific model evaluation
- Risk assessment metrics

### `utils/`
Utility functions and helpers.

**Configuration**
- Model parameter management
- Experiment configuration
- Data source settings

**Visualization**
- Time series plotting
- Model performance charts
- SHAP visualization tools

## Technology Stack
- **Python 3.11**
- **scikit-learn** for traditional ML models
- **XGBoost** for gradient boosting
- **Keras/TensorFlow** for LSTM
- **pandas** for data manipulation
- **SHAP** for explainability
- **matplotlib/seaborn** for visualization

## Code Standards
- PEP 8 compliance
- Type hints for functions
- Comprehensive docstrings
- Unit tests for critical functions
- Logging for debugging and monitoring