# Preprocessing Decisions Log

| Decision | Choice | Rationale | Citation |
|---|---|---|---|
| Missing values < 6h | Forward-fill | Standard for short gaps | Weron (2014) |
| Missing values > 24h | Flag + drop | Avoids long-imputation distortion | — |
| Spike detection | 3σ of 7-day rolling mean | Price spike literature | TBD W4 |
| Train/val/test | 70/15/15 walk-forward | Prevents temporal leakage | Lago et al. (2018) |
| Scaler (nets) | MinMaxScaler, fit on train only | LSTM/GRU stability | — |
| Scaler (trees) | StandardScaler, fit on train only | SVR convergence | — |
