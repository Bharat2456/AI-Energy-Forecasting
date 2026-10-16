# 2. Tech-stack options

| | **A – Easiest** | **B – Intermediate** | **C – Advanced** |
|---|---|---|---|
| Tools | Python, pandas, NumPy, Matplotlib | + scikit-learn, Seaborn, joblib | + TensorFlow/PyTorch, Prophet/statsmodels, MLflow, FastAPI/Streamlit |
| Models | Linear Regression | Linear Reg., Random Forest, Gradient Boosting (+ lag features) | ARIMA/SARIMA, Prophet, LSTM/Transformer, quantile models |
| Difficulty | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ |
| GPU needed | No | No | Recommended for LSTM/Transformer |
| Expected output | one trend/forecast plot, R² | metrics table, model comparison, 7-day forecast, many graphs | prediction intervals, deployed API/dashboard, experiment tracking |
| Run time | seconds | ~1 minute | minutes–hours |

## Selected: **Option B**
For a student who wants quick execution, strong GitHub proof and clear understanding:
* runs on any laptop in ~1 minute, no GPU, no heavy installs;
* tree-based models + lag features routinely match or beat LSTMs on single-site hourly load, and are far easier to explain in interviews;
* gives enough depth (leakage-free features, baseline comparison, backtest, business simulation) to look like real industry work;
* Option C ideas (SARIMA, LSTM, dashboard) are listed as future work and can be added as new modules.
