# 3. Architecture

```
          ┌───────────────┐
 CSV  ───▶│ INPUT DATA    │ timestamp, energy_mwh, temperature_c, humidity_pct
 /sim     └──────┬────────┘
                 ▼
          ┌───────────────┐  duplicates · missing hours · outliers · interpolation
          │ PREPROCESSING │  → data/energy_clean.csv, outputs/preprocessing_report.json
          └──────┬────────┘
                 ▼
          ┌───────────────┐  calendar · cyclical · weather · lags(24,48,72,168h) · rolling
          │ FEATURES      │  → X (23 columns), y
          └──────┬────────┘
                 ▼
          ┌───────────────┐  chronological split  train | validation(60d) | test(90d)
          │ MODEL TRAINING│  baseline + 3 models → pick best on validation
          └──────┬────────┘  → outputs/metrics.*, models/best_model.joblib
                 ▼
          ┌───────────────┐  refit winner on all data → predict hour by hour,
          │ FORECASTING   │  feeding predictions back as lags → forecast_next_7_days.csv
          └──────┬────────┘
        ┌────────┴────────┐
        ▼                 ▼
 ┌─────────────┐   ┌───────────────┐
 │ SIMULATION  │   │ VISUALIZATION │ images/*.png
 │ cost + peaks│   └───────────────┘
 └─────────────┘
```

## Modules
| Module | Responsibility |
|---|---|
| `config.py` | one place for paths, split sizes, lags, prices |
| `data_generator.py` | seeded simulator with deliberate data problems |
| `preprocessing.py` | returns clean frame + report dict |
| `features.py` | `make_features`, `build_dataset`, `FEATURE_COLUMNS` |
| `models.py` | metrics, split, models, selection, feature importance, save/load |
| `forecasting.py` | `recursive_forecast`, climatology weather, `recursive_backtest` |
| `simulation.py` | `procurement_cost`, `run_cost_simulation`, `peak_report` |
| `visualization.py` | one function per graph |
| `pipeline.py` | `run_pipeline()` ties everything together (used by CLI and tests) |

## Data flow & key design decisions
* **No leakage:** every history feature is ≥ 24 h old, matching a real day-ahead process (forecast at midnight for the next day).
* **Chronological split:** time series are never shuffled.
* **Validation picks, test reports:** the model is chosen on validation RMSE, so test scores stay unbiased.
* **Recursive forecast:** beyond 24 h, lags are earlier predictions; future weather = seasonal average (swap for a weather API in production).
