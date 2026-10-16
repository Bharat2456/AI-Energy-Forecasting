# 4. Implementation plan (10 phases)

| # | Phase | What to do | Why | Expected output | Common mistakes | How to verify |
|---|---|---|---|---|---|---|
| 1 | Setup | Python 3.9+, venv, `pip install -r requirements.txt` | isolated, reproducible environment | `python -c "import sklearn"` works | forgetting to activate venv; old Python | `python --version`, `pip list` |
| 2 | Dataset loading | `python main.py --quick` generates `data/energy_raw.csv` (or use `--data`) | need hourly demand + weather | 26k-row CSV, `outputs/data_preview.txt` | wrong column names / timestamp format | open preview; clear error if columns missing |
| 3 | Data cleaning | dedupe, full hourly index, outlier removal, interpolation | models cannot handle NaN; spikes distort training | `energy_clean.csv`, `preprocessing_report.json`, graph 03 | interpolating before removing outliers; dropping rows (breaks lags) | zero NaNs after; consecutive 1-hour steps (unit tests check this) |
| 4 | Feature engineering | calendar, cyclical, weather, lags ≥ 24 h, rolling stats | gives the model the daily/weekly/seasonal memory | 23-column feature table | **leakage** (using lag_1 for a day-ahead model; rolling windows including the present) | `test_no_leakage_lags_are_past_values` |
| 5 | Model building | baseline + Linear/RandomForest/GradientBoosting, chronological split | compare simple vs. complex honestly | fitted models, `training_log.txt` | random `train_test_split` on time series | log shows train period ends before test starts |
| 6 | Evaluation | RMSE, MAE, MAPE, R² on test; 7-day backtest | quantify accuracy and prove the ML beats the baseline | `metrics.csv`, `backtest_7day.json`, graphs 04-07 | judging only on training data; ignoring the baseline | best model RMSE < baseline RMSE |
| 7 | Forecasting | refit best on all data; recursive 168 h forecast | operational output | `forecast_next_7_days.csv`, `best_model.joblib` | forgetting future weather; negative values | 168 rows, hourly, all > 0 |
| 8 | Visualization | 9 PNG graphs | communicate results; GitHub proof | `images/*.png` | unlabelled axes | open images |
| 9 | GitHub publishing | see `06_github_guide.md` | portfolio proof | public repo with README + images | committing `.venv`; no README images | repo page renders images |
| 10 | Final output | README results table, screenshots, LinkedIn/CV bullet | recruiters skim | polished repo | stale numbers in README after re-running | compare README table with `outputs/metrics.csv` |
