# 1. Project explanation

## What is AI-based energy consumption forecasting?
**Simple language:** it is "weather forecasting for electricity use". We look at how much power a city used in the past (and the weather then), learn the patterns — people use more power in the evening, less on Sundays, much more when it is hot — and use them to predict how much will be needed tomorrow or next week.

**Technical language:** supervised time-series regression. Hourly load `y_t` is modelled as `f(calendar_t, weather_t, y_{t-24}, y_{t-48}, y_{t-168}, rolling statistics)`. We fit `f` with regression / ensemble models on a chronological training window, validate on a later block, score on a final unseen block, and produce multi-step forecasts recursively.

## What problem does it solve?
Power must be bought or generated *before* it is consumed and cannot be stored cheaply. Errors cost money both ways: under-forecasting forces emergency purchases (≈1.8× price in our simulation); over-forecasting leaves surplus sold at a discount (≈0.5×). Forecasting also drives capacity planning, maintenance windows and peak-demand management.

## Why it matters by sector
* **Smart cities** – plan street-lighting, EV charging and building loads; trigger demand-response in heat waves.
* **Electricity boards** – day-ahead power purchase agreements, generation scheduling, grid stability.
* **Manufacturing plants** – demand-charge tariffs punish a single high peak; forecasts let plants shift heavy processes.
* **Data centers** – cooling and compute loads are large and forecastable; power contracts are priced on expected usage.
* **Renewable companies** – compare demand with expected solar/wind output; size and schedule battery storage.

## How companies cut cost with forecasts
1. Buy day-ahead energy closer to actual need (less spot/penalty cost).
2. Flatten peaks by shifting flexible load (demand response) before a forecast peak.
3. Schedule maintenance in forecast low-demand periods.
4. Right-size contracts and storage.

## Workflow (as implemented)
| Stage | What happens | Code |
|---|---|---|
| Data collection | 3 years of hourly demand + weather (simulated, or your CSV) | `data_generator.py` |
| Preprocessing | de-duplicate, rebuild hourly index, remove spikes, interpolate gaps | `preprocessing.py` |
| Feature engineering | calendar, cyclical encodings, weather, cooling/heating degrees, lags ≥ 24 h, rolling stats | `features.py` |
| Model training | baseline + Linear Regression + Random Forest + Gradient Boosting | `models.py` |
| Forecasting | refit winner on all data, recursive 168-hour forecast | `forecasting.py` |
| Evaluation | RMSE, MAE, MAPE, R² on the last 90 days; 7-day backtest | `models.py`, `forecasting.py` |
| Visualization | 9 PNG graphs | `visualization.py` |
