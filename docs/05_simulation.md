# 5. Virtual simulation

## How the dataset represents real energy usage
`data_generator.py` reproduces what drives real feeder/substation load: a daily double peak (≈9:00 and ≈20:00), lower weekends/holidays, temperature-driven cooling load, yearly growth, noise — and the **messiness of real telemetry** (dropouts, spikes, missing timestamps, duplicate rows).

## How forecasting simulates real business decisions
Every day an energy retailer buys tomorrow's energy day-ahead. We "replay" the final 90 days: for each hour the planner buys the *forecast* amount at the day-ahead price; if actual demand is higher, the shortfall costs 1.8× on the spot market; if lower, the surplus is resold at 0.5×. (Prices are editable in `src/config.py`.) Three planners are compared: **Oracle** (perfect), **ML model**, **Naive** (same hour last week).

## Step-by-step
1. **Generate** raw data → `python main.py` (data/energy_raw.csv). *Proof:* `outputs/data_preview.txt`.
2. **Clean** it. *Proof:* `outputs/preprocessing_report.json`, `images/03_preprocessing_before_after.png`.
3. **Explore** patterns. *Graphs:* `01_consumption_overview`, `02_demand_patterns`.
4. **Train** baseline + 3 models. *Proof:* `outputs/training_log.txt`.
5. **Evaluate** on the last 90 days. *Proof:* `outputs/metrics.csv`; graphs `04`, `05`, `06`, `07`.
6. **Forecast** the next 7 days. *Proof:* `outputs/forecast_next_7_days.csv`, `images/08_future_forecast.png`.
7. **Backtest** 7-day forecasts at four past dates. *Proof:* `outputs/backtest_7day.json`.
8. **Simulate cost**: replay the test period with each planner. *Proof:* `outputs/simulation_report.json`, `images/09_cost_simulation.png`.
9. **Peak planning**: for each forecast day the report lists the peak hour/MWh and flags days above the historical 95th-percentile load (heat-wave alerts; none in a January forecast) plus the energy a 5 % demand-response shift would move.

## Proof to capture (for GitHub / interviews)
Terminal screenshot of the run, `metrics.csv`, graphs 05 + 08 + 09, passing `unittest` output, repo page.
